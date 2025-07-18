import torch
import numpy as np
from pathlib import Path
import src.constants as const
import src.validation as val
import src.utils as utils
import yaml
import networkx as nx
from tqdm import tqdm
from pdgrapher import Dataset
from pdgrapher._models import PerturbationDiscoveryModel, GCNArgs
from pdgrapher._utils import get_thresholds


def load_perturbation_discovery_model(model_path, edge_index, device):
    """
    Load a trained perturbation discovery model from checkpoint.
    
    Args:
        model_path (str): Path to the saved perturbation_discovery.pt file
        edge_index (torch.Tensor): Edge index for the graph
        device (torch.device): Device to load the model on
    
    Returns:
        PerturbationDiscoveryModel: Loaded model ready for inference
    """
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint["model_state_dict"]
    
    # Extract model parameters from the saved weights
    embed_dim = state_dict["embed_layer_diseased.bias"].shape[1]
    num_vars = state_dict["positional_embeddings.weight"].shape[0]
    dim_gnn = state_dict["mlp.0.bias"].shape[0]
    
    # Determine number of GNN layers
    n_layers_gnn = 1
    if "convs.1.bias" in state_dict:
        n_layers_gnn = 2
    if "convs.2.bias" in state_dict:
        n_layers_gnn = 3
    
    # Create GCNArgs with the inferred parameters
    args = GCNArgs(
        positional_features_dims=16,
        embedding_layer_dim=embed_dim,
        dim_gnn=dim_gnn,
        num_vars=num_vars,
        n_layers_gnn=n_layers_gnn,
        n_layers_nn=2
    )
    
    # Initialize and load the model
    model = PerturbationDiscoveryModel(args, edge_index)
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Ensure edge_index is on the correct device
    if hasattr(model, 'edge_index'):
        model.edge_index = model.edge_index.to(device)
    
    model.eval()
    
    print(f"Loaded PerturbationDiscoveryModel:")
    print(f"  - Number of variables: {num_vars}")
    print(f"  - Embedding dimension: {embed_dim}")
    print(f"  - GNN dimension: {dim_gnn}")
    print(f"  - GNN layers: {n_layers_gnn}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model


def get_predictions(model, data_loader, thresholds, device):
    """
    Make perturbation predictions using the loaded model.
    
    Args:
        model: Trained PerturbationDiscoveryModel
        data_loader: DataLoader with test data
        thresholds: Dictionary of thresholds for input processing
        device: Device for computation
    
    Returns:
        tuple: (predictions, true_sources) - predictions as logits, true_sources as node indices
    """
    model.eval()
    
    predictions = []
    true_sources = []
    
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            # Prepare input data - ensure all tensors are on the same device
            diseased = data.diseased.to(device).view(-1, 1)
            treated = data.treated.to(device).view(-1, 1)
            batch = data.batch.to(device)
            mutations = data.mutations.to(device) if hasattr(data, 'mutations') else None
            
            # Make sure edge_index is on the correct device
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                data.edge_index = data.edge_index.to(device)
            
            # Extract true source for this sample
            true_source_tensor = torch.where(data.intervention == 1)[0]
            if len(true_source_tensor) > 0:
                true_source = int(true_source_tensor[0].cpu())
                true_sources.append(true_source)
            else:
                print(f"WARNING: No true source found in sample {i}")
                true_sources.append(-1)  # placeholder for missing source
            
            # Predict interventions using the perturbation discovery model
            intervention_logits = model(
                torch.cat([diseased, treated], dim=1),
                batch,
                mutilate_mutations=mutations,
                threshold_input=thresholds
            )
            
            intervention_logits = intervention_logits.flatten()
            predictions.append(intervention_logits)
            
            # Progress indicator for large datasets
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1} samples...")
    
    print(f"Completed prediction for {len(predictions)} samples")
    return predictions, true_sources



def distance_metrics(true_sources, pred_sources, edge_index) -> dict:
    """
    Calculate the average distance between predicted and true source nodes.
    
    Args:
        true_sources: list of true source node indices
        pred_sources: list of predicted source node indices
        edge_index: edge index tensor for the graph
        
    Returns:
        dict: dictionary with the average distance to the source
    """
    dists_to_source = []

    for i, true_source in enumerate(
        tqdm(true_sources, desc="calculate distances", disable=const.ON_CLUSTER)
    ):
        pred_source = pred_sources[i]
        if true_source == pred_source:
            dists_to_source.append(0)
        else:
            # get the graph from the data instance
            nx_graph = nx.from_edgelist(edge_index.t().tolist())

            # calculate the shortest path distance
            try:
                dist = nx.shortest_path_length(nx_graph, source=true_source, target=pred_source)
            except nx.NetworkXNoPath:
                dist = float("inf")
            dists_to_source.append(dist)

    return {
        "avg dist to source": np.mean(dists_to_source),
    }


def supervised_metrics(
    pred_label_set: list,
    true_sources: list,
    pred_sources: list,
    edge_index: torch.Tensor,
) -> dict:
    """
    Performs supervised evaluation metrics for models that predict whether each node is a source or not.
    
    Args:
        pred_label_set: list of predicted labels for each data instance
        true_sources: list of true source node indices
        pred_sources: list of predicted source node indices
        edge_index: edge index tensor for distance calculations
        
    Returns:
        dict: dictionary containing the evaluation metrics
    """
    metrics = {}

    print("Evaluating Model ...")

    # Graph-level metrics (source detection)
    metrics |= val.prediction_metrics(pred_label_set, true_sources)
    metrics |= distance_metrics(true_sources, pred_sources, edge_index)
    metrics |= val.TP_FP_metrics(true_sources, pred_sources)

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = round(value, 3)
        print(f"{key}: {metrics[key]}")

    return metrics


def main():
    assert const.MODEL == "pdgrapher", "This script is only for PDGrapher perturbation discovery validation."
    model_path = "examples/PDGrapher/tunedperturbation_discovery.pt"
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    edge_index = torch.load(const.PROCESSED_EDGE_INDEX_PATH)
    model = load_perturbation_discovery_model(model_path, edge_index, device)
    data_path = Path(const.PROCESSED_PATH)
    
    # Load dataset
    print("Loading dataset...")
    dataset = Dataset(
        forward_path=str(data_path / "data_forward.pt"),
        backward_path=str(data_path / "data_backward.pt"),
        splits_path=str(data_path / "splits" / "splits.pt")
    )
    
    # Get thresholds (needed for model input processing)
    thresholds = get_thresholds(dataset)
    thresholds = {k: torch.tensor(v).to(device) for k, v in thresholds.items()}
    
    # Get test data
    print("Preparing test data...")
    (train_loader_forward, train_loader_backward,
        val_loader_forward, val_loader_backward,
        test_loader_forward, test_loader_backward
    ) = dataset.get_dataloaders(batch_size=1)
    
    print("Running inference on test data...")
    predictions, true_sources = get_predictions(model, test_loader_backward, thresholds, device)
    pred_sources = [int(pred.argmax().cpu().numpy()) for pred in predictions]
    
    print("Evaluating predictions...")
    metrics_dict = {}
    metrics_dict["network"] = const.NETWORK
    metrics_dict["metrics"] = supervised_metrics(
        predictions, true_sources, pred_sources, edge_index
    )
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))
    
    print("Saving results...")
    utils.save_metrics(metrics_dict, const.MODEL_NAME, const.NETWORK)


if __name__ == "__main__":
    main()
