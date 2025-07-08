#!/usr/bin/env python3
"""
Validation script for PDGrapher Perturbation Discovery Model
This script loads a trained perturbation discovery model and evaluates it on test data.
"""

import torch
import numpy as np
from pathlib import Path
import src.constants as const
import src.validation as val
import src.utils as utils
import yaml
import networkx as nx
from tqdm import tqdm

# Import PDGrapher components
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
    # Load checkpoint to inspect the saved model architecture
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters from the saved weights
    # Based on the training code, we can infer the parameters from layer sizes
    state_dict = checkpoint["model_state_dict"]
    
    # Get dimensions from the saved state dict
    embed_dim = state_dict["embed_layer_diseased.bias"].shape[1]  # embedding dimension
    num_vars = state_dict["positional_embeddings.weight"].shape[0]  # number of variables/genes
    dim_gnn = state_dict["mlp.0.bias"].shape[0]  # GNN hidden dimension
    
    # Check if there are multiple GNN layers
    n_layers_gnn = 1
    if "convs.1.bias" in state_dict:
        n_layers_gnn = 2
    if "convs.2.bias" in state_dict:
        n_layers_gnn = 3
    
    # Create GCNArgs with the correct parameters
    args = GCNArgs(
        positional_features_dims=16,  # This is typically consistent
        embedding_layer_dim=embed_dim,
        dim_gnn=dim_gnn,
        num_vars=num_vars,
        n_layers_gnn=n_layers_gnn,
        n_layers_nn=2  # This is typically 2 based on the training code
    )
    
    # Initialize the model
    model = PerturbationDiscoveryModel(args, edge_index)
    
    # Load the trained weights
    model.load_state_dict(state_dict)
    model.to(device)
    
    # Explicitly move edge_index to device - this is often the source of CUDA errors
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
        dict: Dictionary containing predictions and ground truth
    """
    model.eval()
    
    predictions = []
    
    with torch.no_grad():
        for data in data_loader:
            # Prepare input data - ensure ALL tensors are on the same device
            diseased = data.diseased.to(device).view(-1, 1)
            treated = data.treated.to(device).view(-1, 1)
            batch = data.batch.to(device)
            mutations = data.mutations.to(device) if hasattr(data, 'mutations') else None
            
            # Make sure edge_index is on the correct device (this is often the culprit)
            if hasattr(data, 'edge_index') and data.edge_index is not None:
                data.edge_index = data.edge_index.to(device)
            
            # Predict interventions using the perturbation discovery model
            intervention_logits = model(
                torch.cat([diseased, treated], dim=1),
                batch,
                mutilate_mutations=mutations,
                threshold_input=thresholds
            )
            intervention_logits = intervention_logits.flatten()
            predictions.append(intervention_logits)
            
    
    return predictions


def extract_true_sources(data_loader, device):
    """
    Extract true sources from the data loader.
    
    Args:
        data_loader: DataLoader with test data
        device: Device for computation
    
    Returns:
        list: List of true source genes
    """
    true_sources = []
    
    for data in data_loader:
        intervention = data.intervention.to(device)
        source_node = torch.where(intervention == 1)[0]
        true_sources.append(int(source_node[0].cpu()))
    
    return true_sources


def distance_metrics(true_sources, pred_sources, edge_index) -> dict:
    """
    Get the average min matching distance and the average distance to the source in general.
    :param true_sources: list of true source node indices
    :param pred_sources: list of predicted source node indices
    :param data_set: list of data instances containing true labels
    :return: dictionary with the average minimum matching distance and average distance to the source
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
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param raw_data_set: the raw data set containing true labels
    :param processed_data_set: the processed data set with PyTorch Geometric format
    :param true_sources: list of true source node indices
    :param pred_sources: list of predicted source node indices
    :return: dictionary containing the evaluation metrics
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
    model_path = "examples/PDGrapher/tunedperturbation_discovery.pt"
    data_path = Path(const.DATA_PATH) / const.MODEL / "processed"
    edge_index_path = data_path / "edge_index.pt"
    

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    data_path = Path(data_path)
    edge_index_path = edge_index_path
    print(f"Loading edge index from: {edge_index_path}")
    edge_index = torch.load(edge_index_path)
    
    print(f"Loading model from: {model_path}")
    model = load_perturbation_discovery_model(model_path, edge_index, device)
    
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
    test_split = dataset.splits["test_index_backward"]
    (train_loader_forward, train_loader_backward,
        val_loader_forward, val_loader_backward,
        test_loader_forward, test_loader_backward
    ) = dataset.get_dataloaders(batch_size=1) # TODO batchsize is a quickfix

    predictions = get_predictions(model, test_loader_backward, thresholds, device)
    
    true_sources = extract_true_sources(test_loader_backward, device)
    pred_sources = [int(pred.argmax().cpu().numpy()) for pred in predictions]

    metrics_dict = {}
    metrics_dict["network"] = const.NETWORK
    metrics_dict["metrics"] = supervised_metrics(
        predictions, true_sources, pred_sources, edge_index
    )
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))
    utils.save_metrics(metrics_dict, const.MODEL_NAME, const.NETWORK)


if __name__ == "__main__":
    main()
