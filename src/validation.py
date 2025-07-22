""" Unified validation script for both PDGrapher and GAT models. """
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score
from pathlib import Path

import src.constants as const
from src import utils
# Model imports
from architectures.GAT import GAT
try:
    from pdgrapher import Dataset
    from pdgrapher._models import PerturbationDiscoveryModel, GCNArgs
    from pdgrapher._utils import get_thresholds
    PDGRAPHER_AVAILABLE = True
except ImportError:
    PDGRAPHER_AVAILABLE = False
    print("Warning: PDGrapher not available. PDGrapher validation will be skipped.")


class ModelValidator:
    """Unified validator for different model types."""
    
    def __init__(self, model_type: str, model_name: str = None):
        self.model_type = model_type.lower()
        self.model_name = model_name or utils.latest_model_name()
        self.model_path = f"{const.MODEL_PATH}/{const.MODEL}/{self.model_name}_latest.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing validator for {self.model_type} using device: {self.device}")
        
    def load_model(self):
        """Load the appropriate model based on model type."""
        if self.model_type == "gat":
            return self._load_gat_model()
        elif self.model_type == "pdgrapher":
            return self._load_pdgrapher_model()
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _load_gat_model(self):
        """Load GAT model using the new saving format."""
        model = GAT()
        print(f"Loading GAT model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model with config: {checkpoint.get('model_config', {})}")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_pdgrapher_model(self):
        """Load PDGrapher perturbation discovery model."""
        if not PDGRAPHER_AVAILABLE:
            raise ImportError("PDGrapher is not available. Please install it first.")
            
        edge_index = torch.load(const.PROCESSED_EDGE_INDEX_PATH, map_location=self.device)
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
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
        model.to(self.device)
        model.eval()
        
        print(f"Loaded PerturbationDiscoveryModel with {sum(p.numel() for p in model.parameters()):,} parameters")
        return model
    
    def load_test_data(self):
        """Load test data based on model type."""
        if self.model_type == "gat":
            return self._load_gat_test_data()
        elif self.model_type == "pdgrapher":
            return self._load_pdgrapher_test_data()
    
    def _load_gat_test_data(self):
        """Load test data for GAT model."""
        # Load processed test data
        processed_test_data = utils.load_processed_data(split="test")
        
        # Load corresponding raw data for validation metrics
        raw_test_data = self._load_raw_test_data_from_indices()
        
        return raw_test_data, processed_test_data
    
    def _load_raw_test_data_from_indices(self):
        """Load raw test data based on split indices."""
        processed_dir = Path(f"data/processed/{const.MODEL}")
        splits = torch.load(processed_dir / "splits" / "splits.pt", weights_only=False)
        test_indices = splits["test_index_backward"]
        
        raw_data_dir = Path("data/raw")
        raw_files = sorted(list(raw_data_dir.glob("*.pt")))
        
        raw_test_data = []
        for idx in test_indices:
            raw_data = torch.load(raw_files[idx], weights_only=False)
            raw_test_data.append(raw_data)
        
        return raw_test_data
    
    def _load_pdgrapher_test_data(self):
        """Load test data for PDGrapher model."""
        data_path = Path(const.PROCESSED_PATH)
        
        dataset = Dataset(
            forward_path=str(data_path / "data_forward.pt"),
            backward_path=str(data_path / "data_backward.pt"),
            splits_path=str(data_path / "splits" / "splits.pt")
        )
        
        # Get test data loader
        (_, _, _, _, _, test_loader_backward) = dataset.get_dataloaders(batch_size=1)
        
        return dataset, test_loader_backward
    
    def get_predictions(self, model, test_data):
        """Get model predictions based on model type."""
        if self.model_type == "gat":
            return self._get_gat_predictions(model, test_data[1])  # processed_test_data
        elif self.model_type == "pdgrapher":
            return self._get_pdgrapher_predictions(model, test_data[0], test_data[1])  # dataset, test_loader
    
    def _get_gat_predictions(self, model, processed_test_data):
        """Get predictions from GAT model."""
        predictions = []
        
        for data in tqdm(processed_test_data, desc="GAT predictions", disable=const.ON_CLUSTER):
            data = data.to(self.device)
            with torch.no_grad():
                pred = model(data).detach().squeeze()
                pred_probs = torch.sigmoid(pred)
            predictions.append(pred_probs.cpu())
        
        return predictions
    
    def _get_pdgrapher_predictions(self, model, dataset, test_loader):
        """Get predictions from PDGrapher model."""
        # Get thresholds for model input processing
        thresholds = get_thresholds(dataset)
        thresholds = {k: torch.tensor(v).to(self.device) for k, v in thresholds.items()}
        
        predictions = []
        
        with torch.no_grad():
            for _, data in enumerate(tqdm(test_loader, desc="PDGrapher predictions", disable=const.ON_CLUSTER)):
                # Move all relevant tensors to the correct device
                for attr in ['diseased', 'treated', 'batch', 'mutations', 'edge_index', 'intervention']:
                    if hasattr(data, attr):
                        tensor = getattr(data, attr)
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            setattr(data, attr, tensor.to(self.device))
                diseased = data.diseased.view(-1, 1)
                treated = data.treated.view(-1, 1)
                batch = data.batch
                mutations = data.mutations if hasattr(data, 'mutations') else None

                # Predict interventions
                intervention_logits = model(
                    torch.cat([diseased, treated], dim=1),
                    batch,
                    mutilate_mutations=mutations,
                    threshold_input=thresholds
                )

                intervention_logits = intervention_logits.flatten()
                predictions.append(intervention_logits.cpu())
        
        return predictions
    
    def extract_true_sources(self, test_data):
        """Extract true source nodes based on model type."""
        if self.model_type == "gat":
            return self._extract_gat_true_sources(test_data[1])  # processed_test_data
        elif self.model_type == "pdgrapher":
            return self._extract_pdgrapher_true_sources(test_data[1])  # test_loader
    
    def _extract_gat_true_sources(self, processed_test_data):
        """Extract true sources from GAT processed data."""
        true_sources = []
        for data in processed_test_data:
            source_node = torch.where(data.y == 1)[0][0].item()
            true_sources.append(source_node)
        return true_sources
    
    def _extract_pdgrapher_true_sources(self, test_loader):
        """Extract true sources from PDGrapher data."""
        true_sources = []
        for i, data in enumerate(test_loader):
            # Ensure intervention is on CPU for indexing
            intervention = data.intervention
            if intervention.device != torch.device("cpu"):
                intervention = intervention.cpu()
            true_source_tensor = torch.where(intervention == 1)[0]
            if len(true_source_tensor) > 0:
                true_source = int(true_source_tensor[0])
                true_sources.append(true_source)
            else:
                print(f"WARNING: No true source found in sample {i}")
                true_sources.append(-1)
        return true_sources
    
    def get_raw_data_for_stats(self, test_data):
        """Get raw data for statistics calculation."""
        if self.model_type == "gat":
            return test_data[0]  # raw_test_data
        elif self.model_type == "pdgrapher":
            # For PDGrapher, we need to load raw data separately
            return self._load_raw_test_data_from_indices()


def distance_metrics(true_sources, pred_sources) -> dict:
    """Calculate distance metrics between predicted and true sources."""
    dists_to_source = []


    for i, true_source in enumerate(tqdm(true_sources, desc="calculate distances", disable=const.ON_CLUSTER)):
        pred_source = pred_sources[i]
        if true_source == pred_source:
            dists_to_source.append(0)
        else:
            # Create NetworkX graph from edge index
            G, gene_to_idx = utils.get_graph_data_from_topo()
            idx_to_gene = {v: k for k, v in gene_to_idx.items()}
            true_source = idx_to_gene[true_source]
            pred_source = idx_to_gene[pred_source]
            
            try:
                dist = nx.shortest_path_length(G, source=true_source, target=pred_source)
            except nx.NetworkXNoPath:
                dist = float("inf")
            dists_to_source.append(dist)

    return {"avg dist to source": np.mean(dists_to_source)}


def TP_FP_metrics(true_sources: list, pred_sources: list) -> dict:
    """Calculate true positive and false positive rates."""
    true_positive = 0
    false_positive = 0

    for true_source, pred_source in zip(true_sources, pred_sources):
        if true_source == pred_source:
            true_positive += 1
        else:
            false_positive += 1

    total_instances = len(true_sources)
    true_positive_rate = true_positive / total_instances
    false_positive_rate = false_positive / total_instances
    
    f1 = f1_score(true_sources, pred_sources, average='weighted')

    return {
        "true positive rate": true_positive_rate,
        "false positive rate": false_positive_rate,
        "f1 score": f1,
    }


def prediction_metrics(pred_label_set: list, true_sources: list) -> dict:
    """Calculate prediction ranking and probability metrics."""
    source_ranks = []
    predictions_for_source = []
    general_predictions = []
    in_top3 = []
    in_top5 = []

    for i, pred_labels in enumerate(tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)):
        true_source = true_sources[i]
        ranked_predictions = utils.ranked_source_predictions(pred_labels).tolist()

        source_rank = ranked_predictions.index(true_source)
        source_ranks.append(source_rank)
        predictions_for_source.append(pred_labels[true_source].item())
        general_predictions += pred_labels.tolist()

        in_top3.append(source_rank < 3)
        in_top5.append(source_rank < 5)

    return {
        "avg rank of source": np.mean(source_ranks),
        "avg prediction for source": np.mean(predictions_for_source),
        "avg prediction over all nodes": np.mean(general_predictions),
        "min prediction over all nodes": min(general_predictions),
        "max prediction over all nodes": max(general_predictions),
        "source in top 3": np.mean(in_top3),
        "source in top 5": np.mean(in_top5),
    }


def node_classification_metrics(pred_label_set: list, processed_data: list = None) -> dict:
    """Calculate node-level binary classification metrics."""
    # For GAT models, we can calculate node-level metrics
    if processed_data is None:
        return {}
    
    all_preds = []
    all_labels = []
    
    for i, (pred_probs, data) in enumerate(zip(pred_label_set, processed_data)):
        if isinstance(pred_probs, torch.Tensor):
            pred_list = pred_probs.flatten().tolist()
        else:
            pred_list = pred_probs
        all_preds.extend(pred_list)
        
        if isinstance(data.y, torch.Tensor):
            label_list = data.y.flatten().int().tolist()
        else:
            label_list = data.y
        all_labels.extend(label_list)
    
    # Calculate metrics
    unique_labels = set(all_labels)
    if len(unique_labels) > 1:
        auc_roc = roc_auc_score(all_labels, all_preds)
    else:
        auc_roc = 0.0
    
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    precision = precision_score(all_labels, binary_preds, average='binary', zero_division=0)
    recall = recall_score(all_labels, binary_preds, average='binary', zero_division=0)
    f1_binary = f1_score(all_labels, binary_preds, average='binary', zero_division=0)
    
    return {
        "node_auc_roc": auc_roc,
        "node_precision": precision,
        "node_recall": recall,
        "node_f1": f1_binary,
        "total_nodes": len(all_labels),
        "positive_nodes": sum(all_labels),
        "predicted_positive": sum(binary_preds)
    }


def data_stats(raw_data_set: list) -> dict:
    """Calculate data statistics."""
    n_nodes = []
    n_sources = []
    n_nodes_affected = []
    percent_affected = []
    
    for data in tqdm(raw_data_set, desc="get data stats"):
        n_nodes = data.num_nodes
        indicator = torch.tensor(data.binary_perturbation_indicator)
        diff = torch.tensor(data.difference) 
        
        n_sources.append(len(torch.where(indicator == 1)[0].tolist()))
        n_nodes_affected.append(len(torch.where(diff != 0)[0].tolist()))
        percent_affected.append(n_nodes_affected[-1] / n_nodes)

    stats = {
        "graph stats": {
            "number of nodes": n_nodes,
        },
        "infection stats": {
            "avg number of sources": round(float(np.mean(n_sources)), 3),
            "avg portion of affected nodes": round(float(np.mean(percent_affected)), 3),
            "std portion of affected nodes": round(float(np.std(percent_affected)), 3),
        },
    }

    return stats


def supervised_metrics(
    pred_label_set: list,
    true_sources: list,
    pred_sources: list,
    processed_data: list = None,
    model_type: str = "gat"
) -> dict:
    """Perform supervised evaluation metrics."""
    metrics = {}

    print("Evaluating Model ...")

    # Common metrics for all models
    metrics.update(prediction_metrics(pred_label_set, true_sources))
    metrics.update(distance_metrics(true_sources, pred_sources))
    metrics.update(TP_FP_metrics(true_sources, pred_sources))
    
    # Node-level metrics only for GAT (since it does node classification)
    if model_type.lower() == "gat" and processed_data is not None:
        metrics.update(node_classification_metrics(pred_label_set, processed_data))

    # Round numerical values
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            metrics[key] = round(value, 3)
        print(f"{key}: {metrics[key]}")

    return metrics


def main():
    """Main validation function."""

    model_type = const.MODEL.lower()
    network = const.NETWORK
    model_name = const.MODEL_NAME
    print(f"Starting validation for {model_type} model on {network} network")
    
    # Initialize validator
    validator = ModelValidator(model_type, model_name)
    
    # Load model and data
    model = validator.load_model()
    test_data = validator.load_test_data()
    
    # Get predictions
    pred_labels = validator.get_predictions(model, test_data)
    
    # Extract true and predicted sources
    true_sources = validator.extract_true_sources(test_data)
    pred_sources = [pred.argmax().item() for pred in pred_labels]
    
    # Get raw data for statistics
    raw_test_data = validator.get_raw_data_for_stats(test_data)
    
    # Determine processed data for node-level metrics (GAT only)
    processed_data = test_data[1] if model_type == "gat" else None
    
    # Calculate metrics
    metrics_dict = {
        "network": network,
        "model_type": model_type,
        "metrics": supervised_metrics(
            pred_labels, true_sources, pred_sources, processed_data, model_type
        ),
        "data stats": data_stats(raw_test_data),
        "parameters": yaml.full_load(open("params.yaml", "r"))
    }
    
    # Save results
    model_save_name = validator.model_name
    utils.save_metrics(metrics_dict, model_save_name, network)
    
    print(f"Validation complete! Results saved for {model_type} model: {model_save_name}")


if __name__ == "__main__":
    main()