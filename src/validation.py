""" Unified validation script for both PDGrapher and GAT models. """
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from pathlib import Path

import src.constants as const
from src import utils
# Model imports
from architectures.GAT import GAT
from pdgrapher import Dataset
from pdgrapher._models import PerturbationDiscoveryModel, GCNArgs
from pdgrapher._utils import get_thresholds


class ModelValidator:
    """Unified validator for different model types."""
    
    def __init__(self, model_type: str, model_name: str = None):
        self.model_type = model_type.lower()
        self.model_name = model_name
        self.model_path = f"{const.MODEL_PATH}/{const.MODEL}/{self.model_name}_latest.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize cached true sources for PDGrapher
        self._pdgrapher_true_sources = None
        
        print(f"Initializing validator for {self.model_type}")
        print(f"Model identifier: {self.model_name}")
        print(f"Model path: {self.model_path}")
        print(f"Using device: {self.device}")
        
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
        
        positional_features_dim = state_dict["positional_embeddings.weight"].shape[1]
        

        args = GCNArgs(
            positional_features_dims=positional_features_dim,
            embedding_layer_dim=embed_dim,
            dim_gnn=dim_gnn,
            num_vars=num_vars,
            n_layers_gnn=n_layers_gnn,
            n_layers_nn=const.LAYERS,
        )
        
        print(f"Model config (from saved weights): positional_features_dim={positional_features_dim}, "
              f"embedding_layer_dim={embed_dim}, dim_gnn={dim_gnn}, num_vars={num_vars}, "
              f"n_layers_gnn={n_layers_gnn}")
        
        # Initialize and load the model
        model = PerturbationDiscoveryModel(args, edge_index)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Loaded PerturbationDiscoveryModel with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Model configuration loaded from checkpoint: {checkpoint.get('model_config', 'Not available')}")
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
        splits = torch.load(const.SPLITS_PATH, weights_only=False)
        test_indices = splits["test_index_backward"]
        
        raw_data_dir = Path(const.RAW_PATH)
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
            splits_path=str(const.SPLITS_PATH)
        )
        # Get test data loader with reduced workers for cluster compatibility
        (_, _, _, _, _, test_loader_backward) = dataset.get_dataloaders(batch_size=1, num_workers=0)

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
        # Load thresholds directly from processed data directory
        thresholds_path = Path(const.PROCESSED_PATH) / "thresholds.pt"

        thresholds = torch.load(thresholds_path, weights_only=False)
        # Ensure thresholds are on the correct device
        thresholds = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(v)).to(self.device) 
                     for k, v in thresholds.items()}
        
        print(f"Loaded thresholds with keys: {list(thresholds.keys())}")
        
        predictions = []
        true_sources = []
        
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, desc="PDGrapher predictions", disable=const.ON_CLUSTER)):
                # Extract true source first (before moving to device)
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

                # Predict interventions using the correct threshold structure for PDGrapher
                # PDGrapher model expects threshold_input with "diseased" and "treated" keys
                threshold_input = {
                    "diseased": thresholds["diseased"],
                    "treated": thresholds["treated"]
                }
                intervention_logits = model(
                    torch.cat([diseased, treated], dim=1),
                    batch,
                    mutilate_mutations=mutations,
                    threshold_input=threshold_input
                )

                intervention_logits = intervention_logits.flatten()
                predictions.append(intervention_logits.cpu())
        
        self._pdgrapher_true_sources = true_sources
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
        # True sources were already extracted during prediction phase
        # to ensure exact same data ordering
        if hasattr(self, '_pdgrapher_true_sources'):
            return self._pdgrapher_true_sources
        else:
            # Fallback - but this shouldn't happen with the fixed code
            print("WARNING: True sources not found from prediction phase, falling back to separate extraction")
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
    precision = precision_score(true_sources, pred_sources, average='weighted', zero_division=0)
    recall = recall_score(true_sources, pred_sources, average='weighted', zero_division=0)
    f1 = f1_score(true_sources, pred_sources, average='weighted')

    return {
        "true positive rate": true_positive_rate,
        "false positive rate": false_positive_rate,
        "precision": precision,
        "recall": recall,
        "f1 score": f1,
    }


def prediction_metrics(pred_label_set: list, true_sources: list) -> dict:
    """Calculate prediction ranking and probability metrics."""
    print("Calculating prediction ranking and probabilities ...")
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
        "accuracy": np.mean(np.array(source_ranks) == 0),
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
            "number of possible sources": data.num_possible_sources,
        },
        "infection stats": {
            "avg number of sources": round(float(np.mean(n_sources)), 3),
            "avg portion of affected nodes": round(float(np.mean(percent_affected)), 3),
            "std portion of affected nodes": round(float(np.std(percent_affected)), 3),
        },
    }

    return stats


def gene_specific_metrics(
    pred_label_set: list,
    true_sources: list,
    pred_sources: list,
    genes_of_interest: list,
    gene_mapping: dict = None,
    raw_test_data: list = None
) -> dict:
    """Calculate metrics specifically for genes of interest."""
    if not genes_of_interest or not gene_mapping:
        return {}
    
    # Create reverse mapping from index to gene name
    idx_to_gene = {v: k for k, v in gene_mapping.items()}
    
    # Filter for genes of interest that exist in the mapping
    valid_genes = [gene for gene in genes_of_interest if gene in gene_mapping]
    if not valid_genes:
        print(f"Warning: None of the genes of interest {genes_of_interest} found in gene mapping")
        return {}
    
    gene_metrics_dict = {}
    
    for gene_name in valid_genes:
        gene_idx = gene_mapping[gene_name]
        gene_metrics = {
            "gene_name": gene_name,
            "gene_index": gene_idx,
            "total_samples": len(true_sources),
            "times_true_source": 0,
            "times_predicted_source": 0,
            "correct_predictions": 0,
            "avg_prediction_prob": 0.0,
            "avg_rank": 0.0,
            "prediction_probs_when_true": [],
            "ranks_when_true": [],
            "in_top3_count": 0,
            "in_top5_count": 0,
        }
        
        # Collect all prediction probabilities for this gene across all samples
        all_probs_for_gene = []
        
        for i, (pred_probs, true_source, pred_source) in enumerate(zip(pred_label_set, true_sources, pred_sources)):
            # Get prediction probability for this gene
            if isinstance(pred_probs, torch.Tensor):
                gene_prob = pred_probs[gene_idx].item()
            else:
                gene_prob = pred_probs[gene_idx]
            
            all_probs_for_gene.append(gene_prob)
            
            # Check if this gene was the true source
            if true_source == gene_idx:
                gene_metrics["times_true_source"] += 1
                gene_metrics["prediction_probs_when_true"].append(gene_prob)
                
                # Calculate rank for this gene
                if isinstance(pred_probs, torch.Tensor):
                    ranked_predictions = utils.ranked_source_predictions(pred_probs).tolist()
                else:
                    # Convert to tensor for ranking function
                    pred_tensor = torch.tensor(pred_probs)
                    ranked_predictions = utils.ranked_source_predictions(pred_tensor).tolist()
                
                gene_rank = ranked_predictions.index(gene_idx)
                gene_metrics["ranks_when_true"].append(gene_rank)
                
                # Check if correctly predicted
                if pred_source == gene_idx:
                    gene_metrics["correct_predictions"] += 1
            
            # Check if this gene was predicted as source
            if pred_source == gene_idx:
                gene_metrics["times_predicted_source"] += 1
        
        # Calculate aggregate metrics
        gene_metrics["avg_prediction_prob"] = np.mean(all_probs_for_gene)
        
        if gene_metrics["times_true_source"] > 0:
            gene_metrics["avg_rank_when_true"] = np.mean(gene_metrics["ranks_when_true"])
            gene_metrics["avg_prob_when_true"] = np.mean(gene_metrics["prediction_probs_when_true"])
            gene_metrics["in_top3_when_true"] = sum(1 for rank in gene_metrics["ranks_when_true"] if rank < 3)
            gene_metrics["in_top5_when_true"] = sum(1 for rank in gene_metrics["ranks_when_true"] if rank < 5)
            gene_metrics["accuracy_when_true"] = gene_metrics["correct_predictions"] / gene_metrics["times_true_source"]
        else:
            gene_metrics["avg_rank_when_true"] = None
            gene_metrics["avg_prob_when_true"] = None
            gene_metrics["in_top3_when_true"] = 0
            gene_metrics["in_top5_when_true"] = 0
            gene_metrics["accuracy_when_true"] = None
        
        # Calculate precision and recall for this gene
        if gene_metrics["times_predicted_source"] > 0:
            gene_metrics["precision"] = gene_metrics["correct_predictions"] / gene_metrics["times_predicted_source"]
        else:
            gene_metrics["precision"] = None
        
        if gene_metrics["times_true_source"] > 0:
            gene_metrics["recall"] = gene_metrics["correct_predictions"] / gene_metrics["times_true_source"]
        else:
            gene_metrics["recall"] = None
        
        # Calculate F1 score
        if gene_metrics["precision"] is not None and gene_metrics["recall"] is not None and \
           (gene_metrics["precision"] + gene_metrics["recall"]) > 0:
            gene_metrics["f1_score"] = 2 * (gene_metrics["precision"] * gene_metrics["recall"]) / \
                                     (gene_metrics["precision"] + gene_metrics["recall"])
        else:
            gene_metrics["f1_score"] = None
        
        # Remove detailed lists for cleaner output (keep only aggregated metrics)
        del gene_metrics["prediction_probs_when_true"]
        del gene_metrics["ranks_when_true"]
        
        # Round numerical values
        for key, value in gene_metrics.items():
            if isinstance(value, float):
                gene_metrics[key] = round(value, 3)
        
        gene_metrics_dict[gene_name] = gene_metrics
    
    return gene_metrics_dict


def supervised_metrics(
    pred_label_set: list,
    true_sources: list,
    pred_sources: list,
    processed_data: list = None,
    model_type: str = "gat",
    raw_test_data: list = None
) -> dict:
    """Perform supervised evaluation metrics."""
    metrics = {}

    print("Evaluating Model ...")

    # Common metrics for all models
    metrics.update(prediction_metrics(pred_label_set, true_sources))
    # not useful for sigle source prediction
    # metrics.update(TP_FP_metrics(true_sources, pred_sources))

    # Gene-specific metrics if enabled and genes of interest are specified
    if const.GENE_METRICS_ENABLED and const.GENES_OF_INTEREST and raw_test_data:
        # Extract gene mapping from first raw data sample
        gene_mapping = raw_test_data[0].gene_mapping if hasattr(raw_test_data[0], 'gene_mapping') else None
        
        if gene_mapping:
            print(f"Calculating gene-specific metrics for: {const.GENES_OF_INTEREST}")
            gene_metrics = gene_specific_metrics(
                pred_label_set, 
                true_sources, 
                pred_sources, 
                const.GENES_OF_INTEREST,
                gene_mapping,
                raw_test_data
            )
            if gene_metrics:
                metrics["gene_specific_metrics"] = gene_metrics
        else:
            print("Warning: Gene mapping not found in raw data, skipping gene-specific metrics")

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
            pred_labels, true_sources, pred_sources, processed_data, model_type, raw_test_data
        ),
        "data stats": data_stats(raw_test_data),
        "parameters": const.params  # Use the loaded parameters from constants
    }
    
    utils.save_metrics(metrics_dict)

    print(f"Validation complete! Results saved for {model_type} model with {network} network")


if __name__ == "__main__":
    main()