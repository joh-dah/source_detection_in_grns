""" Unified validation script for both PDGrapher and GAT models. """
import argparse
import yaml
import numpy as np
import torch
from tqdm import tqdm
import networkx as nx
import os
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr, linregress
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
        elif self.model_type == "pdgraphernognn":
            return self._load_pdgrapher_nognn_model()
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
            
        edge_index = torch.load(const.EXPERIMENT_EDGE_INDEX_PATH, map_location=self.device)
        
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
    
    def _load_pdgrapher_nognn_model(self):
        """Load PDGrapherNoGNN perturbation discovery model."""
        from architectures.PDGrapherNoGNN import PerturbationDiscoveryModelNoGNN, NoGNNArgs
        
        print(f"Loading PDGrapherNoGNN model from: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        state_dict = checkpoint["model_state_dict"]
        
        # Extract model configuration - handle both saved formats
        model_config = checkpoint.get("model_config", {})
        positional_features_dims = model_config.get("positional_features_dims", 64)
        embed_dim = model_config.get("embedding_layer_dim", 64)
        dim_gnn = model_config.get("dim_gnn", 64)
        num_vars = model_config.get("num_vars", const.N_NODES)
        n_layers_gnn = model_config.get("n_layers_gnn", const.LAYERS)
        
        # Create model args (no edge_index needed)
        args = NoGNNArgs(
            positional_features_dims=positional_features_dims,
            embedding_layer_dim=embed_dim,
            dim_gnn=dim_gnn,
            num_vars=num_vars,
            n_layers_gnn=n_layers_gnn,
            n_layers_nn=const.LAYERS,
        )
        
        print(f"Model config (from saved weights): positional_features_dims={positional_features_dims}, "
              f"embedding_layer_dim={embed_dim}, dim_gnn={dim_gnn}, num_vars={num_vars}, "
              f"n_layers_gnn={n_layers_gnn}")
        
        # Initialize and load the model (no edge_index needed)
        model = PerturbationDiscoveryModelNoGNN(args, num_vars)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        
        print(f"Loaded PerturbationDiscoveryModelNoGNN with {sum(p.numel() for p in model.parameters()):,} parameters")
        print(f"Model configuration loaded from checkpoint: {checkpoint.get('model_config', 'Not available')}")
        return model
    
    def load_test_data(self):
        """Load test data based on model type."""
        if self.model_type == "gat":
            return self._load_gat_test_data()
        elif self.model_type == "pdgrapher":
            return self._load_pdgrapher_test_data()
        elif self.model_type == "pdgraphernognn":
            return self._load_pdgrapher_nognn_test_data()

    def _load_gat_test_data(self):
        """Load test data for GAT model."""
        # Load processed test data
        processed_test_data = utils.load_processed_data(split="test")
        
        # Try to load corresponding raw data for validation metrics
        try:
            raw_test_data = self._load_raw_test_data_from_indices()
        except (FileNotFoundError, Exception) as e:
            print(f"Warning: Could not load raw test data for GAT: {e}")
            raw_test_data = None
        
        return raw_test_data, processed_test_data
    
    def _load_raw_test_data_from_indices(self):
        """Load raw test data based on split indices."""
        try:
            splits = torch.load(const.SPLITS_PATH, weights_only=False)
            test_indices = splits["test_index_backward"]
            
            raw_data_dir = Path(const.RAW_PATH)
            if not raw_data_dir.exists():
                raise FileNotFoundError(f"Raw data directory does not exist: {raw_data_dir}")
                
            raw_files = sorted(list(raw_data_dir.glob("*.pt")))
            if not raw_files:
                raise FileNotFoundError(f"No raw data files found in: {raw_data_dir}")
            
            raw_test_data = []
            for idx in test_indices:
                if idx >= len(raw_files):
                    raise IndexError(f"Test index {idx} exceeds number of raw files {len(raw_files)}")
                raw_data = torch.load(raw_files[idx], weights_only=False)
                raw_test_data.append(raw_data)
            
            return raw_test_data
        except Exception as e:
            print(f"Error loading raw test data: {e}")
            raise
    
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
    
    def _load_pdgrapher_nognn_test_data(self):
        """Load test data for PDGrapherNoGNN model (same as PDGrapher but without edge dependencies)."""
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
        elif self.model_type == "pdgraphernognn":
            return self._get_pdgrapher_nognn_predictions(model, test_data[0], test_data[1])  # dataset, test_loader
    
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
                    if len(true_source_tensor) > 1:
                        print(f"WARNING: Multiple sources found in sample {i}: {true_source_tensor.tolist()}")
                        print(f"Using first source: {true_source_tensor[0].item()} for single-source evaluation")
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
                # PDGrapher model expects threshold_input with forward/backward keys 
                threshold_input = {
                    "diseased": thresholds["forward"],   # Use forward thresholds for diseased samples
                    "treated": thresholds["backward"]    # Use backward thresholds for treated samples
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
    
    def _get_pdgrapher_nognn_predictions(self, model, dataset, test_loader):
        """Get predictions from PDGrapherNoGNN model (same logic as PDGrapher but without edge dependencies)."""
        predictions = []
        true_sources = []
        
        # Load thresholds
        thresholds_path = Path(const.PROCESSED_PATH) / "thresholds.pt"
        thresholds = torch.load(thresholds_path, weights_only=False)
        # Ensure thresholds are on the correct device
        thresholds = {k: (v.detach().clone() if isinstance(v, torch.Tensor) else torch.tensor(v)).to(self.device) 
                     for k, v in thresholds.items()}
        
        print(f"Loaded thresholds with keys: {list(thresholds.keys())}")
        
        # Use the same threshold structure as PDGrapher
        if "forward" in thresholds and "backward" in thresholds:
            threshold_input = {"diseased": thresholds["forward"], "treated": thresholds["backward"]}
        else:
            # Fallback to backward thresholds if the standard keys don't exist
            threshold_input = {"diseased": thresholds["backward"], "treated": thresholds["backward"]}
        
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(tqdm(test_loader, desc="PDGrapherNoGNN predictions", disable=const.ON_CLUSTER)):
                # Extract true source for tracking (same logic as PDGrapher)
                intervention = data.intervention
                if intervention.device != torch.device("cpu"):
                    intervention = intervention.cpu()
                true_source_tensor = torch.where(intervention == 1)[0]
                if len(true_source_tensor) > 0:
                    if len(true_source_tensor) > 1:
                        print(f"WARNING: Multiple sources found in sample {i}: {true_source_tensor.tolist()}")
                        print(f"Using first source: {true_source_tensor[0].item()} for single-source evaluation")
                    true_source = int(true_source_tensor[0])
                    true_sources.append(true_source)
                else:
                    print(f"WARNING: No true source found in sample {i}")
                    true_sources.append(-1)
                
                # Move all relevant tensors to the correct device (no edge_index needed)
                for attr in ['diseased', 'treated', 'batch', 'mutations', 'intervention']:
                    if hasattr(data, attr):
                        tensor = getattr(data, attr)
                        if tensor is not None and isinstance(tensor, torch.Tensor):
                            setattr(data, attr, tensor.to(self.device))
                diseased = data.diseased.view(-1, 1)
                treated = data.treated.view(-1, 1)
                batch = data.batch
                mutations = data.mutations
                
                # Run model inference (no edge_index passed)
                intervention_logits = model(
                    torch.cat([diseased, treated], dim=1),
                    batch,
                    mutilate_mutations=mutations,
                    threshold_input=threshold_input
                )

                intervention_logits = intervention_logits.flatten()
                predictions.append(intervention_logits.cpu())
        
        self._pdgrapher_nognn_true_sources = true_sources
        return predictions
    
    def extract_true_sources(self, test_data):
        """Extract true source nodes based on model type."""
        if self.model_type == "gat":
            return self._extract_gat_true_sources(test_data[1])  # processed_test_data
        elif self.model_type == "pdgrapher":
            return self._extract_pdgrapher_true_sources(test_data[1])  # test_loader
        elif self.model_type == "pdgraphernognn":
            return self._extract_pdgrapher_nognn_true_sources(test_data[1])  # test_loader
    
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
    
    def _extract_pdgrapher_nognn_true_sources(self, test_loader):
        """Extract true sources from PDGrapherNoGNN data (same logic as PDGrapher)."""
        # True sources were already extracted during prediction phase
        # to ensure exact same data ordering
        if hasattr(self, '_pdgrapher_nognn_true_sources'):
            return self._pdgrapher_nognn_true_sources
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
        elif self.model_type == "pdgrapher" or self.model_type == "pdgraphernognn":
            # For PDGrapher/PDGrapherNoGNN, try to load raw data but handle failure gracefully
            try:
                return self._load_raw_test_data_from_indices()
            except (FileNotFoundError, Exception) as e:
                print(f"Warning: Could not load raw data for stats: {e}")
                return None


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
    in_top5 = []
    in_top20 = []
    in_top40 = []
    in_top80 = []

    for i, pred_labels in enumerate(tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)):
        # Handle both single and multiple sources
        if isinstance(true_sources[i], (list, tuple)):
            # Multiple sources case
            true_source_list = true_sources[i]
        else:
            # Single source case
            true_source_list = [true_sources[i]]
        
        ranked_predictions = utils.ranked_source_predictions(pred_labels).tolist()

        # Calculate metrics for each true source and take the best/average
        sample_ranks = []
        sample_predictions = []
        sample_in_top5 = []
        sample_in_top20 = []
        sample_in_top40 = []
        sample_in_top80 = []
        
        for true_source in true_source_list:
            if true_source == -1:  # Skip invalid sources
                continue
                
            source_rank = ranked_predictions.index(true_source)
            sample_ranks.append(source_rank)
            sample_predictions.append(pred_labels[true_source].item())
            
            sample_in_top5.append(source_rank < 5)
            sample_in_top20.append(source_rank < 20)
            sample_in_top40.append(source_rank < 40)
            sample_in_top80.append(source_rank < 80)
        
        if sample_ranks:  # Only add if we have valid sources
            # For multi-source: use best rank (most optimistic) or average
            source_ranks.append(min(sample_ranks))  # Best rank among true sources
            predictions_for_source.extend(sample_predictions)
            general_predictions += pred_labels.tolist()
            
            # For top-k: sample is successful if ANY true source is in top-k
            in_top5.append(any(sample_in_top5))
            in_top20.append(any(sample_in_top20))
            in_top40.append(any(sample_in_top40))
            in_top80.append(any(sample_in_top80))

    return {
        "accuracy": np.mean(np.array(source_ranks) == 0),
        "avg rank of source": np.mean(source_ranks),
        "avg prediction for source": np.mean(predictions_for_source),
        "avg prediction over all nodes": np.mean(general_predictions),
        "min prediction over all nodes": min(general_predictions) if general_predictions else 0,
        "max prediction over all nodes": max(general_predictions) if general_predictions else 0,
        "source in top 5": np.mean(in_top5),
        "source in top 20": np.mean(in_top20),
        "source in top 40": np.mean(in_top40),
        "source in top 80": np.mean(in_top80),
    }


def multi_source_prediction_metrics(pred_label_set: list, test_data_loader) -> dict:
    """
    Calculate prediction metrics that can handle multiple true sources per sample.
    Returns metrics for multi-source scenarios.
    """
    print("Calculating multi-source prediction metrics...")
    
    multi_source_samples = 0
    all_source_ranks = []
    all_predictions_for_sources = []
    multi_source_accuracy = []  # How many sources we get right per sample
    
    for i, (pred_labels, data) in enumerate(zip(pred_label_set, test_data_loader)):
        # Extract all true sources for this sample
        intervention = data.intervention
        if intervention.device != torch.device("cpu"):
            intervention = intervention.cpu()
        true_source_indices = torch.where(intervention == 1)[0].tolist()
        
        if len(true_source_indices) > 1:
            multi_source_samples += 1
            
            # Get rankings for all nodes
            ranked_predictions = utils.ranked_source_predictions(pred_labels).tolist()
            
            # Calculate metrics for each true source
            sample_ranks = []
            sample_predictions = []
            correct_in_top_k = {"top1": 0, "top5": 0, "top20": 0, "top40": 0, "top80": 0}
            
            for true_source in true_source_indices:
                rank = ranked_predictions.index(true_source)
                sample_ranks.append(rank)
                sample_predictions.append(pred_labels[true_source].item())
                
                # Count how many sources are correctly identified in top-k
                if rank == 0:
                    correct_in_top_k["top1"] += 1
                if rank < 5:
                    correct_in_top_k["top5"] += 1
                if rank < 20:
                    correct_in_top_k["top20"] += 1
                if rank < 40:
                    correct_in_top_k["top40"] += 1
                if rank < 80:
                    correct_in_top_k["top80"] += 1
            
            all_source_ranks.extend(sample_ranks)
            all_predictions_for_sources.extend(sample_predictions)
            
            # Calculate accuracy as fraction of sources correctly identified
            total_sources = len(true_source_indices)
            multi_source_accuracy.append({
                "sample_id": i,
                "total_sources": total_sources,
                "correct_top1": correct_in_top_k["top1"],
                "correct_top5": correct_in_top_k["top5"],
                "correct_top20": correct_in_top_k["top20"], 
                "correct_top40": correct_in_top_k["top40"],
                "correct_top80": correct_in_top_k["top80"],
                "accuracy_top1": correct_in_top_k["top1"] / total_sources,
                "accuracy_top5": correct_in_top_k["top5"] / total_sources,
                "accuracy_top20": correct_in_top_k["top20"] / total_sources,
                "accuracy_top40": correct_in_top_k["top40"] / total_sources,
                "accuracy_top80": correct_in_top_k["top80"] / total_sources,
            })
    
    if multi_source_samples == 0:
        return {"multi_source_metrics": "No multi-source samples found"}
    
    # Calculate aggregate metrics
    avg_accuracy_top1 = np.mean([sample["accuracy_top1"] for sample in multi_source_accuracy])
    avg_accuracy_top20 = np.mean([sample["accuracy_top20"] for sample in multi_source_accuracy])
    avg_accuracy_top40 = np.mean([sample["accuracy_top40"] for sample in multi_source_accuracy])
    avg_accuracy_top80 = np.mean([sample["accuracy_top80"] for sample in multi_source_accuracy])
    
    return {
        "multi_source_samples": multi_source_samples,
        "avg_rank_of_sources": np.mean(all_source_ranks),
        "avg_prediction_for_sources": np.mean(all_predictions_for_sources),
        "avg_source_accuracy_top1": avg_accuracy_top1,
        "avg_source_accuracy_top20": avg_accuracy_top20,
        "avg_source_accuracy_top40": avg_accuracy_top40,
        "avg_source_accuracy_top80": avg_accuracy_top80,
        "perfect_samples_top1": sum(1 for s in multi_source_accuracy if s["accuracy_top1"] == 1.0),
        "perfect_samples_top20": sum(1 for s in multi_source_accuracy if s["accuracy_top20"] == 1.0),
        "perfect_samples_top40": sum(1 for s in multi_source_accuracy if s["accuracy_top40"] == 1.0),
        "perfect_samples_top80": sum(1 for s in multi_source_accuracy if s["accuracy_top80"] == 1.0),
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
    if raw_data_set is None or len(raw_data_set) == 0:
        print("Warning: No raw data available for statistics calculation")
        return {
            "graph stats": {
                "number of nodes": const.N_NODES,
                "number of possible sources": "N/A (no raw data)",
            },
            "infection stats": {
                "avg number of sources": "N/A (no raw data)",
                "avg portion of affected nodes": "N/A (no raw data)", 
                "std portion of affected nodes": "N/A (no raw data)",
            },
        }
    
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


def compute_idcg(num_correct: int, num_nodes: int) -> float:
    """
    Computes the Ideal Discounted Cumulative Gain (IDCG) for the given
    number of correct interventions and total number of nodes.
    Exact implementation from PDGrapher compute_pd_accuracy.py
    """
    idcg = 0
    for rank in range(1, num_correct + 1):  # Ideal ranking: 1 to num_correct
        gain = 1 - (rank / num_nodes)  # Gain function
        discount = 1 / np.log2(rank + 1)  # Logarithmic discount
        idcg += gain * discount
    return idcg


def pdgrapher_perturbation_discovery_metrics(pred_label_set: list, test_data_loader) -> dict:
    """
    Calculate PDGrapher-style perturbation discovery metrics.
    Implements exact calculations from compute_pd_accuracy.py and test_new_cell_line.py
    """
    print("Calculating PDGrapher perturbation discovery metrics...")
    
    all_recall_at_1 = []
    all_recall_at_10 = []
    all_recall_at_100 = []
    all_recall_at_1000 = []
    all_perc_partially_accurate_predictions = []
    all_rankings = []
    all_rankings_dcg = []
    
    # Get actual num_nodes from the first data sample
    first_data = next(iter(test_data_loader))
    num_nodes = int(first_data.num_nodes / len(torch.unique(first_data.batch)))
    
    for i, (pred_labels, data) in enumerate(zip(pred_label_set, test_data_loader)):
        # Extract intervention information
        intervention = data.intervention
        if intervention.device != torch.device("cpu"):
            intervention = intervention.cpu()
        
        # Get true interventions
        correct_interventions = set(torch.where(intervention.view(-1, num_nodes))[1].tolist())
        
        if len(correct_interventions) == 0:
            print(f"Warning: No true interventions found in sample {i}")
            continue
        
        # Get predicted interventions ranked by score
        if isinstance(pred_labels, torch.Tensor):
            if pred_labels.dim() > 1:
                pred_scores = pred_labels.view(-1)
            else:
                pred_scores = pred_labels
        else:
            pred_scores = torch.tensor(pred_labels)
        
        # Ensure we have the right number of scores
        if len(pred_scores) != num_nodes:
            print(f"Warning: Prediction length {len(pred_scores)} != num_nodes {num_nodes} in sample {i}")
            continue
            
        predicted_interventions = torch.argsort(pred_scores, descending=True).tolist()
        
        # Calculate ranking score for each correct intervention
        sample_rankings = []
        for ci in list(correct_interventions):
            rank_pos = predicted_interventions.index(ci)
            ranking_score = 1 - (rank_pos / num_nodes)
            sample_rankings.append(ranking_score)
        
        # Calculate DCG (Discounted Cumulative Gain)
        dcg = 0
        for ci in list(correct_interventions):
            rank = predicted_interventions.index(ci) + 1  # 1-based indexing for DCG
            gain = 1 - (rank / num_nodes)
            discount = 1 / np.log2(rank + 1)
            dcg += gain * discount
        
        # Calculate NDCG (Normalized DCG)
        idcg = compute_idcg(len(correct_interventions), num_nodes)
        ndcg = dcg / idcg if idcg > 0 else 0
        
        # Calculate Recall@K metrics
        recall_at_1 = len(set(predicted_interventions[:1]).intersection(correct_interventions)) / len(correct_interventions)
        recall_at_10 = len(set(predicted_interventions[:10]).intersection(correct_interventions)) / len(correct_interventions)
        recall_at_100 = len(set(predicted_interventions[:100]).intersection(correct_interventions)) / len(correct_interventions)
        recall_at_1000 = len(set(predicted_interventions[:1000]).intersection(correct_interventions)) / len(correct_interventions)
        
        # Calculate Jaccard similarity for partial accuracy
        jaccards = len(correct_interventions.intersection(set(predicted_interventions[:len(correct_interventions)]))) / len(correct_interventions.union(set(predicted_interventions[:len(correct_interventions)])))
        
        # Store sample-level metrics
        all_recall_at_1.append(recall_at_1)
        all_recall_at_10.append(recall_at_10)
        all_recall_at_100.append(recall_at_100)
        all_recall_at_1000.append(recall_at_1000)
        all_rankings.extend(sample_rankings)  # Individual ranking scores
        all_rankings_dcg.append(ndcg)
        all_perc_partially_accurate_predictions.append(1 if jaccards != 0 else 0)
    
    if len(all_recall_at_1) == 0:
        print("Warning: No valid samples found for PDGrapher metrics calculation")
        return {}
    
    # Calculate aggregate metrics
    metrics = {
        "recall@1": round(np.mean(all_recall_at_1), 4),
        "recall@10": round(np.mean(all_recall_at_10), 4),
        "recall@100": round(np.mean(all_recall_at_100), 4),
        "recall@1000": round(np.mean(all_recall_at_1000), 4),
        "ranking_score": round(np.mean(all_rankings), 4),
        "ranking_score_dcg": round(np.mean(all_rankings_dcg), 4),
        "perc_partially_accurate_predictions": round(100 * np.mean(all_perc_partially_accurate_predictions), 2),
        
        # Standard deviations for statistical reporting
        "recall@1_std": round(np.std(all_recall_at_1), 4),
        "recall@10_std": round(np.std(all_recall_at_10), 4),
        "recall@100_std": round(np.std(all_recall_at_100), 4),
        "recall@1000_std": round(np.std(all_recall_at_1000), 4),
        "ranking_score_std": round(np.std(all_rankings), 4),
        "ranking_score_dcg_std": round(np.std(all_rankings_dcg), 4),
        "perc_partially_accurate_predictions_std": round(100 * np.std(all_perc_partially_accurate_predictions), 2),
        
        # Additional useful metrics
        "total_samples": len(all_recall_at_1),
        "avg_topk": round(np.mean([r * num_nodes for r in all_rankings]), 4),  # Average rank position
    }
    
    return metrics


def pdgrapher_response_prediction_metrics(pred_label_set: list, true_labels: list, test_data_loader=None) -> dict:
    """
    Calculate PDGrapher-style response prediction metrics.
    Implements exact calculations from train.py _test_one_pass method.
    Note: This requires actual expression data, not just source predictions.
    """
    print("Calculating PDGrapher response prediction metrics...")
    
    if test_data_loader is None:
        print("Warning: No test data loader provided for response prediction metrics")
        return {}
    
    # Extract predicted and true expression values
    all_pred_expression = []
    all_true_expression = []
    
    # This would need to be adapted based on your specific data structure
    # Since we're focusing on perturbation discovery, we'll return placeholder values
    print("Note: Response prediction metrics require expression prediction data")
    print("Currently only implementing perturbation discovery metrics")
    
    return {
        "forward_mae": -1,
        "forward_mse": -1,
        "forward_r2": -1,
        "forward_r2_scgen": -1,
        "forward_spearman": -1,
        "backward_mae": -1,
        "backward_mse": -1,
        "backward_r2": -1,
        "backward_r2_scgen": -1,
        "backward_spearman": -1,
        "backward_avg_topk": -1
    }


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
            "in_top20_count": 0,
            "in_top40_count": 0,
            "in_top80_count": 0,
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
            gene_metrics["in_top20_when_true"] = sum(1 for rank in gene_metrics["ranks_when_true"] if rank < 20)
            gene_metrics["in_top40_when_true"] = sum(1 for rank in gene_metrics["ranks_when_true"] if rank < 40)
            gene_metrics["in_top80_when_true"] = sum(1 for rank in gene_metrics["ranks_when_true"] if rank < 80)
            gene_metrics["accuracy_when_true"] = gene_metrics["correct_predictions"] / gene_metrics["times_true_source"]
        else:
            gene_metrics["avg_rank_when_true"] = None
            gene_metrics["avg_prob_when_true"] = None
            gene_metrics["in_top20_when_true"] = 0
            gene_metrics["in_top40_when_true"] = 0
            gene_metrics["in_top80_when_true"] = 0
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
    raw_test_data: list = None,
    test_data_loader = None
) -> dict:
    """Perform supervised evaluation metrics."""
    metrics = {}

    print("Evaluating Model ...")

    # Common metrics for all models (single-source evaluation)
    metrics.update(prediction_metrics(pred_label_set, true_sources))
    
    # Multi-source metrics if we have access to the test data loader
    if test_data_loader is not None and model_type in ["pdgrapher", "pdgraphernognn"]:
        multi_metrics = multi_source_prediction_metrics(pred_label_set, test_data_loader)
        if "multi_source_metrics" not in multi_metrics:  # Only if we found multi-source samples
            metrics["multi_source_metrics"] = multi_metrics
    
    # PDGrapher-specific perturbation discovery metrics
    if test_data_loader is not None and model_type in ["pdgrapher", "pdgraphernognn"]:
        print("Computing PDGrapher-style perturbation discovery metrics...")
        pdgrapher_pd_metrics = pdgrapher_perturbation_discovery_metrics(pred_label_set, test_data_loader)
        metrics["pdgrapher_perturbation_discovery"] = pdgrapher_pd_metrics
        
        # Also compute response prediction metrics structure (placeholder for now)
        pdgrapher_rp_metrics = pdgrapher_response_prediction_metrics(pred_label_set, true_sources, test_data_loader)
        metrics["pdgrapher_response_prediction"] = pdgrapher_rp_metrics
    
    # not useful for sigle source prediction
    # metrics.update(TP_FP_metrics(true_sources, pred_sources))

    # Gene-specific metrics if enabled and genes of interest are specified
    if (const.GENE_METRICS_ENABLED and 
        hasattr(const, 'GENES_OF_INTEREST') and 
        const.GENES_OF_INTEREST and 
        raw_test_data is not None and 
        len(raw_test_data) > 0):
        
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
    else:
        if not hasattr(const, 'GENE_METRICS_ENABLED') or not const.GENE_METRICS_ENABLED:
            print("Gene-specific metrics disabled in configuration")
        elif not hasattr(const, 'GENES_OF_INTEREST') or not const.GENES_OF_INTEREST:
            print("No genes of interest specified, skipping gene-specific metrics")
        elif raw_test_data is None:
            print("No raw data available, skipping gene-specific metrics")

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
    
    # Get test_data_loader for multi-source metrics (PDGrapher models only)
    test_data_loader = test_data[1] if model_type in ["pdgrapher", "pdgraphernognn"] else None
    
    # Calculate metrics
    metrics_dict = {
        "network": network,
        "model_type": model_type,
        "metrics": supervised_metrics(
            pred_labels, true_sources, pred_sources, processed_data, model_type, raw_test_data, test_data_loader
        ),
        "parameters": const.params  # Use the loaded parameters from constants
    }
    
    # Only include data stats if raw data is available
    if raw_test_data is not None:
        metrics_dict["data stats"] = data_stats(raw_test_data)
    else:
        print("Raw data not available, skipping data statistics")
    
    utils.save_metrics(metrics_dict)

    print(f"Validation complete! Results saved for {model_type} model with {network} network")


if __name__ == "__main__":
    main()