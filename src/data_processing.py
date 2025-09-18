""" Creates new processed data based on the selected model. """
from pathlib import Path
from src import constants as const
from src import utils
import networkx as nx
import numpy as np
import os
import shutil
from torch_geometric.data import Data, Dataset
import torch
import multiprocessing as mp
from tqdm import tqdm
from torch_geometric.utils import add_remaining_self_loops, to_undirected, from_networkx


class SDDataset(Dataset):
    """
    Unified dataset class for both PDGrapher and GNN models.
    Handles both forward/backward modes (PDGrapher) and individual file mode (GNN).
    Always reads raw data from data/raw and writes processed data to data/processed/{model}.
    """
    def __init__(self, model_type, processing_func, transform=None, pre_transform=None, mode="individual", raw_data_dir=const.RAW_PATH, edge_index=None, edge_attr=None):
        """
        Args:
            model_type: Type of model (e.g., "GAT", "GCNSI", "PDGrapher")
            processing_func: Function to process raw data for this model
            transform: Optional transform to be applied on a sample
            pre_transform: Optional pre-transform to be applied on a sample (deprecated, use processing_func)
            mode: "individual" for GNN models, "forward"/"backward" for PDGrapher
            raw_data_dir: Directory containing the raw .pt files
            edge_index: Pre-loaded edge index tensor (optional)
            edge_attr: Pre-loaded edge attributes tensor (optional)
        """
        self.model_type = model_type
        self.processing_func = processing_func or pre_transform  # Backward compatibility
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(const.PROCESSED_PATH)
        self.transform = transform
        self.mode = mode
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.raw_files = sorted(self.raw_data_dir.glob("*.pt"))
        self.size = len(self.raw_files)

        print(f"Found {len(self.raw_files)} raw data files in {self.raw_data_dir}")

        super().__init__(str(self.raw_data_dir), transform, self.processing_func, None)

    @property
    def raw_file_names(self):
        return [f.name for f in self.raw_files]

    @property
    def processed_file_names(self):
        if self.mode in ["forward", "backward"]:
            return [f"data_{self.mode}.pt"]  # Single file for PDGrapher-style
        else:
            return [f"{i}.pt" for i in range(self.size)]  # Individual files for GNN-style

    @property
    def processed_dir(self):
        return self.processed_data_dir

    def process(self):
        """Process raw data according to the mode."""
        if self.processing_func is not None:
            if self.mode in ["forward", "backward"]:
                # PDGrapher-style: process all files into a single list
                self._process_as_list()
            else:
                # GNN-style: process each file individually with multiprocessing
                self._process_individually()

    def _process_as_list(self):
        """Process all files into a single list (for PDGrapher-style datasets)"""
        print(f"Processing {len(self.raw_files)} files as single list for mode: {self.mode}")
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data_list = []
        for raw_path in tqdm(self.raw_files, desc=f"Processing {self.mode} data"):
            data = torch.load(raw_path, weights_only=False)
            processed_data = self.processing_func(data)
            processed_data_list.append(processed_data)
        
        output_file = self.processed_data_dir / f"data_{self.mode}.pt"
        torch.save(processed_data_list, output_file)
        print(f"Saved {len(processed_data_list)} {self.mode} data objects to {output_file}")

    def _process_individually(self):
        """Process each file individually with multiprocessing (for GNN-style datasets)"""
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

        params = [
            (self.processing_func, str(self.raw_files[i]), i, str(self.processed_data_dir), self.edge_index, self.edge_attr)
            for i in range(self.size)
        ]
        with mp.get_context("spawn").Pool(const.N_CORES) as pool:
            print(f"Processing data set using multiprocessing ({const.N_CORES} cores)...")
            list(tqdm(pool.imap_unordered(process_single, params), total=self.size))

    def len(self):
        if self.mode in ["forward", "backward"]:
            # For PDGrapher-style, load the list and return its length
            data_file = self.processed_data_dir / f"data_{self.mode}.pt"
            if data_file.exists():
                data_list = torch.load(data_file, weights_only=False)
                return len(data_list)
            return 0
        else:
            return self.size

    def get(self, idx):
        if self.mode in ["forward", "backward"]:
            # For PDGrapher-style, load from the list
            data_file = self.processed_data_dir / f"data_{self.mode}.pt"
            data_list = torch.load(data_file, weights_only=False)
            return data_list[idx]
        else:
            # For GNN-style, load individual file
            path = self.processed_data_dir / f"{idx}.pt"
            data = torch.load(path, weights_only=False)
            return data
    

def process_single(args):
    processing_func, raw_path, idx, processed_dir, edge_index, edge_attr = args
    data = torch.load(raw_path, weights_only=False)
    if processing_func is not None:
        if processing_func == process_data:
            data = process_data(data, edge_index, edge_attr)
        else:
            data = processing_func(data)
    torch.save(data, os.path.join(processed_dir, f"{idx}.pt"))


# def filter_data_arrays(data: Data, node_mapping_info: dict) -> Data:
#     """
#     Filter data arrays to remove values corresponding to removed nodes.
    
#     Args:
#         data: Original data object with arrays corresponding to all original nodes
#         node_mapping_info: Information about removed nodes and index mappings
        
#     Returns:
#         Modified data object with filtered arrays
#     """
#     if not node_mapping_info['removed_nodes']:
#         # No nodes were removed, return original data
#         return data
    
#     old_to_new_idx = node_mapping_info['old_to_new_idx']
#     remaining_indices = sorted(old_to_new_idx.keys())
#     remaining_nodes = node_mapping_info['remaining_nodes']
#     filtered_data = Data()
    
#     # Get the original gene mapping to understand the source gene
#     original_gene_mapping = data.gene_mapping
#     source_gene = data.perturbed_gene
    
#     # Check if the source gene is still in the remaining nodes
#     if source_gene not in remaining_nodes:
#         raise ValueError(f"Source gene '{source_gene}' was removed from the graph, but source genes should be protected!")
    
#     # Create new gene mapping for remaining nodes
#     new_gene_mapping = {gene: new_idx for new_idx, gene in enumerate(remaining_nodes)}
    
#     for key in data.keys:
#         value = data[key]
        
#         if key in ['original', 'perturbed', 'difference']:
#             # Filter these arrays normally
#             if isinstance(value, (list, np.ndarray)):
#                 filtered_value = [value[i] for i in remaining_indices]
#                 setattr(filtered_data, key, filtered_value)
#             elif hasattr(value, '__getitem__') and hasattr(value, '__len__'):
#                 filtered_value = value[remaining_indices]
#                 setattr(filtered_data, key, filtered_value)
#             else:
#                 setattr(filtered_data, key, value)
#         elif key == 'binary_perturbation_indicator':
#             # Reconstruct the binary perturbation indicator for the remaining nodes
#             new_indicator = [1 if gene == source_gene else 0 for gene in remaining_nodes]
#             setattr(filtered_data, key, new_indicator)
#         elif key == 'gene_mapping':
#             setattr(filtered_data, key, new_gene_mapping)
#         elif key == 'num_nodes':
#             setattr(filtered_data, key, len(remaining_nodes))
#         else:
#             # Copy other attributes as-is
#             setattr(filtered_data, key, value)
    
#     # Verify that we still have exactly one source
#     binary_indicator = getattr(filtered_data, 'binary_perturbation_indicator', [])
#     if sum(binary_indicator) != 1:
#         raise ValueError(f"After filtering, expected exactly 1 source node, but found {sum(binary_indicator)} in binary_perturbation_indicator")
    
#     return filtered_data


# def process_data_with_node_filtering(data: Data, edge_index: torch.Tensor = None, edge_attr: torch.Tensor = None, node_mapping_info: dict = None) -> Data:
#     """
#     Enhanced version of process_data that filters out removed nodes.
    
#     Args:
#         data: input data to be processed
#         edge_index: Pre-loaded edge index tensor (optional)
#         edge_attr: Pre-loaded edge attributes tensor (optional)
#         node_mapping_info: Information about removed nodes (optional)
        
#     Returns:
#         processed data with expanded features and labels, filtered for remaining nodes
#     """
#     if node_mapping_info and node_mapping_info['removed_nodes']:
#         data = filter_data_arrays(data, node_mapping_info)

#     healthy_norm, perturbed_norm = normalize_paired_tensors(data.original, data.perturbed)
#     data.x = torch.stack([healthy_norm, perturbed_norm], dim=1).float()

#     data.edge_index = edge_index
#     data.edge_attr = edge_attr

#     data.y = torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32)
    
#     return data


# def backward_data_with_node_filtering(data: Data, node_mapping_info: dict = None) -> Data:
#     """
#     Enhanced version of backward_data that filters out removed nodes.
#     """
#     if node_mapping_info and node_mapping_info['removed_nodes']:
#         data = filter_data_arrays(data, node_mapping_info)

#     diseased_norm, treated_norm = normalize_paired_tensors(data.original, data.perturbed)
    
#     return Data(
#         perturbagen_name=data.perturbed_gene,
#         diseased=diseased_norm,
#         intervention=torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32),
#         treated=treated_norm,
#         gene_symbols=list(data.gene_mapping.keys()),
#         mutations=torch.zeros(data.num_nodes, dtype=torch.float32),
#         batch=torch.arange(data.num_nodes, dtype=torch.long),
#         num_nodes=data.num_nodes,
#     )


# def forward_data_with_node_filtering(data: Data, node_mapping_info: dict = None) -> Data:
#     """
#     Enhanced version of forward_data that filters out removed nodes.
#     """
#     if node_mapping_info and node_mapping_info['removed_nodes']:
#         data = filter_data_arrays(data, node_mapping_info)

#     healthy_norm, diseased_norm = normalize_paired_tensors(data.original, data.perturbed)
    
#     return Data(
#         healthy=healthy_norm,
#         diseased=diseased_norm,
#         mutations=torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32),
#         gene_symbols=list(data.gene_mapping.keys()),
#         batch=torch.arange(data.num_nodes, dtype=torch.long),
#         num_nodes=data.num_nodes,
#     )


def normalize_paired_tensors(tensor1, tensor2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes two related tensors using the same min-max scale.
    This preserves the relative relationship between the two states.
    
    :param tensor1: First tensor (e.g., healthy expression) - expects list input
    :param tensor2: Second tensor (e.g., diseased expression) - expects list input
    :return: Tuple of normalized tensors
    """
    # Convert lists to torch tensors
    tensor1 = torch.tensor(tensor1, dtype=torch.float32)
    tensor2 = torch.tensor(tensor2, dtype=torch.float32)
    
    # Apply normalization if enabled in constants
    if const.NORMALIZE_DATA:
        # Find global min and max across both tensors
        combined = torch.cat([tensor1, tensor2])
        min_val = combined.min()
        max_val = combined.max()
        range_val = (max_val - min_val + 1e-8)  # Add small epsilon to avoid division by zero
        
        # Apply same normalization to both tensors
        tensor1_norm = (tensor1 - min_val) / range_val
        tensor2_norm = (tensor2 - min_val) / range_val
        
        return tensor1_norm, tensor2_norm
    
    return tensor1, tensor2


def backward_data(data: Data) -> Data:
    """
    Transforms a Data object containing original and perturbed gene expression data into a new Data object
    with normalized diseased and treated expression profiles, intervention indicators, and gene metadata.
    """
    diseased_norm, treated_norm = normalize_paired_tensors(data.original, data.perturbed)
    
    return Data(
        perturbagen_name=data.perturbed_gene,
        diseased=diseased_norm,
        intervention=torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32),
        treated=treated_norm,
        gene_symbols=list(data.gene_mapping.keys()),
        mutations=torch.zeros(data.num_nodes, dtype=torch.float32),  # ✅ FIXED: No background mutations
        batch=torch.arange(data.num_nodes, dtype=torch.long),  # ✅ ADDED: Batch indices
        num_nodes=data.num_nodes,
    )


def forward_data(data: Data) -> Data:
    """
    Transforms a Data object containing original and perturbed gene expression data into a new Data object
    with normalized healthy and diseased expression profiles, mutation indicators, and gene metadata.
    """
    healthy_norm, diseased_norm = normalize_paired_tensors(data.original, data.perturbed)
    
    return Data(
        healthy=healthy_norm,
        diseased=diseased_norm,
        mutations=torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32),
        gene_symbols=list(data.gene_mapping.keys()),
        batch=torch.arange(data.num_nodes, dtype=torch.long),  # ✅ ADDED: Batch indices
        num_nodes=data.num_nodes,
    )


def process_data(data: Data, edge_index: torch.Tensor = None, edge_attr: torch.Tensor = None) -> Data:
    """
    Features and Labels for the model.
    :param data: input data to be processed.
    :param edge_index: Pre-loaded edge index tensor (optional)
    :param edge_attr: Pre-loaded edge attributes tensor (optional)
    :return: processed data with expanded features and labels
    """
    healthy_norm, perturbed_norm = normalize_paired_tensors(data.original, data.perturbed)
    # stack the healthy and perturbed tensors to create x
    data.x = torch.stack([healthy_norm, perturbed_norm], dim=1).float()

    data.edge_index = edge_index
    data.edge_attr = edge_attr

    # Convert binary_perturbation_indicator to tensor
    data.y = torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32)
    
    return data


def store_edge_index_for_pdgrapher(edge_index: torch.Tensor):
    """
    Process the raw edge index and edge attributes, and store them in the model-specific processed directory.
    
    Args:
        model_type: Type of model (e.g., "GAT", "GCNSI", "PDGrapher")
    """
    print("Processing and storing edge index...")

    edge_index = add_remaining_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)
    processed_dir = Path(const.PROCESSED_PATH)
    processed_dir.mkdir(parents=True, exist_ok=True)
    torch.save(edge_index, const.EXPERIMENT_EDGE_INDEX_PATH)


#TODO PLACEHOLDER Double check and shorten
# def compute_thresholds_uniform(values):
#     min_val, max_val = min(values), max(values)
#     return torch.linspace(min_val, max_val, 501)

def compute_and_store_thresholds(datasets):
    """
    Compute thresholds for discretizing expression values as required by PDGrapher.
    These thresholds convert continuous expression values to discrete categories (0-499).
    
    How it works:
    1. Collect all expression values from training data
    2. Compute percentile-based thresholds (501 values for 500 categories)
    3. During inference: find which threshold bin an expression value falls into
    4. Use that bin index as the discrete category for embedding lookup
    """
    print("Computing thresholds for PDGrapher...")
    
    # Collect all expression values - ONLY from training data to avoid data leakage
    all_diseased = []
    all_treated = []
    all_healthy = []
    
    # PERFORMANCE FIX: Load data files once instead of loading for each sample
    processed_dir = Path(const.PROCESSED_PATH)
    
    # Load backward data efficiently - load file once, not per sample
    if "backward" in datasets:
        backward_file = processed_dir / "data_backward.pt"
        if backward_file.exists():
            print("Loading backward data for threshold computation...")
            backward_data_list = torch.load(backward_file, weights_only=False)
            print(f"Processing {len(backward_data_list)} backward samples for thresholds...")
            for data in tqdm(backward_data_list, desc="Processing backward data"):
                all_diseased.extend(data.diseased.tolist())
                all_treated.extend(data.treated.tolist())
    
    # Load forward data efficiently - load file once, not per sample  
    if "forward" in datasets:
        forward_file = processed_dir / "data_forward.pt"
        if forward_file.exists():
            print("Loading forward data for threshold computation...")
            forward_data_list = torch.load(forward_file, weights_only=False)
            print(f"Processing {len(forward_data_list)} forward samples for thresholds...")
            for data in tqdm(forward_data_list, desc="Processing forward data"):
                all_healthy.extend(data.healthy.tolist())
    
    # Compute percentile-based thresholds (0.2% increments as in PDGrapher)
    # Creates 501 threshold values -> 500 discrete categories (0-499)
    def compute_thresholds(values):
        import numpy as np
        
        # Handle different input types
        if not values:
            raise ValueError("No values provided for threshold computation")
        
        # If values is a list of floats (from .tolist()), convert directly
        if isinstance(values[0], (int, float)):
            print("Processing list of float values for threshold computation...")
            all_values = values
        else:
            # If values is a list of tensors, flatten and extract
            print("Processing list of tensors for threshold computation...")
            all_values = []
            for value_tensor in values:
                if hasattr(value_tensor, 'flatten'):
                    all_values.extend(value_tensor.flatten().tolist())
                else:
                    all_values.append(float(value_tensor))
        
        print(f"Total values: {len(all_values)}")
        
        # Use approximate quantiles with a fixed sample size
        sample_size = 50_000  # Guaranteed manageable size
        
        if len(all_values) <= sample_size:
            # Use all values if dataset is small enough
            sampled_values = all_values
            print(f"Using all {len(sampled_values)} values for quantile computation")
        else:
            # Randomly sample for large datasets
            import random
            sampled_values = random.sample(all_values, sample_size)
            sampling_rate = sample_size / len(all_values)
            print(f"Sampled {len(sampled_values)} values (sampling rate: {sampling_rate:.4f}) for quantile computation")
        
        # Compute percentile-based thresholds for PDGrapher (501 thresholds for 500 bins)
        percentiles = np.linspace(0, 100, 501)  # 0%, 0.2%, 0.4%, ..., 99.8%, 100%
        quantiles = np.percentile(sampled_values, percentiles)
        
        return torch.tensor(quantiles, dtype=torch.float32)
    
    # Create thresholds in format expected by PDGrapher model
    thresholds = {}
    
    # PDGrapher expects separate thresholds for diseased and treated states
    if all_diseased:
        thresholds["diseased"] = compute_thresholds(all_diseased)
        print(f"Diseased thresholds: min={thresholds['diseased'].min():.3f}, max={thresholds['diseased'].max():.3f}")
    
    if all_treated:
        thresholds["treated"] = compute_thresholds(all_treated)
        print(f"Treated thresholds: min={thresholds['treated'].min():.3f}, max={thresholds['treated'].max():.3f}")
    
    # Also store combined thresholds for backward compatibility
    if all_diseased and all_treated:
        backward_values = all_diseased + all_treated
        thresholds["backward"] = compute_thresholds(backward_values)
        print(f"Backward (combined) thresholds: min={thresholds['backward'].min():.3f}, max={thresholds['backward'].max():.3f}")
    
    # For forward direction, combine healthy and diseased
    if all_healthy and all_diseased:
        forward_values = all_healthy + all_diseased 
        thresholds["forward"] = compute_thresholds(forward_values)
        print(f"Forward thresholds: min={thresholds['forward'].min():.3f}, max={thresholds['forward'].max():.3f}")
    
    # Store thresholds
    thresholds_path = Path(const.PROCESSED_PATH) / "thresholds.pt"
    torch.save(thresholds, thresholds_path)
    print(f"Thresholds saved to {thresholds_path}")
    print(f"Each threshold tensor has {501} values for {500} discrete categories")
    
    return thresholds


def create_datasets_for_model(model_type, raw_data_dir=const.RAW_PATH):
    """
    Create datasets based on model type using the unified SDDataset approach.
    
    Args:
        model_type: Type of model ("pdgrapher", "GCNSI", "GAT", etc.)
        raw_data_dir: Directory containing raw data files
    """

    G = utils.load_perturbed_graph()
    
    if model_type == "pdgrapher":
        # Create both forward and backward datasets for PDGrapher
        print("Creating PDGrapher-style datasets...")

        store_edge_index_for_pdgrapher(G.edge_index)
        
        # Create backward dataset
        backward_dataset = SDDataset(
            model_type=model_type,
            processing_func=backward_data,
            mode="backward",
            raw_data_dir=raw_data_dir,
        )
        
        # Create forward dataset  
        forward_dataset = SDDataset(
            model_type=model_type,
            processing_func=forward_data,
            mode="forward",
            raw_data_dir=raw_data_dir,
        )
        
        print("PDGrapher-style datasets created successfully!")
        
        # Compute and store thresholds for PDGrapher
        compute_and_store_thresholds({"backward": backward_dataset, "forward": forward_dataset})
        
        return {"backward": backward_dataset, "forward": forward_dataset}
        
    elif model_type == "pdgrapher_nognn":
        # Create both forward and backward datasets for PDGrapherNoGNN (no edge_index needed)
        print("Creating PDGrapherNoGNN-style datasets...")
        
        # Create backward dataset
        backward_dataset = SDDataset(
            model_type=model_type,
            processing_func=backward_data,
            mode="backward",
            raw_data_dir=raw_data_dir,
        )
        
        # Create forward dataset  
        forward_dataset = SDDataset(
            model_type=model_type,
            processing_func=forward_data,
            mode="forward",
            raw_data_dir=raw_data_dir,
        )
        
        print("PDGrapherNoGNN-style datasets created successfully!")
        
        # Compute and store thresholds for PDGrapherNoGNN
        compute_and_store_thresholds({"backward": backward_dataset, "forward": forward_dataset})
        
        return {"backward": backward_dataset, "forward": forward_dataset}
        
    elif model_type in ["gcnsi", "gat"]:
        # Create individual file dataset for GNN models
        print(f"Creating {model_type} dataset...")
        
        dataset = SDDataset(
            model_type=model_type,
            processing_func=process_data,
            mode="individual",
            raw_data_dir=raw_data_dir,
            edge_index = G.edge_index,
            edge_attr = G.edge_attr,
        )
        
        print(f"{model_type} dataset created successfully!")
        return {"main": dataset}
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """
    Creates new processed data based on the selected model.
    Uses experiment-specific path since processed data depends on the perturbed graph.
    """   
    print(f"Creating processed data for {const.MODEL} in experiment-specific location...")
    
    # Check if processed data already exists in experiment-specific location
    processed_dir = Path(const.PROCESSED_PATH)
    if processed_dir.exists() and any(processed_dir.iterdir()):
        print(f"Processed data already exists at {processed_dir}")
        print("Loading existing processed datasets...")
        datasets = create_datasets_for_model(const.MODEL)
        return datasets
    
    print(f"Creating new processed data at {processed_dir}")
    
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets using the unified approach
    datasets = create_datasets_for_model(const.MODEL)

    print(f"Data processing complete for {const.MODEL}!")
    return datasets

if __name__ == "__main__":
    main()

# Compatibility wrapper for PDGrapher (if needed for existing code)
class PDGrapherDataset:
    """
    Thin compatibility wrapper around SDDataset for PDGrapher-style access.
    Use SDDataset directly when possible.
    """
    def __init__(self, root, data_type="backward"):
        # Extract model type from the usage context
        model_type = "PDGrapher"
        
        if data_type == "backward":
            processing_func = backward_data
        elif data_type == "forward": 
            processing_func = forward_data
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
            
        self.dataset = SDDataset(
            model_type=model_type,
            processing_func=processing_func,
            mode=data_type,
            raw_data_dir=const.RAW_PATH
        )
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]
    
    def get_data_list(self):
        """Return the complete data list."""
        return [self.dataset[i] for i in range(len(self.dataset))]