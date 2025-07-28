""" Creates new processed data based on the selected model. """
from pathlib import Path
from src import constants as const
from src import utils
from src.create_splits import create_data_splits
import networkx as nx
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
    def __init__(self, model_type, processing_func, transform=None, pre_transform=None, mode="individual", raw_data_dir="data/raw", edge_index=None, edge_attr=None):
        """
        Args:
            model_type: Type of model (e.g., "GAT", "GCNSI", "PDGrapher")
            processing_func: Function to process raw data for this model
            transform: Optional transform to be applied on a sample
            pre_transform: Optional pre-transform to be applied on a sample (deprecated, use processing_func)
            mode: "individual" for GNN models, "forward"/"backward" for PDGrapher
            raw_data_dir: Directory containing the raw .pt files
        """
        self.model_type = model_type
        self.processing_func = processing_func or pre_transform  # Backward compatibility
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(f"data/processed/{model_type}")
        self.transform = transform
        self.mode = mode
        self.edge_index = edge_index
        self.edge_attr = edge_attr

        
        # Create processed data directory
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load and sort raw file paths
        self.raw_files = sorted(self.raw_data_dir.glob("*.pt"))
        self.size = len(self.raw_files)
        
        if not self.raw_files:
            raise ValueError(f"No raw .pt files found in {self.raw_data_dir}")
        
        print(f"Found {len(self.raw_files)} raw data files in {self.raw_data_dir}")
        
        # Call parent constructor
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
        
        # Ensure processed directory exists
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        processed_data_list = []
        for raw_path in tqdm(self.raw_files, desc=f"Processing {self.mode} data"):
            data = torch.load(raw_path, weights_only=False)
            processed_data = self.processing_func(data)
            processed_data_list.append(processed_data)
        
        # Save as single file
        output_file = self.processed_data_dir / f"data_{self.mode}.pt"
        torch.save(processed_data_list, output_file)
        print(f"Saved {len(processed_data_list)} {self.mode} data objects to {output_file}")

    def _process_individually(self):
        """Process each file individually with multiprocessing (for GNN-style datasets)"""
        # Ensure processed directory exists
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
    # load raw data
    data = torch.load(raw_path, weights_only=False)
    # process data with pre-loaded edge_index and edge_attr
    if processing_func is not None:
        if processing_func == process_data:
            # Pass edge_index and edge_attr to process_data function
            data = processing_func(data, edge_index, edge_attr)
        else:
            data = processing_func(data)
    # save data object with numeric index
    torch.save(data, os.path.join(processed_dir, f"{idx}.pt"))


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
        mutations=torch.tensor(data.binary_perturbation_indicator, dtype=torch.float32),
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
    torch.save(edge_index, const.PROCESSED_EDGE_INDEX_PATH)


def add_noise_to_graph(G: nx.DiGraph, noise_level: float = 0.1) -> nx.DiGraph:
    """ PLACEHOLDER FUNCTION"""
    # TODO: Implement actual noise addition logic
    return G  # No changes made, just a placeholder for now


def create_datasets_for_model(model_type, raw_data_dir="data/raw"):
    """
    Create datasets based on model type using the unified SDDataset approach.
    
    Args:
        model_type: Type of model ("pdgrapher", "GCNSI", "GAT", etc.)
        raw_data_dir: Directory containing raw data files
    """

    G, _ = utils.get_graph_data_from_topo(Path(const.TOPO_PATH) / f"{const.NETWORK}.topo")
    G = add_noise_to_graph(G, 0.1)
    # create a torch geometric edge_index from the graph
    G = from_networkx(G, group_edge_attrs=['weight'])
    G.edge_attr = G.edge_attr.float()
    
    if model_type == "pdgrapher":
        # Create both forward and backward datasets for PDGrapher
        print("Creating PDGrapher-style datasets...")

        store_edge_index_for_pdgrapher(G.edge_index)
        
        # Create backward dataset
        backward_dataset = SDDataset(
            model_type=model_type,
            processing_func=backward_data,
            mode="backward",
            raw_data_dir=raw_data_dir
        )
        
        # Create forward dataset  
        forward_dataset = SDDataset(
            model_type=model_type,
            processing_func=forward_data,
            mode="forward",
            raw_data_dir=raw_data_dir
        )
        
        print("PDGrapher-style datasets created successfully!")
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
            edge_attr = G.edge_attr
        )
        
        print(f"{model_type} dataset created successfully!")
        return {"main": dataset}
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def main():
    """
    Creates new processed data based on the selected model.
    """   
    print(f"Creating new processed data for {const.MODEL}...")
    
    # Model-specific processed directory
    processed_dir = Path(f"data/processed/{const.MODEL}")
    
    if processed_dir.exists():
        shutil.rmtree(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Create datasets using the unified approach
    datasets = create_datasets_for_model(const.MODEL)
    create_data_splits()

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