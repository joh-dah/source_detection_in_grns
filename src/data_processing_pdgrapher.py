""" Creates new processed data based on the selected model. """
import argparse
import glob
from pathlib import Path
from src import constants as const
import networkx as nx
import numpy as np
import os
import shutil
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch
import multiprocessing as mp
from tqdm import tqdm
from src.create_splits import create_splits_for_data
from torch_geometric.utils import add_remaining_self_loops, to_undirected


def process_all_data(root_path, data_type="backward"):
    """
    Process all data files and save as a single list.
    
    Args:
        root_path: Path to the data directory
        data_type: Either "backward" or "forward" to determine processing type
    """
    raw_dir = Path(root_path) / "raw"
    processed_dir = Path(root_path) / "processed"
    processed_dir.mkdir(exist_ok=True)
    
    # Load and sort raw file paths
    raw_files = sorted(raw_dir.glob("*.pt"))
    
    if not raw_files:
        print(f"No .pt files found in {raw_dir}")
        return
    
    print(f"Processing {len(raw_files)} files for {data_type} data...")
    
    # Process all files
    processed_data_list = []
    for raw_path in tqdm(raw_files, desc=f"Processing {data_type} data"):
        # Load raw data
        data = torch.load(raw_path, weights_only=False)
        
        # Process based on data type
        if data_type == "backward":
            processed_data = backward_data(data)
        elif data_type == "forward":
            processed_data = forward_data(data)
        else:
            raise ValueError(f"Unknown data_type: {data_type}")
        
        processed_data_list.append(processed_data)
    
    # Save the complete list to a single file
    output_file = processed_dir / f"data_{data_type}.pt"
    torch.save(processed_data_list, output_file)
    print(f"Saved {len(processed_data_list)} {data_type} data objects to {output_file}")


def process_single(args):
    pre_transform, raw_path, idx, processed_dir = args
    # load raw data
    data = torch.load(raw_path, weights_only=False)
    # process data
    if pre_transform is not None:
        data = pre_transform(data)
    # save data object with numeric index
    torch.save(data, os.path.join(processed_dir, f"{idx}.pt"))


class SDDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        self.root = Path(root)
        self._raw_dir = self.root / "raw"
        self._processed_dir = self.root / "processed"
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = pre_filter

        # Load and sort raw file paths
        self.raw_files = sorted(self._raw_dir.glob("*.pt"))
        self.size = len(self.raw_files)

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [f.name for f in self.raw_files]

    @property
    def processed_file_names(self):
        return [f"{i}.pt" for i in range(self.size)]

    def process(self):
        if self.pre_transform is not None:
            params = [
                (self.pre_transform, str(self.raw_files[i]), i, str(self._processed_dir))
                for i in range(self.size)
            ]
            with mp.get_context("spawn").Pool(const.N_CORES) as pool:
                print(f"Processing data set using multiprocessing ({const.N_CORES} cores)...")
                list(tqdm(pool.imap_unordered(process_single, params), total=self.size))

    def len(self):
        return self.size

    def get(self, idx):
        path = self._processed_dir / f"{idx}.pt"
        data = torch.load(path, weights_only=False)
        return data


def normalize_paired_tensors(tensor1: torch.Tensor, tensor2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Normalizes two related tensors using the same min-max scale.
    This preserves the relative relationship between the two states.
    
    :param tensor1: First tensor (e.g., healthy expression)
    :param tensor2: Second tensor (e.g., diseased expression)
    :return: Tuple of normalized tensors
    """
    tensor1 = torch.from_numpy(tensor1)
    tensor2 = torch.from_numpy(tensor2)
    
    # Ensure tensors are float type
    tensor1 = tensor1.float()
    tensor2 = tensor2.float()
    
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
    with normalized healthy and diseased expression profiles, mutation indicators, and gene metadata.

    Args:
        data (Data): Input Data object with the following attributes:
            - original: Raw healthy gene expression values.
            - perturbed: Raw diseased gene expression values.
            - perturbed_gene_list: Tensor indicating mutated genes (binary mask).
            - gene_symbols: List of gene symbol strings.
            - num_nodes: Number of genes (nodes).

    Returns:
        Data: A new Data object with the following attributes:
            - perturbagen_name: Name of the perturbagen (gene).
            - diseased: Normalized diseased gene expression tensor.
            - intervention: Binary tensor indicating whether the gene is treated (1) or not (0).
            - treated: Normalized treated gene expression tensor.
            - gene_symbols: List of gene symbol strings.
            - mutations: Tensor indicating mutated genes (binary mask).
            - num_nodes: Number of genes (nodes).
    """
    # Normalize both tensors with the same scale
    diseased_norm, treated_norm = normalize_paired_tensors(data.original, data.perturbed)
    
    return Data(
        perturbagen_name=data.perturbed_gene,
        diseased=diseased_norm,
        intervention=torch.tensor(data.perturbed_gene_list, dtype=torch.float32),
        treated=treated_norm,
        gene_symbols=data.gene_symbols,
        mutations=torch.tensor(data.perturbed_gene_list, dtype=torch.float32),
        num_nodes=data.num_nodes,
    )


def forward_data(data: Data) -> Data:
    """
    Transforms a Data object containing original and perturbed gene expression data into a new Data object
    with normalized healthy and diseased expression profiles, mutation indicators, and gene metadata.

    Args:
        data (Data): Input Data object with the following attributes:
            - original: Raw healthy gene expression values.
            - perturbed: Raw diseased gene expression values.
            - perturbed_gene_list: Tensor indicating mutated genes (binary mask).
            - gene_symbols: List of gene symbol strings.
            - num_nodes: Number of genes (nodes).

    Returns:
        Data: A new Data object with the following attributes:
            - healthy: Normalized healthy gene expression tensor.
            - diseased: Normalized diseased gene expression tensor.
            - mutations: Tensor indicating mutated genes (binary mask).
            - gene_symbols: List of gene symbol strings.
            - num_nodes: Number of genes (nodes).
    """
    # Normalize both tensors with the same scale
    healthy_norm, diseased_norm = normalize_paired_tensors(data.original, data.perturbed)
    
    return Data(
        healthy=healthy_norm,
        diseased=diseased_norm,
        mutations=torch.tensor(data.perturbed_gene_list, dtype=torch.float32),
        gene_symbols=data.gene_symbols,
        num_nodes=data.num_nodes,
    )


def main():
    """
    Creates new processed data based on the selected model.
    """   
    print(f"Creating new processed data...")
    # Remove old processed data
    processed_dir = Path(const.DATA_PATH) / "processed"
    if processed_dir.exists():
        shutil.rmtree(processed_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create PDGrapher-style datasets (both forward and backward)
    create_pdgrapher_style_datasets(const.DATA_PATH)


def store_edge_index(root_path=None):
    """
    process the raw edge index and store it in the processed directory.
    
    Args:
        edge_index: Edge index to store (if None, it will be created from PPI)
        root_path: Path to the data directory
    """
    if root_path is None:
        root_path = const.DATA_PATH
    print("Storing edge index...")
    # Load edge index
    edge_index_file = Path(root_path) / "edge_index" / "raw_edge_index.pt"
    if not edge_index_file.exists():
        raise FileNotFoundError(f"Edge index file not found: {edge_index_file}")    
    edge_index = torch.load(edge_index_file, weights_only=False)

    edge_index = add_remaining_self_loops(edge_index)[0]
    edge_index = to_undirected(edge_index)
    
    # Save edge index
    processed_dir = Path(root_path) / "processed"
    torch.save(edge_index, processed_dir / "edge_index.pt")


def create_pdgrapher_style_datasets(root_path):
    """
    Create both forward and backward datasets in PDGrapher style.
    This creates separate data_forward.pt and data_backward.pt files.
    Also creates the splits.pt file needed for training.
    
    Args:
        root_path: Path to the data directory containing raw data
    """
    print("Creating PDGrapher-style datasets...")
    
    store_edge_index(root_path=root_path)
    # Process backward data (diseased + intervention -> treated)
    process_all_data(root_path, data_type="backward")
    
    # Process forward data (healthy -> mutations -> diseased)
    process_all_data(root_path, data_type="forward")
    
    print("PDGrapher-style datasets created successfully!")

    create_splits_for_data(root_path, nfolds=const.N_FOLDS, splits_type='random')




class PDGrapherDataset:
    """
    Simple dataset class for loading PDGrapher-style data from single files.
    """
    def __init__(self, root, data_type="backward"):
        self.root = Path(root)
        self.data_type = data_type
        self.processed_dir = Path(const.DATA_PATH) / "processed"
        
        # Load the data list
        data_file = self.processed_dir / f"data_{data_type}.pt"
        if data_file.exists():
            self.data_list = torch.load(data_file, weights_only=False)
        else:
            self.data_list = []
            print(f"Warning: {data_file} not found!")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx]
    
    def get_data_list(self):
        """Return the complete data list."""
        return self.data_list


if __name__ == "__main__":
    main()

