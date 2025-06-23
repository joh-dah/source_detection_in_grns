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
from tqdm import tqdm # Make sure const.N_CORES is defined


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


def create_distance_labels(
    graph: nx.DiGraph, initial_values: torch.tensor
) -> torch.tensor:
    """
    Creates the labels for the GCNR model. Each label is the distance of the node to the nearest source.
    :param graph: graph for which to create the distance labels
    :param initial_values: initial values indicating the source nodes
    :return: distance labels
    """
    distances = []
    # extract all sources from prob_model
    sources = torch.where(initial_values == 1)[0].tolist()
    for source in sources:
        distances.append(nx.single_source_shortest_path_length(graph, source))
    # get min distance for each node
    min_distances = []
    for node in graph.nodes:
        min_distances.append(min([distance[node] for distance in distances]))

    return torch.tensor(np.expand_dims(min_distances, axis=1)).float()


def normalize_datapoint(x: torch.Tensor) -> torch.Tensor:
    """
    Normalizes the features of a data point.
    :param data: input data to be normalized.
    :return: normalized data
    """
    if const.NORMALIZE_DATA:
        original_expr = x[:, 0]
        delta_expr = x[:, 1]

        min_val = original_expr.min()
        max_val = original_expr.max()
        range_val = (max_val - min_val + 1e-8)
    
        # Min-Max auf Original
        norm_orig = (original_expr - min_val) / range_val
        # Δ in gleiche Skala bringen
        norm_delta = delta_expr / range_val

        x = torch.stack([norm_orig, norm_delta], dim=1)
    return x


def process_data(data: Data) -> Data:
    """
    Features and Labels for the model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = normalize_datapoint(data.x)
    data.edge_attr = data.edge_attr.unsqueeze(1).float()
    if const.MODEL == "GCNR":
        # For GCNR, we create distance labels based on the graph structure
        data.y = create_distance_labels(to_networkx(data, to_undirected=False), data.y)
    elif const.MODEL == "GCNSI":
        # For GCNSI, we assume y is already in the correct format
        data.y = data.y.unsqueeze(1).float()
    elif const.MODEL == "GAT":
        # For GAT we treat every node as class and therefore switch to classification
        # get the index of the source node
        source_node = torch.where(data.y == 1)[0]
        assert source_node.numel() == 1, f"Expected exactly one source node, got {source_node.numel()}"
        data.y = source_node.item() # Convert to single label for GAT
    return data


def main():
    """
    Creates new processed data based on the selected model.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to create validation or training data",
    )
    parser.add_argument(
        "--network", 
        type=str, 
        help="name of the network that should be used"
    )
    args = parser.parse_args()

    train_or_val = "validation" if args.validation else "training"
    path = Path(const.DATA_PATH) / train_or_val

    print("Removing old processed data...")
    shutil.rmtree(path / "processed", ignore_errors=True)

    print("Creating new processed data...")

    # triggers the process function of the dataset
    SDDataset(
        path,
        pre_transform=process_data,
    )


if __name__ == "__main__":
    main()
