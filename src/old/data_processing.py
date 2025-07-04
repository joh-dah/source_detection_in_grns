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



# def paper_input(current_status: torch.tensor, edge_index: torch.tensor) -> torch.tensor:
#     """
#     Prepares the input features for the GCNSI model according to the paper:
#     https://dl.acm.org/doi/abs/10.1145/3357384.3357994
#     :param current_status: the current infection status
#     :param edge_index: edge_index of a graph
#     :return: prepared input features
#     """
#     Y = np.array(current_status)
#     g = to_networkx(Data(edge_index=edge_index), to_undirected=False).to_undirected()
#     S = nx.normalized_laplacian_matrix(g)
#     V3 = Y.copy()
#     Y = [-1 if x == 0 else 1 for x in Y]
#     V4 = [-1 if x == -1 else 0 for x in Y]
#     I = np.identity(len(Y))
#     a = const.ALPHA
#     d1 = Y
#     temp = (1 - a) * np.linalg.inv(I - a * S)
#     d2 = np.squeeze(np.asarray(temp.dot(Y)))
#     d3 = np.squeeze(np.asarray(temp.dot(V3)))
#     d4 = np.squeeze(np.asarray(temp.dot(V4)))
#     X = torch.from_numpy(np.column_stack((d1, d2, d3, d4))).float()
#     return X


def paper_input(current_status: torch.tensor, edge_index: torch.tensor) -> torch.tensor: #TODO: this is chatGPTs take on how to handle directed,cyclic graphs. check if valid
    Y = np.array(current_status[:, 1])  # second feature column -> diff expression
    g = to_networkx(Data(edge_index=edge_index), to_undirected=False)

    A = nx.to_numpy_array(g)
    D_out = np.diag(A.sum(axis=1))
    D_inv = np.linalg.pinv(D_out)


    # Transition matrix for diffusion
    P = D_inv @ A

    V3 = Y.copy()
    Y = [-1 if x == 0 else 1 for x in Y]
    V4 = [-1 if x == -1 else 0 for x in Y]
    I = np.identity(len(Y))

    a = const.ALPHA
    d1 = Y
    temp = (1 - a) * np.linalg.inv(I - a * P)
    d2 = np.squeeze(np.asarray(temp.dot(Y)))
    d3 = np.squeeze(np.asarray(temp.dot(V3)))
    d4 = np.squeeze(np.asarray(temp.dot(V4)))

    X = torch.from_numpy(np.column_stack((d1, d2, d3, d4))).float()
    return X


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


def process_gcnsi_data(data: Data) -> Data:
    """
    Features and Labels for the GCNSI model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = paper_input(data.x, data.edge_index)
    # expand labels to 2D tensor
    data.y = data.y.unsqueeze(1).float()
    return data


def process_simplified_gcnsi_data(data: Data) -> Data:
    """
    Simplified features and Labels for the GCNSI model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = data.x.float()  # Assume x is already shape [N, 2]
    data.y = data.y.unsqueeze(1).float()
    return data



def process_gcnr_data(data: Data) -> Data:
    """
    Features and Labels for the GCNR model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = paper_input(data.x, data.edge_index)
    data.y = create_distance_labels(to_networkx(data, to_undirected=False), data.y)
    return data


def process_simplified_gcnr_data(data: Data) -> Data:
    """
    Features and Labels for the GCNR model.
    :param data: input data to be processed.
    :return: processed data with expanded features and labels
    """
    data.x = data.x.float()
    data.y = data.y.unsqueeze(1).float()
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
    if const.MODEL == "GCNSI":
        if const.SMALL_INPUT:
            pre_transform_function = process_simplified_gcnsi_data
        else:
            pre_transform_function = process_gcnsi_data
    elif const.MODEL == "GCNR":
        if const.SMALL_INPUT:
            pre_transform_function = process_simplified_gcnr_data
        else:
            pre_transform_function = process_gcnr_data

    # triggers the process function of the dataset
    SDDataset(
        path,
        pre_transform=pre_transform_function,
    )


if __name__ == "__main__":
    main()
