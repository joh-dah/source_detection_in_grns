""" Utility functions for data loading and machine learning. """
from pathlib import Path
import json
import os
import torch
import src.constants as const
import src.data_processing_combined as dp
import glob
import pandas as pd
import networkx as nx
import networkx as nx
import pandas as pd
from datetime import datetime


def latest_model_name():
    """
    Extracts the name of the latest trained model.
    Gets the name of the newest file in the model folder,
    that is not the "latest.pth" file and splits the path to extract the name.
    """
    model_files = glob.glob(f"{const.MODEL_PATH}/*.pth")
    model_files = [file for file in model_files if "latest" not in file]
    last_model_file = max(model_files, key=os.path.getctime)
    model_name = os.path.split(last_model_file)[1].split(".")[0]
    return model_name


def save_model(model, name: str):
    """
    Saves model state to path.
    :param model: model with state
    :param name: name of model
    """
    Path(const.MODEL_PATH).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{const.MODEL_PATH}/{name}.pth")


def load_model(model, model_path: str):
    """
    Loads model state from path.
    :param model: model
    :param model_name: path to model
    :return: model with loaded state
    """
    print(f"loading model: {model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
    return model


def ranked_source_predictions(
    predictions: torch.tensor, n_nodes: int = None
) -> torch.tensor:
    """
    Return nodes ranked by predicted probability of beeing source.
    Selects the n nodes with the highest probability.
    :param predictions: list of predictions of nodes beeing source
    :param n_nodes: amount of nodes to return
    :return: list of nodes ranked by predicted probability of beeing source
    """
    if n_nodes is None:
        n_nodes = predictions.shape[0]
    if const.MODEL in ["GAT", "GCNSI", "pdgrapher"]:
        top_nodes = torch.topk(predictions.flatten(), n_nodes).indices
    else:
        raise ValueError(f"Model {const.MODEL} not supported for ranked source predictions.")
    return  top_nodes


def save_metrics(metrics: dict, model_name: str, network: str):
    """
    Save dictionary with metrics as json in reports folder.
    One "latest.json" is created and named after the corresponding model.
    :params metrics: dictionary containing metrics
    :params model_name: name of the corresponding model
    """
    (Path(const.REPORT_PATH) / model_name).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d_%H%M")
    filename = f"{network}_{timestamp}.json"
    with open(
        os.path.join((Path(const.REPORT_PATH) / model_name), filename), "w"
    ) as file:
        json.dump(metrics, file, indent=4)
    # with open(os.path.join(const.REPORT_PATH, "latest.json"), "w") as file:
    #     json.dump(metrics, file, indent=4)



def load_processed_data(split="train"):
    """
    Load processed data using the new data structure.
    
    Args:
        split: "train", "val", or "test"
    
    Returns:
        Dataset: Dataset for the specified split
    """
    processed_dir = Path(const.PROCESSED_PATH)
    
    # Load splits - they are stored in splits/splits.pt subdirectory
    splits_file = processed_dir / "splits" / "splits.pt"
    splits = torch.load(splits_file, weights_only=False)
    split_indices = splits[f'{split}_index']
    
    # Simple dataset class that loads individual processed files
    class ProcessedDataset(torch.utils.data.Dataset):
        def __init__(self, processed_dir, indices):
            self.processed_dir = processed_dir
            self.indices = indices
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            file_idx = self.indices[idx]
            file_path = self.processed_dir / f"{file_idx}.pt"
            return torch.load(file_path, weights_only=False)
    
    return ProcessedDataset(processed_dir, split_indices)


def create_topo_file_from_graph(network_name, G: nx.DiGraph, dir):
    """
    Create a topo file as expected by racipe from a nx Graph
    and store it in the const.TOPO_PATH directory.
    :param G: nx Graph
    """
    new_file_path = Path(dir) / f"{network_name}.topo" 
    # save graph to a trrust.topo file with the header Source Target Type
    with open(new_file_path, "w") as f:
        f.write("Source Target Type\n")
        for u, v, d in G.edges(data='weight'):
            f.write(f"{u} {v} {d}\n")


def get_graph_data_from_topo(filepath=None):
    """
    Reads a .topo file and returns:
    - A NetworkX directed graph with gene names as node labels and 'Type' as edge weight.
    - A mapping from gene names to integer indices (useful for ML models like PyG).
    
    :param filepath: path to the topology file
    :return: G_named (NetworkX DiGraph), gene_to_idx (dict)
    """
    if filepath is None:
        filepath = Path(const.TOPO_PATH) / f"{const.NETWORK}.topo"

    df = pd.read_csv(filepath, sep=r"\s+")

    # Create gene-to-index mapping for optional ML use
    genes = sorted(set(df['Source']).union(df['Target']))
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # Build NetworkX DiGraph with weights
    edges_with_weights = list(zip(df['Source'], df['Target'], df['Type']))
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_with_weights)

    return G, gene_to_idx


def load_raw_data(path: str = None):
    """
    Load raw data.
    :param split: split of the data to load, can be "train", "validation" or "test"
    :return: list of raw Data objects
    """
    print("Load raw data...")

    if path is None:
        path = Path(const.RAW_PATH)

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return [torch.load(p, weights_only=False) for p in path.glob("*.pt")]
