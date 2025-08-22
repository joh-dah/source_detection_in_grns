""" Utility functions for data loading and machine learning. """
from pathlib import Path
import json
import os
import torch
import src.constants as const
import src.data_processing as dp
import glob
import pandas as pd
import networkx as nx
import networkx as nx
import pandas as pd
from datetime import datetime
import pickle


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
    if const.MODEL.lower() in ["gat", "gcnsi", "pdgrapher"]:
        top_nodes = torch.topk(predictions.flatten(), n_nodes).indices
    else:
        raise ValueError(f"Model {const.MODEL} not supported for ranked source predictions.")
    return  top_nodes


def get_current_time() -> str:
    """
    Get the current timestamp in the format MMDD_HHMM.
    
    Returns:
        str: Current timestamp formatted as MMDD_HHMM.
    """
    return datetime.now().strftime("%m%d_%H%M")


def save_metrics(metrics: dict, method_name: str = None):
    """
    Save dictionary with metrics as json in reports folder.
    One "latest.json" is created and named after the corresponding model.
    :params metrics: dictionary containing metrics
    :params model_name: name of the corresponding model
    """
    timestamp = datetime.now().strftime("%m%d_%H%M")
    if method_name is None:
        method_name = const.MODEL
    report_dir = Path(const.REPORT_PATH) / const.EXPERIMENT
    report_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{method_name}_{timestamp}.json"
    with open(report_dir / filename, "w") as file:
        json.dump(metrics, file, indent=4)


def extract_gat_true_sources(processed_test_data):
    """Extract true sources from GAT processed data."""
    true_sources = []
    for data in processed_test_data:
        source_node = torch.where(data.y == 1)[0][0].item()
        true_sources.append(source_node)
    return true_sources


def load_processed_data(split="train", model_type=None):
    """
    Load processed data using the new data structure.
    
    Args:
        split: "train", "val", or "test"
    
    Returns:
        Dataset: Dataset for the specified split
    """
    splits_file = Path(const.SPLITS_PATH)
    splits = torch.load(splits_file, weights_only=False)
    split_indices = splits[f'{split}_index_backward']

    #TODO remove debug
    print(f"Train indices: {len(splits['train_index_backward'])}")
    print(f"Head of train indices: {splits['train_index_backward'][:5]}")
    print(f"Validation indices: {len(splits['val_index_backward'])}")
    print(f"Head of validation indices: {splits['val_index_backward'][:5]}")
    print(f"Test indices: {len(splits['test_index_backward'])}")
    print(f"Head of test indices: {splits['test_index_backward'][:5]}")

    if model_type is None:
        model_type = const.MODEL.lower()
    data_path = Path(const.DATA_PATH) / f"processed/{model_type}"

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
    
    return ProcessedDataset(data_path, split_indices)


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


def load_perturbed_graph():
    """
    Load the perturbed graph from the data directory stored as graph.pkl
    """
    graph_path = Path(const.DATA_PATH) / "graph.pkl"
    if not graph_path.exists():
        raise FileNotFoundError(f"Graph file not found: {graph_path}. Please run graph perturbation first.")
    
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    
    return G


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


def load_raw_test_data():
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
