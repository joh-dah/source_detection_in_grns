""" Utility functions for data loading and machine learning. """
from pathlib import Path
import json
import os
import torch
import src.constants as const
import src.data_processing as dp
import glob
import torch_geometric.datasets as datasets
import torch_geometric.transforms as T
import networkx as nx
import pandas as pd


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


def load_model(model, path: str):
    """
    Loads model state from path.
    :param model: model
    :param path: path to model
    :return: model with loaded state
    """
    print(f"loading model: {path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
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
    if const.MODEL == "GCNSI":
        top_nodes = torch.topk(predictions.flatten(), n_nodes).indices
    elif const.MODEL == "GCNR":
        top_nodes = torch.topk(predictions.flatten(), n_nodes, largest=False).indices
    return top_nodes


def save_metrics(metrics: dict, model_name: str, network: str):
    """
    Save dictionary with metrics as json in reports folder.
    One "latest.json" is created and named after the corresponding model.
    :params metrics: dictionary containing metrics
    :params model_name: name of the corresponding model
    """
    (Path(const.REPORT_PATH) / model_name).mkdir(parents=True, exist_ok=True)
    with open(
        os.path.join((Path(const.REPORT_PATH) / model_name), f"{network}.json"), "w"
    ) as file:
        json.dump(metrics, file, indent=4)
    # with open(os.path.join(const.REPORT_PATH, "latest.json"), "w") as file:
    #     json.dump(metrics, file, indent=4)


def load_processed_data(validation: bool = False):
    """
    Load processed data
    :param validation: whether to load validation or training data (default: load training data)
    :return: processed data
    """
    print("Load processed data...")

    if const.MODEL == "GCNSI" and const.SMALL_INPUT:
        pre_transform = dp.process_simplified_gcnsi_data
    elif const.MODEL == "GCNR" and const.SMALL_INPUT:
        pre_transform = dp.process_simplified_gcnr_data
    elif const.MODEL == "GCNSI":
        pre_transform = dp.process_gcnsi_data
    elif const.MODEL == "GCNR":
        pre_transform = dp.process_gcnr_data

    train_or_val = "validation" if validation else "training"
    # TODO why dont we need to add "processed" here but not raw?
    path = Path(const.DATA_PATH) / train_or_val

    data = dp.SDDataset(
        path,
        pre_transform=pre_transform,
    )

    return data



def load_raw_data(validation: bool = False):
    """
    Load raw data.
    :param validation: whether to load validation or training data (default: training)
    :return: list of raw Data objects
    """
    print("Load raw data...")

    split_dir = "validation" if validation else "training"
    path = Path(const.DATA_PATH) / split_dir / "raw"

    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")

    return [torch.load(p, weights_only=False) for p in path.glob("*.pt")]
