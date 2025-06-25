""" Initiates the validation of the classifier specified in the constants file. """
import argparse
import yaml
import os.path
import numpy as np
import torch
import torch_geometric
from tqdm import tqdm
import networkx as nx
from sklearn.metrics import roc_auc_score, roc_curve
import rpasdt.algorithm.models as rpasdt_models
from rpasdt.algorithm.simulation import perform_source_detection_simulation
from rpasdt.algorithm.taxonomies import SourceDetectionAlgorithm
from torch_geometric.utils.convert import from_networkx
from sklearn.metrics import f1_score

import src.data_processing as dp
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
from architectures.GAT import GAT
import src.constants as const
from src import visualization as vis
from src import utils


def distance_metrics(true_sources, pred_sources, data_set: list) -> dict:
    """
    Get the average min matching distance and the average distance to the source in general.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param data_set: list of data instances containing true labels
    :return: dictionary with the average minimum matching distance and average distance to the source
    """
    dists_to_source = []

    for i, true_source in enumerate(
        tqdm(true_sources, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        # get the distance from true_source to pred_source
        # both true_source and pred_source are indices of the nodes in the graph
        pred_source = pred_sources[i]
        if true_source == pred_source:
            dists_to_source.append(0)
        else:
            # get the graph from the data instance
            nx_graph = nx.from_edgelist(data_set[i].edge_index.t().tolist())

            # calculate the shortest path distance
            try:
                dist = nx.shortest_path_length(nx_graph, source=true_source, target=pred_source)
            except nx.NetworkXNoPath:
                dist = float("inf")
            dists_to_source.append(dist)

    return {
        "avg dist to source": np.mean(dists_to_source),
    }


def TP_FP_metrics(true_sources: list, pred_sources: list) -> dict:
    """
    Calculate the true positive and false positive rates based on the true and predicted sources.
    :param true_sources: list of true sources for each data instance
    :param pred_sources: list of predicted sources for each data instance
    :return: dictionary with the true positive rate and false positive rate.
    """
    true_positive = 0
    false_positive = 0

    for true_source, pred_source in zip(true_sources, pred_sources):
        if true_source == pred_source:
            true_positive += 1
        else:
            false_positive += 1

    total_instances = len(true_sources)
    true_positive_rate = true_positive / total_instances if total_instances > 0 else 0
    false_positive_rate = false_positive / total_instances if total_instances > 0 else 0

    return {
        "true positive rate": true_positive_rate,
        "false positive rate": false_positive_rate,
        "f1 score": f1_score(true_sources, pred_sources, average='weighted')
    }



def prediction_metrics(pred_label_set: list, true_sources: list) -> dict:
    """
    Get the average rank of the source, the average prediction for the source
    and additional metrics that help to evaluate the prediction.
    :param pred_label_set: predicted labels for each data instance in the data set
    :param data_set: a data set containing true labels
    :return: dictionary with prediction metrics
    """
    source_ranks = []
    predictions_for_source = []
    general_predictions = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_source = true_sources[i]
        ranked_predictions = (utils.ranked_source_predictions(pred_labels)).tolist()

        source_ranks.append(ranked_predictions.index(true_source))
        predictions_for_source.append(pred_labels[true_source].item())
        general_predictions += pred_labels.tolist()

    return {
        "avg rank of source": np.mean(source_ranks),
        "avg prediction for source": np.mean(predictions_for_source),
        "avg prediction over all nodes": np.mean(general_predictions),
        "min prediction over all nodes": min(general_predictions),
        "max prediction over all nodes": max(general_predictions),
    }


def supervised_metrics(
    pred_label_set: list,
    raw_data_set: list,
    true_sources: list,
    pred_sources: list,
) -> dict:
    """
    Performs supervised evaluation metrics for models that predict whether each node is a source or not.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param data_set: the data set containing true labels
    :param model_name: name of the model being evaluated
    :param threshold: threshold for the predicted labels
    :return: dictionary containing the evaluation metrics
    """
    metrics = {}

    print("Evaluating Model ...")

    metrics |= prediction_metrics(pred_label_set, true_sources)
    metrics |= distance_metrics(true_sources, pred_sources, raw_data_set)
    metrics |= TP_FP_metrics(true_sources, pred_sources)

    for key, value in metrics.items():
        metrics[key] = round(value, 3)
        print(f"{key}: {metrics[key]}")

    return metrics


def data_stats(raw_data_set: list) -> dict:
    """
    Calculates various graph-related statistics and infection-related statistics for the provided raw data set.
    :param raw_data_set: the raw data set.
    :return: dictionary containing the calculated statistics
    """
    n_nodes = []
    n_sources = []
    centrality = []
    n_nodes_affected = []
    precent_affected = []
    for data in tqdm(raw_data_set, desc="get data stats"):
        n_nodes.append(len(data.y))
        n_sources.append(len(torch.where(data.y == 1)[0].tolist()))
        n_nodes_affected.append(len(torch.where(data.x[1] != 0)[0].tolist()))
        precent_affected.append(n_nodes_affected[-1] / n_nodes[-1])
        graph = nx.from_edgelist(data.edge_index.t().tolist())
        centrality.append(np.mean(list(nx.degree_centrality(graph).values())))

    stats = {
        "graph stats": {
            "avg number of nodes": np.mean(n_nodes),
            "avg centrality": np.mean(centrality),
        },
        "infection stats": {
            "avg number of sources": np.mean(n_sources),
            "avg portion of affected nodes": np.mean(precent_affected),
            "std portion of affected nodes": np.std(precent_affected),
        },
    }

    for key, value in stats.items():
        for k, v in value.items():
            stats[key][k] = round(v, 3)

    return stats


def predictions(model: torch.nn.Module, data_set: dp.SDDataset) -> list:
    """
    Generate predictions using the specified model for the given data set.
    :param model: the model used for making predictions
    :param data_set: the data set to make predictions on
    :return: predictions generated by the model
    """
    predictions = []
    for data in tqdm(
        data_set, desc="make predictions with model", disable=const.ON_CLUSTER
    ):
        pred = model(data).detach()[0]
        print(f"Prediction: {pred}")
        predictions.append(pred)

    return predictions


def main():
    """
    Initiates the validation of the classifier specified in the constants file.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", 
        type=str, 
        help="name of the network that should be used"
    )
    args = parser.parse_args()
    network = args.network
    model_name = (
        utils.latest_model_name() if const.MODEL_NAME is None else const.MODEL_NAME
    )

    # assert that model is gat
    assert const.MODEL == "GAT", "This validation script only supports GAT models."
    model = GAT()

    model = utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))
    raw_val_data = utils.load_raw_data(validation=True)
    processed_val_data = utils.load_processed_data(validation=True)
    pred_labels = predictions(model, processed_val_data)
    true_sources = [data.y for data in processed_val_data]
    pred_sources = [pred.argmax().item() for pred in pred_labels]

    metrics_dict = {}
    metrics_dict["network"] = network
    metrics_dict["metrics"] = supervised_metrics(
        pred_labels, raw_val_data, true_sources, pred_sources
    )
    metrics_dict["data stats"] = data_stats(raw_val_data)
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))
    utils.save_metrics(metrics_dict, model_name, network)


if __name__ == "__main__":
    main()
