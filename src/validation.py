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


"""
IMPORTANT CHANGES FOR PER-NODE PREDICTIONS:

This validation script has been updated to handle models that make per-node predictions
instead of single graph-level predictions. Key changes:

1. predictions() function now returns per-node probability scores for each graph
2. Added node-level classification metrics (AUC-ROC, precision, recall, F1)
3. Graph-level metrics work by finding the node with highest prediction
4. Assumes exactly one source per graph (simplified logic)

The script now provides both:
- Graph-level metrics: How well does the model identify the correct source node?
- Node-level metrics: How well does the model classify each node as source/non-source?
"""


def distance_metrics(true_sources, pred_sources, data_set: list) -> dict:
    """
    Get the average min matching distance and the average distance to the source in general.
    :param true_sources: list of true source node indices
    :param pred_sources: list of predicted source node indices
    :param data_set: list of data instances containing true labels
    :return: dictionary with the average minimum matching distance and average distance to the source
    """
    dists_to_source = []

    for i, true_source in enumerate(
        tqdm(true_sources, desc="calculate distances", disable=const.ON_CLUSTER)
    ):
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
    true_positive_rate = true_positive / total_instances
    false_positive_rate = false_positive / total_instances
    
    f1 = f1_score(true_sources, pred_sources, average='weighted')

    return {
        "true positive rate": true_positive_rate,
        "false positive rate": false_positive_rate,
        "f1 score": f1,
    }



def prediction_metrics(pred_label_set: list, true_sources: list) -> dict:
    """
    Get the average rank of the source, the average prediction for the source,
    and additional metrics that help to evaluate the prediction.
    Also computes how often the true source is in the top 3 and top 5 predictions.
    :param pred_label_set: predicted labels for each data instance in the data set (per-node probabilities)
    :param true_sources: list of true source node indices
    :return: dictionary with prediction metrics
    """
    source_ranks = []
    predictions_for_source = []
    general_predictions = []
    in_top3 = []
    in_top5 = []

    for i, pred_labels in enumerate(
        tqdm(pred_label_set, desc="evaluate model", disable=const.ON_CLUSTER)
    ):
        true_source = true_sources[i]
        ranked_predictions = (utils.ranked_source_predictions(pred_labels)).tolist()

        source_rank = ranked_predictions.index(true_source)
        source_ranks.append(source_rank)
        predictions_for_source.append(pred_labels[true_source].item())
        general_predictions += pred_labels.tolist()

        in_top3.append(source_rank < 3)
        in_top5.append(source_rank < 5)

    return {
        "avg rank of source": np.mean(source_ranks),
        "avg prediction for source": np.mean(predictions_for_source),
        "avg prediction over all nodes": np.mean(general_predictions),
        "min prediction over all nodes": min(general_predictions),
        "max prediction over all nodes": max(general_predictions),
        "source in top 3": np.mean(in_top3),
        "source in top 5": np.mean(in_top5),
    }


def supervised_metrics(
    pred_label_set: list,
    raw_data_set: list,
    processed_data_set: list,
    true_sources: list,
    pred_sources: list,
) -> dict:
    """
    Performs supervised evaluation metrics for models that predict whether each node is a source or not.
    :param pred_label_set: list of predicted labels for each data instance in the data set
    :param raw_data_set: the raw data set containing true labels
    :param processed_data_set: the processed data set with PyTorch Geometric format
    :param true_sources: list of true source node indices
    :param pred_sources: list of predicted source node indices
    :return: dictionary containing the evaluation metrics
    """
    metrics = {}

    print("Evaluating Model ...")

    # Graph-level metrics (source detection)
    metrics |= prediction_metrics(pred_label_set, true_sources)
    metrics |= distance_metrics(true_sources, pred_sources, raw_data_set)
    metrics |= TP_FP_metrics(true_sources, pred_sources)
    
    # Node-level metrics (binary classification)
    metrics |= node_classification_metrics(pred_label_set, processed_data_set)

    for key, value in metrics.items():
        if isinstance(value, (int, float)):
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
    :return: predictions generated by the model (per-node predictions)
    """
    predictions = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    for data in tqdm(
        data_set, desc="make predictions with model", disable=const.ON_CLUSTER
    ):
        data = data.to(device)
        with torch.no_grad():
            pred = model(data).detach().squeeze()  # shape: [N] for N nodes
            # Apply sigmoid to get probabilities for each node being a source
            pred_probs = torch.sigmoid(pred)
        predictions.append(pred_probs.cpu())

    return predictions


def extract_true_sources(processed_data) -> list:
    """
    Extract true source nodes from the processed data (exactly one source per graph).
    :param processed_data: list of PyTorch Geometric data instances
    :return: list of source node indices (one per graph)
    """
    true_sources = []
    
    for data in processed_data:
        # Find the single source node (y == 1)
        source_node = torch.where(data.y == 1)[0][0].item()
        true_sources.append(source_node)
    
    return true_sources


def node_classification_metrics(pred_label_set: list, processed_data: list) -> dict:
    """
    Calculate AUC-ROC and other binary classification metrics for per-node source detection.
    :param pred_label_set: list of per-node prediction probabilities
    :param processed_data: list of data instances with true labels
    :return: dictionary with classification metrics
    """
    all_preds = []
    all_labels = []
    
    for i, (pred_probs, data) in enumerate(zip(pred_label_set, processed_data)):
        # Ensure pred_probs is flattened
        if isinstance(pred_probs, torch.Tensor):
            pred_list = pred_probs.flatten().tolist()
        else:
            pred_list = pred_probs
        all_preds.extend(pred_list)
        
        # Ensure labels are flattened integers
        if isinstance(data.y, torch.Tensor):
            label_list = data.y.flatten().int().tolist()
        else:
            label_list = data.y
        all_labels.extend(label_list)
    
    # Calculate AUC-ROC if we have both classes
    unique_labels = set(all_labels)
    if len(unique_labels) > 1:
        auc_roc = roc_auc_score(all_labels, all_preds)
        fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
    else:
        auc_roc = 0.0
        fpr, tpr, thresholds = None, None, None
    
    # Calculate binary predictions using 0.5 threshold
    binary_preds = [1 if p > 0.5 else 0 for p in all_preds]
    
    # Calculate precision, recall, F1
    from sklearn.metrics import precision_score, recall_score, f1_score
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

    model = utils.load_model(model, model_name)
    raw_test_data = utils.load_raw_data(split="test")
    processed_test_data = utils.load_processed_data(split="test")
    pred_labels = predictions(model, processed_test_data)
    
    # Extract true sources (exactly one per graph)
    true_sources = extract_true_sources(processed_test_data)
    
    # Get predicted sources (node with highest prediction probability)
    pred_sources = [pred.argmax().item() for pred in pred_labels]

    metrics_dict = {}
    metrics_dict["network"] = network
    metrics_dict["metrics"] = supervised_metrics(
        pred_labels, raw_test_data, processed_test_data, true_sources, pred_sources
    )
    metrics_dict["data stats"] = data_stats(raw_test_data)
    metrics_dict["parameters"] = yaml.full_load(open("params.yaml", "r"))
    utils.save_metrics(metrics_dict, model_name, network)


if __name__ == "__main__":
    main()
