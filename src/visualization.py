""" Visualize graphs with the associated predictions. """
import argparse
from typing import Union
import glob
import os
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.colors import Colormap
from pathlib import Path
import src.constants as const
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from tqdm import tqdm
from src import utils
from architectures.GCNR import GCNR
from architectures.GCNSI import GCNSI
from src.data_processing import SDDataset, process_gcnr_data, process_gcnsi_data


def plot_graph_with_colors(
    g: nx.Graph,
    node_values: np.ndarray,
    title: str,
    cmap: Union[Colormap, str] = "viridis",
    max_abs_value: float = None,
    layout: callable = nx.circular_layout,
):
    """
    Plots a graph with nodes colored according to node_values and labeled using node names.
    
    :param g: NetworkX graph
    :param node_values: 1D array of node values to map to colors
    :param title: plot title and filename
    :param cmap: colormap (default 'viridis')
    :param layout: layout function for node positioning
    """
    pos = layout(g)

    if max_abs_value:
        min_val, max_val = -max_abs_value, max_abs_value
    else:
        min_val = node_values.min()
        max_val = node_values.max()

    plt.figure(figsize=(8, 8))

    # Draw nodes with thickness depending on 'source' attribute
    node_border_widths = [
        5 if g.nodes[n].get("source") == 1 else 1 for n in g.nodes
    ]

    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=node_values,
        node_size=500,
        cmap=cmap,
        vmin=min_val,
        vmax=max_val,
        linewidths=node_border_widths,
        edgecolors="black",
    )

    # Set edge colors: green = activating (1), red = inhibiting (2)
    edge_colors = [
        "green" if g[u][v].get("weight") == 1 else "red"
        for u, v in g.edges
    ]
    nx.draw_networkx_edges(
        g,
        pos=pos,
        edge_color=edge_colors,
        width=3,
    )

    # Draw node labels (use actual node names from graph)
    nx.draw_networkx_labels(
        g, 
        pos={node: (x, y - 0.08) for node, (x, y) in pos.items()}, 
        font_size=10, 
        font_color="black"
    )

    # Add colorbar
    ax = plt.gca()
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.9)

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def plot_matching_graph(
    g: nx.Graph, matching: list, new_edges: list, title: str = "matching_graph"
):
    """
    Plots the matching graph to debug the min-matching distance metric.
    """
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)

    pos = nx.kamada_kawai_layout(g)

    plt.figure(figsize=(20, 20))
    edge_colors = [
        "green" if edge in matching else "red" if edge in new_edges else "black"
        for edge in g.edges
    ]
    colors = ["red" if node[0] == "s" else "blue" for node in g.nodes]
    nx.draw(
        g,
        pos=pos,
        with_labels=True,
        node_color=colors,
        edge_color=edge_colors,
        node_size=150,
    )
    nx.draw_networkx_edge_labels(
        g, pos, edge_labels=nx.get_edge_attributes(g, "weight")
    )
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def plot_roc_curve(
    true_positives: np.ndarray,
    false_positives: np.ndarray,
    thresholds: np.ndarray,
    model_name: str,
    network: str,
):
    """
    Plot ROC curves.
    :param false_positives: the false positives rates
    :param true_positives: the true positives rates
    :param model_name: the name of the model that is evaluated (used for saving the plot)
    :param network: the name of the network the model is evaluated on (used for saving the plot)
    """
    print("Visualize ROC curve:")
    (Path(const.ROC_PATH) / model_name).mkdir(parents=True, exist_ok=True)
    plt.scatter(
        false_positives,
        true_positives,
        c=thresholds,
        cmap="viridis",
        label="ROC curve",
    )
    plt.plot(
        false_positives,
        true_positives,
        color="black",
        linestyle="-",
        alpha=0.5,
    )
    plt.colorbar(label="Threshold")
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", label="Random guess")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC curve for {model_name} on {network}")
    plt.legend()
    plt.savefig(Path(const.ROC_PATH) / model_name / f"{network}_roc.png")
    plt.close()



def load_model_by_type(model_name: str):
    """
    Load the model based on the type defined in constants.
    """
    if const.MODEL == "GCNR":
        model = GCNR()
    elif const.MODEL == "GCNSI":
        model = GCNSI()
    return utils.load_model(model, os.path.join(const.MODEL_PATH, f"{model_name}.pth"))


def visualize_graph_predictions(data, model, processed_data, network, index):
    """
    Visualizes initial, current, and predicted node statuses of a (sub)graph.
    """
    g = build_graph(data)
    predictions = get_model_predictions(model, processed_data)

    real_node_names = {v: k for k, v in data.node_mapping.items()}

    subgraph, node_order, _ = extract_relevant_subgraph(g, data.y.numpy(), real_node_names)
    relabeled_data = reorder_node_data(data, node_order, predictions)

    plot_graph_states(
        subgraph,
        relabeled_data["initial_status"],
        relabeled_data["diff_status"],
        relabeled_data["current_status"],
        relabeled_data["predictions"],
        network,
        index
    )



def build_graph(data):
    """
    Builds a NetworkX directed graph with node and edge attributes.
    """
    initial_status = data.y.numpy()
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    g = nx.DiGraph()
    g.add_nodes_from(range(len(initial_status)))
    nx.set_node_attributes(g, dict(enumerate(initial_status)), "source")
    g.add_edges_from(edge_index.t().tolist())

    if isinstance(edge_attr, torch.Tensor):
        edge_attr = edge_attr.tolist()

    edge_attr_dict = {(u, v): attr for (u, v), attr in zip(edge_index.t().tolist(), edge_attr)}
    nx.set_edge_attributes(g, edge_attr_dict, "weight")

    return g


def get_model_predictions(model, processed_data):
    """
    Gets and processes model predictions.
    """
    pred = model(processed_data)
    pred = process_predictions(pred)
    return pred


def extract_relevant_subgraph(g, initial_status, real_node_names, min_subset_size=500):
    """
    Returns a relabeled subgraph with real node names and the node order.
    """
    if len(g.nodes()) <= min_subset_size:
        subset_indices = list(g.nodes())
    else:
        source_node = np.where(initial_status == 1)[0][0]
        subset_indices = list(nx.dfs_preorder_nodes(g, source_node))[:min_subset_size]

    subgraph = g.subgraph(subset_indices).copy()
    subgraph = nx.convert_node_labels_to_integers(subgraph, first_label=0, label_attribute="old_index")

    # Relabel nodes using real node names
    label_mapping = {
        i: real_node_names.get(data["old_index"], data["old_index"])
        for i, data in subgraph.nodes(data=True)
    }
    subgraph = nx.relabel_nodes(subgraph, label_mapping)

    # new_order = list of original indices in the order used in subgraph
    new_order = [data["old_index"] for _, data in subgraph.nodes(data=True)]

    return subgraph, new_order, label_mapping


def reorder_node_data(data, new_order, predictions):
    """
    Reorders node status and prediction arrays according to new graph node order.
    """
    binary_perturbation_marker = data.y.numpy()[new_order]
    initial_status = data.x[new_order, 0]
    diff_status = data.x[new_order, 1]  # Assuming second column is diff status
    current_status = initial_status + diff_status
    predictions = predictions[new_order]

    return {
        "binary_perturbation_marker": binary_perturbation_marker,
        "initial_status": initial_status,
        "diff_status": diff_status,
        "current_status": current_status,
        "predictions": predictions
    }


def plot_graph_states(g, initial_status, diff_status, current_status, predictions, network, index):
    """
    Plots the initial, current, and predicted states of a graph.
    """
    if const.MODEL == "GCNSI":
        predictions_cmap = LinearSegmentedColormap.from_list(
            "predictions", ["blue", "red"]
        )
    elif const.MODEL == "GCNR":
        predictions_cmap = LinearSegmentedColormap.from_list(
            "predictions", ["red", "blue"]
        )
    # Plot initial infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(initial_status,  dtype=float),
        f"{network}_initial_{index}",
        cmap=None,
    )

    # Plot current infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(current_status, dtype=float),
        f"{network}_current_{index}",
        cmap=None,
    )

        # Plot current infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(diff_status, dtype=float),
        f"{network}_diff_{index}",
        max_abs_value=max(np.abs(diff_status)),
        cmap=LinearSegmentedColormap.from_list(
        "predictions", ["red", "white", "blue"]
    ),
    )

    # Plot predicted infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(predictions, dtype=float),
        f"{network}_prediction_{index}",
        cmap=predictions_cmap,
    )


def process_predictions(pred):
    """
    Process predictions based on the model type and return the processed predictions,
    colormap, and number of colors.
    """
    if const.MODEL == "GCNSI":
        pred = torch.sigmoid(pred)
        pred = torch.round(pred)
    return pred




def main():
    """
    Visualize graphs with the associated predictions.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--network", 
        type=str, 
        help="name of the network that should be used"
    )
    args = parser.parse_args()
    n_graphs = 5
    model_name = (
        utils.latest_model_name() if const.MODEL_NAME is None else const.MODEL_NAME
    )
    network = args.network


    # remove old figures
    for file in glob.glob(f"{const.FIGURES_PATH}/*"):
        os.remove(file)
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)
    

    model = load_model_by_type(model_name)
    processed_val_data = utils.load_processed_data(validation=True)[:n_graphs]
    raw_val_data = utils.load_raw_data(validation=True)[:n_graphs]

    print("Visualize example predictions:")
    for i, data in tqdm(enumerate(raw_val_data)):
        visualize_graph_predictions(data, model, processed_val_data[i], network, i)


if __name__ == "__main__":
    main()
