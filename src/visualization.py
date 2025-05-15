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
    layout: any = nx.kamada_kawai_layout,
):
    """
    Plots graph and colors nodes according to node_values.
    :param g: graph
    :param node_values: values for nodes
    :param max_colored_value: highest node value that should be mapped to a unique color
    :param title: title of plot
    :param cmap: colormap to use
    :param layout: graph plotting layout to use
    """
    pos = layout(g)

    min_colored_value = node_values.min()
    max_colored_value = node_values.max()
 
    plt.figure(figsize=(8, 8))

    nodes = nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=node_values,
        node_size=200,
        cmap=cmap,
        vmin=0,
        vmax=node_values.max(),
        linewidths=[5 if node["source"] == 1 else 1 for node in g.nodes.values()],
    )
    # set edge color according to edge attribute
    # set green for activating edges (1) and red for inhibiting edges (2)
    edge_colors = [
        "green" if g[u][v]["weight"] == 1 else "red" for u, v in g.edges
    ]
    nx.set_edge_attributes(g, edge_colors, "color")
    nx.draw_networkx_edges(
        g,
        pos=pos,
        edge_color=edge_colors,
        width=2,
    )


    # Get the current axes
    ax = plt.gca()

    # Add colorbar with custom int ticks
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=0, vmax=max_colored_value)
    )
    colorbar = plt.colorbar(sm, ax=ax, shrink=0.9)
    colorbar.set_ticks(np.arange(min_colored_value, max_colored_value + 1, 1))
    colorbar.set_ticklabels(np.arange(min_colored_value, max_colored_value + 1, 1))

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
    Visualize the predictions for a single graph.
    """
    initial_status = data.y.numpy()
    status = data.x
    edge_index = data.edge_index
    edge_attr = data.edge_attr

    print(f"initial status: {initial_status}")
    print(f"status: {status}")

    g = nx.DiGraph()
    g.add_nodes_from(range(len(initial_status)))
    nx.set_node_attributes(g, dict(enumerate(initial_status)), "source")
    g.add_edges_from(edge_index.t().tolist())

    # Ensure edge_attr is a list or numpy array
    edge_attr = edge_attr.tolist() if isinstance(edge_attr, torch.Tensor) else edge_attr

    # Create a dictionary mapping edges to their attributes
    edge_attr_dict = {
        (u, v): attr for (u, v), attr in zip(g.edges, edge_attr)
    }

    # Set edge attributes
    nx.set_edge_attributes(g, edge_attr_dict, "weight")


    pred = model(processed_data)
    pred, predictions_cmap, n_colors = process_predictions(pred)

    # the graph is to big to plot all nodes. therefore, we only plot a subsection
    
    min_subset_size = 30

    # If the graph has fewer nodes than min_subset_size, plot the entire graph
    if len(g.nodes()) <= min_subset_size:
        subset_indices = list(g.nodes())
    else:
        # Select the first node that is one in the initial status
        source_node = np.where(initial_status == 1)[0][0]
        # Do a depth-first search to find min_subset_size nodes connected to the source node
        subset_indices = list(nx.dfs_preorder_nodes(g, source_node))
        # If the subset is larger than min_subset_size, truncate the subset
        if len(subset_indices) > min_subset_size:
            subset_indices = subset_indices[:min_subset_size]

    # Create a new graph with only the nodes and corresponding edges in the subset
    g = g.subgraph(subset_indices).copy()

    # Relabel nodes to ensure indices are consecutive and get the mapping
    g = nx.convert_node_labels_to_integers(g, first_label=0, label_attribute="old_label")

    # Extract the mapping from the node attributes
    mapping = {new_label: data["old_label"] for new_label, data in g.nodes(data=True)}

    # Extract the new order of nodes based on the mapping
    new_order = [node_data["old_label"] for _, node_data in g.nodes(data=True)]

    # Update status, initial_status, and pred to match the new node indices
    status = status[new_order]
    initial_status = initial_status[new_order]
    pred = pred[new_order]

    sir_cmap = ListedColormap(["blue", "red", "gray"])


    # initial infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(initial_status, dtype=int),
        f"{network}_initial_{index}",
        cmap=None,
    )

    # current infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(status, dtype=int),
        f"{network}_current_{index}",
        cmap=None,
    )

    # predicted graph
    plot_graph_with_colors(
        g,
        np.fromiter(pred, dtype=float),
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
        n_colors = 1
        predictions_cmap = LinearSegmentedColormap.from_list(
            "predictions", ["blue", "red"]
        )
    elif const.MODEL == "GCNR":
        n_colors = 1
        predictions_cmap = LinearSegmentedColormap.from_list(
            "predictions", ["red", "blue"]
        )
    return pred, predictions_cmap, n_colors




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
