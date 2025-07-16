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
from architectures.GAT import GAT
import random
from matplotlib.patches import Patch


N_GRAPHS = 5  # Number of graphs to visualize


def get_color_scale_range(node_values: np.ndarray, max_abs_value: float = None):
    """
    Determine color scale limits based on node values or a fixed max_abs_value.

    :param node_values: Array of node values
    :param max_abs_value: Optional max absolute value for symmetric scaling
    :return: Tuple of (min_val, max_val)
    """
    if max_abs_value:
        return -max_abs_value, max_abs_value
    return node_values.min(), node_values.max()


def add_node_border_legend(ax):
    """
    Adds a legend explaining node border colors and widths.
    :param ax: Matplotlib axis
    """
    legend_elements = [
        Patch(facecolor="white", edgecolor="purple", linewidth=3.5, label="True + Predicted Source"),
        Patch(facecolor="white", edgecolor="green", linewidth=2.5, label="True Source"),
        Patch(facecolor="white", edgecolor="orange", linewidth=2.5, label="Predicted Source"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.0, label="Other Node"),
    ]
    ax.legend(
        handles=legend_elements,
        loc="upper left",  # You can customize this
        frameon=False,
        fontsize=6,
        title="Node Borders"
    )


def get_node_borders(nodes, source_node: int, predicted_source: int):
    """
    Generate border widths and colors for each node based on source annotations.

    :param nodes: Iterable of node indices
    :param source_node: True source node index
    :param predicted_source: Predicted source node index
    :return: Tuple (list of linewidths, list of edgecolors)
    """
    widths = []
    colors = []
    for n in nodes:
        is_real = n == source_node
        is_pred = predicted_source is not None and n == predicted_source
        if is_real and is_pred:
            widths.append(7)
            colors.append("purple")  # Both real and predicted
        elif is_real:
            widths.append(5)
            colors.append("green")   # Real source
        elif is_pred:
            widths.append(5)
            colors.append("orange")  # Predicted source
        else:
            widths.append(1)
            colors.append("black")   # Regular node
    return widths, colors


def draw_edges(g, pos):
    """
    Draw graph edges, colored by their 'weight' attribute.

    :param g: NetworkX graph
    :param pos: Node position dictionary
    """
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


def draw_labels(g, pos):
    """
    Draw node labels slightly below each node.

    :param g: NetworkX graph
    :param pos: Node position dictionary
    """
    label_pos = {node: (x, y - 0.08) for node, (x, y) in pos.items()}
    nx.draw_networkx_labels(
        g,
        pos=label_pos,
        font_size=10,
        font_color="black"
    )


def plot_graph_with_colors(
    g: nx.Graph,
    node_values: np.ndarray,
    title: str,
    cmap: Union[Colormap, str] = "viridis",
    max_abs_value: float = None,
    layout: callable = nx.spring_layout,
    source_node: int = None,
    predicted_source: int = None,
):
    """
    Plots a graph with nodes colored by a given value array, highlights source/predicted nodes,
    and saves the plot as a PNG.

    :param g: NetworkX graph
    :param node_values: 1D array of node values to map to node colors
    :param title: Plot title and filename (without extension)
    :param cmap: Colormap to use for nodes (default 'viridis')
    :param max_abs_value: Optional max absolute value for symmetric color scaling
    :param layout: Layout function for node positioning (default nx.circular_layout)
    :param source_node: Optional true source node index for highlighting
    :param predicted_source: Optional predicted source node index for highlighting
    """
    pos = layout(g, seed=11)  # Fixed seed for reproducibility

    min_val, max_val = get_color_scale_range(node_values, max_abs_value)
    node_border_widths, node_border_colors = get_node_borders(
        g.nodes, source_node, predicted_source
    )

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_nodes(
        g,
        pos=pos,
        node_color=node_values,
        node_size=500,
        cmap=cmap,
        vmin=min_val,
        vmax=max_val,
        linewidths=node_border_widths,
        edgecolors=node_border_colors,
    )
    draw_edges(g, pos)
    draw_labels(g, pos)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_val, vmax=max_val))
    ax = plt.gca()
    plt.colorbar(sm, ax=ax, shrink=0.9)

    add_node_border_legend(plt.gca())

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{const.FIGURES_PATH}/{title}.png")
    plt.close()


def visualize_graph_predictions(raw_data, model, processed_data, network, index):
    """
    Visualizes initial, current, and predicted node statuses of a (sub)graph.
    """
    device = next(model.parameters()).device  # Get device from model
    processed_data = processed_data.to(device)  # Move data to same device as model
    
    g = build_graph(raw_data)
    # Get per-node predictions and apply sigmoid
    with torch.no_grad():
        predictions = torch.sigmoid(model(processed_data).squeeze()).detach().cpu()

    # Use node_mapping from processed_data, not raw_data, since model was trained on processed_data
    if hasattr(processed_data, 'node_mapping'):
        real_node_names = {v: k for k, v in processed_data.node_mapping.items()}
    else:
        real_node_names = {v: k for k, v in raw_data.node_mapping.items()}
    
    source_node = torch.where(processed_data.y == 1)[0][0].item()  # Get the true source node
    predicted_source = predictions.argmax().item()  # Node with highest prediction

    subgraph, node_order, _ = extract_relevant_subgraph(g, source_node, real_node_names)
    relabeled_data = reorder_node_data(raw_data, node_order, predictions)

    plot_graph_states(
        subgraph,
        real_node_names[source_node],
        real_node_names[predicted_source],
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


def extract_relevant_subgraph(g, source_node, real_node_names, min_subset_size=500):
    """
    Returns a relabeled subgraph with real node names and the node order.
    """
    if len(g.nodes()) <= min_subset_size:
        subset_indices = list(g.nodes())
    else:
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
    # Convert new_order to tensor indices if needed
    if isinstance(new_order, list):
        new_order = torch.tensor(new_order)
    
    initial_status = data.x[new_order, 0]
    diff_status = data.x[new_order, 1]  # Assuming second column is diff status
    current_status = initial_status + diff_status
    
    # Make sure predictions can be indexed with new_order
    if len(predictions) > max(new_order):
        predictions_reordered = predictions[new_order]
    else:
        # If there's a dimension mismatch, just take the first len(new_order) predictions
        predictions_reordered = predictions[:len(new_order)]

    return {
        "initial_status": initial_status,
        "diff_status": diff_status,
        "current_status": current_status,
        "predictions": predictions_reordered
    }


def plot_graph_states(g, source_node, predicted_source, initial_status, diff_status, current_status, predictions, network, index):
    """
    Plots the initial, current, and predicted states of a graph.
    """
    predictions_cmap = LinearSegmentedColormap.from_list(
        "predictions", ["blue", "red"]
    )

    # Plot initial infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(initial_status,  dtype=float),
        f"{network}_initial_{index}",
        cmap=None,
        source_node=source_node,
        predicted_source=predicted_source,
    )

    # Plot current infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(current_status, dtype=float),
        f"{network}_current_{index}",
        cmap=None,
        source_node=source_node,
        predicted_source=predicted_source,
    )

    # Plot diff infection graph
    plot_graph_with_colors(
        g,
        np.fromiter(diff_status, dtype=float),
        f"{network}_diff_{index}",
        max_abs_value=max(np.abs(diff_status)),
        cmap=LinearSegmentedColormap.from_list(
            "predictions", ["red", "white", "blue"]
        ),
        source_node=source_node,
        predicted_source=predicted_source,
    )

    plot_graph_with_colors(
        g,
        np.fromiter(predictions, dtype=float),
        f"{network}_prediction_{index}",
        cmap=predictions_cmap,
        source_node=source_node,
        predicted_source=predicted_source,
    )


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
    model_name = (
        utils.latest_model_name() if const.MODEL_NAME is None else const.MODEL_NAME
    )
    network = args.network


    # remove old figures
    for file in glob.glob(f"{const.FIGURES_PATH}/*"):
        os.remove(file)
    Path(const.FIGURES_PATH).mkdir(parents=True, exist_ok=True)
    
    assert const.MODEL == "GAT", "This visualization script only supports GAT models."
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GAT().to(device)
    # Load the trained model weights
    model = utils.load_model(model, model_name)
    model.eval()  # Set to evaluation mode
    all_processed_val_data = utils.load_processed_data(split="val")
    all_raw_val_data = utils.load_raw_data(split="val")
    indices = random.sample(range(len(all_processed_val_data)), N_GRAPHS)
    processed_val_data = [all_processed_val_data[i] for i in indices]
    raw_val_data = [all_raw_val_data[i] for i in indices]

    for i, data in tqdm(enumerate(raw_val_data)):
        visualize_graph_predictions(data, model, processed_val_data[i], network, i)


if __name__ == "__main__":
    main()
