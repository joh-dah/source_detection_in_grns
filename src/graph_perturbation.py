""" Creates new processed data based on the selected model. """
from pathlib import Path
from src import constants as const
from src import utils
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import add_remaining_self_loops, to_undirected, from_networkx
import pickle

def store_graph(graph: nx.DiGraph, name: str = None) -> None:
    """
    Store the graph with all information (edge atr. etc) in the experiment data directory.
    :param graph: nx.DiGraph
    """
    print("Storing Graph...")
    data_dir = Path(const.DATA_PATH)
    data_dir.mkdir(parents=True, exist_ok=True)  # Ensure experiment directory exists

    if name is None:
        name = "graph"
    graph_path = data_dir / f"{name}.pkl"
    with open(graph_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"NetworkX graph saved to {graph_path}")


def sample_edges_to_remove(G: nx.DiGraph, fraction: float) -> list:
    """
    Sample edges to remove while protecting connectivity.
    Returns a list of edges that can be safely removed.
    """
    num_edges = G.number_of_edges()
    edges_to_remove = int(num_edges * fraction)
    if edges_to_remove == 0:
        return []

    # Compute spanning tree on undirected version to protect connectivity
    G_undirected = G.to_undirected()
    if not nx.is_connected(G_undirected):
        # If graph is not connected, get spanning forest
        spanning_edges = set()
        for component in nx.connected_components(G_undirected):
            subgraph = G_undirected.subgraph(component)
            spanning_tree = nx.minimum_spanning_tree(subgraph)
            spanning_edges.update(spanning_tree.edges())
    else:
        spanning_tree = nx.minimum_spanning_tree(G_undirected)
        spanning_edges = set(spanning_tree.edges())

    # Convert spanning edges to directed format (both directions)
    protected_edges = set()
    for u, v in spanning_edges:
        if G.has_edge(u, v):
            protected_edges.add((u, v))
        if G.has_edge(v, u):
            protected_edges.add((v, u))

    # Sample only from removable edges
    all_edges = list(G.edges())
    removable_edges = [edge for edge in all_edges if edge not in protected_edges]
    
    if len(removable_edges) == 0:
        print("Warning: No edges can be removed without breaking connectivity.")
        return []

    edges_to_actually_remove = min(edges_to_remove, len(removable_edges))
    np.random.shuffle(removable_edges)
    
    if edges_to_actually_remove < edges_to_remove:
        print(f"Warning: Only sampling {edges_to_actually_remove} out of {edges_to_remove} edges for removal (limited by connectivity constraints).")
    
    return removable_edges[:edges_to_actually_remove]


def sample_edges_to_add(G: nx.DiGraph, fraction: float) -> list:
    """
    Sample edges to add based on degree probabilities.
    Returns a list of (u, v, weight) tuples for edges to add.
    """
    num_edges = G.number_of_edges()
    edges_to_add = int(num_edges * fraction)
    if edges_to_add == 0:
        return []

    nodes = list(G.nodes())
    out_degrees = np.array([G.out_degree(n) for n in nodes], dtype=float)
    in_degrees = np.array([G.in_degree(n) for n in nodes], dtype=float)

    # Avoid division by zero
    out_probs = out_degrees / out_degrees.sum() if out_degrees.sum() > 0 else np.ones_like(out_degrees) / len(nodes)
    in_probs = in_degrees / in_degrees.sum() if in_degrees.sum() > 0 else np.ones_like(in_degrees) / len(nodes)

    # Generate all possible edges not already in the graph (excluding self-loops)
    possible_edges = [(u, v) for u in nodes for v in nodes if u != v and not G.has_edge(u, v)]
    if len(possible_edges) == 0:
        print("Warning: No possible edges to add.")
        return []

    # Compute probabilities for each possible edge
    edge_probs = np.array([out_probs[nodes.index(u)] * in_probs[nodes.index(v)] for u, v in possible_edges])
    edge_probs /= edge_probs.sum()  # Normalize

    # Sample edges to add
    edges_to_actually_add = min(edges_to_add, len(possible_edges))
    chosen_indices = np.random.choice(len(possible_edges), size=edges_to_actually_add, replace=False, p=edge_probs)
    
    edges_with_weights = []
    for idx in chosen_indices:
        u, v = possible_edges[idx]
        weight = np.random.choice([1, 2])
        edges_with_weights.append((u, v, weight))
    
    if edges_to_actually_add < edges_to_add:
        print(f"Warning: Only sampling {edges_to_actually_add} out of {edges_to_add} edges for addition (limited by possible edges).")
    
    return edges_with_weights


def check_connectivity(G: nx.DiGraph) -> None:
    """
    Check if graph is weakly connected and raise error if not.
    """
    if not nx.is_weakly_connected(G):
        num_components = nx.number_weakly_connected_components(G)
        component_sizes = [len(c) for c in nx.weakly_connected_components(G)]
        raise RuntimeError(
            f"Graph is not weakly connected after perturbation! "
            f"Found {num_components} disconnected components with sizes: {component_sizes}. "
        )


def apply_graph_noise(G: nx.DiGraph, missing_edges_fraction: float, wrong_edges_fraction: float) -> nx.DiGraph:
    """
    Apply noise to graph by removing and adding edges in one step.
    """
    print("Graph before noise:")
    print(f"Number of nodes: {G.number_of_nodes()}, Number of edges: {G.number_of_edges()}")
    
    # Sample edges to remove and add
    edges_to_remove = sample_edges_to_remove(G, missing_edges_fraction)
    edges_to_add = sample_edges_to_add(G, wrong_edges_fraction)
    
    # Apply mutations
    G_perturbed = G.copy()
    
    # Remove sampled edges
    for u, v in edges_to_remove:
        G_perturbed.remove_edge(u, v)
    
    print(f"Removed {len(edges_to_remove)} edges")
    print(f"Graph after removing edges: {G_perturbed.number_of_nodes()} nodes, {G_perturbed.number_of_edges()} edges")
    
    # Add sampled edges
    for u, v, weight in edges_to_add:
        G_perturbed.add_edge(u, v, weight=weight)
    
    print(f"Added {len(edges_to_add)} edges")
    print(f"Graph after adding edges: {G_perturbed.number_of_nodes()} nodes, {G_perturbed.number_of_edges()} edges")
    
    # Final connectivity check
    check_connectivity(G_perturbed)
    
    return G_perturbed


def add_noise_to_graph(G: nx.DiGraph, random_graph=False) -> nx.DiGraph:
    if random_graph:
        print("Creating random graph...")
        # create a graph with G.number_of_nodes() and only self loops with weight 1 or 2
        G_random = nx.DiGraph()
        G_random.add_nodes_from(G.nodes(data=True))
        for node in G_random.nodes():
            G_random.add_edge(node, node, weight=np.random.choice([1, 2]))
        print("Random graph created.")
        return G_random
        # # create a random graph with the same number of nodes and edges
        # G_random = nx.barabasi_albert_graph(G.number_of_nodes(), G.number_of_edges() // G.number_of_nodes())
        # G_random = nx.DiGraph(G_random)  # Convert to directed graph
        # # Assign random weight (1 or 2) to each edge
        # for u, v in G_random.edges():
        #     G_random[u][v]['weight'] = np.random.choice([1, 2])
        # print("Random graph created.")
        # return G_random
    
    return apply_graph_noise(G, const.GRAPH_NOISE["missing_edges"], const.GRAPH_NOISE["wrong_edges"])


def main():
    print(f"Perturbing graph from {const.NETWORK} with noise: {const.GRAPH_NOISE}")

    G, _ = utils.get_graph_data_from_topo(Path(const.TOPO_PATH) / f"{const.NETWORK}.topo")
    
    # Store original graph in PyTorch Geometric format
    G_original_pyg = from_networkx(G, group_edge_attrs=['weight'])
    G_original_pyg.edge_attr = G_original_pyg.edge_attr.float()
    store_graph(G_original_pyg, name="original")
    
    # Apply perturbations
    G_perturbed = add_noise_to_graph(G, const.RANDOM_GRAPH)

    # Store perturbed graph in PyTorch Geometric format
    G_perturbed_pyg = from_networkx(G_perturbed, group_edge_attrs=['weight'])
    G_perturbed_pyg.edge_attr = G_perturbed_pyg.edge_attr.float()
    store_graph(G_perturbed_pyg, name="graph")


if __name__ == "__main__":
    main()
