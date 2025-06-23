""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import os
import random
import argparse
from typing import Optional
from pathlib import Path
import numpy as np
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import networkx as nx
from tqdm import tqdm
import src.constants as const
from torch_geometric.data import Data, Dataset
from torch_geometric.utils.convert import to_networkx
import torch
import src.utils as utils
import multiprocessing as mp


def random_generator(seed, root_seed):
    return np.random.default_rng([seed, root_seed])


def iterate_until(threshold_infected: float, model: ep.SIModel) -> int:
    """
    Runs the given model until the given percentage of nodes is infected.
    :param threshold_infected: maximum percentage of infected nodes
    :param model: model to run
    :param config: configuration of the model
    :return: number of iterations until threshold was reached
    """

    iterations = 0
    threshold = int(threshold_infected * len(model.status))
    n_infected_nodes = sum([x if x == 1 else 0 for x in model.status.values()])
    while n_infected_nodes <= threshold and iterations < 100:
        n_infected_nodes = sum([x if x == 1 else 0 for x in model.status.values()])
        model.iteration(False)
        iterations += 1

    return iterations


def signal_propagation(seed: int, graph: nx.Graph, root_seed: int):
    """
    Simulates signal propagation on the given graph.
    :param graph: graph to simulate signal propagation on
    :return: list of infected nodes
    """
    model = ep.SIModel(
        graph,
        seed=int(random_generator(seed + 1, root_seed).integers(0, 10e5)),
    )
    config = mc.Configuration()
    beta = random_generator(seed, root_seed).uniform(*const.BETA)
    config.add_model_parameter("beta", beta)

    # randomly choose const.N_SOURCES nodes to be infected
    source_nodes = random_generator(seed + 2, root_seed).choice(
        list(graph.nodes), const.N_SOURCES, replace=False
    )
    infected_dict = {int(node): 1 for node in source_nodes}
    config.add_model_initial_configuration("Infected", infected_dict)
    model.set_initial_status(config)

    threshold_infected = random_generator(seed + 3, root_seed).uniform(
        *const.RELATIVE_INFECTED
    )
    iterations = iterate_until(threshold_infected, model)
    return model, iterations, source_nodes, beta, threshold_infected


def generate_metrics(graph: nx.Graph, data: Data):
    """
    Generates metrics for the given graph and model.
    :param graph: graph to generate metrics for
    :param model: model to generate metrics for
    :param data: data point to save metrics to
    """
    data.metrics = dict(
        diameter=nx.diameter(graph),
        average_shortest_path_length=nx.average_shortest_path_length(
            graph, method="unweighted"
        ),
        average_clustering_coefficient=nx.average_clustering(graph),
        average_degree=np.mean([x[1] for x in graph.degree]),
        n_nodes=len(graph.nodes),
        n_edges=len(graph.edges),
        avg_degree_centrality=np.mean(list(nx.degree_centrality(graph).values())),
    )


def create_data(params: tuple):
    """
    Creates a single data point. Consisting of a graph and the result of a signal propagation model on that graph.
    The data point is saved to the given path.
    :param i: index of the data point (used for seeding)
    :param file: path to save the data point to (including file name)
    :param existing_data: existing data point, if supplied the signal propagation will be performed on the given graph
    """

    i, path, network, metrics, root_seed = params

    seed = i * 20
    graph = to_networkx(
        network, to_undirected=False, remove_self_loops=True
    ).to_undirected()

    (
        prop_model,
        iterations,
        source_nodes,
        beta,
        threshold_infected,
    ) = signal_propagation(seed + 15, graph, root_seed)
    X = torch.tensor(list(prop_model.status.values()), dtype=torch.float)
    y = torch.tensor(list(prop_model.initial_status.values()), dtype=torch.float)
    edge_index = (
        torch.tensor(list(graph.to_directed().edges), dtype=torch.long).t().contiguous()
    )
    data = Data(
        x=X,
        y=y,
        edge_index=edge_index,
        settings=dict(
            beta=beta,
            threshold_infected=threshold_infected,
            iterations=iterations,
            source_nodes=source_nodes,
            currently_infected=sum(
                [x if x == 1 else 0 for x in prop_model.status.values()]
            ),
        ),
    )
    if metrics:
        generate_metrics(graph, data)
    torch.save(data, path / f"{i}.pt")


def create_data_set(
    path: Path,
    root_seed: int,
    network: Data,
    desired_dataset_size: int,
    generate_graph_metrics: bool = True,
):
    """
    Performs desired_dataset_size times signal propagation on the given graph.
    Parameters of the signal propagation are chosen randomly from ranges given in params.yaml.
    The resulting data will be saved to the given path.
    :param path: path to save the created data set to
    :param root_seed: seed to use for random number generation
    :param network: cellline network on which the signal propagation should be performed
    :param desired_dataset_size: number of signal propagations to perform per graph
    :param generate_graph_metrics: whether to generate graph metrics or not
    """

    path /= "raw"
    Path(path).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(path):
        os.remove(os.path.join(path, file_name))

    inputs = [
        (
            j,
            path,
            network,
            j == 0 and generate_graph_metrics,
            root_seed,
        )
        for j in range(desired_dataset_size)
    ]

    with mp.Pool(const.N_CORES) as pool:
        print(f"Creating data set using multiprocessing ({pool})")
        list(
            tqdm(
                pool.imap_unordered(create_data, inputs),
                total=len(inputs),
            )
        )


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
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
    root_seed = const.ROOT_SEED_VALIDATION if args.validation else const.ROOT_SEED_TRAINING
    size = const.VALIDATION_SIZE if args.validation else const.TRAINING_SIZE

    create_data_set(
        path=path,
        root_seed=root_seed,
        network=utils.get_cellline_network(args.network),
        desired_dataset_size=size,
    )

    print(f"{args.network} {train_or_val} Data:")



if __name__ == "__main__":
    main()
