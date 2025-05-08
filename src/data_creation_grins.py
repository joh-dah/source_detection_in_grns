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
from grins.racipe_run import gen_topo_param_files, run_all_replicates
from multiprocessing import Pool
import jax.numpy as jnp

#TODO Move to constants/params
num_replicates = 1
num_params = 10
num_init_conds = 1


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


# def create_data(params: tuple):
#     """
#     Creates a single data point. Consisting of a graph and the result of a signal propagation model on that graph.
#     The data point is saved to the given path.
#     :param i: index of the data point (used for seeding)
#     :param file: path to save the data point to (including file name)
#     :param existing_data: existing data point, if supplied the signal propagation will be performed on the given graph
#     """

#     i, path, network, metrics, root_seed = params

#     seed = i * 20
#     graph = to_networkx(
#         network, to_undirected=False, remove_self_loops=True
#     ).to_undirected()

#     (
#         prop_model,
#         iterations,
#         source_nodes,
#         beta,
#         threshold_infected,
#     ) = signal_propagation(seed + 15, graph, root_seed)
#     X = torch.tensor(list(prop_model.status.values()), dtype=torch.float)
#     y = torch.tensor(list(prop_model.initial_status.values()), dtype=torch.float)
#     edge_index = (
#         torch.tensor(list(graph.to_directed().edges), dtype=torch.long).t().contiguous()
#     )
#     data = Data(
#         x=X,
#         y=y,
#         edge_index=edge_index,
#         settings=dict(
#             beta=beta,
#             threshold_infected=threshold_infected,
#             iterations=iterations,
#             source_nodes=source_nodes,
#             currently_infected=sum(
#                 [x if x == 1 else 0 for x in prop_model.status.values()]
#             ),
#         ),
#     )
#     if metrics:
#         generate_metrics(graph, data)
#     torch.save(data, path / f"{i}.pt")

# og expression values
# expression values after perturbation
# edge_index (with moa)


# def create_data_set(
#     path: Path,
#     root_seed: int,
#     network: Data,
#     desired_dataset_size: int,
#     generate_graph_metrics: bool = True,
# ):
#     """
#     Performs desired_dataset_size times signal propagation on the given graph.
#     Parameters of the signal propagation are chosen randomly from ranges given in params.yaml.
#     The resulting data will be saved to the given path.
#     :param path: path to save the created data set to
#     :param root_seed: seed to use for random number generation
#     :param network: cellline network on which the signal propagation should be performed
#     :param desired_dataset_size: number of signal propagations to perform per graph
#     :param generate_graph_metrics: whether to generate graph metrics or not
#     """

#     path /= "raw"
#     Path(path).mkdir(parents=True, exist_ok=True)

#     for file_name in os.listdir(path):
#         os.remove(os.path.join(path, file_name))

#     inputs = [
#         (
#             j,
#             path,
#             network,
#             j == 0 and generate_graph_metrics,
#             root_seed,
#         )
#         for j in range(desired_dataset_size)
#     ]

#     with mp.Pool(const.N_CORES) as pool:
#         print(f"Creating data set using multiprocessing ({pool})")
#         list(
#             tqdm(
#                 pool.imap_unordered(create_data, inputs),
#                 total=len(inputs),
#             )
#         )



def grins_simulation(network_name, dest_dir):
    topo_file = f"{const.TOPO_PATH}/{network_name}.topo"
    dest_dir /= "raw"
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(dest_dir):
        os.remove(os.path.join(dest_dir, file_name))

    #TODO parallize:

    print("gen topo files")

    gen_topo_param_files(
        topo_file,
        dest_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method="Sobol",
    )

    print("run replicants")

    # Run time-series simulations
    run_all_replicates(
        topo_file,
        dest_dir,
        tsteps=jnp.array([25.0, 75.0, 100.0]),
        max_steps=2048,
        batch_size=4000,
    )


    # Run steady-state simulations
    run_all_replicates(
        topo_file,
        dest_dir,
        batch_size=4000,
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
    dest_dir = Path(const.DATA_PATH) / train_or_val
    root_seed = const.ROOT_SEED_VALIDATION if args.validation else const.ROOT_SEED_TRAINING
    size = const.VALIDATION_SIZE if args.validation else const.TRAINING_SIZE

    grins_simulation(args.network, dest_dir)

    # create_data_set(
    #     path=path,
    #     root_seed=root_seed,
    #     network=utils.get_cellline_network(args.network),
    #     desired_dataset_size=size,
    # )

    print(f"{args.network} {train_or_val} Data:")



if __name__ == "__main__":
    main()
