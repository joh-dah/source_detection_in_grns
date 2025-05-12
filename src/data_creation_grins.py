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
import grins.racipe_run as rr
from multiprocessing import Pool
import jax.numpy as jnp
import pandas as pd

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


    # Functiont to generate all the parameters related files with replicates
def gen_topo_param_files(
    topo_file: str,
    save_dir: str = ".",
    num_replicates: int = 3,
    num_params: int = 2**10,
    num_init_conds: int = 2**7,
    sampling_method: Union[str, dict] = "Sobol",
):
    """
    Generate parameter files for simulation.

    Parameters
    ----------
    topo_file : str
        The path to the topo file.
    save_dir : str, optional
        The directory where the parameter files will be saved. Defaults to ".".
    num_params : int, optional
        The number of parameter files to generate. Defaults to 2**10.
    num_init_conds : int, optional
        The number of initial condition files to generate. Defaults to 2**7.
    sampling_method : Union[str, dict], optional
        The method to use for sampling the parameter space. Defaults to 'Sobol'. For a finer control over the parameter generation look at the documentation of the gen_param_range_df function and gen_param_df function.

    Returns
    -------
    None
        The parameter files and initial conditions are generated and saved in the specified replicate directories.
    """
    # Get the name of the topo file
    topo_name = topo_file.split("/")[-1].split(".")[0]
    # Parse the topo file
    topo_df = rr.parse_topos(topo_file)
    # # Generate the parameter names
    # param_names = gen_param_names(topo_df)
    # Get the unique nodes in the topo file
    # unique_nodes = sorted(set(param_names[1] + param_names[2]))
    # Generate the required directory structure
    rr.gen_sim_dirstruct(topo_file, save_dir, num_replicates)
    # Specify directory where all the generated ode system file will be saved
    sim_dir = f"{save_dir}/{topo_file.split('/')[-1].split('.')[0]}"
    # Generate the ODE system for diffrax
    rr.gen_diffrax_odesys(topo_df, topo_name, sim_dir)
    # Generate the parameter dataframe and save in each of the replicate folders
    for rep in range(1, num_replicates + 1):
        # Generate the parameter range dataframe
        param_range_df = rr.gen_param_range_df(
            topo_df, num_params, sampling_method=sampling_method
        )
        # Save the parameter range dataframe
        param_range_df.to_csv(
            f"{sim_dir}/{rep:03}/{topo_name}_param_range_{rep:03}.csv",
            index=False,
            sep="\t",
        )
        # # Generate the parameter dataframe with the default values
        param_df = rr.gen_param_df(param_range_df, num_params)
        # print(param_df)
        param_df.to_parquet(
            f"{sim_dir}/{rep:03}/{topo_name}_params_{rep:03}.parquet", index=False
        )
        # Generate the initial conditions dataframe
        initcond_df = gen_init_cond(topo_df=topo_df, num_init_conds=num_init_conds)
        # print(initcond_df)
        initcond_df.to_parquet(
            f"{sim_dir}/{rep:03}/{topo_name}_init_conds_{rep:03}.parquet",
            index=False,
        )
    print(f"Parameter and Intial Condition files generated for {topo_name}")
    return None


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

og expression values
expression values after perturbation
edge_index (with moa)


def create_data_set(
    path: Path,
    steady_states_path,
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
    steady_states_df = pd.read_parquet(steady_states_path)  
    avg_time_to_reach_ss = steady_states_df["Time"].avg 

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

    steady_states_dir = dest_dir / "raw" / "steady_states"

    create_data_set(
        path=path,
        root_seed=root_seed,
        network=utils.get_cellline_network(args.network),
        desired_dataset_size=size,
    )

    print(f"{args.network} {train_or_val} Data:")



if __name__ == "__main__":
    main()
