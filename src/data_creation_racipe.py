""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import src.constants as const
import grins.racipe_run as rr
import pandas as pd
import jax.numpy as jnp
from torch_geometric.data import Data
import torch


def get_steady_state_df(dest_dir, network_name) -> pd.DataFrame:
    """
    Reads the steady state from the racipe output file.
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    :return: steady state and time
    """
    steady_state_file = dest_dir / network_name / "001" / f'{network_name}_steadystate_solutions_001.parquet'
    # check if file exists
    if not steady_state_file.exists():
        print(f"Steady state file {steady_state_file} does not exist")
        return None
    # read the steady state file
    return pd.read_parquet(steady_state_file)


def get_time_series_df(dest_dir, network_name) -> pd.DataFrame:
    """
    Reads the time series from the racipe output file.
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    :return: time series
    """
    time_series_file = dest_dir / network_name / "001" / f'{network_name}_timeseries_solutions_001.parquet'
    # check if file exists
    if not time_series_file.exists():
        print(f"Time series file {time_series_file} does not exist")
        return None
    # read the time series file
    return pd.read_parquet(time_series_file)


def create_init_conds_file(
    desired_initial_state_df: pd.DataFrame,
    racipe_dir: Path,
    network_name: str,
    genes
):
    """
    Creates the init_conds file for racipe.
    :param initial_perturbed_state_df: steady state to perturb
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    """
    # remove the columns Time, ParamNum, Replicate, and State
    desired_initial_state_df = desired_initial_state_df[genes]
    # Add column InitCondNum with sequential values starting from 1
    desired_initial_state_df["InitCondNum"] = range(1, len(desired_initial_state_df) + 1)
    # File path
    init_conds_file = racipe_dir / "001" / f"{network_name}_init_conds_001.parquet"
    # Check if file exists
    if not init_conds_file.exists():
        print(f"Init conds file {init_conds_file} does not exist")
        return None
    # Overwrite the init_conds file
    desired_initial_state_df.to_parquet(
        init_conds_file,
        index=False,
    )


def run_racipe_steady_state(topo_file, dest_dir, sample_count):
    num_replicates = 1
    num_init_conds = int(np.cbrt(sample_count))
    num_params = sample_count // num_init_conds
    #TODO: make deterministic with seed
    rr.gen_topo_param_files(
        topo_file,
        dest_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method="Sobol",
    )
    rr.run_all_replicates(
        topo_file,
        dest_dir,
        batch_size=4000,
        normalize=False,
        discretize=False,
    )


def calculate_perturbation_delay(steady_state_df: pd.DataFrame) -> float:
    """
    Calculates the delay that should be used for the perturbation.
    For that the Time column of the racipe output is used.
    The average time is calculated and divided by 2.
    :param steady_state_df: steady state to perturb
    :return: delay for the perturbation
    """
    avg_time = steady_state_df["Time"].mean()
    return avg_time / 2  # TODO: make this more sophisticated


def create_adjusted_param_file(racipe_dir, network_name, steady_state_df, gene_list):
    """
    Create a new parameter file. For a steady state x that was generated with parameters y, 
    row x in the parameter file should contain the parameters y. To do so a new dataframe is created.
    For every steady state the according row (defined in the column ParamNum) of the old parameter file is copied.
    :param topo_file: path to the topo file
    :param racipe_dir: directory where the racipe output is saved
    """
    param_file = racipe_dir / "001" / f'{network_name}_params_001.parquet'
    # check if file exists
    if not param_file.exists():
        print(f"Param file {param_file} does not exist")
        return None
    # read the parameter file
    param_df = pd.read_parquet(param_file)
    # create a new dataframe with the same columns as the parameter file
    new_param_df = pd.DataFrame(columns=param_df.columns)
    perturbed_nodes = []
    # for every steady state copy the row of the parameter file
    for index, row in steady_state_df.iterrows():
        # get the row of the parameter file
        param_row = param_df[param_df["ParamNum"] == row["ParamNum"]]
        param_row["ParamNum"] = index + 1  # set the new ParamNum
        # Add perturbation
        perturbed_gene = np.random.choice(gene_list)
        perturbed_nodes.append(perturbed_gene)
        param_row[f"Prod_{perturbed_gene}"] = 0
        param_row[f"Deg_{perturbed_gene}"] = 1
        perturbed_nodes
        new_param_df = pd.concat([new_param_df, param_row], ignore_index=True)
    # save the new parameter file
    new_param_file = racipe_dir / "001" / f'{network_name}_params_001.parquet'
    new_param_df.to_parquet(new_param_file, index=False)
    return perturbed_nodes


def get_edge_index_from_topo(filepath, node_to_idx):
    """
    Convert a .topo file to PyTorch Geometric edge_index and edge_attr tensors.

    Parameters:
        filepath (str): Path to the .topo file.
        node_to_idx (dict): Dictionary mapping node names to integer indices.

    Returns:
        edge_index (torch.LongTensor): Tensor of shape [2, num_edges].
        edge_attr (torch.LongTensor): Tensor of edge types.
    """
    df = pd.read_csv(filepath, sep=r"\s+")

    # Check that all nodes exist in the mapping
    all_nodes = set(df['Source']).union(df['Target'])
    missing_nodes = all_nodes - set(node_to_idx.keys())
    if missing_nodes:
        raise ValueError(f"The following nodes are missing from node_to_idx: {missing_nodes}")

    # Map to integer indices
    src = df['Source'].map(node_to_idx).values
    tgt = df['Target'].map(node_to_idx).values

    # Build edge_index and edge_attr
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(df['Type'].values, dtype=torch.long)

    return edge_index, edge_attr


def get_perturbed_states(topofile, dest_dir, timepoint, combinations, network_name, genes):
    """
    Simulate the perturbation of the network using racipe.
    :param topofile: path to the topo file
    :param dest_dir: directory where the racipe output is saved
    :param timepoint: time point for the perturbation
    :param combinations: combinations of nodes to perturb
    """
    if timepoint is None:
        # simulate steady state
        rr.run_all_replicates(
            topofile,
            dest_dir,
            predefined_combinations=combinations,
            normalize=False,
            discretize=False,
            batch_size=4000,
        )
        df = get_steady_state_df(dest_dir, network_name)

    else:
        # simulate time series
        rr.run_all_replicates(
            topofile,
            dest_dir,
            tsteps=jnp.array([timepoint]),
            predefined_combinations=combinations,
            normalize=False,
            discretize=False,
            batch_size=4000,
        )
        df = get_time_series_df(dest_dir, network_name)

    perturbed_states = [[x for x in df[genes].values[i]] for i in range(len(df))]

    return perturbed_states


def create_data_set(
    dest_dir: Path,
    topo_file: str,
    network_name: str,
    desired_dataset_size: int,
):
    shutil.rmtree(dest_dir, ignore_errors=True)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    racipe_dir = dest_dir / network_name

    metadata_columns = ["Time", "SteadyStateFlag", "ParamNum", "InitCondNum", "State"]
    print(f"Creating data set for {network_name} in {dest_dir}")
    run_racipe_steady_state(topo_file, dest_dir, desired_dataset_size)
    steady_state_df = get_steady_state_df(dest_dir, network_name)

    # get difference of steady_state_df.columns and metadata_columns
    genes = steady_state_df.columns.difference(metadata_columns).tolist()
    print(f"Genes: {genes}")

    initial_states = [[x for x in steady_state_df[genes].values[i]] for i in range(desired_dataset_size)]
    node_to_idx = {gene: id for id, gene in enumerate(genes)}
    perturbation_delay = calculate_perturbation_delay(steady_state_df)
    perturbed_nodes = create_adjusted_param_file(racipe_dir, network_name, steady_state_df, genes)
    create_init_conds_file(steady_state_df, racipe_dir, network_name, genes)
    combinations_to_generate = jnp.array([[i, i] for i in range(len(perturbed_nodes))])

    rr.run_all_replicates(
        topo_file,
        dest_dir,
        tsteps=jnp.array([perturbation_delay]),
        predefined_combinations=combinations_to_generate,
        normalize=False,
        discretize=False,
        batch_size=4000,
    )

    edge_index, edge_attr = get_edge_index_from_topo(topo_file, node_to_idx)
    delayed_perturbed_states = get_perturbed_states(
        topo_file,
        dest_dir,
        None,
        combinations_to_generate,
        network_name,
        genes
    )
    # print([
    #     abs(delayed_perturbed_states[i][j] - initial_states[i][j])
    #       for j in range(len(delayed_perturbed_states[0]))
    #       for i in range(len(delayed_perturbed_states))])

    # import matplotlib.pyplot as plt

    # # Parameters
    # delays = list(range(13, 101, 5))
    # total_differences = []

    # # Loop over delays
    # for delay in delays:
    #     delayed_perturbed_states = get_perturbed_states(
    #         topo_file,
    #         dest_dir,
    #         delay,
    #         combinations_to_generate,
    #         network_name
    #     )

    #     difference = sum(
    #         abs(delayed_perturbed_states[i][j] - initial_states[i][j])
    #         for j in range(len(delayed_perturbed_states[0]))
    #         for i in range(len(delayed_perturbed_states))
    #     )
        
    #     total_differences.append(difference)

    # # Plot
    # plt.figure(figsize=(8, 5))
    # plt.plot(delays, total_differences, marker='o')
    # plt.xlabel("Perturbation Delay")
    # plt.ylabel("Total Absolute Difference")
    # plt.title("Effect of Perturbation Delay on Total State Difference")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig(f"perturbation_delay_vs_difference_{pd.Timestamp.now()}.png")
    # plt.close()



    #

    for i in range(desired_dataset_size):

        # print(f"Initial state: {initial_states[i]}")
        # print(f"Delayed state: {delayed_perturbed_states[i]}")
        # print(f"Input:  {[delayed_perturbed_states[i][j] - initial_states[i][j] for j in range(len(delayed_perturbed_states[0]))]}")

        X = torch.tensor(
            [delayed_perturbed_states[i][j] - initial_states[i][j] for j in range(len(delayed_perturbed_states[0]))],
            dtype=torch.float
        )

        source_nodes = perturbed_nodes[i]
        y = torch.tensor(
            [1 if node in source_nodes else 0 for node in node_to_idx.keys()],
            dtype=torch.float
        )
        data = Data(
            x=X,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr,
            node_mapping=node_to_idx,
            settings=dict(
            source_nodes=source_nodes,
            ),
        )
        torch.save(data, dest_dir / f"{i}.pt")




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
    dest_dir = Path(const.DATA_PATH) / train_or_val / "raw"
    topo_file = f"{const.TOPO_PATH}/{args.network}.topo"
    sample_count = const.VALIDATION_SIZE if args.validation else const.TRAINING_SIZE

    shutil.rmtree(dest_dir, ignore_errors=True)

    print(f"{args.network} {train_or_val} Data:")

    create_data_set(
        dest_dir,
        topo_file,
        args.network,
        sample_count,
    )


if __name__ == "__main__":
    main()
