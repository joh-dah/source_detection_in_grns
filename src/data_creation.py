""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import math
from pathlib import Path
import shutil
from asyncssh import logger
import numpy as np
import src.constants as const
import src.utils as utils
import grins.racipe_run as rr
import grins.gen_params as gen_params
import grins.gen_diffrax_ode as gen_ode
import pandas as pd
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import networkx as nx
import multiprocessing as mp
import logging
from pathlib import Path
import csv
from datetime import datetime
from pathlib import Path
import time
import psutil
import os


def log_gene_metrics(
    gene: str,
    experiment: str,
    start_time: float,
    end_time: float,
    subnetwork_nodes: int = 0,
    subnetwork_edges: int = 0,
    biggest_hub_subnetwork: int = 0,
):
    # Gene metrics go in shared data path since they're part of data creation
    log_path = Path(const.SHARED_DATA_PATH) / f"{experiment}_gene_metrics.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    duration = round(end_time - start_time, 2)

    row = {
        "timestamp": datetime.now().isoformat(),
        "experiment": experiment,
        "gene": gene,
        "start_time": int(start_time),
        "end_time": int(end_time),
        "duration_sec": duration,
        "subnetwork_nodes": subnetwork_nodes,
        "subnetwork_edges": subnetwork_edges,
        "biggest_hub_subnetwork": biggest_hub_subnetwork,
    }

    header = list(row.keys())
    file_exists = log_path.exists()

    with open(log_path, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


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


def get_timeseries_df(dest_dir, network_name) -> pd.DataFrame:
    """
    Reads the time series from the racipe output file.
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    :return: time series and state
    """
    timeseries_file = dest_dir / network_name / "001" / f'{network_name}_timeseries_solutions_001.parquet'
    # check if file exists
    if not timeseries_file.exists():
        print(f"Time series file {timeseries_file} does not exist")
        return None
    # read the time series file
    return pd.read_parquet(timeseries_file)


def update_init_conds_file(
    desired_initial_state_df: pd.DataFrame,
    considered_nodes: list,
    racipe_dir: Path,
    network_name: str,
):
    """
    Creates the init_conds file for racipe.
    :param initial_perturbed_state_df: steady state to perturb
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    """
    # Drop columns that are not in considered_nodes and make a copy
    desired_initial_state_df = desired_initial_state_df.loc[:, considered_nodes].copy()
    
    # Add column InitCondNum with sequential values starting from 1
    desired_initial_state_df["InitCondNum"] = range(1, len(desired_initial_state_df) + 1)

    # Store file
    init_conds_file = racipe_dir / "001" / f"{network_name}_init_conds_001.parquet"
    desired_initial_state_df.to_parquet(
        init_conds_file,
        index=False,
    )


def generate_relevant_param_names(nodes, edges):
    param_names = []
    for node in nodes:
        param_names += [f"Prod_{node}", f"Deg_{node}"]
    for source, target, weight in edges:
        if weight == 3:
            continue
        elif weight == 1: 
            param_names += [f"ActFld_{source}_{target}",f"Thr_{source}_{target}", f"Hill_{source}_{target}"]
        elif weight == 2:
            param_names += [f"InhFld_{source}_{target}", f"Thr_{source}_{target}", f"Hill_{source}_{target}"]
    return param_names


def update_param_files(
    subnetwork_dir: str,
    subnetwork_name: str,
    sub_params: list,
):
    """
    Read the params file and update it to the new graphs name.
    Remove all unneeded parameters.
    :param dest_dir: directory where the racipe output is saved
    :param network_name: name of the network
    """
    param_file = subnetwork_dir / "001" / f'{subnetwork_name}_params_001.parquet'
    # check if file exists
    if not param_file.exists():
        print(f"Param file {param_file} does not exist")
        return None
    # read the parameter file
    og_param_df = pd.read_parquet(param_file)
    sub_params += ["ParamNum"]  # always keep ParamNum
    new_param_df = og_param_df[sub_params]
    # store file
    new_param_df.to_parquet(
        param_file,
        index=False,
    )


def simulate_loss_of_function(G, gene_to_perturb):
    """
    Simulate a loss of function by changing the mode of activation of all outgoing 
    edges of the gene_to_perturb to 3 (inactive).
    :param G: the graph
    :param gene_to_perturb: the gene to perturb
    """
    # get the edges of the gene_to_perturb
    edges = G.edges(gene_to_perturb)
    # set the edge weight to 3 (inactive)
    for edge in edges:
        G[edge[0]][edge[1]]['weight'] = 3  # assuming 'attr' is the key for the mode of activation
    return G


def update_ode_file(subnetwork_dir: str, subnetwork_name: str, perturbed_gene: str = None):
    """
    Update the ode file to reflect the new subgraph and perturbation.
    """
    topo_file = Path(subnetwork_dir) / f"{subnetwork_name}.topo"
    topo_df = gen_params.parse_topos(topo_file)
    # Create the ODE file
    nodes, param_names = gen_ode.gen_diffrax_odesys(
        topo_df,
        topo_name=subnetwork_name,
        save_dir=subnetwork_dir,
        perturbed_gene=perturbed_gene,
    )
    return nodes, param_names


def update_files_to_perturbed_subgraph(subnetwork_name, raw_data_dir, perturbed_subgraph, og_steady_state, perturbed_gene):
    """
    Update the files to the subgraph scope. This is done by removing all nodes and edges that are not in the subgraph.
    A new directory will be created that can be used by racipe to simulate the perturbed steady state.
    :param network_name: name of the original network
    :param subnetwork_name: name of the subnetwork
    :param perturbed_subgraph: the perturbed subgraph
    """
    subnetwork_dir = Path(raw_data_dir) / subnetwork_name
    utils.create_topo_file_from_graph(subnetwork_name, perturbed_subgraph, subnetwork_dir)
    nodes_in_subgraph, param_names = update_ode_file(subnetwork_dir, subnetwork_name, perturbed_gene)
    update_param_files(subnetwork_dir, subnetwork_name, param_names)
    update_init_conds_file(og_steady_state, nodes_in_subgraph, subnetwork_dir, subnetwork_name)


def perturb_graph(G, gene_to_perturb, og_steady_state_df, subnetwork_name, raw_data_dir):

    metadata = {}

    #store what parameters where used for the steady state (init_cond: param_num)
    params_per_steady_state = [[row_id+1, row["ParamNum"]] for row_id, row in og_steady_state_df.iterrows()]

    # Step 1 & 2: get subgraph induced by all reachable nodes
    all_nodes = sorted(G.nodes())
    reachable_nodes = nx.descendants(G, gene_to_perturb)
    # if there is an edge in G from a node in reachable_nodes to gene_to_perturb, add gene_to_perturb to reachable nodes
    for node in reachable_nodes | {gene_to_perturb}:
        if G.has_edge(node, gene_to_perturb):
            reachable_nodes.add(gene_to_perturb)
            break
    source_reachable = gene_to_perturb in reachable_nodes
    # include the gene to perturb for the racipe computations but be aware that it is not part of the steady state 
    reachable_nodes = sorted(reachable_nodes.union({gene_to_perturb}))
    subnetwork = G.subgraph(reachable_nodes).copy()

    metadata["num_nodes_subgraph"] = len(subnetwork.nodes())
    metadata["num_edges_subgraph"] = len(subnetwork.edges())
    # find the biggest hub in the subnetwork
    metadata["biggest_hub_subnetwork"] = max(dict(subnetwork.degree()).values())

    # Step 3: perturb the subnetwork edges
    perturbed_subgraph = simulate_loss_of_function(subnetwork, gene_to_perturb)

    # Update your files based on perturbed subgraph
    update_files_to_perturbed_subgraph(subnetwork_name, raw_data_dir, perturbed_subgraph, og_steady_state_df, gene_to_perturb)

    rr.run_all_replicates(
        topo_file=subnetwork_name,
        save_dir=raw_data_dir,
        batch_size=4000,
        predefined_combinations=params_per_steady_state, #TODO check if this is right
        tsteps=const.TIME_STEPS,
        normalize=False,
        discretize=True,
    )
    if const.TIME_STEPS is None:
        subnetwork_perturbed_state_df = get_steady_state_df(raw_data_dir, subnetwork_name)
    else:
        subnetwork_perturbed_state_df = get_timeseries_df(raw_data_dir, subnetwork_name)

    if not source_reachable:
        # source was only added for racipe computations and needs to be removed
        reachable_nodes.remove(gene_to_perturb)
    index_positions = [all_nodes.index(node) for node in reachable_nodes]
    og_states = [str(s).replace("'", "") for s in og_steady_state_df["State"].tolist()]
    sub_states = [str(s).replace("'", "") for s in subnetwork_perturbed_state_df["State"].tolist()]


    def merge_states(og_state, sub_state):
        og_state = list(og_state)  # Convert to list for mutability
        for idx, pos in enumerate(index_positions):
            og_state[pos] = sub_state[idx]
        return "".join(og_state)

    merged_states = [
        merge_states(og_state, sub_state)
        for og_state, sub_state in zip(og_states, sub_states)
    ]
    og_steady_state_df["State"] = merged_states
    subnetwork_perturbed_state_df["State"] = merged_states
    # replace values in og state with subnetwork state. og_steady_state_df has more columns than subnetwork_steady_state_df
    # overwrite only the columns that are in subnetwork_steady_state_df
    perturbed_steady_state_df = og_steady_state_df.copy()
    for col in subnetwork_perturbed_state_df.columns:
        if col in perturbed_steady_state_df.columns:
            perturbed_steady_state_df[col] = subnetwork_perturbed_state_df[col]

    return perturbed_steady_state_df, metadata


def remove_near_duplicate_combinations(metadata_df, init_states, perturbed_states, gene_name):
    """
    Remove near duplicate combinations of init_steady_state, perturbed_steady_state and perturbed_gene.
    
    :param metadata_df: DataFrame with metadata
    :param init_states: Array of initial discrete steady states
    :param perturbed_states: Array of perturbed discrete steady states
    :param gene_name: Name of the perturbed gene
    :return: Filtered indices to keep
    """
    print("Removing near duplicate combinations...")
    original_count = len(init_states)
    
    # Create combinations as tuples for easy comparison
    combinations = []
    for i in range(len(init_states)):
        combo = (init_states[i], perturbed_states[i], gene_name)
        combinations.append(combo)
    
    # Find unique combinations and their first occurrence indices
    seen = set()
    indices_to_keep = []
    
    for i, combo in enumerate(combinations):
        if combo not in seen:
            seen.add(combo)
            indices_to_keep.append(i)
    
    duplicates_removed = original_count - len(indices_to_keep)
    if duplicates_removed > 0:
        print(f"Gene {gene_name}: Removed {duplicates_removed} near-duplicate combinations out of {original_count}")
    
    return indices_to_keep


def nx_to_pyg_edges(G, gene_to_idx):
    """
    Converts a NetworkX DiGraph with 'weight' attributes to PyTorch Geometric edge_index and edge_attr tensors.
    Assumes nodes are named with gene strings and `gene_to_idx` maps them to integers.
    
    :param G: NetworkX DiGraph
    :param gene_to_idx: Mapping from node names to integer indices
    :return: edge_index (torch.LongTensor), edge_attr (torch.Tensor)
    """
    src = []
    tgt = []
    weights = []

    for u, v, data in G.edges(data=True):
        src.append(gene_to_idx[u])
        tgt.append(gene_to_idx[v])
        weights.append(data.get("weight", 1.0))  # default to 1.0 if no weight present

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.float).unsqueeze(-1)  # Add dimension for edge features

    return edge_index, edge_attr


def process_gene(
    gene_to_perturb,
    G,
    gene_to_idx,
    raw_data_dir,
    topo_file,
    og_network_name,
    perturbations_per_gene,
    num_genes_to_perturb
):
    start = time.time()

    subnetwork_name = f"{og_network_name}_{gene_to_perturb}"
    num_init_conds = math.sqrt(perturbations_per_gene/10)
    num_params = math.ceil(num_init_conds*10)
    num_init_conds = math.ceil(num_init_conds)
    rr.gen_topo_param_files(
        topo_file,
        topo_name=subnetwork_name,
        save_dir=raw_data_dir,
        num_replicates=1,
        num_params=num_params,
        num_init_conds=num_init_conds,
        sampling_method="Sobol",
    )

    rr.run_all_replicates(
        topo_file,
        topo_name=subnetwork_name,
        save_dir=raw_data_dir,
        batch_size=4000,
        normalize=False,
        discretize=True
    )


    steady_state_df = get_steady_state_df(raw_data_dir, subnetwork_name)
    init_discrete_steady_states = steady_state_df["State"].values
    og_steady_state_df = steady_state_df[gene_to_idx.keys()].copy()
    perturbed_steady_state_df, subnetwork_metadata = perturb_graph(G, gene_to_perturb, steady_state_df, subnetwork_name, raw_data_dir)
    perturbed_discrete_steady_states = perturbed_steady_state_df["State"].values
    perturbed_steady_state_df = perturbed_steady_state_df[gene_to_idx.keys()]
    difference_to_og_steady_state = perturbed_steady_state_df - og_steady_state_df

    # Collect metadata to return instead of writing to file directly
    metadata_for_return = pd.DataFrame({
        "perturbed_gene": [gene_to_perturb] * len(init_discrete_steady_states),
        "init_steady_state": init_discrete_steady_states,
        "perturbed_steady_state": perturbed_discrete_steady_states,
    })

    # Remove near duplicates if enabled
    if const.REMOVE_NEAR_DUPLICATES:
        indices_to_keep = remove_near_duplicate_combinations(
            metadata_for_return, 
            init_discrete_steady_states, 
            perturbed_discrete_steady_states, 
            gene_to_perturb
        )
        
        # Filter all arrays and dataframes
        init_discrete_steady_states = init_discrete_steady_states[indices_to_keep]
        perturbed_discrete_steady_states = perturbed_discrete_steady_states[indices_to_keep]
        og_steady_state_df = og_steady_state_df.iloc[indices_to_keep].reset_index(drop=True)
        perturbed_steady_state_df = perturbed_steady_state_df.iloc[indices_to_keep].reset_index(drop=True)
        difference_to_og_steady_state = difference_to_og_steady_state.iloc[indices_to_keep].reset_index(drop=True)
        metadata_for_return = metadata_for_return.iloc[indices_to_keep].reset_index(drop=True)

    local_datapoints = []
    for row_id, diff_row in difference_to_og_steady_state.iterrows():
        og_row = og_steady_state_df.iloc[row_id]
        perturbed_row = perturbed_steady_state_df.iloc[row_id]

        # Create binary perturbation indicator
        binary_perturbation_indicator = [1 if gene == gene_to_perturb else 0 for gene in gene_to_idx.keys()]
                    
        data = Data(
            original=og_row.values.astype(np.float32),
            perturbed=perturbed_row.values.astype(np.float32),
            difference=diff_row.values.astype(np.float32),
            binary_perturbation_indicator=binary_perturbation_indicator,
            perturbed_gene=gene_to_perturb,
            gene_mapping=gene_to_idx,
            num_nodes=len(gene_to_idx),
            num_possible_sources=num_genes_to_perturb,
        )
        local_datapoints.append(data)


    # Save the data files
    for i, data in enumerate(local_datapoints):
        idx = f"{gene_to_perturb}_{i}"
        torch.save(data, raw_data_dir / f"{idx}.pt")

    shutil.rmtree(raw_data_dir / subnetwork_name, ignore_errors=True)

    end = time.time()
    p = psutil.Process(os.getpid())

    log_gene_metrics(
        gene=gene_to_perturb,
        experiment=const.EXPERIMENT,
        start_time=start,
        end_time=end,
        subnetwork_nodes=subnetwork_metadata.get("num_nodes_subgraph"),
        subnetwork_edges=subnetwork_metadata.get("num_edges_subgraph"),
        biggest_hub_subnetwork=subnetwork_metadata.get("biggest_hub_subnetwork", 0),
    )

    return metadata_for_return


def create_data_set(
    raw_data_dir: Path,
    topo_file: str,
    og_network_name: str,
    desired_dataset_size: int,
):
    G, gene_to_idx = utils.get_graph_data_from_topo(topo_file)

    genes_with_outgoing_edges = [gene for gene in G.nodes() if G.out_degree(gene) > 0]
    print(f"Genes that will be used as sources: {genes_with_outgoing_edges}")
    
    perturbations_per_gene = math.ceil(desired_dataset_size / len(genes_with_outgoing_edges))
    print(f"Perturbations per gene: {perturbations_per_gene}")

    args = [
        (
            gene,
            G,
            gene_to_idx,
            raw_data_dir,
            topo_file,
            og_network_name,
            perturbations_per_gene,
            len(genes_with_outgoing_edges),
        )
        for gene in genes_with_outgoing_edges
    ]

    ctx = mp.get_context("spawn")  # use "spawn" instead of default "fork"
    with ctx.Pool(processes=const.N_CORES) as pool:
        results = list(tqdm(pool.starmap(process_gene, args), total=len(args), desc="Processing genes"))
    
    # Collect all metadata and save to file after multiprocessing is complete
    # Steady state metadata goes in shared data path
    ss_metadata_file = Path(const.SHARED_DATA_PATH) / "steady_state_metadata.csv"
    ss_metadata_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure directory exists
    if ss_metadata_file.exists():
        ss_metadata_df = pd.read_csv(ss_metadata_file)
    else:
        ss_metadata_df = pd.DataFrame(columns=["perturbed_gene", "init_steady_state", "perturbed_steady_state"])
    
    # Concatenate all metadata from the processes
    all_new_metadata = pd.concat(results, ignore_index=True)
    ss_metadata_df = pd.concat([ss_metadata_df, all_new_metadata], ignore_index=True)
    ss_metadata_df.to_csv(ss_metadata_file, index=False)


def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    Uses shared data path so data can be reused between experiments with same data_creation config.
    """
    from src.data_utils import data_exists

    # Check if shared data already exists
    if data_exists(const.SHARED_DATA_PATH, const.N_SAMPLES):
        print(f"Shared data already exists at {const.SHARED_DATA_PATH}")
        print("Skipping data creation...")
        return

    print(f"Creating new shared data at {const.SHARED_DATA_PATH}")
    
    # Clean and create shared data directory
    dest_dir = Path(const.RAW_PATH)
    if dest_dir.exists():
        shutil.rmtree(dest_dir, ignore_errors=True)
    dest_dir.mkdir(parents=True, exist_ok=True)

    topo_file = f"{const.TOPO_PATH}/{const.NETWORK}.topo"
    sample_count = const.N_SAMPLES

    print(f"Creating data set with {sample_count} samples for {const.NETWORK} in {dest_dir}")

    create_data_set(
        dest_dir,
        topo_file,
        const.NETWORK,
        sample_count,
    )

    print(f"Shared data creation complete! Data saved to {const.SHARED_DATA_PATH}")
    print("This data can now be reused by experiments with the same data_creation configuration.")


if __name__ == "__main__":
    main()
