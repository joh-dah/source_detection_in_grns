""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import argparse
import math
from pathlib import Path
import shutil
import numpy as np
import src.constants as const
import src.utils as utils
import grins.racipe_run as rr
import pandas as pd
import jax.numpy as jnp
from torch_geometric.data import Data
import torch
from tqdm import tqdm
import networkx as nx


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
    # Drop columns that are not in considered_nodes
    desired_initial_state_df = desired_initial_state_df[considered_nodes]
    # Add column InitCondNum with sequential values starting from 1
    desired_initial_state_df["InitCondNum"] = range(1, len(desired_initial_state_df) + 1)
    print(f"Shape of init conds: {desired_initial_state_df.shape}")
    # store file
    init_conds_file = racipe_dir / "001" / f"{network_name}_init_conds_001.parquet"
    desired_initial_state_df.to_parquet(
        init_conds_file,
        index=False,
    )


def generate_relevant_param_names(nodes, edges):
    param_names = []
    for node in nodes:
        param_names += [f"Prod_{node}", f"Deg_{node}"]
    for source, target, weight in edges: #TODO
        if weight == 3:  # inactive
            continue  # skip inactive edges
        if weight == 1: 
            param_names += [f"ActFld_{source}_{target}",f"Thr_{source}_{target}", f"Hill_{source}_{target}"]
        if weight == 2:
            param_names += [f"InhFld_{source}_{target}", f"Thr_{source}_{target}", f"Hill_{source}_{target}"]
    return param_names


def update_param_files(
    subnetwork_dir: str,
    subnetwork_name: str,
    considered_nodes: list,
    considered_edges: list
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
    sub_params = generate_relevant_param_names(considered_nodes, considered_edges)
    sub_params += ["ParamNum"]  # always keep ParamNum
    new_param_df = og_param_df[sub_params]
    # store file
    new_param_df.to_parquet(
        param_file,
        index=False,
    )


def get_graph_data_from_topo(filepath):
    """
    Reads a .topo file and returns:
    - A NetworkX directed graph with gene names as node labels and 'Type' as edge weight.
    - A mapping from gene names to integer indices (useful for ML models like PyG).
    
    :param filepath: path to the topology file
    :return: G_named (NetworkX DiGraph), gene_to_idx (dict)
    """
    import pandas as pd
    import networkx as nx

    df = pd.read_csv(filepath, sep=r"\s+")

    # Create gene-to-index mapping for optional ML use
    genes = set(df['Source']).union(df['Target'])
    gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}

    # Build NetworkX DiGraph with weights
    edges_with_weights = list(zip(df['Source'], df['Target'], df['Type']))
    G = nx.DiGraph()
    G.add_weighted_edges_from(edges_with_weights)

    return G, gene_to_idx


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

    # print all edge weights to check if the perturbation was successful
    for u, v, d in G.edges(data=True):
        print(f"Edge {u} -> {v} has weight {d['weight']}")
    return G



def update_files_to_perturbed_subgraph(subnetwork_name, raw_data_dir, perturbed_subgraph, og_steady_state):
    """
    Update the files to the subgraph scope. This is done by removing all nodes and edges that are not in the subgraph.
    A new directory will be created that can be used by racipe to simulate the perturbed steady state.
    :param network_name: name of the original network
    :param subnetwork_name: name of the subnetwork
    :param perturbed_subgraph: the perturbed subgraph
    """
    subnetwork_dir = Path(raw_data_dir) / subnetwork_name
    utils.create_topo_file_from_graph(subnetwork_name, perturbed_subgraph, subnetwork_dir)
    nodes_in_subgraph = list(perturbed_subgraph.nodes())
    edges_in_subgraph = list(perturbed_subgraph.edges(data='weight'))
    update_init_conds_file(og_steady_state, nodes_in_subgraph, subnetwork_dir, subnetwork_name)
    update_param_files(subnetwork_dir, subnetwork_name, nodes_in_subgraph, edges_in_subgraph)


def perturb_graph(G, gene_to_perturb, og_steady_state_df, subnetwork_name, raw_data_dir):

    #store what parameters where used for the steady state (init_cond: param_num)
    params_per_steady_state = {row_id+1: row["ParamNum"] for row_id, row in og_steady_state_df.iterrows()}

    # Step 1 & 2: get subgraph induced by all reachable nodes
    reachable_nodes = nx.descendants(G, gene_to_perturb) | {gene_to_perturb}
    subnetwork = G.subgraph(reachable_nodes).copy()

    # Step 3: perturb the subnetwork edges
    perturbed_subgraph = simulate_loss_of_function(subnetwork, gene_to_perturb)

    # Update your files based on perturbed subgraph
    update_files_to_perturbed_subgraph(subnetwork_name, raw_data_dir, perturbed_subgraph, og_steady_state_df)

    rr.run_all_replicates(
        subnetwork_name,
        raw_data_dir,
        batch_size=4000,
        predefined_combinations=params_per_steady_state, #TODO check if this is right
        normalize=False,
        discretize=False,
    )

    subnetwork_steady_state_df = get_steady_state_df(raw_data_dir, subnetwork_name)
    # replace values in og state with subnetwork state. og_steady_state_df has more columns than subnetwork_steady_state_df
    # overwrite only the columns that are in subnetwork_steady_state_df
    perturbed_steady_state_df = og_steady_state_df.copy()
    for col in subnetwork_steady_state_df.columns:
        if col in perturbed_steady_state_df.columns:
            perturbed_steady_state_df[col] = subnetwork_steady_state_df[col]

    return perturbed_steady_state_df


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
        weights.append(data.get("weight", 1))  # default to 1 if no weight present

    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    edge_attr = torch.tensor(weights, dtype=torch.long)

    return edge_index, edge_attr


def create_data_set(
    raw_data_dir: Path,
    topo_file: str,
    og_network_name: str,
    desired_dataset_size: int,
):
    shutil.rmtree(raw_data_dir, ignore_errors=True)
    Path(raw_data_dir).mkdir(parents=True, exist_ok=True)
    print(f"Creating data set for {og_network_name} in {raw_data_dir}")

    G, gene_to_idx = get_graph_data_from_topo(topo_file)
    edge_index, edge_attr = nx_to_pyg_edges(G, gene_to_idx)

    # get all genes that have outgoing edges
    genes_with_outgoing_edges = [gene for gene in G.nodes() if G.out_degree(gene) > 0]

    perturbations_per_gene = math.ceil(desired_dataset_size / len(genes_with_outgoing_edges))
    #### TODO: find a good way to calculate this. there are some hints in the racipe documentation on how to choose the number of init_conds add params
    # also document well how the params and dataset size is calculated
    num_init_conds = int(np.cbrt(perturbations_per_gene))
    num_params = perturbations_per_gene // num_init_conds
    datapoint_id = 0

    for gene_to_perturb in tqdm(genes_with_outgoing_edges, desc="Creating data set", total=len(genes_with_outgoing_edges)):

        subnetwork_name = f"{og_network_name}_{gene_to_perturb}"
        #TODO: make deterministic with seed
        print(f"Generating {num_params} params and {num_init_conds} init conds for {gene_to_perturb}")
        # generate steady state for the original graph
        rr.gen_topo_param_files(
            topo_file,
            topo_name=subnetwork_name,
            save_dir=raw_data_dir,
            num_replicates = 1,
            num_params=num_params,
            num_init_conds=num_init_conds,
            sampling_method="Sobol",
        )
        print(f"Running racipe to generate initial steady state")
        rr.run_all_replicates(
            topo_file,
            topo_name=subnetwork_name,
            save_dir=raw_data_dir,
            batch_size=4000,
            normalize=False,
            discretize=False,
        )
        #### done

        steady_state_df = get_steady_state_df(raw_data_dir, subnetwork_name)
        og_steady_state_df = steady_state_df[gene_to_idx.keys()].copy()
        perturbed_steady_state_df = perturb_graph(G, gene_to_perturb, steady_state_df, subnetwork_name, raw_data_dir)
        perturbed_steady_state_df = perturbed_steady_state_df[gene_to_idx.keys()]
        difference_to_og_steady_state = perturbed_steady_state_df - og_steady_state_df
        #TODO: maybe normalize?

        for row_id, row in difference_to_og_steady_state.iterrows():
            
            X = torch.tensor(
                [row],  #TODO somehow make sure that the order of the genes is the same as in the original steady state
                dtype=torch.float
            )

            y = torch.tensor(
                [1 if gene == gene_to_perturb else 0 for gene in gene_to_idx.keys()],
                dtype=torch.float
            )
            
            data = Data(
                x=X,
                y=y,
                edge_index=edge_index,
                edge_attr=edge_attr,
                node_mapping=gene_to_idx
            )
            torch.save(data, raw_data_dir / f"{datapoint_id}.pt")
            datapoint_id += 1

        #remove the files that were created by racipe
        shutil.rmtree(raw_data_dir / subnetwork_name, ignore_errors=True)



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
