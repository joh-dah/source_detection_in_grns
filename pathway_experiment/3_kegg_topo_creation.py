from Bio.KEGG.KGML import KGML_parser
from Bio.KEGG import REST
from io import StringIO
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import os

def download_pathway(pathway_id):
    kgml = REST.kegg_get(pathway_id, "kgml").read()
    return KGML_parser.read(StringIO(kgml))

def kegg_to_L1000():
    """
    Fetch gene symbols for a list of KEGG gene IDs.
    """
    df_gene_info = pd.read_csv('pathway_experiment/data_phase1/gene_info.txt', sep='\t')  
    # create a map by using the columns 'gene_id' and 'gene_symbol'
    L1000_gene_map = df_gene_info.set_index('pr_gene_id')['pr_gene_symbol'].to_dict()
    # to every gene_id in the mapping add "hsa:" as a prefix
    L1000_gene_map = {f"hsa:{k}": v for k, v in L1000_gene_map.items()}
    return L1000_gene_map

def create_topo_file_from_graph(network_name, G: nx.DiGraph, dir):
    """
    Create a topo file as expected by racipe from a nx Graph
    and store it in the const.TOPO_PATH directory.
    :param G: nx Graph
    """
    new_file_path = Path(dir) / f"{network_name}.topo" 
    # save graph to a trrust.topo file with the header Source Target Type
    with open(new_file_path, "w") as f:
        f.write("Source Target Type\n")
        for u, v, d in G.edges(data='interaction'):
            f.write(f"{u} {v} {d}\n")
    print(f"✅ Saved topo file for graph with {G.number_of_edges()} edges  and {G.number_of_nodes()} nodes to {new_file_path}")



def create_data_for_pathway(pathway_name, pathway_id):
    pathway = download_pathway(pathway_id)
    kegg_to_L1000_map = kegg_to_L1000()
    activation = ["activation", "expression"]
    inhibition = ["inhibition", "repression"]
    other = ["compound", "hidden compound", "indirect effect", "state change", "binding/association", 
            "dissociation", "missing interaction", "phosphorylation", "dephosphorylation", "glycosylation", 
            "ubiquitination", "methylation"]

    G = nx.DiGraph()
    # A single KEGG pathway entry (node) can correspond to multiple gene IDs (e.g., family members); keep a list per node ID.
    node_id_to_gene_ids = {}      # maps KEGG node ID to list of gene IDs (e.g. ['hsa:7157', 'hsa:1029'])

    added_genes = 0
    skipped_entirely = 0

    # Add gene nodes
    # KEGG encodes a "gene" entry that can be a set of genes; split and map each gene ID individually.
    for node in pathway.genes:
        kegg_gene_ids = node.name.split()
        for kegg_gene_id in kegg_gene_ids:
            if kegg_gene_id not in kegg_to_L1000_map:
                continue
            else:
                G.add_node(kegg_gene_id, label=kegg_to_L1000_map[kegg_gene_id], type="gene")
                added_genes += 1
        # Store only the gene IDs that could be mapped; a KEGG node may map to multiple L1000 genes.
        node_id_to_gene_ids[node.id] = [gid for gid in kegg_gene_ids if gid in kegg_to_L1000_map]

    print(node_id_to_gene_ids)

    print(f"✅ Added {added_genes} Genes to the graph.")

    # Add edges based on relations

    activating_edges = 0
    inhibiting_edges = 0
    skipped_edges = 0

    for rel in pathway.relations:
        src_id = rel.entry1.id
        tgt_id = rel.entry2.id

        # Only add edge if both sides are gene-type nodes
        if src_id in node_id_to_gene_ids and tgt_id in node_id_to_gene_ids:
            src_genes = node_id_to_gene_ids[src_id]
            tgt_genes = node_id_to_gene_ids[tgt_id]

            for subtype in rel.subtypes:
                interaction = subtype[0]
                # Determine the type of interaction based on the subtype
                if interaction in activation:
                    interaction = "1"
                    activating_edges += len(src_genes) * len(tgt_genes)
                elif interaction in inhibition:
                    interaction = "2"
                    inhibiting_edges += len(src_genes) * len(tgt_genes)
                elif interaction in other:
                    interaction = "3"
                    skipped_edges += len(src_genes) * len(tgt_genes)
                    continue
                else:
                    "⚠️ Unknown interaction type: {interaction} (subtype: {subtype})"

                # Connect each source gene to each target gene
                for src_gene in src_genes:
                    for tgt_gene in tgt_genes:
                        G.add_edge(src_gene, tgt_gene, interaction=interaction)

        
    G = nx.relabel_nodes(G, {node: G.nodes[node]["label"] for node in G.nodes() if "label" in G.nodes[node]})

    print(f"✅ Added {G.number_of_edges()} edges to the graph.")
    print(f"✅ {activating_edges} activating edges, {inhibiting_edges} inhibiting edges, "
        f"{skipped_edges} skipped edges due to unknown interaction types.")

    # remove isolated nodes
    G.remove_nodes_from(list(nx.isolates(G)))
    print(f"✅ Removed isolated nodes. Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    wcc = list(nx.weakly_connected_components(G))
    print(f"✅ Graph has {len(wcc)} weakly connected components.")
    wcc_sizes = [len(c) for c in wcc]
    print(f"✅ Sizes of weakly connected components: {wcc_sizes}")

    # choose largest weakly connected component
    largest_wcc = max(wcc, key=len)
    G = G.subgraph(largest_wcc).copy()
    print(f"✅ Selected largest weakly connected component. Graph now has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    create_topo_file_from_graph(pathway_name, G, "topos")
    L1000_to_idx_map = {gene: idx for idx, gene in enumerate(G.nodes())}
    # Convert to undirected if requested (common for GNNs)
    # if to_undirected:
    #     G = G.to_undirected()

    # # Create edge_index tensor
    # edges = list(G.edges())
    # edge_list = [(L1000_to_idx_map[u], L1000_to_idx_map[v]) for u, v in edges]
    # edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    # # store edge_index as a torch file
    # torch.save(edge_index, f"experiment_data/{pathway_name}_edge_index.pt")
    # print(f"✅ Saved edge_index tensor with shape {edge_index.shape} to experiment_data/{pathway_name}_edge_index.pt")
    import pickle
    #create dir 
    os.makedirs(f"data/shared/{pathway_name}", exist_ok=True)
    with open(f"data/shared/{pathway_name}/gene_to_idx.pkl", "wb") as f:
        pickle.dump(L1000_to_idx_map, f)
    print(f"✅ Saved gene to index mapping to experiment_data/gene_to_idx.pkl")
    with open(f"data/shared/{pathway_name}/kegg_to_L1000_map.pkl", "wb") as f:
        # from the keys of kegg_to_L1000_map cut the "hsa:" prefix
        kegg_to_L1000_map_no_prefix = {k.split("hsa:")[-1]: v for k, v in kegg_to_L1000_map.items()}
        pickle.dump(kegg_to_L1000_map_no_prefix, f)
    print(f"✅ Saved gene to index mapping to experiment_data/kegg_to_L1000_map.pkl")
    return G.number_of_nodes()

def main():
    pathway_id_dict = {
        "age_rage_signaling_in_diabetic_complications": "hsa04933",
        "apelin_signaling": "hsa04371",
        "breast_cancer": "hsa05224",
        "fluid_shear_stress_and_atherosclerosis": "hsa05418",
        "foxo_signaling": "hsa04068",
        "insulin_resistance": "hsa04931",
        "insulin_signaling": "hsa04910",
        "neurotrophin_signaling": "hsa04722",
        "parathyroid_hormone_synthesis_secretion_and_action": "hsa04928",
        "signalings_regulating_pluripotency_of_stem_cells": "hsa04550",
        "t_cell_receptor_signaling": "hsa04660",
    }

    node_counts = {}

    for pathway_name, pathway_id in pathway_id_dict.items():
        print(f"Processing pathway: {pathway_name} with ID: {pathway_id}")
        if not os.path.exists(f"data/shared/{pathway_name}/data_creation_complete.txt"):
            n_nodes = create_data_for_pathway(pathway_name, pathway_id)
            node_counts[pathway_name] = n_nodes

    print("Node counts for each pathway:")
    for pathway_name, n_nodes in node_counts.items():
        print(f"\"{pathway_name}\": {n_nodes},")

if __name__ == "__main__":
    main()