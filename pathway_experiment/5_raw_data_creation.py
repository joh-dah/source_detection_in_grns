from Bio.KEGG.KGML import KGML_parser
from Bio.KEGG import REST
from io import StringIO
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import os
import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from cmapPy.pandasGEXpress.parse import parse
import pickle
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from pathlib import Path


# print current directory
DATA_DIR = "pathway_experiment/data_phase1"
GCTX_FILE = os.path.join(DATA_DIR, "level5.gctx")
gctx_meta = parse(GCTX_FILE)



def create_data_for_pathway(pathway_name, pathway_id):
    PATHWAY_DIR = f"data/shared/{pathway_name}"
    PERTURB_FILE = f"{PATHWAY_DIR}/perturbation_pairs.csv"
    OUTPUT_DIR = f"{PATHWAY_DIR}/raw"
    print(f"Creating raw data for pathway: {pathway_name} (ID: {pathway_id}) in {OUTPUT_DIR}")
    # Load perturbation info
    df_perturbation_pairs = pd.read_csv(PERTURB_FILE, sep="\t")
    df_perturbation_pairs
    with open(f"{PATHWAY_DIR}/gene_to_idx.pkl", "rb") as f:
        L1000_to_idx_map = pickle.load(f)
    with open(f"{PATHWAY_DIR}/kegg_to_L1000_map.pkl", "rb") as f:
        kegg_to_L1000_map = pickle.load(f)

    L1000_to_kegg_map = {v: k for k, v in kegg_to_L1000_map.items()}
    genes_of_interest = [str(L1000_to_kegg_map[gene]) for gene in L1000_to_idx_map.keys()]
    sigs_of_interest = df_perturbation_pairs['perturbed_sig_id'].tolist() + df_perturbation_pairs['unperturbed_sig_id'].tolist()
    sigs_of_interest = list(set(sigs_of_interest))
    print(f"Number of unique genes of interest: {len(genes_of_interest)}")
    print(f"Number of unique sigs of interest: {len(sigs_of_interest)}")
    data_gct = parse(GCTX_FILE, rid=genes_of_interest, cid=sigs_of_interest)
    kegg_to_L1000_idx_map = {}
    for kegg_id, l1000_gene in kegg_to_L1000_map.items():
        if l1000_gene in L1000_to_idx_map:
            idx = L1000_to_idx_map[l1000_gene]
            kegg_to_L1000_idx_map[kegg_id] = idx
    sorted_kegg_ids = sorted(kegg_to_L1000_idx_map.keys(), key=lambda kegg_id: kegg_to_L1000_idx_map[kegg_id])
    # pick some KEGG IDs to test
    test_ids = sorted_kegg_ids[:5]

    # cache first-column values before sorting
    colname = data_gct.data_df.columns[0]
    before_vals = {k: data_gct.data_df.loc[k, colname] for k in test_ids}

    # perform sorting
    data_gct.data_df = data_gct.data_df.loc[sorted_kegg_ids]

    # verify that the order is correct
    for i, kegg_id in enumerate(data_gct.data_df.index):
        l1000_gene = kegg_to_L1000_map.get(kegg_id, None)
        if l1000_gene is not None:
            expected_idx = L1000_to_idx_map[l1000_gene]
            if expected_idx != i:
                print(f"Mismatch at position {i}: kegg_id {kegg_id} maps to {l1000_gene} which has index {expected_idx}")

    print("First 5 L1000 indices:", list(L1000_to_idx_map.keys())[:5])
    print("Kegg IDs for first 5 L1000 indices:", [L1000_to_kegg_map[idx] for idx in list(L1000_to_idx_map.keys())[:5]])
    print("First 5 sorted KEGG IDs:", sorted_kegg_ids[:5])
    print("First 5 row indices in data_gct.data_df:", data_gct.data_df.index[:5].tolist())

    # test whether values moved with their index
    for k in test_ids:
        new_pos = data_gct.data_df.index.get_loc(k)
        after_val = data_gct.data_df.iloc[new_pos][colname]

        if after_val != before_vals[k]:
            print(f"Value mismatch for {k}: expected {before_vals[k]}, found {after_val} at new position {new_pos}")
        else:
            print(f"OK: {k} kept its value ({after_val}) at new position {new_pos}")

    num_possible_sources=df_perturbation_pairs['perturbed_gene_id'].nunique()
    # remove output directory if it exists
    if os.path.exists(OUTPUT_DIR):
        import shutil
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Create a mapping from (gene_id, cell_line, pert_type) -> group_id ONCE (outside loop)
    print("Creating group mapping...")
    group_mapping = {}
    group_ids_list = []
    unique_groups = []

    for idx, row in df_perturbation_pairs.iterrows():
        # Create group key
        group_key = (
            row['perturbed_gene_id'],
            row.get('cell_line', 'unknown'),
            row.get('perturbation_type', 'unknown')
        )

        # Assign group ID (incrementing)
        if group_key not in group_mapping:
            group_id = len(group_mapping)
            group_mapping[group_key] = group_id
            unique_groups.append(group_key)
        else:
            group_id = group_mapping[group_key]
        
        group_ids_list.append(group_id)

    print(f"Created {len(unique_groups)} unique groups")

    # Convert to numpy array for storage
    group_ids = np.array(group_ids_list, dtype=np.int32)

    # Now loop through data creation and save files
    print("Creating data files...")
    for idx, row in tqdm(df_perturbation_pairs.iterrows(), total=df_perturbation_pairs.shape[0]):
        unpert_sig_id = row['unperturbed_sig_id']
        pert_sig_id = row['perturbed_sig_id']
        pert_gene_id = row['perturbed_gene_id']
        L1000_gene = kegg_to_L1000_map.get(str(pert_gene_id), None)

        # Extract gene expression vectors
        original = data_gct.data_df[unpert_sig_id].values.astype(np.float32)
        perturbed = data_gct.data_df[pert_sig_id].values.astype(np.float32)
        difference = perturbed - original

        # One-hot perturbation indicator
        binary_perturbation_indicator = np.zeros(len(genes_of_interest), dtype=np.float32)
        binary_perturbation_indicator[L1000_to_idx_map[L1000_gene]] = 1.0

        # Create Data object
        data_obj = Data(
            original=original,
            perturbed=perturbed,
            difference=difference,
            binary_perturbation_indicator=binary_perturbation_indicator,
            perturbed_gene=L1000_gene,
            gene_mapping=L1000_to_idx_map,
            num_nodes=len(L1000_to_idx_map),
            num_possible_sources=num_possible_sources
        )

        # Save individual file
        out_file = os.path.join(OUTPUT_DIR, f"{idx}.pt")
        torch.save(data_obj, out_file)

    # Save metadata ONCE after all data files to the shared data path
    print("Saving metadata...")
    sample_metadata = {
        "group_ids": group_ids,
        "unique_groups": unique_groups,
        "group_mapping": group_mapping,
        "n_samples": len(group_ids),
        "n_unique_groups": len(unique_groups),
    }

    # Determine the shared data path (same directory where raw data is stored)

    metadata_file = f"{PATHWAY_DIR}/sample_metadata.pt"
    torch.save(sample_metadata, metadata_file)

    # create a file to indicate that data creation is complete
    with open(f"{PATHWAY_DIR}/data_creation_complete.txt", "w") as f:
        f.write("Data creation complete.\n")

    print(f"Done! All data objects saved.")
    print(f"Metadata saved to {metadata_file}: {len(group_ids)} samples in {len(unique_groups)} groups")

  

    

def main():
    pathway_id_dict = {
        "combined": 00000
    }

    for pathway_name, pathway_id in pathway_id_dict.items():
        print(f"Processing pathway: {pathway_name} with ID: {pathway_id}")
        if not os.path.exists(f"data/shared/{pathway_name}/data_creation_complete.txt"):
            n_nodes = create_data_for_pathway(pathway_name, pathway_id)


if __name__ == "__main__":
    main()