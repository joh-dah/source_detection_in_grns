from Bio.KEGG.KGML import KGML_parser
from Bio.KEGG import REST
from io import StringIO
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import torch
import os

import pandas as pd
from pathlib import Path
parent_dir = "pathway_experiment/data_phase1"
df_cell_info = pd.read_csv(f'{parent_dir}/cell_info.txt', sep='\t')     # Metadata for cell lines (the samples of the data matrices)
df_gene_info = pd.read_csv(f'{parent_dir}/gene_info.txt', sep='\t')     # Metadata for genes (the features of the data matrices)
df_inst_info = pd.read_csv(f'{parent_dir}/inst_info.txt', sep='\t')     # Metadata for L1000 experiments (the data matrices)
df_pert_info = pd.read_csv(f'{parent_dir}/pert_info.txt', sep='\t')     # Metadata for perturbations applied in L1000 experiments
df_sig_info = pd.read_csv(f'{parent_dir}/sig_info.txt', sep='\t')       # Metadata for level 5 profiles

print(f"{len(df_cell_info)} cell lines, {len(df_gene_info)} genes, {len(df_inst_info)} experiments, {len(df_pert_info)} perturbations, {len(df_sig_info)} signatures loaded.")

df_sig_with_gene_info = df_sig_info.merge(df_gene_info, left_on='pert_iname', right_on='pr_gene_symbol', how='inner')
# select columns sig_id, pert_id, pr_gene_id, pr_gene_symbol, pert_type, cell_id
df_sig_with_gene_info = df_sig_with_gene_info[['sig_id', 'pert_id', 'pr_gene_id', 'pr_gene_symbol', 'pert_type', 'cell_id']]
df_sig_with_gene_info


def create_data_for_pathway(pathway_name, pathway_id):
    # load pw_in_cancer_gene_to_idx.pkl to get all gene names in the pathway
    experiment_data_dir = f'data/shared/{pathway_name}'
    import pickle
    with open(f'{experiment_data_dir}/gene_to_idx.pkl', 'rb') as f:
        gene_to_idx = pickle.load(f)
    gene_names = list(gene_to_idx.keys())
    print(f"Loaded {len(gene_names)} gene names from {experiment_data_dir}/gene_to_idx.pkl")

    # sanity check: make sure all gene names are in df_gene_info
    for gene in gene_names:
        assert gene in df_gene_info['pr_gene_symbol'].values, f"Gene {gene} not found in df_gene_info"
    # filter df_perturbed_genes to only include rows where pr_gene_symbol is in the list of gene symbols in the graph G
    print(f"ℹ️ Filtering perturbed genes from {len(df_sig_with_gene_info)} to only those present in the pathway graph.")
    df_perturbed_genes = df_sig_with_gene_info[df_sig_with_gene_info['pr_gene_symbol'].isin(gene_names)]
    print(f"✅ Filtered perturbed genes to {len(df_perturbed_genes)} genes present in the pathway graph.")
    # print pert types with their counts
    pert_type_counts = df_perturbed_genes['pert_type'].value_counts()
    print("ℹ️ Perturbation types in filtered perturbed genes:")
    for pert_type, count in pert_type_counts.items():
        print(f" - {pert_type}: {count}")
    # get unique values for cell_id
    cell_lines = df_perturbed_genes['cell_id'].unique().tolist()
    print(f"Found {len(cell_lines)} unique cell lines with perturbations.")

    control_pert_types = ['ctl_untrt.cns', 'ctl_untrt', 'ctl_vehicle.cns', 'ctl_vehicle']
    control_samples = []
    for cell_line in cell_lines:
        df_cell = df_sig_info[df_sig_info['cell_id'] == cell_line]
        for pert_type in control_pert_types:
            df_control = df_cell[df_cell['pert_type'] == pert_type]
            if not df_control.empty:
                break
        if not df_control.empty:
            control_sample = df_control.iloc[0]['sig_id']
            pert_type_found = df_control.iloc[0]['pert_type']
            control_samples.append({'cell_id': cell_line, 'control_sample': control_sample, 'pert_type': pert_type_found})
        else:
            print(f"⚠️ No control sample found for cell line {cell_line} with {len(df_cell)} samples.")
    df_control_samples = pd.DataFrame(control_samples)
    print(f"✅ Created control samples dataframe with {len(df_control_samples)} entries.")
    df_control_samples
    df_perturbation_pairs = df_perturbed_genes.merge(df_control_samples, on='cell_id', how='inner')
    df_renamed_perturbation_pairs = df_perturbation_pairs[['control_sample', 'sig_id', 'pr_gene_symbol', 'pr_gene_id', 'cell_id', 'pert_type_x']].rename(
        columns={
            'control_sample': 'unperturbed_sig_id',
            'sig_id': 'perturbed_sig_id',
            'pr_gene_symbol': 'perturbed_gene',
            'pr_gene_id': 'perturbed_gene_id',
            'pert_type_x': 'perturbation_type',
            'cell_id': 'cell_line'
        }
    ) 

    out_path = f"data/shared/{pathway_name}/perturbation_pairs.csv"
    df_renamed_perturbation_pairs.to_csv(out_path, sep='\t', index=False)
    print(f"✅ Wrote {len(df_renamed_perturbation_pairs)} perturbation pairs with perturbed_gene to {out_path}")


def main():
    pathway_id_dict = {
        "combined": "0000",
    }


    for pathway_name, pathway_id in pathway_id_dict.items():
        print(f"Processing pathway: {pathway_name} with ID: {pathway_id}")
        if not os.path.exists(f"data/shared/{pathway_name}/data_creation_complete.txt"):
            n_nodes = create_data_for_pathway(pathway_name, pathway_id)

if __name__ == "__main__":
    main()