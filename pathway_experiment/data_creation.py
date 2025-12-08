import os
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from cmapPy.pandasGEXpress.parse import parse

# print current directory
print("Current working directory:", os.getcwd())

# Paths
DATA_DIR = "data_phase1"
GCTX_FILE = os.path.join(DATA_DIR, "level5.gctx")
PERTURB_FILE = os.path.join(DATA_DIR, "perturbation_pairs.txt")
OUTPUT_DIR = "../../data/pathway/raw"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load perturbation info
df_perturb = pd.read_csv(PERTURB_FILE, sep="\t")

# Create global gene mapping from GCTX row IDs (we'll load this first)
# Load only row metadata to get gene symbols
gctx_meta = parse(GCTX_FILE)
genes = list(gctx_meta.row_metadata_df.index)
gene_to_idx = {gene: idx for idx, gene in enumerate(genes)}
num_genes_to_perturb = len(df_perturb)

# Collect all signatures we need (unperturbed + perturbed)
sig_ids_of_interest = pd.unique(df_perturb[['unperturbed_sig_id', 'perturbed_sig_id']].values.ravel())

# Load only the needed subset of the GCTX
data_gct = parse(GCTX_FILE, rid=genes, cid=sig_ids_of_interest)

# Convert the GCTX data matrix to a DataFrame for easier indexing
df_gctx = data_gct.data_df  # rows = genes, columns = sig_ids

# Iterate over perturbation pairs and create PyG Data objects
for idx, row in df_perturb.iterrows():
    unpert_sig = row['unperturbed_sig_id']
    pert_sig = row['perturbed_sig_id']
    pert_gene = row['perturbed_gene']

    # Extract gene expression vectors
    original = df_gctx[unpert_sig].values.astype(np.float32)
    perturbed = df_gctx[pert_sig].values.astype(np.float32)
    difference = perturbed - original

    # One-hot perturbation indicator
    binary_perturbation_indicator = np.zeros(len(genes), dtype=np.float32)
    if pert_gene in gene_to_idx:
        binary_perturbation_indicator[gene_to_idx[pert_gene]] = 1.0

    # Create Data object
    data_obj = Data(
        original=original,
        perturbed=perturbed,
        difference=difference,
        binary_perturbation_indicator=binary_perturbation_indicator,
        perturbed_gene=pert_gene,
        gene_mapping=gene_to_idx,
        num_nodes=len(gene_to_idx),
        num_possible_sources=num_genes_to_perturb
    )

    # Save individual file
    out_file = os.path.join(OUTPUT_DIR, f"{pert_sig}.pt")
    torch.save(data_obj, out_file)

print("Done! All data objects saved.")
