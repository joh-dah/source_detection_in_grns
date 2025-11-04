#!/usr/bin/env python3
import os, sys, pandas as pd
import matplotlib.pyplot as plt
import re, ast
from math import ceil, sqrt

parent = "reports"

# --- argument parsing ---
metric = sys.argv[1] if len(sys.argv) > 1 else "source in top 20"
normalize = any(a.lower() == "normalize=true" for a in sys.argv[2:])
print(f"Comparing random graph results for metric '{metric}'{' with normalization' if normalize else ''}...")

all_folders = [f for f in os.listdir(parent) if os.path.isdir(os.path.join(parent, f))]

def base_name(folder):
    return re.sub(r'(_fr|_sl|_r33|_r66)(?=_|$)', '', folder)

groups = {}
for f in all_folders:
    b = base_name(f)
    groups.setdefault(b, []).append(f)

def sort_variants(base, variants):
    base_only = [v for v in variants if not re.search(r'_fr|_sl|_r33|_r66', v)]
    r33 = [v for v in variants if '_r33' in v]
    r66 = [v for v in variants if '_r66' in v]
    fr = [v for v in variants if '_fr' in v]
    sl = [v for v in variants if '_sl' in v]
    return base_only + r33 + r66 + fr + sl

groups = {b: sort_variants(b, v) for b, v in groups.items() if len(v) > 1}

if not groups:
    print("No matching folder groups found.")
    sys.exit(0)

n_groups = len(groups)
n_cols = ceil(sqrt(n_groups))
n_rows = ceil(n_groups / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
colors = {
    "base": "#2ca02c",  # strong blue
    "r33": "#1f77b4",   # vivid orange
    "r66": "#ff7f0e",   # green
    "fr":  "#d62728",   # red
    "sl":  "#9467bd",   # purple
}

# collect pdgrapher base - fr differences per experiment
experiment_names = []
pdgrapher_diffs = []
# collect pdgraphernognn(base) - pdgrapher(_fr) differences per experiment
experiment_names_2 = []
pdgraphernognn_minus_pdgrapherfr = []

def extract_metric(row, metric_path):
    """Extract nested metric, falling back to top-level column."""
    keys = metric_path.split(".")
    try:
        value = row
        for k in keys:
            if isinstance(value, str):
                try:
                    value = ast.literal_eval(value)
                except Exception:
                    return None
            value = value.get(k, None) if isinstance(value, dict) else None
            if value is None:
                break
        if value is not None:
            return value
    except Exception:
        pass
    return row.get(keys[-1], None)

for idx, (base, variants) in enumerate(groups.items()):
    row, col = divmod(idx, n_cols)
    ax = axes[row][col]

    dfs, num_nodes_per_file = [], []
    available_variants = []

    for v in variants:
        path = os.path.join(parent, v, "all_results.csv")
        if not os.path.exists(path):
            print(f"Warning: missing file for variant '{v}': {path}. Skipping this variant.")
            continue
        try:
            df = pd.read_csv(path)
        except Exception as e:
            print(f"Warning: failed to read '{path}' for variant '{v}': {e}. Skipping this variant.")
            continue
        if df is None or df.empty:
            print(f"Warning: file '{path}' is empty or has no rows. Skipping variant '{v}'.")
            continue

        dfs.append(df)
        available_variants.append(v)

        # Find first non-empty num_nodes value (same for all rows in this file)
        num_nodes = df["num_nodes"].dropna().iloc[0] if "num_nodes" in df.columns and df["num_nodes"].notna().any() else None
        num_nodes_per_file.append(num_nodes)

    means, stds = [], []
    model_types_set = set()

    for df, num_nodes in zip(dfs, num_nodes_per_file):
        vals = {}
        for mt, group in df.groupby("model_type"):
            metrics = []
            for r in group.to_dict(orient="records"):
                val = extract_metric(r, metric)
                if val is None:
                    continue
                if normalize and num_nodes:
                    try:
                        val = val / float(num_nodes)
                    except Exception:
                        pass
                metrics.append(val)
            if metrics:
                vals[mt] = (sum(metrics) / len(metrics), pd.Series(metrics).std())
        means.append(pd.Series({k: v[0] for k, v in vals.items()}))
        stds.append(pd.Series({k: v[1] for k, v in vals.items()}))
        model_types_set.update(vals.keys())

    model_types = sorted(model_types_set)
    selected_model_types = ["pdgrapher", "pdgraphernognn", "overlap", "random"]
    print("reducing to the following model types:", selected_model_types)
    model_types = selected_model_types

    # compute pdgrapher base - fr difference for this experiment (if both present)
    try:
        base_idx = next((i for i, v in enumerate(available_variants) if not re.search(r'_fr|_sl|_r33|_r66', v)), None)
    except StopIteration:
        base_idx = None
    fr_idx = next((i for i, v in enumerate(available_variants) if '_fr' in v), None)

    base_mean_val = None
    fr_mean_val = None
    if base_idx is not None and base_idx < len(means):
        base_mean_val = means[base_idx].get('pdgrapher', None)
    if fr_idx is not None and fr_idx < len(means):
        fr_mean_val = means[fr_idx].get('pdgrapher', None)

    if base_mean_val is not None and fr_mean_val is not None:
        pdgrapher_diffs.append(base_mean_val - fr_mean_val)
        experiment_names.append(base)
    else:
        # not all experiments will have both variants; report and skip
        print(f"Note: skipping pdgrapher diff for '{base}' (base: {base_mean_val}, fr: {fr_mean_val})")

    # now compute pdgraphernognn(base) - pdgrapher(_fr) for this experiment
    # base variant index (same as above base_idx) contains pdgraphernognn entry
    pdgraphernognn_base_val = None
    pdgrapher_fr_val = None
    if base_idx is not None and base_idx < len(means):
        pdgraphernognn_base_val = means[base_idx].get('pdgraphernognn', None)
    if fr_idx is not None and fr_idx < len(means):
        pdgrapher_fr_val = means[fr_idx].get('pdgrapher', None)

    if pdgraphernognn_base_val is not None and pdgrapher_fr_val is not None:
        pdgraphernognn_minus_pdgrapherfr.append(pdgraphernognn_base_val - pdgrapher_fr_val)
        experiment_names_2.append(base)
    else:
        print(f"Note: skipping pdgraphernognn vs pdgrapher_fr diff for '{base}' (pdgraphernognn: {pdgraphernognn_base_val}, pdgrapher_fr: {pdgrapher_fr_val})")


    x = range(len(model_types))
    if not dfs:
        print(f"Warning: all variants for base '{base}' are missing or unreadable. Skipping this group.")
        # disable this subplot to avoid empty plot
        ax.axis("off")
        continue

    w = 0.8 / len(available_variants)

    used_keys = set()
    for i, v in enumerate(available_variants):
        key = "fr" if "_fr" in v else "sl" if "_sl" in v else "r33" if "_r33" in v else "r66" if "_r66" in v else "base"
        # use short key as legend label and only add it once
        label = key if key not in used_keys else "_nolegend_"
        used_keys.add(key)
        ax.bar(
            [xi - 0.4 + w / 2 + i * w for xi in x],
            [means[i].get(mt, 0) for mt in model_types],
            w,
            yerr=[stds[i].get(mt, 0) for mt in model_types],
            label=label,
            color=colors[key],
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=45, ha="right")
    ax.set_title(base + (" (normalized)" if normalize else ""))
    ax.set_ylabel(metric + (" / num_nodes" if normalize else ""))
    # place legend outside the axes on the right (deduplicated short keys)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), borderaxespad=0, frameon=False)

# Disable unused subplots
for idx in range(n_groups, n_rows * n_cols):
    row, col = divmod(idx, n_cols)
    axes[row][col].axis("off")

plt.tight_layout()
outname = f"{parent}/random_graph_analysis_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
plt.savefig(outname, dpi=300)
print("✅ Saved plot to", outname)

# Create an additional barplot: one bar per experiment showing pdgrapher (base - fr)
if pdgrapher_diffs:
    fig2, ax2 = plt.subplots(figsize=(max(6, len(pdgrapher_diffs) * 1.2), 5))
    xs = range(len(pdgrapher_diffs))
    bar_colors = ["#2ca02c" if d >= 0 else "#d62728" for d in pdgrapher_diffs]
    ax2.bar(xs, pdgrapher_diffs, color=bar_colors)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(experiment_names, rotation=45, ha="right")
    ylabel = metric + (" / num_nodes" if normalize else "")
    ax2.set_ylabel(f"pdgrapher base - fr ({ylabel})")
    ax2.set_title(f"pdgrapher base - fr difference per experiment{(' (normalized)' if normalize else '')}")
    plt.tight_layout()
    outname2 = f"{parent}/random_graph_pdgrapher_base_minus_fr_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
    plt.savefig(outname2, dpi=300)
    print("✅ Saved pdgrapher diff plot to", outname2)
else:
    print("No pdgrapher differences computed (missing base or _fr variants for all groups).")

# Create the second additional barplot: pdgraphernognn(base) - pdgrapher(_fr)
if pdgraphernognn_minus_pdgrapherfr:
    fig3, ax3 = plt.subplots(figsize=(max(6, len(pdgraphernognn_minus_pdgrapherfr) * 1.2), 5))
    xs2 = range(len(pdgraphernognn_minus_pdgrapherfr))
    bar_colors2 = ["#2ca02c" if d >= 0 else "#d62728" for d in pdgraphernognn_minus_pdgrapherfr]
    ax3.bar(xs2, pdgraphernognn_minus_pdgrapherfr, color=bar_colors2)
    ax3.set_xticks(xs2)
    ax3.set_xticklabels(experiment_names_2, rotation=45, ha="right")
    ylabel2 = metric + (" / num_nodes" if normalize else "")
    ax3.set_ylabel(f"pdgraphernognn(base) - pdgrapher(_fr) ({ylabel2})")
    ax3.set_title(f"pdgraphernognn(base) - pdgrapher(_fr) per experiment{(' (normalized)' if normalize else '')}")
    plt.tight_layout()
    outname3 = f"{parent}/random_graph_pdgraphernognn_base_minus_pdgrapher_fr_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
    plt.savefig(outname3, dpi=300)
    print("✅ Saved pdgraphernognn vs pdgrapher_fr diff plot to", outname3)
else:
    print("No pdgraphernognn vs pdgrapher_fr differences computed (missing variants for all groups).")
