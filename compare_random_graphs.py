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
        """Return the experiment base name (everything before the first underscore).

        Examples:
            'exp2' -> 'exp2'
            'exp2_fr' -> 'exp2'
            'exp2_500_1000' -> 'exp2'
        """
        return folder.split('_', 1)[0]

groups = {}
for f in all_folders:
    b = base_name(f)
    groups.setdefault(b, []).append(f)

def sort_variants(base, variants):
    """Sort variants so that the exact base folder (no underscore) comes first
    (if present), then all other variants that start with the base as a prefix.

    The exact naming/contents of the suffixes is ignored; they are preserved
    and later used as legend labels.
    """
    # put exact base (e.g. 'exp2') first if present
    base_only = [v for v in variants if v == base]
    # all other variants that start with base + '_' sorted lexicographically
    others = sorted([v for v in variants if v != base and v.startswith(base + '_')])
    # include any remaining variants (defensive) after that
    remaining = sorted([v for v in variants if v not in base_only + others])
    return base_only + others + remaining

groups = {b: sort_variants(b, v) for b, v in groups.items() if len(v) > 1}

if not groups:
    print("No matching folder groups found.")
    sys.exit(0)

n_groups = len(groups)
n_cols = ceil(sqrt(n_groups))
n_rows = ceil(n_groups / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows), squeeze=False)
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
    # base is the exact folder equal to the base name (no underscore)
    base_idx = next((i for i, v in enumerate(available_variants) if v == base), None)
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
    # per-variant legend: use suffix (part after base_) or 'Base' when exact match
    color_cycle = plt.rcParams.get('axes.prop_cycle').by_key().get('color', list(colors.values()))
    for i, v in enumerate(available_variants):
        if v == base:
            label = "Base"
        elif v.startswith(base + '_'):
            label = v[len(base) + 1:]
        else:
            label = v
        clr = color_cycle[i % len(color_cycle)]
        ax.bar(
            [xi - 0.4 + w / 2 + i * w for xi in x],
            [means[i].get(mt, 0) for mt in model_types],
            w,
            yerr=[stds[i].get(mt, 0) for mt in model_types],
            label=label,
            color=clr,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_types, rotation=45, ha="right")
    ax.set_title(base)
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

# helper to compute mean, std, n for a given folder and model_type
def get_metric_stats(folder, model_type):
    path = os.path.join(parent, folder, "all_results.csv")
    if not os.path.exists(path):
        print(f"Warning: missing file for variant '{folder}': {path}.")
        return None
    df = pd.read_csv(path)
    if df is None or df.empty:
        print(f"Warning: file '{path}' is empty or has no rows.")
        return None
    # find num_nodes for potential normalization
    num_nodes = df["num_nodes"].dropna().iloc[0] if "num_nodes" in df.columns and df["num_nodes"].notna().any() else None
    metrics = []
    for r in df.to_dict(orient="records"):
        if r.get("model_type") != model_type:
            continue
        val = extract_metric(r, metric)
        if val is None:
            continue
        if normalize and num_nodes:
            try:
                val = val / float(num_nodes)
            except Exception:
                pass
        metrics.append(float(val))
    if not metrics:
        return None
    s = pd.Series(metrics)
    n = len(metrics)
    # sample std (ddof=1) is appropriate; if n < 2, fall back to 0 and warn
    if n < 2:
        std = 0.0
        print(f"Note: only {n} sample(s) for model_type '{model_type}' in '{folder}'; setting std=0 for error calculations.")
    else:
        std = float(s.std(ddof=1))
    mean = float(s.mean())
    return mean, std, n

# Create an additional barplot: one bar per experiment showing pdgrapher (base - fr) with error bars
pdgrapher_diff_errs = []
pdgrapher_diff_vals = []
pdgrapher_diff_names = []

for base in experiment_names:
    variants = groups.get(base, [])
    base_variant = next((v for v in variants if v == base), None)
    fr_variant = next((v for v in variants if '_fr' in v), None)
    if base_variant is None or fr_variant is None:
        print(f"Note: cannot compute pdgrapher diff for '{base}' (missing base or _fr variant).")
        continue
    base_stats = get_metric_stats(base_variant, "pdgrapher")
    fr_stats = get_metric_stats(fr_variant, "pdgrapher")
    if base_stats is None or fr_stats is None:
        print(f"Note: skipping pdgrapher diff for '{base}' due to missing data.")
        continue
    base_mean, base_std, base_n = base_stats
    fr_mean, fr_std, fr_n = fr_stats
    diff = base_mean - fr_mean
    # variance of difference = var(base_mean) + var(fr_mean) with var(mean)=std^2 / n
    var = (base_std ** 2) / base_n + (fr_std ** 2) / fr_n
    err = sqrt(var) if var >= 0 else 0.0
    pdgrapher_diff_vals.append(diff)
    pdgrapher_diff_errs.append(err)
    pdgrapher_diff_names.append(base)

if pdgrapher_diff_vals:
    # sort ascending (lowest value first)
    order = sorted(range(len(pdgrapher_diff_vals)), key=lambda i: pdgrapher_diff_vals[i])
    vals_sorted = [pdgrapher_diff_vals[i] for i in order]
    errs_sorted = [pdgrapher_diff_errs[i] for i in order]
    names_sorted = [pdgrapher_diff_names[i] for i in order]

    fig2, ax2 = plt.subplots(figsize=(max(10, len(vals_sorted) * 1.2), 7))
    xs = range(len(vals_sorted))
    bar_colors = ["#d62728" if d >= 0 else "#2ca02c" for d in vals_sorted]
    ax2.bar(xs, vals_sorted, color=bar_colors, yerr=errs_sorted, error_kw={"capsize": 5})
    ax2.set_xticks(xs)
    ax2.set_xticklabels(names_sorted, rotation=45, ha="right")
    ylabel = metric + (" / num_nodes" if normalize else "")
    ax2.set_ylabel(f"correct graph - random graph ({ylabel})")
    ax2.set_title(f"Difference in pdgrapher performance between correct and random graph")
    plt.tight_layout()
    outname2 = f"{parent}/random_graph_pdgrapher_base_minus_fr_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
    plt.savefig(outname2, dpi=300)
    print("✅ Saved pdgrapher diff plot to", outname2)
else:
    print("No pdgrapher differences computed (missing base or _fr variants for all groups).")

# Create the second additional barplot: pdgraphernognn(base) - pdgrapher(_fr) with error bars
pdgraphernognn_diff_vals = []
pdgraphernognn_diff_errs = []
pdgraphernognn_diff_names = []

for base in experiment_names_2:
    variants = groups.get(base, [])
    base_variant = next((v for v in variants if v == base), None)
    fr_variant = next((v for v in variants if '_fr' in v), None)
    if base_variant is None or fr_variant is None:
        print(f"Note: cannot compute pdgraphernognn vs pdgrapher_fr diff for '{base}' (missing base or _fr variant).")
        continue
    base_stats = get_metric_stats(base_variant, "pdgraphernognn")
    fr_stats = get_metric_stats(fr_variant, "pdgrapher")
    if base_stats is None or fr_stats is None:
        print(f"Note: skipping pdgraphernognn vs pdgrapher_fr diff for '{base}' due to missing data.")
        continue
    base_mean, base_std, base_n = base_stats
    fr_mean, fr_std, fr_n = fr_stats
    diff = base_mean - fr_mean
    var = (base_std ** 2) / base_n + (fr_std ** 2) / fr_n
    err = sqrt(var) if var >= 0 else 0.0
    pdgraphernognn_diff_vals.append(diff)
    pdgraphernognn_diff_errs.append(err)
    pdgraphernognn_diff_names.append(base)

if pdgraphernognn_diff_vals:
    # sort ascending (lowest value first)
    order2 = sorted(range(len(pdgraphernognn_diff_vals)), key=lambda i: pdgraphernognn_diff_vals[i])
    vals2_sorted = [pdgraphernognn_diff_vals[i] for i in order2]
    errs2_sorted = [pdgraphernognn_diff_errs[i] for i in order2]
    names2_sorted = [pdgraphernognn_diff_names[i] for i in order2]

    fig3, ax3 = plt.subplots(figsize=(max(6, len(vals2_sorted) * 1.2), 5))
    xs2 = range(len(vals2_sorted))
    bar_colors2 = ["#2ca02c" if d >= 0 else "#d62728" for d in vals2_sorted]
    ax3.bar(xs2, vals2_sorted, color=bar_colors2, yerr=errs2_sorted, error_kw={"capsize": 5})
    ax3.set_xticks(xs2)
    ax3.set_xticklabels(names2_sorted, rotation=45, ha="right")
    ylabel2 = metric + (" / num_nodes" if normalize else "")
    ax3.set_ylabel(f"pdgraphernognn - pdgrapher random graph ({ylabel2})")
    ax3.set_title(f"pdgraphernognn - pdgrapher random graph per experiment")
    plt.tight_layout()
    outname3 = f"{parent}/random_graph_pdgraphernognn_base_minus_pdgrapher_fr_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
    plt.savefig(outname3, dpi=300)
    print("✅ Saved pdgraphernognn vs pdgrapher_fr diff plot to", outname3)
else:
    print("No pdgraphernognn vs pdgrapher_fr differences computed (missing variants for all groups).")

# Overall comparison across experiments: average pdgraphernognn(base) vs pdgrapher(_fr)
overall_base_vals = []
overall_fr_vals = []
overall_names = []
for base in groups.keys():
    variants = groups.get(base, [])
    base_variant = next((v for v in variants if v == base), None)
    fr_variant = next((v for v in variants if '_fr' in v), None)
    if base_variant is None or fr_variant is None:
        continue
    base_stats = get_metric_stats(base_variant, "pdgraphernognn")
    fr_stats = get_metric_stats(fr_variant, "pdgrapher")
    if base_stats is None or fr_stats is None:
        continue
    overall_base_vals.append(base_stats[0])
    overall_fr_vals.append(fr_stats[0])
    overall_names.append(base)

if overall_base_vals:
    n = len(overall_base_vals)
    s_base = pd.Series(overall_base_vals)
    s_fr = pd.Series(overall_fr_vals)
    mean_base = float(s_base.mean())
    mean_fr = float(s_fr.mean())
    std_base = float(s_base.std(ddof=1)) if n > 1 else 0.0
    std_fr = float(s_fr.std(ddof=1)) if n > 1 else 0.0

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    xs = [0, 1]
    vals = [mean_fr, mean_base]
    errs = [std_fr, std_base]
    labels = ["pdgrapher random graph", "pdgraphernognn"]
    colors4 = ["#d62728", "#2ca02c"]
    bars = ax4.bar(xs, vals, color=colors4, yerr=errs, error_kw={"capsize": 10})
    ax4.set_xticks(xs)
    ax4.set_xticklabels(labels, rotation=20)
    ylabel4 = metric + (" / num_nodes" if normalize else "")
    ax4.set_ylabel(ylabel4)
    ax4.set_title(f"Overall mean across all experiments")

    # Annotate each bar with its mean value (printed above/below the error cap)
    magnitude = max((abs(v) for v in vals), default=1.0)
    offset = magnitude * 0.02
    for i, bar in enumerate(bars):
        mean_val = vals[i]
        err_val = errs[i] if i < len(errs) else 0.0
        if mean_val >= 0:
            y = mean_val + (err_val if err_val else 0.0) + offset
            va = "bottom"
        else:
            y = mean_val - (err_val if err_val else 0.0) - offset
            va = "top"
        ax4.text(bar.get_x() + bar.get_width() / 2, y, f"{mean_val:.3f}", ha="center", va=va, fontsize=9)

    plt.tight_layout()
    outname4 = f"{parent}/random_graph_overall_pdgraphernognn_base_vs_pdgrapher_fr_{metric.replace(' ', '_')}{'_norm' if normalize else ''}.png"
    plt.savefig(outname4, dpi=300)
    print("✅ Saved overall comparison plot to", outname4)
else:
    print("No experiments with both pdgraphernognn (base) and pdgrapher (_fr) found for overall comparison.")
