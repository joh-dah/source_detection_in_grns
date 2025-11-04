#!/usr/bin/env python3
"""
Script to create bar plots comparing PDGrapher performance across different experiments.
Each bar represents the average metric for PDGrapher in that experiment folder,
with error bars showing standard deviation.

Usage:
    python create_pdgrapher_comparison.py
    python create_pdgrapher_comparison.py --output-dir custom_output/
"""

import argparse
import sys
import os
import json
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


def load_model_data_from_experiments(reports_base_path: str = "reports", model_prefix: str = "pdgrapher") -> pd.DataFrame:
    """Load model data (files with prefix) from all experiment folders and combine into a single DataFrame.

    Args:
        reports_base_path: top-level reports directory containing experiment subfolders
        model_prefix: filename prefix to look for (e.g. 'pdgrapher', 'pdgrapher_nognn', 'gat')
    """
    all_data = []
    
    if not os.path.exists(reports_base_path):
        print(f"Reports directory {reports_base_path} does not exist!")
        return pd.DataFrame()
    
    # Get all experiment directories at top level
    experiment_dirs = [d for d in os.listdir(reports_base_path) 
                      if os.path.isdir(os.path.join(reports_base_path, d))]
    
    print(f"Found experiment directories: {experiment_dirs}")
    
    for exp_dir in experiment_dirs:
        exp_path = os.path.join(reports_base_path, exp_dir)
        print(f"Loading data for prefix '{model_prefix}_' from {exp_path}...")

        # Find all JSON files for the requested model prefix in this experiment
        pattern = os.path.join(exp_path, f"{model_prefix}_*.json")
        model_files = glob.glob(pattern)

        if not model_files:
            print(f"  No files with prefix '{model_prefix}_' found in {exp_path}")
            continue

        print(f"  Found {len(model_files)} files for prefix '{model_prefix}_'")

        # Load data from all files for this model in this experiment
        for file_path in model_files:
            try:
                with open(file_path, 'r') as f:
                    content = json.load(f)
                
                # Extract metrics
                metrics = content.get('metrics', {})
                if not metrics:
                    print(f"    No metrics found in {file_path}")
                    continue
                
                # Create a record for this file
                record = {
                    'experiment': exp_dir,
                    'file': Path(file_path).name,
                    'file_path': file_path,
                    'network': content.get('network', 'unknown'),
                    'model_type': content.get('model_type', model_prefix)
                }
                
                # Add all top-level metrics
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, dict):
                        # Handle nested metrics (like pdgrapher_perturbation_discovery)
                        if metric_name == 'pdgrapher_perturbation_discovery':
                            for nested_metric, nested_value in metric_value.items():
                                record[nested_metric] = nested_value
                    else:
                        record[metric_name] = metric_value
                
                all_data.append(record)
                
            except Exception as e:
                print(f"    Error loading {file_path}: {e}")
    
    if not all_data:
        print(f"No data with prefix '{model_prefix}_' found in any experiments!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} records (prefix '{model_prefix}_') from {len(experiment_dirs)} experiments")
    
    return df


def create_model_comparison_plots(df: pd.DataFrame, output_dir: str, model_prefix: str = "pdgrapher"):
    """Create bar plots comparing a model's performance across experiments.

    Args:
        df: combined DataFrame with records
        output_dir: where to save plots
        model_prefix: prefix used to select files (used for titles and filenames)
    """

    if df.empty:
        print("No data available for comparison")
        return

    model_label = model_prefix.replace('_', ' ').title()

    metrics_config = [
        ("accuracy", f"{model_label} Accuracy Comparison Across Experiments", "Accuracy", f"{model_prefix}_accuracy_comparison.png", True),
        ("avg rank of source", f"{model_label} Average Rank of Source Comparison Across Experiments", "Average Rank of Source", f"{model_prefix}_avg_rank_comparison.png", False),
        ("source in top 5", f"{model_label} Source in Top 5 Comparison Across Experiments", "Source in Top 5 (%)", f"{model_prefix}_source_in_top5_comparison.png", True),
        ("source in top 20", f"{model_label} Source in Top 20 Comparison Across Experiments", "Source in Top 20 (%)", f"{model_prefix}_source_in_top20_comparison.png", True),
        ("ranking_score_dcg", f"{model_label} Ranking Score DCG Comparison Across Experiments", "Ranking Score DCG", f"{model_prefix}_ranking_score_dcg_comparison.png", True),
    ]

    for column, title, ylabel, filename, higher_better in metrics_config:
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping {title}")
            continue

        metric_df = df.dropna(subset=[column])
        if metric_df.empty:
            print(f"Warning: No data available for metric '{column}'. Skipping plot.")
            continue

        stats_df = metric_df.groupby('experiment')[column].agg(['mean', 'std', 'count']).reset_index()
        stats_df['std'] = stats_df['std'].fillna(0)

        stats_df = stats_df.sort_values('mean', ascending=not higher_better)

        plt.figure(figsize=(15, 8))
        bars = plt.bar(range(len(stats_df)), stats_df['mean'],
                      yerr=stats_df['std'], capsize=5,
                      alpha=0.7, color='steelblue',
                      error_kw={'color': 'black', 'capthick': 2})

        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Experiment', fontsize=12, fontweight='bold')
        plt.ylabel(ylabel, fontsize=12, fontweight='bold')
        plt.xticks(range(len(stats_df)), stats_df['experiment'], rotation=45, ha='right')

        for i, (mean_val, std_val, count) in enumerate(zip(stats_df['mean'], stats_df['std'], stats_df['count'])):
            label_text = f'{mean_val:.3f}\n(n={int(count)})'
            plt.text(i, mean_val + std_val + (max(stats_df['mean']) * 0.02), label_text, ha='center', va='bottom', fontsize=9)

        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()

        output_path = os.path.join(output_dir, filename)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Saved {column} comparison plot: {output_path}")
        print(f"\n{column.upper().replace('_', ' ')} Summary:")
        print("Experiment | Mean | Std | Count")
        print("-" * 40)
        for _, row in stats_df.iterrows():
            print(f"{row['experiment']:<20} | {row['mean']:>6.3f} | {row['std']:>5.3f} | {int(row['count']):>5}")


def main():
    parser = argparse.ArgumentParser(description='Create model comparison plots across experiments')
    parser.add_argument('--output-dir', default='reports',
                        help='Output directory for plots (default: reports)')
    parser.add_argument('--reports-dir', default='reports',
                        help='Directory containing experiment folders (default: reports)')
    parser.add_argument('--model', default='pdgrapher',
                        help="Which model to compare. Supported: 'pdgrapher', 'pdgrapher_nognn' (or 'pdgraphernognn'), 'gat'")

    args = parser.parse_args()

    # Normalize model prefix (accept alias without underscore)
    model_arg = args.model.lower()
    if model_arg == 'pdgraphernognn':
        model_prefix = 'pdgrapher_nognn'
    else:
        model_prefix = model_arg

    print(f"Creating comparison plots for model prefix: '{model_prefix}'")
    print(f"Scanning experiments in: {args.reports_dir}")
    print(f"Output directory: {args.output_dir}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data for the requested model prefix
    df = load_model_data_from_experiments(args.reports_dir, model_prefix=model_prefix)

    if df.empty:
        print(f"No data found for model prefix '{model_prefix}'. Cannot create comparison plots.")
        return

    # Create comparison plots
    create_model_comparison_plots(df, args.output_dir, model_prefix=model_prefix)

    # Save combined data for further analysis
    output_csv = os.path.join(args.output_dir, f'{model_prefix}_comparison_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved combined data: {output_csv}")

    print(f"\n{model_prefix} comparison complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()