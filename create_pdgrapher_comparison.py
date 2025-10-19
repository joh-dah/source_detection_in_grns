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


def load_pdgrapher_data_from_experiments(reports_base_path: str = "reports") -> pd.DataFrame:
    """Load PDGrapher data from all experiment folders and combine into a single DataFrame."""
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
        print(f"Loading PDGrapher data from {exp_path}...")
        
        # Find all PDGrapher JSON files in this experiment
        pdgrapher_files = glob.glob(os.path.join(exp_path, "pdgrapher_*.json"))
        
        if not pdgrapher_files:
            print(f"  No PDGrapher files found in {exp_path}")
            continue
        
        print(f"  Found {len(pdgrapher_files)} PDGrapher files")
        
        # Load data from all PDGrapher files in this experiment
        for file_path in pdgrapher_files:
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
                    'model_type': content.get('model_type', 'pdgrapher')
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
        print("No PDGrapher data found in any experiments!")
        return pd.DataFrame()
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} PDGrapher records from {len(experiment_dirs)} experiments")
    
    return df


def create_pdgrapher_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create bar plots comparing PDGrapher performance across experiments."""
    
    if df.empty:
        print("No data available for comparison")
        return
    
    # Define metrics to plot
    metrics_config = [
        {
            'column': 'accuracy',
            'title': 'PDGrapher Accuracy Comparison Across Experiments',
            'ylabel': 'Accuracy',
            'filename': 'pdgrapher_accuracy_comparison.png',
            'higher_better': True
        },
        {
            'column': 'avg rank of source',
            'title': 'PDGrapher Average Rank of Source Comparison Across Experiments',
            'ylabel': 'Average Rank of Source',
            'filename': 'pdgrapher_avg_rank_comparison.png',
            'higher_better': False
        },
        {
            'column': 'source in top 5',
            'title': 'PDGrapher Source in Top 5 Comparison Across Experiments',
            'ylabel': 'Source in Top 5 (%)',
            'filename': 'pdgrapher_source_in_top5_comparison.png',
            'higher_better': True
        },
        {
            'column': 'source in top 20',
            'title': 'PDGrapher Source in Top 20 Comparison Across Experiments',
            'ylabel': 'Source in Top 20 (%)',
            'filename': 'pdgrapher_source_in_top20_comparison.png',
            'higher_better': True
        },
        {
            'column': 'ranking_score_dcg',
            'title': 'PDGrapher Ranking Score DCG Comparison Across Experiments',
            'ylabel': 'Ranking Score DCG',
            'filename': 'pdgrapher_ranking_score_dcg_comparison.png',
            'higher_better': True
        }
    ]
    
    # Create plots for each metric
    for metric_config in metrics_config:
        column = metric_config['column']
        
        # Check if the column exists in the data
        if column not in df.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping {metric_config['title']}")
            continue
        
        # Filter out rows with missing values for this metric
        metric_df = df.dropna(subset=[column])
        
        if metric_df.empty:
            print(f"Warning: No data available for metric '{column}'. Skipping plot.")
            continue
        
        # Calculate mean and std for each experiment
        stats_df = metric_df.groupby('experiment')[column].agg(['mean', 'std', 'count']).reset_index()
        stats_df['std'] = stats_df['std'].fillna(0)  # Fill NaN std (when count=1) with 0
        
        # Sort experiments by mean value for better visualization
        if metric_config['higher_better']:
            stats_df = stats_df.sort_values('mean', ascending=False)
        else:
            stats_df = stats_df.sort_values('mean', ascending=True)
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Create bar plot with error bars
        bars = plt.bar(range(len(stats_df)), stats_df['mean'], 
                      yerr=stats_df['std'], capsize=5, 
                      alpha=0.7, color='steelblue', 
                      error_kw={'color': 'black', 'capthick': 2})
        
        # Customize the plot
        plt.title(metric_config['title'], fontsize=16, fontweight='bold')
        plt.xlabel('Experiment', fontsize=12, fontweight='bold')
        plt.ylabel(metric_config['ylabel'], fontsize=12, fontweight='bold')
        
        # Set x-axis labels
        plt.xticks(range(len(stats_df)), stats_df['experiment'], rotation=45, ha='right')
        
        # Add value labels on top of bars
        for i, (mean_val, std_val, count) in enumerate(zip(stats_df['mean'], stats_df['std'], stats_df['count'])):
            label_text = f'{mean_val:.3f}\n(n={int(count)})'
            plt.text(i, mean_val + std_val + (max(stats_df['mean']) * 0.02), 
                    label_text, ha='center', va='bottom', fontsize=9)
        
        # Add grid for better readability
        plt.grid(axis='y', alpha=0.3)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, metric_config['filename'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {column} comparison plot: {output_path}")
        
        # Print summary statistics
        print(f"\n{column.upper().replace('_', ' ')} Summary:")
        print("Experiment | Mean | Std | Count")
        print("-" * 40)
        for _, row in stats_df.iterrows():
            print(f"{row['experiment']:<20} | {row['mean']:>6.3f} | {row['std']:>5.3f} | {int(row['count']):>5}")


def main():
    parser = argparse.ArgumentParser(description='Create PDGrapher comparison plots across experiments')
    parser.add_argument('--output-dir', default='reports',
                       help='Output directory for plots (default: reports)')
    parser.add_argument('--reports-dir', default='reports',
                       help='Directory containing experiment folders (default: reports)')
    
    args = parser.parse_args()
    
    print("Creating PDGrapher comparison plots...")
    print(f"Scanning experiments in: {args.reports_dir}")
    print(f"Output directory: {args.output_dir}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load PDGrapher data from all experiments
    df = load_pdgrapher_data_from_experiments(args.reports_dir)
    
    if df.empty:
        print("No PDGrapher data found. Cannot create comparison plots.")
        return
    
    # Create comparison plots
    create_pdgrapher_comparison_plots(df, args.output_dir)
    
    # Save combined data for further analysis
    output_csv = os.path.join(args.output_dir, 'pdgrapher_comparison_data.csv')
    df.to_csv(output_csv, index=False)
    print(f"\nSaved combined PDGrapher data: {output_csv}")
    
    print(f"\nPDGrapher comparison complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()