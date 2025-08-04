#!/usr/bin/env python3
"""
Script to compare source detection method results from JSON files.
Creates a comprehensive comparison report with visualizations.
"""

import json
import os
import glob
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Any
import src.constants as const


def load_timing_data(experiment: str) -> pd.DataFrame:
    """Load timing data from CSV file created by SLURM pipeline."""
    timing_file = Path("timing_logs") / f"{experiment}_timing.csv"
    
    if not timing_file.exists():
        print(f"Warning: Timing file {timing_file} not found. Runtime plot will be skipped.")
        return pd.DataFrame()
    
    try:
        timing_df = pd.read_csv(timing_file)
        print(f"Loaded timing data with {len(timing_df)} entries")
        return timing_df
    except Exception as e:
        print(f"Error loading timing data: {e}")
        return pd.DataFrame()


def process_timing_data(timing_df: pd.DataFrame) -> pd.DataFrame:
    """Process timing data to calculate total runtime per method and stage."""
    if timing_df.empty:
        return pd.DataFrame()
    
    # Convert duration to hours for better readability
    timing_df['duration_hours'] = timing_df['duration_seconds'] / 3600
    
    # Combine all baseline methods into a single "baseline" entry
    timing_df_processed = timing_df.copy()
    timing_df_processed.loc[timing_df_processed['method'].str.startswith('baseline'), 'method'] = 'baseline'
    
    # Group by method and stage, summing durations (in case of multiple runs)
    processed_timing = timing_df_processed.groupby(['method', 'stage']).agg({
        'duration_hours': 'max'
    }).reset_index()
    
    # Define stage order for consistent plotting
    stage_order = ['data_creation', 'data_processing', 'training', 'evaluation']
    processed_timing['stage'] = pd.Categorical(processed_timing['stage'], categories=stage_order, ordered=True)
    
    return processed_timing


def create_runtime_comparison_plot(timing_df: pd.DataFrame, output_dir: str):
    """Create a stacked bar plot comparing runtime across methods and stages."""
    if timing_df.empty:
        print("No timing data available. Skipping runtime plot.")
        return
    
    print("Creating runtime comparison plot...")
    
    # Create pivot table for stacked bar chart
    pivot_df = timing_df.pivot(index='method', columns='stage', values='duration_hours')
    pivot_df = pivot_df.fillna(0)  # Fill missing stages with 0
    
    print("Pivot table for runtime data:")
    print(pivot_df)
    print("\nTotal times per method:")
    print(pivot_df.sum(axis=1))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for each stage
    stage_colors = {
        'data_creation': '#1f77b4',      # Blue
        'data_processing': '#ff7f0e',    # Orange  
        'training': '#2ca02c',           # Green
        'evaluation': '#d62728'          # Red
    }
    
    # Get available stages in the correct order
    available_stages = [stage for stage in ['data_creation', 'data_processing', 'training', 'evaluation'] 
                       if stage in pivot_df.columns]
    
    # Create stacked bars
    bottom = np.zeros(len(pivot_df))
    bars = []
    
    for stage in available_stages:
        if stage in pivot_df.columns:
            bar = ax.bar(pivot_df.index, pivot_df[stage], bottom=bottom, 
                        label=stage.replace('_', ' ').title(), 
                        color=stage_colors.get(stage, '#gray'))
            bars.append(bar)
            bottom += pivot_df[stage]
    
    # Customize plot
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Runtime (Hours)', fontsize=12, fontweight='bold')
    ax.set_title('Runtime Comparison by Method and Pipeline Stage', fontsize=14, fontweight='bold')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars for total runtime
    for i, method in enumerate(pivot_df.index):
        total_time = pivot_df.loc[method].sum()
        if total_time > 0:  # Only add label if there's actually runtime data
            # Convert total_time (in hours) to seconds
            total_seconds = int(round(total_time * 3600))
            days = total_seconds // 86400
            hours = (total_seconds % 86400) // 3600
            minutes = (total_seconds % 3600) // 60
            seconds = total_seconds % 60
            label = f"{days}d - {hours:02}:{minutes:02}:{seconds:02}"
            y_offset = pivot_df.sum(axis=1).max() * 0.02  # 2% of max height
            ax.text(i, total_time + y_offset, label, 
                    ha='center', va='bottom', fontweight='bold')
            print(f"Adding label for {method}: total_time={total_time:.6f}, label={label}, y_pos={total_time + y_offset:.6f}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'runtime_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Also create a summary table
    summary_table = pivot_df.copy()
    summary_table['Total'] = summary_table.sum(axis=1)
    summary_table.to_csv(os.path.join(output_dir, 'runtime_summary.csv'))
    
    print("Runtime comparison plot saved!")
    print("\nRuntime Summary (hours):")
    print(summary_table.round(2))


def load_json_files(directory: str) -> List[Dict[str, Any]]:
    """Load all JSON files from directory and extract relevant data."""
    json_files = glob.glob(os.path.join(directory, "*.json"))
    data = []
    
    print(f"Found {len(json_files)} JSON files in {directory}")
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                content = json.load(f)
                
            # Extract filename without extension for method identification
            filename = Path(file_path).stem
            
            # Extract metrics and other relevant info
            entry = {
                'filename': filename,
                'filepath': file_path,
                'metrics': content.get('metrics', {}),
                'network': content.get('network', 'unknown'),
                'model_type': content.get('model_type', filename.split('_')[0] if '_' in filename else 'unknown')
            }
            
            # Extract graph stats if available
            if 'data stats' in content:
                graph_stats = content['data stats'].get('graph stats', {})
                entry['num_nodes'] = graph_stats.get('number of nodes')
                entry['num_possible_sources'] = graph_stats.get('number of possible sources')
                
                infection_stats = content['data stats'].get('infection stats', {})
                entry['avg_num_sources'] = infection_stats.get('avg number of sources')
                entry['avg_portion_affected'] = infection_stats.get('avg portion of affected nodes')
            
            # Extract training parameters if available
            if 'parameters' in content:
                entry['parameters'] = content['parameters']
            
            data.append(entry)
            
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return data

def create_metrics_dataframe(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """Convert loaded data to a pandas DataFrame for easier analysis."""
    rows = []
    
    for entry in data:
        row = {
            'filename': entry['filename'],
            'model_type': entry['model_type'],
            'network': entry['network'],
            'num_nodes': entry.get('num_nodes'),
            'num_possible_sources': entry.get('num_possible_sources'),
            'avg_num_sources': entry.get('avg_num_sources'),
            'avg_portion_affected': entry.get('avg_portion_affected')
        }
        
        # Add all metrics as separate columns
        metrics = entry['metrics']
        for metric_name, metric_value in metrics.items():
            row[metric_name] = metric_value
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    return df

def create_comparison_plots(df: pd.DataFrame, output_dir: str):
    """Create comprehensive comparison plots."""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Key metrics to compare
    key_metrics = [
        'avg rank of source',
        'precision',
        'recall',
        'f1 score',
    ]
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    print(f"Creating plots for metrics: {available_metrics}")
    
    # Overview comparison across all methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Source Detection Methods Comparison Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Average rank of source (lower is better)
    if 'avg rank of source' in df.columns:
        ax1 = axes[0, 0]
        df_plot = df.dropna(subset=['avg rank of source'])
        sns.barplot(data=df_plot, x='model_type', y='avg rank of source', ax=ax1)
        ax1.set_title('Average Rank of Source (Lower is Better)')
        ax1.set_ylabel('Average Rank')
        ax1.tick_params(axis='x', rotation=45)
    
    # Plot 2: Precision
    if 'precision' in df.columns:
        ax2 = axes[1, 0]
        df_plot = df.dropna(subset=['precision'])
        sns.barplot(data=df_plot, x='model_type', y='precision', ax=ax2)
        ax2.set_title('Precision (Higher is Better)')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
    
    # Plot 3: Recall
    if 'recall' in df.columns:
        ax3 = axes[1, 1]
        df_plot = df.dropna(subset=['recall'])
        sns.barplot(data=df_plot, x='model_type', y='recall', ax=ax3)
        ax3.set_title('Recall (Higher is Better)')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)

    # Plot 4: F1 Score (higher is better)
    if 'f1 score' in df.columns:
        ax4 = axes[0, 1]
        df_plot = df.dropna(subset=['f1 score'])
        sns.barplot(data=df_plot, x='model_type', y='f1 score', ax=ax4)
        ax4.set_title('F1 Score (Higher is Better)')
        ax4.set_ylabel('F1 Score')
        ax4.tick_params(axis='x', rotation=45)
    
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():

    const.EXPERIMENT = "XS_run"  # Example experiment name, adjust as needed
    results_dir = f"{const.REPORT_PATH}/{const.EXPERIMENT}"

    results_dir = "reports/XS_run"
    
    print(f"Loading results from: {results_dir}")
    
    # Check if input directory exists
    if not os.path.exists(results_dir):
        print(f"Error: Input directory {results_dir} does not exist!")
        return
    
    # Load data
    data = load_json_files(results_dir)
    
    if not data:
        print("No JSON files found or no valid data loaded!")
        return
    
    # Create DataFrame
    df = create_metrics_dataframe(data)
    
    print(f"Loaded {len(df)} results")
    print(f"Model types: {df['model_type'].unique()}")
    print(f"Networks: {df['network'].unique()}")
    
    # Save raw data as CSV for further analysis
    df.to_csv(os.path.join(results_dir, 'all_results.csv'), index=False)
    
    # Load and process timing data
    timing_df = load_timing_data(const.EXPERIMENT)
    processed_timing = process_timing_data(timing_df)
    
    # Create visualizations
    print("Creating comparison plots...")
    create_comparison_plots(df, results_dir)
    
    # Create runtime comparison plot
    create_runtime_comparison_plot(processed_timing, results_dir)
    
    print(f"\nComparison complete! Results saved to: {results_dir}")
    print("Generated files:")
    print("- all_results.csv: Raw data in CSV format")
    print("- *.png: Comparison plots")
    print("- runtime_summary.csv: Runtime breakdown by method and stage")
    print("- runtime_comparison.png: Runtime comparison plot")

if __name__ == "__main__":
    main()
