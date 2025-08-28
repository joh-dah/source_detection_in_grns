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
    
    # Debug: Check for duplicates before pivoting
    print("Timing data before pivot:")
    print(timing_df.head(10))
    print(f"Shape: {timing_df.shape}")
    
    # Check for duplicate method-stage combinations
    duplicates = timing_df.groupby(['method', 'stage']).size()
    duplicate_entries = duplicates[duplicates > 1]
    if not duplicate_entries.empty:
        print("WARNING: Found duplicate method-stage combinations:")
        print(duplicate_entries)
        print("Taking the maximum duration for duplicates...")
        # Remove duplicates by taking the maximum duration for each method-stage combination
        timing_df = timing_df.groupby(['method', 'stage']).agg({'duration_hours': 'max'}).reset_index()
    
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
        'accuracy',
        'avg rank of source',
        'source in top 3',
        'source in top 5',
    ]
    
    # Filter to metrics that exist in the dataframe
    available_metrics = [m for m in key_metrics if m in df.columns]
    
    print(f"Creating plots for metrics: {available_metrics}")
    
    # Overview comparison across all methods
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Source Detection Methods Comparison Overview', fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy (higher is better)
    if 'accuracy' in df.columns:
        ax1 = axes[0, 0]
        df_plot = df.dropna(subset=['accuracy'])
        sns.barplot(data=df_plot, x='model_type', y='accuracy', ax=ax1)
        ax1.set_title('Accuracy (Higher is Better)')
        ax1.set_ylabel('Accuracy')
        ax1.tick_params(axis='x', rotation=45)

    # Plot 2: Average rank of source (lower is better)
    if 'avg rank of source' in df.columns:
        ax2 = axes[0, 1]
        df_plot = df.dropna(subset=['avg rank of source'])
        sns.barplot(data=df_plot, x='model_type', y='avg rank of source', ax=ax2)
        ax2.set_title('Average Rank of Source (Lower is Better)')
        ax2.set_ylabel('Average Rank')
        ax2.tick_params(axis='x', rotation=45)

    # Plot 3: Source in top 3 (higher is better)
    if 'source in top 3' in df.columns:
        ax3 = axes[1, 0]
        df_plot = df.dropna(subset=['source in top 3'])
        sns.barplot(data=df_plot, x='model_type', y='source in top 3', ax=ax3)
        ax3.set_title('Source in Top 3 (Higher is Better)')
        ax3.set_ylabel('Source in Top 3 (%)')
        ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Source in top 5 (higher is better)
    if 'source in top 5' in df.columns:
        ax4 = axes[1, 1]
        df_plot = df.dropna(subset=['source in top 5'])
        sns.barplot(data=df_plot, x='model_type', y='source in top 5', ax=ax4)
        ax4.set_title('Source in Top 5 (Higher is Better)')
        ax4.set_ylabel('Source in Top 5 (%)')
        ax4.tick_params(axis='x', rotation=45)  
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'overview_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()


def load_data_from_all_runs(reports_base_path: str = "reports") -> pd.DataFrame:
    """Load data from all run directories and combine into a single DataFrame."""
    all_data = []
    
    if not os.path.exists(reports_base_path):
        print(f"Reports directory {reports_base_path} does not exist!")
        return pd.DataFrame()
    
    # Get all run directories
    run_dirs = [d for d in os.listdir(reports_base_path) 
               if os.path.isdir(os.path.join(reports_base_path, d))]
    
    print(f"Found run directories: {run_dirs}")
    
    for run_dir in run_dirs:
        run_path = os.path.join(reports_base_path, run_dir)
        print(f"Loading data from {run_path}...")
        
        # Load JSON files from this run
        data = load_json_files(run_path)
        
        if data:
            # Create DataFrame for this run
            df = create_metrics_dataframe(data)
            df['run'] = run_dir  # Add run identifier
            all_data.append(df)
        else:
            print(f"No data found in {run_path}")
    
    if not all_data:
        print("No data loaded from any runs!")
        return pd.DataFrame()
    
    # Combine all runs
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Combined data from {len(run_dirs)} runs, total {len(combined_df)} records")
    
    return combined_df


def create_cross_run_comparison_plots(df: pd.DataFrame, methods: List[str], output_dir: str):
    """Create bar charts comparing specified methods across different runs for multiple metrics."""
    
    if df.empty:
        print("No data available for cross-run comparison")
        return
    
    # Handle "all" option
    if methods == ["all"] or (len(methods) == 1 and methods[0].lower() == "all"):
        available_methods = sorted(df['model_type'].unique())
        print(f"Using all available methods: {available_methods}")
        methods = available_methods
    
    # Filter data for specified methods
    filtered_df = df[df['model_type'].isin(methods)].copy()
    
    if filtered_df.empty:
        print(f"No data found for methods: {methods}")
        print(f"Available methods: {df['model_type'].unique()}")
        return
    
    print(f"Creating cross-run comparison plots for methods: {methods}")
    
    # Define metrics to plot
    metrics_config = [
        {
            'column': 'accuracy',
            'title': 'Accuracy Comparison Across Different Runs',
            'ylabel': 'Accuracy',
            'filename': 'combined_cross_run_accuracy_comparison.png',
            'format': '%.3f',
            'higher_better': True
        },
        {
            'column': 'avg rank of source',
            'title': 'Average Rank of Source Comparison Across Different Runs',
            'ylabel': 'Average Rank of Source',
            'filename': 'combined_cross_run_avg_rank_comparison.png',
            'format': '%.2f',
            'higher_better': False
        },
        {
            'column': 'source in top 3',
            'title': 'Source in Top 3 Comparison Across Different Runs',
            'ylabel': 'Source in Top 3 (%)',
            'filename': 'combined_cross_run_top3_comparison.png',
            'format': '%.1f',
            'higher_better': True
        },
        {
            'column': 'source in top 5',
            'title': 'Source in Top 5 Comparison Across Different Runs',
            'ylabel': 'Source in Top 5 (%)',
            'filename': 'combined_cross_run_top5_comparison.png',
            'format': '%.1f',
            'higher_better': True
        }
    ]
    
    # Create plots for each metric
    for metric_config in metrics_config:
        column = metric_config['column']
        
        # Check if the column exists in the data
        if column not in filtered_df.columns:
            print(f"Warning: Column '{column}' not found in data. Skipping {metric_config['title']}")
            continue
        
        # Filter out rows with missing values for this metric
        metric_df = filtered_df.dropna(subset=[column])
        
        if metric_df.empty:
            print(f"Warning: No data available for metric '{column}'. Skipping plot.")
            continue
        
        plt.figure(figsize=(15, 10))
        ax = sns.barplot(data=metric_df, x='run', y=column, hue='model_type', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        
        # Add value labels for seaborn plot
        for container in ax.containers:
            ax.bar_label(container, fmt=metric_config['format'], fontsize=6, rotation=90, padding=2)
        
        plt.title(metric_config['title'], fontsize=16, fontweight='bold')
        plt.xlabel('Run', fontsize=12, fontweight='bold')
        plt.ylabel(metric_config['ylabel'], fontsize=12, fontweight='bold')
        
        # Adjust legend position for many methods
        if len(methods) > 5:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        else:
            plt.legend()
        
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        output_path = os.path.join(output_dir, metric_config['filename'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved {column} comparison plot: {output_path}")
    
    # Print summary statistics for all methods and metrics
    print("\nSummary Statistics:")
    for metric_config in metrics_config:
        column = metric_config['column']
        if column in filtered_df.columns:
            print(f"\n{column.upper().replace('_', ' ')}:")
            for method in methods:
                method_data = filtered_df[filtered_df['model_type'] == method]
                if not method_data.empty and not method_data[column].isna().all():
                    valid_data = method_data[column].dropna()
                    if not valid_data.empty:
                        print(f"  {method.upper()}:")
                        print(f"    Mean: {valid_data.mean():.3f}")
                        print(f"    Std:  {valid_data.std():.3f}")
                        print(f"    Min:  {valid_data.min():.3f}")
                        print(f"    Max:  {valid_data.max():.3f}")
                    else:
                        print(f"  {method.upper()}: No valid data")
                else:
                    print(f"  {method.upper()}: No data")


def compare_methods_across_runs(methods: List[str] = None, output_dir: str = None):
    """
    Compare specified methods across all available runs.
    
    Args:
        methods: List of method names to compare (e.g., ["pdgrapher", "overlap"]) 
                 or ["all"] to compare all available methods
        output_dir: Directory to save plots (defaults to reports/)
    """
    
    if output_dir is None:
        output_dir = "reports"
    
    print(f"Comparing methods {methods} across all runs...")
    
    # Load data from all runs
    all_data = load_data_from_all_runs("reports")
    
    if all_data.empty:
        print("No data loaded. Cannot create comparison plots.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create cross-run comparison plots
    create_cross_run_comparison_plots(all_data, methods, output_dir)
    
    # Save combined data for further analysis
    output_csv = os.path.join(output_dir, 'cross_run_comparison_data.csv')
    all_data.to_csv(output_csv, index=False)
    print(f"Saved combined data: {output_csv}")
    
    print(f"\nCross-run comparison complete! Results saved to: {output_dir}")


def main():
    # Import constants only when needed for single experiment analysis
    import src.constants as const
    
    results_dir = f"{const.REPORT_PATH}/{const.EXPERIMENT}"
    print(f"Loading results from: {results_dir}")
    # Load data
    data = load_json_files(results_dir)
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
