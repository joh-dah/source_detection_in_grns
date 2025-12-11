"""Aggregate k-fold cross-validation results."""
import json
import os
from pathlib import Path
from collections import defaultdict
import numpy as np
import src.constants as const


def aggregate_fold_results(k_folds=5):
    """
    Aggregate validation results from all k-folds.
    
    Args:
        k_folds: Number of folds to aggregate
    """
    print(f"Aggregating results from {k_folds} folds for experiment: {const.EXPERIMENT}")
    
    # Paths for results
    report_base = Path(const.REPORT_PATH) / const.EXPERIMENT
    report_base.mkdir(parents=True, exist_ok=True)
    
    # Dictionary to store all fold results
    all_fold_results = defaultdict(lambda: defaultdict(list))
    
    # Collect results from each fold
    for fold_idx in range(k_folds):
        fold_report_dir = Path(const.REPORT_PATH) / const.EXPERIMENT / f"fold_{fold_idx}"
        
        if not fold_report_dir.exists():
            print(f"Warning: No results found for fold {fold_idx} at {fold_report_dir}")
            continue
        
        # Look for results files for each model
        for model_type in ["pdgrapher", "pdgrapher_nognn"]:
            results_file = fold_report_dir / f"{model_type}_results.json"
            
            if not results_file.exists():
                print(f"Warning: No results file for {model_type} fold {fold_idx}")
                continue
            
            try:
                with open(results_file, 'r') as f:
                    fold_metrics = json.load(f)
                
                # Store metrics from this fold
                if 'metrics' in fold_metrics:
                    for metric_name, metric_value in fold_metrics['metrics'].items():
                        # Skip non-numeric metrics
                        if isinstance(metric_value, (int, float)):
                            all_fold_results[model_type][metric_name].append(metric_value)
                        # Handle nested dicts (like gene-specific metrics)
                        elif isinstance(metric_value, dict):
                            for sub_metric, sub_value in metric_value.items():
                                if isinstance(sub_value, (int, float)):
                                    key = f"{metric_name}_{sub_metric}"
                                    all_fold_results[model_type][key].append(sub_value)
                
                print(f"Loaded results for {model_type} fold {fold_idx}")
            except Exception as e:
                print(f"Error loading results for {model_type} fold {fold_idx}: {e}")
    
    # Compute aggregate statistics
    aggregate_results = {}
    
    for model_type in ["pdgrapher", "pdgrapher_nognn"]:
        model_results = {}
        model_results['num_folds'] = k_folds
        
        if model_type in all_fold_results:
            metrics_data = all_fold_results[model_type]
            
            for metric_name, values in metrics_data.items():
                if len(values) > 0:
                    model_results[f"{metric_name}_mean"] = round(float(np.mean(values)), 4)
                    model_results[f"{metric_name}_std"] = round(float(np.std(values)), 4)
                    model_results[f"{metric_name}_min"] = round(float(np.min(values)), 4)
                    model_results[f"{metric_name}_max"] = round(float(np.max(values)), 4)
            
            aggregate_results[model_type] = model_results
    
    # Save aggregate results
    aggregate_file = report_base / f"{const.EXPERIMENT}_kfold_aggregated_results.json"
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate_results, f, indent=2)
    
    print(f"Aggregated results saved to: {aggregate_file}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("K-FOLD AGGREGATION SUMMARY")
    print("="*80)
    
    for model_type, metrics in aggregate_results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Number of folds: {metrics['num_folds']}")
        
        # Print selected key metrics
        for key in sorted(metrics.keys()):
            if '_mean' in key:
                metric_name = key.replace('_mean', '')
                mean_val = metrics[key]
                std_val = metrics.get(f"{metric_name}_std", 0)
                print(f"  {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("="*80 + "\n")
    
    return aggregate_results


def main():
    """Main aggregation function."""
    print("Starting K-Fold Results Aggregation...")
    
    # Get k-folds from environment or use default
    k_folds = int(os.environ.get('K_FOLDS', 5))
    print(f"K-Folds: {k_folds}")
    print(f"Experiment: {const.EXPERIMENT}")
    
    # Aggregate results
    aggregate_results = aggregate_fold_results(k_folds=k_folds)
    
    print("K-fold aggregation complete!")


if __name__ == "__main__":
    main()
