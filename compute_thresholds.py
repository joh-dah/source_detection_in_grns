#!/usr/bin/env python3
"""
Standalone script to compute thresholds for PDGrapher models.
Contains the core threshold computation logic that can be imported and reused.
"""
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import sys
import yaml
import hashlib
import json
import os


def get_processed_path(experiment_name):
    """Calculate the processed data path using the same config merging as constants system"""
    # Load configuration using the same merging logic as constants system (without importing constants)
    from pathlib import Path
    import yaml
    
    def deep_merge(base: dict, override: dict) -> dict:
        """Deep merge two dictionaries."""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = deep_merge(result[key], value)
            else:
                result[key] = value
        return result
    
    # Load configs like constants system
    config_dir = Path("configs")
    
    # 1. Load shared config
    shared_config = yaml.safe_load(open(config_dir / "shared.yaml"))
    
    # 2. Load model config
    model_config = yaml.safe_load(open(config_dir / "models" / "pdgrapher.yaml"))
    config = deep_merge(shared_config, model_config)
    
    # 3. Load experiment config
    exp_config = yaml.safe_load(open(config_dir / "experiments" / f"{experiment_name}.yaml"))
    exp_config["experiment"] = experiment_name
    config = deep_merge(config, exp_config)
    
    # Use data_utils functions for consistent path calculation
    from src.data_utils import get_data_fingerprint, get_graph_perturbation_fingerprint
    
    # Get parameters from merged config
    data_creation = config["data_creation"] 
    graph_perturbation = config["graph_perturbation"]
    network = data_creation["network"]
    seed = config.get("seed", 42)
    
    # Calculate hashes using data_utils functions
    shared_hash = get_data_fingerprint(data_creation, network, seed)
    experiment_hash = get_graph_perturbation_fingerprint(graph_perturbation)
    
    return f"data/experiments/{shared_hash}/{experiment_hash}/{experiment_name}/processed/pdgrapher"


def compute_and_store_thresholds(datasets, processed_path):
    """
    Compute thresholds for discretizing expression values as required by PDGrapher.
    These thresholds convert continuous expression values to discrete categories (0-499).
    
    How it works:
    1. Collect all expression values from training data
    2. Compute percentile-based thresholds (501 values for 500 categories)
    3. During inference: find which threshold bin an expression value falls into
    4. Use that bin index as the discrete category for embedding lookup
    
    Args:
        datasets: Dictionary with "backward" and/or "forward" dataset keys
        processed_path: Path to the processed data directory
    
    Returns:
        Dictionary of computed thresholds
    """
    thresholds = {}
    
    print("Computing thresholds from processed data files...")
    print("This ensures consistent discretization for training and validation.")
    
    # Collect all expression values - ONLY from training data to avoid data leakage
    all_diseased = []
    all_treated = []
    all_healthy = []
    
    # PERFORMANCE FIX: Load data files once instead of loading for each sample
    processed_dir = Path(processed_path)
    
    # Load backward data efficiently - load file once, not per sample
    if "backward" in datasets:
        backward_file = processed_dir / "data_backward.pt"
        if backward_file.exists():
            print("Loading backward data for threshold computation...")
            backward_data_list = torch.load(backward_file, weights_only=False)
            print(f"Processing {len(backward_data_list)} backward samples for thresholds...")
            for data in tqdm(backward_data_list, desc="Processing backward data"):
                all_diseased.extend(data.diseased.tolist())
                all_treated.extend(data.treated.tolist())
    
    # Load forward data efficiently - load file once, not per sample  
    if "forward" in datasets:
        forward_file = processed_dir / "data_forward.pt"
        if forward_file.exists():
            print("Loading forward data for threshold computation...")
            forward_data_list = torch.load(forward_file, weights_only=False)
            print(f"Processing {len(forward_data_list)} forward samples for thresholds...")
            for data in tqdm(forward_data_list, desc="Processing forward data"):
                all_healthy.extend(data.healthy.tolist())
    
    # Compute percentile-based thresholds (0.2% increments as in PDGrapher)
    # Creates 501 threshold values -> 500 discrete categories (0-499)
    def compute_thresholds(values):
        # Handle different input types
        if not values:
            raise ValueError("No values provided for threshold computation")
        
        # If values is a list of floats (from .tolist()), convert directly
        if isinstance(values[0], (int, float)):
            print("Processing list of float values for threshold computation...")
            all_values = values
        else:
            # If values is a list of tensors, flatten and extract
            print("Processing list of tensors for threshold computation...")
            all_values = []
            for value_tensor in values:
                if hasattr(value_tensor, 'flatten'):
                    all_values.extend(value_tensor.flatten().tolist())
                else:
                    all_values.extend([float(value_tensor)])
        
        # Convert to numpy array for percentile computation
        all_values = np.array(all_values)
        print(f"Computing thresholds from {len(all_values)} expression values")
        print(f"Value range: {all_values.min():.3f} to {all_values.max():.3f}")
        
        # Create 501 threshold values (percentiles from 0 to 100, 0.2% increments)
        percentiles = np.linspace(0, 100, 501)
        threshold_values = np.percentile(all_values, percentiles)
        
        return torch.tensor(threshold_values, dtype=torch.float32)
    
    # For backward direction, combine diseased and treated (input and output states)
    if all_diseased and all_treated:
        backward_values = all_diseased + all_treated
        thresholds["backward"] = compute_thresholds(backward_values)
        print(f"Backward (combined) thresholds: min={thresholds['backward'].min():.3f}, max={thresholds['backward'].max():.3f}")
    
    # For forward direction, combine healthy and diseased
    if all_healthy and all_diseased:
        forward_values = all_healthy + all_diseased 
        thresholds["forward"] = compute_thresholds(forward_values)
        print(f"Forward thresholds: min={thresholds['forward'].min():.3f}, max={thresholds['forward'].max():.3f}")
    
    # Store thresholds
    thresholds_path = Path(processed_path) / "thresholds.pt"
    torch.save(thresholds, thresholds_path)
    print(f"Thresholds saved to {thresholds_path}")
    print(f"Each threshold tensor has {501} values for {500} discrete categories")
    
    return thresholds


def compute_thresholds_for_existing_data(experiment_name, force_overwrite=False):
    """
    Compute thresholds when you have existing processed data but missing thresholds.pt
    This is the main function to call when running this script standalone.
    
    Args:
        experiment_name: Name of the experiment
        force_overwrite: Whether to overwrite existing thresholds.pt
    """
    # Get the processed data path
    processed_path = get_processed_path(experiment_name)
    processed_dir = Path(processed_path)
    
    print(f"Computing thresholds for experiment: {experiment_name}")
    print(f"Processed data directory: {processed_dir}")
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Check if thresholds already exist
    thresholds_path = processed_dir / "thresholds.pt"
    if thresholds_path.exists() and not force_overwrite:
        print(f"Thresholds file already exists: {thresholds_path}")
        print("Use --force to overwrite existing thresholds")
        return torch.load(thresholds_path, weights_only=False)
    
    # Check which data files exist
    data_backward_path = processed_dir / "data_backward.pt"
    data_forward_path = processed_dir / "data_forward.pt"
    
    datasets = {}
    if data_backward_path.exists():
        datasets["backward"] = True
        print(f"Found backward data: {data_backward_path}")
    if data_forward_path.exists():
        datasets["forward"] = True  
        print(f"Found forward data: {data_forward_path}")
    
    if not datasets:
        raise FileNotFoundError(f"No processed data files found in {processed_dir}")
    
    print(f"Computing thresholds from datasets: {list(datasets.keys())}")
    
    # Compute and store thresholds
    thresholds = compute_and_store_thresholds(datasets, processed_path)
    
    # Validate thresholds
    print("\nThreshold validation:")
    for direction, threshold_tensor in thresholds.items():
        if len(threshold_tensor) != 501:
            print(f"WARNING: {direction} thresholds has {len(threshold_tensor)} values, expected 501")
        else:
            print(f"✓ {direction} thresholds: {len(threshold_tensor)} values (correct)")
    
    print(f"\n✓ Threshold computation completed successfully!")
    print(f"Thresholds saved to: {thresholds_path}")
    
    return thresholds


def main():
    """Command line interface for threshold computation."""
    parser = argparse.ArgumentParser(
        description="Compute thresholds for PDGrapher models from existing processed data"
    )
    parser.add_argument(
        "--experiment", 
        type=str, 
        help="Experiment name (will look for configs/experiments/EXPERIMENT.yaml)"
    )
    parser.add_argument(
        "--force", 
        action="store_true", 
        help="Force overwrite existing thresholds.pt"
    )
    
    args = parser.parse_args()
    
    # Get experiment name from argument or environment variable
    experiment_name = args.experiment or os.environ.get('EXPERIMENT_NAME')
    
    if not experiment_name:
        print("ERROR: No experiment name provided!")
        print("Use: python compute_thresholds.py --experiment EXPERIMENT_NAME")
        print("Or set: export EXPERIMENT_NAME=your_experiment")
        sys.exit(1)
    
    try:
        compute_thresholds_for_existing_data(experiment_name, args.force)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
