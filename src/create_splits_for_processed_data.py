#!/usr/bin/env python3
"""
Create splits.pt for pre-processed PDGrapher data.
This script creates train/validation/test splits for your downloaded processed data.
"""
import torch
from sklearn.model_selection import train_test_split
import src.constants as const

def create_splits_for_processed_data():
    """
    Create splits for pre-processed PDGrapher data.
    """
    # Load configuration using the same merging logic as constants system (without importing constants)
    # from pathlib import Path
    # import yaml
    
    # def deep_merge(base: dict, override: dict) -> dict:
    #     """Deep merge two dictionaries."""
    #     result = base.copy()
    #     for key, value in override.items():
    #         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
    #             result[key] = deep_merge(result[key], value)
    #         else:
    #             result[key] = value
    #     return result
    
    # # Load configs like constants system
    # config_dir = Path("configs")
    
    # # 1. Load shared config
    # shared_config = yaml.safe_load(open(config_dir / "shared.yaml"))
    
    # # 2. Load model config
    # model_config = yaml.safe_load(open(config_dir / "models" / "pdgrapher.yaml"))
    # config = deep_merge(shared_config, model_config)
    
    # # 3. Load experiment config
    # exp_config = yaml.safe_load(open(config_dir / "experiments" / "og_pdgrapher_data.yaml"))
    # exp_config["experiment"] = "og_pdgrapher_data"
    # config = deep_merge(config, exp_config)
    
    # # Use data_utils functions for consistent path calculation
    # from src.data_utils import get_data_fingerprint, get_graph_perturbation_fingerprint
    
    # # Get parameters from merged config
    # data_creation = config["data_creation"]
    # graph_perturbation = config["graph_perturbation"]
    # network = data_creation["network"]
    # seed = config.get("seed", 42)
    # experiment_name = "og_pdgrapher_data"
    
    # # Calculate hashes using data_utils functions
    # shared_hash = get_data_fingerprint(data_creation, network, seed)
    # experiment_hash = get_graph_perturbation_fingerprint(graph_perturbation)
    
    # # Build the correct experiment data path
    # processed_path = Path(f"data/experiments/{shared_hash}/{experiment_hash}/{experiment_name}/processed/pdgrapher")
    
    processed_path = const.PROCESSED_PATH
    seed = const.SEED
    print(f"Creating splits for processed data in: {processed_path}")
    
    # Load the processed data to get sample counts
    data_forward = torch.load(processed_path / "data_forward.pt", weights_only=False)
    data_backward = torch.load(processed_path / "data_backward.pt", weights_only=False)
    
    n_forward = len(data_forward)
    n_backward = len(data_backward)
    
    print(f"Forward samples: {n_forward}")
    print(f"Backward samples: {n_backward}")
    
    # Create splits for forward data
    forward_indices = list(range(n_forward))
    training_share = 0.6  # Default training share
    test_size = 1.0 - training_share
    
    train_indices_forward, temp_indices_forward = train_test_split(
        forward_indices, test_size=test_size, random_state=seed
    )
    val_indices_forward, test_indices_forward = train_test_split(
        temp_indices_forward, test_size=0.5, random_state=seed
    )
    
    # Create splits for backward data
    backward_indices = list(range(n_backward))
    train_indices_backward, temp_indices_backward = train_test_split(
        backward_indices, test_size=test_size, random_state=seed
    )
    val_indices_backward, test_indices_backward = train_test_split(
        temp_indices_backward, test_size=0.5, random_state=seed
    )
    
    # Create splits dictionary
    splits = {
        'train_index_forward': train_indices_forward,
        'val_index_forward': val_indices_forward,
        'test_index_forward': test_indices_forward,
        'train_index_backward': train_indices_backward,
        'val_index_backward': val_indices_backward,
        'test_index_backward': test_indices_backward,
    }
    
    # Save splits to processed data directory
    splits_file = processed_path / "splits.pt"
    torch.save(splits, splits_file)
    
    print(f"\n✅ Created splits and saved to: {splits_file}")
    print(f"📊 Split sizes:")
    print(f"   Forward - Train: {len(train_indices_forward)}, Val: {len(val_indices_forward)}, Test: {len(test_indices_forward)}")
    print(f"   Backward - Train: {len(train_indices_backward)}, Val: {len(val_indices_backward)}, Test: {len(test_indices_backward)}")
    
    return splits

if __name__ == "__main__":
    print("Creating splits for processed PDGrapher data...")
    create_splits_for_processed_data()
    print("✅ Splits created successfully!")
