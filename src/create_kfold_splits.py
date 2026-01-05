"""Create k-fold cross-validation splits for experiments."""
from pathlib import Path
from sklearn.model_selection import KFold, GroupKFold
import torch
import pandas as pd
import numpy as np
import src.constants as const


def split_test_val_indices(test_val_indices, all_indices, group_ids=None):
    """
    Split test_val indices into validation and test sets (50-50 split).
    
    Args:
        test_val_indices: Indices of test+val samples
        all_indices: All available indices
        group_ids: Optional group IDs for group-based splitting
        
    Returns:
        Tuple of (val_indices, test_indices)
    """
    if group_ids is not None:
        # Use GroupKFold for group-aware splitting
        test_val_groups = group_ids[test_val_indices]
        test_val_fold = GroupKFold(n_splits=2)
        test_val_splits = list(test_val_fold.split(test_val_indices, groups=test_val_groups))
        val_indices_rel, test_indices_rel = test_val_splits[0]
        val_indices = test_val_indices[val_indices_rel]
        test_indices = test_val_indices[test_indices_rel]
    else:
        # Use standard KFold for non-grouped splitting
        val_split = KFold(n_splits=2, shuffle=True, random_state=const.SEED)
        val_split_pairs = list(val_split.split(test_val_indices))
        val_indices_rel, test_indices_rel = val_split_pairs[0]
        val_indices = test_val_indices[val_indices_rel]
        test_indices = test_val_indices[test_indices_rel]
    
    return val_indices, test_indices


def create_group_kfold_splits(use_processed_data=False):
    """
    Create k-fold cross-validation splits with group constraints to prevent data leakage.
    
    Groups are defined by the tuple (perturbed_gene_id, cell_line, perturbation_type).
    All samples within a group are assigned to the same fold.
    
    Args:
        use_processed_data: Whether to use processed or raw data
        
    Returns:
        Dictionary containing all fold splits
    """
    print(f"Creating {const.K_FOLDS}-fold splits with group constraints")
    print(f"Group key: (perturbed_gene_id, cell_line, perturbation_type)")
    outdir = Path(const.SPLITS_FILE).parent
    outdir.mkdir(parents=True, exist_ok=True)

    # Load metadata file that contains grouping information
    metadata_file = Path(const.SHARED_DATA_PATH) / "sample_metadata.pt"
    if not metadata_file.exists():
        raise FileNotFoundError(
            f"Metadata file not found at {metadata_file}. "
            "Make sure to save sample_metadata.pt during data creation."
        )

    metadata = torch.load(metadata_file, weights_only=False)
    print(f"Loaded metadata for {len(metadata)} samples")
    
    # Extract group IDs from metadata
    group_ids = metadata.get("group_ids", None)
    if group_ids is None:
        raise ValueError("Metadata does not contain 'group_ids'. Check data creation script.")
    
    # Verify consistency
    if use_processed_data:
        data_path = Path(const.PROCESSED_PATH)
        n_samples = len(list(data_path.glob("*.pt")))
    else:
        data_path = Path(const.RAW_PATH)
        n_samples = len(list(data_path.glob("*.pt")))
    
    if len(group_ids) != n_samples:
        raise ValueError(
            f"Metadata mismatch: {len(group_ids)} group IDs but {n_samples} data files found"
        )
    
    print(f"Found {len(np.unique(group_ids))} unique groups")
    
    # Create indices and group assignments
    indices = np.arange(n_samples)
    
    # Initialize GroupKFold
    gkfold = GroupKFold(n_splits=const.K_FOLDS)
    
    # Store all fold splits
    all_splits = {}
    
    fold_idx = 0
    for train_idx, test_val_idx in gkfold.split(indices, groups=group_ids):
        # Split test_val_idx into validation and test sets
        val_indices, test_idx = split_test_val_indices(test_val_idx, indices, group_ids=group_ids)
        train_indices = train_idx
        fold_key = fold_idx + 1  # Folds numbered from 1
        all_splits[fold_key] = {
            "train_index_forward": train_indices.tolist(),
            "val_index_forward": val_indices.tolist(),
            "test_index_forward": test_idx.tolist(),
            "train_index_backward": train_indices.tolist(),
            "val_index_backward": val_indices.tolist(),
            "test_index_backward": test_idx.tolist(),
        }
        
        print(f"Fold {fold_idx}:")
        print(f"  Train: {len(train_indices)} samples ({100*len(train_indices)/n_samples:.1f}%), "
              f"Val: {len(val_indices)} samples ({100*len(val_indices)/n_samples:.1f}%), "
              f"Test: {len(test_idx)} samples ({100*len(test_idx)/n_samples:.1f}%)")
        print(f"  Train groups: {len(np.unique(group_ids[train_indices]))}, "
              f"Val groups: {len(np.unique(group_ids[val_indices]))}, "
              f"Test groups: {len(np.unique(group_ids[test_idx]))}")
        
        # Verify no overlap
        assert len(set(train_indices) & set(val_indices)) == 0, "Train and val overlap!"
        assert len(set(train_indices) & set(test_idx)) == 0, "Train and test overlap!"
        assert len(set(val_indices) & set(test_idx)) == 0, "Val and test overlap!"
        
        fold_idx += 1
    
    # Save all folds to a single file
    splits_file = Path(const.SPLITS_FILE).parent / "splits_kfold.pt"
    torch.save(all_splits, splits_file)
    print(f"\nK-fold splits saved to {splits_file}")
    print(f"Total folds created: {len(all_splits)}")
    
    # Also save metadata about the splits
    split_metadata = {
        "method": "GroupKFold",
        "group_key": "(perturbed_gene_id, cell_line, perturbation_type)",
        "n_unique_groups": len(np.unique(group_ids)),
        "n_folds": const.K_FOLDS,
        "total_samples": n_samples,
    }
    
    split_metadata_file = Path(const.SPLITS_FILE).parent / "splits_metadata.pt"
    torch.save(split_metadata, split_metadata_file)
    print(f"Split metadata saved to {split_metadata_file}")

    return all_splits


def create_kfold_splits(use_processed_data=False):
    """
    Create k-fold cross-validation splits for the data.
    
    If metadata with group information is available, uses GroupKFold to prevent data leakage.
    Otherwise falls back to standard KFold.
    
    Args:
        use_processed_data: Whether to use processed or raw data
        
    Returns:
        Dictionary containing all fold splits
    """
    # Check if metadata file exists for group-based splitting
    metadata_file = Path(const.SHARED_DATA_PATH) / "sample_metadata.pt"
    if metadata_file.exists():
        print("Metadata file found, using group-based k-fold splitting...")
        return create_group_kfold_splits(use_processed_data=use_processed_data)
    
    # Fallback to standard KFold if no metadata
    print("No metadata file found, using standard k-fold splitting...")
    print("WARNING: This may lead to data leakage if similar samples exist in different folds!")
    
    print(f"Creating {const.K_FOLDS}-fold splits for shared data in: {const.SHARED_DATA_PATH}")
    outdir = Path(const.SPLITS_FILE).parent
    outdir.mkdir(parents=True, exist_ok=True)

    if use_processed_data:
        data_path = Path(const.PROCESSED_PATH)
        print(f"Creating splits for processed data in: {data_path}")
        n_samples_forward = len(torch.load(data_path / "data_forward.pt", weights_only=False))
        n_samples_backward = len(torch.load(data_path / "data_backward.pt", weights_only=False))
    else:
        data_path = Path(const.RAW_PATH)
        print(f"Creating splits for raw data in: {data_path}")
        n_samples_forward = len(list(data_path.glob("*.pt")))
        n_samples_backward = n_samples_forward

    if n_samples_forward == 0:
        raise ValueError(f"No data files found in {data_path}")

    forward_indices = list(range(n_samples_forward))
    backward_indices = list(range(n_samples_backward))

    print(f"Total samples forward: {n_samples_forward}")
    print(f"Total samples backward: {n_samples_backward}")
    print(f"Creating {const.K_FOLDS} folds...")

    # Initialize KFold with proper random state for reproducibility
    kfold_forward = KFold(n_splits=const.K_FOLDS, shuffle=True, random_state=const.SEED)
    kfold_backward = KFold(n_splits=const.K_FOLDS, shuffle=True, random_state=const.SEED + 1)

    # Store all fold splits
    all_splits = {}

    # Generate splits for each fold
    for fold_idx, ((train_idx_fwd, test_idx_fwd), (train_idx_bwd, test_idx_bwd)) in enumerate(
        zip(kfold_forward.split(forward_indices), kfold_backward.split(backward_indices))
    ):
        # Split test indices into validation and test sets (50-50 split of test portion)
        # This gives us 80% train, 10% val, 10% test (same as group k-fold)
        val_indices_fwd, test_indices_fwd = split_test_val_indices(test_idx_fwd, np.array(forward_indices))
        val_indices_bwd, test_indices_bwd = split_test_val_indices(test_idx_bwd, np.array(backward_indices))
        
        # Convert to lists for consistency
        train_indices_forward = train_idx_fwd.tolist() if isinstance(train_idx_fwd, np.ndarray) else list(train_idx_fwd)
        val_indices_forward = val_indices_fwd.tolist() if isinstance(val_indices_fwd, np.ndarray) else list(val_indices_fwd)
        test_indices_forward = test_indices_fwd.tolist() if isinstance(test_indices_fwd, np.ndarray) else list(test_indices_fwd)

        train_indices_backward = train_idx_bwd.tolist() if isinstance(train_idx_bwd, np.ndarray) else list(train_idx_bwd)
        val_indices_backward = val_indices_bwd.tolist() if isinstance(val_indices_bwd, np.ndarray) else list(val_indices_bwd)
        test_indices_backward = test_indices_bwd.tolist() if isinstance(test_indices_bwd, np.ndarray) else list(test_indices_bwd)

        # Store fold-specific splits
        fold_key = fold_idx + 1 
        all_splits[fold_key] = {
            "train_index_forward": train_indices_forward,
            "val_index_forward": val_indices_forward,
            "test_index_forward": test_indices_forward,
            "train_index_backward": train_indices_backward,
            "val_index_backward": val_indices_backward,
            "test_index_backward": test_indices_backward,
        }

        print(f"Fold {fold_idx}:")
        print(f"  Forward - Train: {len(train_indices_forward)} ({100*len(train_indices_forward)/n_samples_forward:.1f}%), "
              f"Val: {len(val_indices_forward)} ({100*len(val_indices_forward)/n_samples_forward:.1f}%), "
              f"Test: {len(test_indices_forward)} ({100*len(test_indices_forward)/n_samples_forward:.1f}%)")
        print(f"  Backward - Train: {len(train_indices_backward)} ({100*len(train_indices_backward)/n_samples_backward:.1f}%), "
              f"Val: {len(val_indices_backward)} ({100*len(val_indices_backward)/n_samples_backward:.1f}%), "
              f"Test: {len(test_indices_backward)} ({100*len(test_indices_backward)/n_samples_backward:.1f}%)")

    # Save all folds to a single file
    splits_file = Path(const.SPLITS_FILE).parent / "splits_kfold.pt"
    torch.save(all_splits, splits_file)
    print(f"\nK-fold splits saved to {splits_file}")
    print(f"Total folds created: {len(all_splits)}")

    return all_splits


def main():
    """Main function for k-fold splitting."""
    print("Starting K-Fold Data Splitting...")
    print(f"K-Folds configuration: {const.K_FOLDS}")
    
    # Check for raw data
    print("Checking for raw data...")
    print(f"{const.N_SAMPLES} samples were created, Duplicate removal is set to {const.REMOVE_NEAR_DUPLICATES}")
    
    min_files_needed = 1 if const.REMOVE_NEAR_DUPLICATES else const.N_SAMPLES
    print(f"At least {min_files_needed} files are expected from data creation.")
    
    raw_data_usable = False
    if Path(const.RAW_PATH).exists() and not const.USE_PROCESSED_DATA:
        file_count = len(list(Path(const.RAW_PATH).glob("*.pt")))
        print(f"Found {file_count} raw files...")
        raw_data_usable = file_count >= min_files_needed
        completion_file = Path(const.SHARED_DATA_PATH) / "data_creation_complete.txt"
        
        # check for the file data_creation_complete.txt
        if not completion_file.exists():
            print("Data creation not complete for raw data. Aborting.")
            raw_data_usable = False
    
    if not raw_data_usable and Path(const.PROCESSED_PATH).exists():
        print("Raw data not usable, checking processed data...")
        file_count = len(list(Path(const.PROCESSED_PATH).glob("*.pt")))
        print(f"Found {file_count} processed files...")
        proc_data_usable = file_count >= min_files_needed
        
        if proc_data_usable:
            print("Creating k-fold splits from processed data...")
            create_kfold_splits(use_processed_data=True)
        else:
            raise FileNotFoundError("Not enough usable data found for k-fold splitting. Aborting")
    elif raw_data_usable:
        print("Creating k-fold splits from raw data...")
        create_kfold_splits(use_processed_data=False)
    else:
        raise FileNotFoundError("No usable data found for k-fold splitting")


if __name__ == "__main__":
    main()
