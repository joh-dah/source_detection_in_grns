"""Create k-fold cross-validation splits for experiments."""
from pathlib import Path
from sklearn.model_selection import KFold
import torch
import src.constants as const


def create_kfold_splits(use_processed_data=False):
    """
    Create k-fold cross-validation splits for the data.
    
    Args:
        use_processed_data: Whether to use processed or raw data
        
    Returns:
        Dictionary containing all fold splits
    """
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
        # Further split training data into train and validation (80-20 split within each fold)
        # Use KFold with 2 splits to get train/val from the training portion
        val_split = KFold(n_splits=2, shuffle=True, random_state=const.SEED + 2 + fold_idx)
        train_val_pairs_fwd = list(val_split.split(train_idx_fwd))
        train_val_pairs_bwd = list(val_split.split(train_idx_bwd))

        # Get the first split (train and val indices within training data)
        train_indices_fwd_subset, val_indices_fwd_subset = train_val_pairs_fwd[0]
        train_indices_bwd_subset, val_indices_bwd_subset = train_val_pairs_bwd[0]

        # Map back to original indices
        train_indices_forward = [train_idx_fwd[i] for i in train_indices_fwd_subset]
        val_indices_forward = [train_idx_fwd[i] for i in val_indices_fwd_subset]
        test_indices_forward = [forward_indices[i] for i in test_idx_fwd]

        train_indices_backward = [train_idx_bwd[i] for i in train_indices_bwd_subset]
        val_indices_backward = [train_idx_bwd[i] for i in val_indices_bwd_subset]
        test_indices_backward = [backward_indices[i] for i in test_idx_bwd]

        # Store fold-specific splits
        fold_key = f"fold_{fold_idx}"
        all_splits[fold_key] = {
            "train_index_forward": train_indices_forward,
            "val_index_forward": val_indices_forward,
            "test_index_forward": test_indices_forward,
            "train_index_backward": train_indices_backward,
            "val_index_backward": val_indices_backward,
            "test_index_backward": test_indices_backward,
        }

        print(f"Fold {fold_idx}:")
        print(f"  Forward - Train: {len(train_indices_forward)}, Val: {len(val_indices_forward)}, Test: {len(test_indices_forward)}")
        print(f"  Backward - Train: {len(train_indices_backward)}, Val: {len(val_indices_backward)}, Test: {len(test_indices_backward)}")

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
