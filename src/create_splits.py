from pathlib import Path
from sklearn.model_selection import train_test_split
import math
import torch
import os
import src.constants as const


def create_data_splits(use_processed_data=False):
    print(f"Creating splits for shared data in: {const.SHARED_DATA_PATH}")
    outdir = Path(const.SPLITS_FILE).parent
    os.makedirs(outdir, exist_ok=True)

    if use_processed_data:
        data_path = Path(const.PROCESSED_PATH)
        print(f"Creating splits for processed data in: {data_path}")
        n_samples = len(torch.load(data_path / "data_forward.pt", weights_only=False))
        assert n_samples == len(torch.load(data_path / "data_backward.pt", weights_only=False)), "Forward and backward data lengths do not match."
    else:
        data_path = Path(const.RAW_PATH)
        print(f"Creating splits for raw data in: {data_path}")
        n_samples = len(list(data_path.glob("*.pt")))

    indices = list(range(n_samples))

    if const.FIXED_TRAINING_SIZE is not None:
        train_size = const.FIXED_TRAINING_SIZE
        test_size = (n_samples - train_size) // 2
    else:
        train_size = int(n_samples * const.TRAINING_SHARE)
        test_size = (n_samples - train_size) // 2

    print(f"Total samples: {n_samples}, Train size: {train_size}, Test size: {test_size}, Validation size: {test_size}")

    train_indices_forward, temp_indices_forward = train_test_split(
        indices, train_size=train_size, random_state=const.SEED
    )
    val_indices_forward, test_indices_forward = train_test_split(
        temp_indices_forward, test_size=test_size, random_state=const.SEED
    )
    train_indices_backward, temp_indices_backward = train_test_split(
        indices, train_size=train_size, random_state=const.SEED+1
    )
    val_indices_backward, test_indices_backward = train_test_split(
        temp_indices_backward, test_size=test_size, random_state=const.SEED+1
    )
    splits = {
        "train_index_forward": train_indices_forward,
        "val_index_forward": val_indices_forward,
        "test_index_forward": test_indices_forward,
        "train_index_backward": train_indices_backward,
        "val_index_backward": val_indices_backward,
        "test_index_backward": test_indices_backward,
    }
    splits_file = outdir / "splits.pt"
    torch.save(splits, splits_file)
    print(f"Splits saved to {splits_file}")
    return splits

def main():
    print("Starting Data Splitting...")
    print("Checking for raw data...")
    print(f"{const.N_SAMPLES} where created, Duplicate removal is set to {const.REMOVE_NEAR_DUPLICATES}")
    min_files_needed = 1 if const.REMOVE_NEAR_DUPLICATES else const.N_SAMPLES
    print(f"At least {min_files_needed} files are expected from data creation.")
    min_files_needed = min(min_files_needed, const.FIXED_TRAINING_SIZE + 2)
    print(f"{const.FIXED_TRAINING_SIZE} samples are reserved for training")
    raw_data_usable = False
    if Path(const.RAW_PATH).exists():
        file_count = len(list(Path(const.RAW_PATH).glob("*.pt")))
        print(f"Found {file_count} raw files...")
        raw_data_usable = file_count >= min_files_needed
    if not raw_data_usable and Path(const.PROCESSED_PATH).exists():
        print("Raw data not usable, checking processed data...")
        file_count = len(list(Path(const.PROCESSED_PATH).glob("*.pt")))
        print(f"Found {file_count} processed files...")
        proc_data_usable = file_count >= min_files_needed
        if proc_data_usable:
            print("Creating splits from processed data...")
            create_data_splits(use_processed_data=True)
        else:
            raise FileNotFoundError("Not enough usable data found for splitting. Aborting")
    elif raw_data_usable:
        print("Creating splits from raw data...")
        create_data_splits(use_processed_data=False)

if __name__ == "__main__":
    main()