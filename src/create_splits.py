import os
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from src import constants as const

def create_data_splits():
    """
    Run the data splitting process.
    Uses shared data path so splits are reused between experiments with same data_creation config.
    """
    # first check, whether processed data is available
    if not Path(const.PROCESSED_PATH).exists():
        raise FileNotFoundError(f"Processed data path {const.PROCESSED_PATH} does not exist. Please run data processing first.")
    # check what files are in the processed data path
    processed_files = list(Path(const.PROCESSED_PATH).glob("*.pt"))
    if len(processed_files) == 0:
        raise FileNotFoundError(f"No processed data files found in {const.PROCESSED_PATH}. Please run data processing first.")

    print(f"Creating splits for shared data in: {const.SHARED_DATA_PATH}")
    outdir = Path(const.SHARED_DATA_PATH) / "splits"
    os.makedirs(outdir, exist_ok=True)
    
    # Check if splits already exist
    splits_file = outdir / "splits.pt"
    if splits_file.exists():
        print(f"Splits already exist at {splits_file}, loading existing splits")
        return torch.load(splits_file)
    
    dataset_size = len(list(Path(const.RAW_PATH).glob("*.pt")))
    print(f"Detected  {dataset_size} individual files")

    indices = list(range(dataset_size))
    
    # Use training_share from config instead of hardcoded values
    test_size = 1.0 - const.TRAINING_SHARE
    train_indices, temp_indices = train_test_split(
        indices, test_size=test_size, random_state=const.SEED
    )
    # Split remaining into val/test equally
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=const.SEED
    )

    splits = {
        'train_index_forward': train_indices,
        'val_index_forward': val_indices,
        'test_index_forward': test_indices,
        'train_index_backward': train_indices,
        'val_index_backward': val_indices,
        'test_index_backward': test_indices,
    }

    # Save splits to shared location
    torch.save(splits, splits_file)
    print(f"Created splits for {dataset_size} individual files and saved to {splits_file}")
    return splits
        

def main():
    """
    Main function to create splits.
    """
    print("Starting data splitting process...")
    create_data_splits()
    print("Splits created successfully!")
    

if __name__ == "__main__":
    main()
