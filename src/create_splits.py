import os
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from src import constants as const

def create_data_splits(use_processed_data: bool = False):
    """
    Run the data splitting process.
    Uses shared data path so splits are reused between experiments with same data_creation config.
    """
    raw_path = Path(const.RAW_PATH)
    processed_path = Path(const.PROCESSED_PATH)

    print(f"Creating splits for shared data in: {const.SHARED_DATA_PATH}")
    outdir = Path(const.SHARED_DATA_PATH) / "splits"
    os.makedirs(outdir, exist_ok=True)

    # Check if splits already exist
    splits_file = outdir / "splits.pt"
    if splits_file.exists():
        print(f"Splits already exist at {splits_file}, loading existing splits")
        return torch.load(splits_file)
    
    if use_processed_data:
        if not processed_path.exists():
            raise FileNotFoundError(f"Processed data path {processed_path} does not exist. Please run data processing first.")
        print(f"Creating splits for processed data in: {processed_path}")

        # Load the processed data to get sample counts
        data_forward = torch.load(processed_path / "data_forward.pt", weights_only=False)
        data_backward = torch.load(processed_path / "data_backward.pt", weights_only=False)
        n_forward = len(data_forward)
        n_backward = len(data_backward)
            # Create splits for forward data
        forward_indices = list(range(n_forward))
        test_size = 1.0 - const.TRAINING_SHARE
        
        train_indices_forward, temp_indices_forward = train_test_split(
            forward_indices, test_size=test_size, random_state=const.SEED
        )
        val_indices_forward, test_indices_forward = train_test_split(
            temp_indices_forward, test_size=0.5, random_state=const.SEED
        )
        
        # Create splits for backward data
        backward_indices = list(range(n_backward))
        train_indices_backward, temp_indices_backward = train_test_split(
            backward_indices, test_size=test_size, random_state=const.SEED
        )
        val_indices_backward, test_indices_backward = train_test_split(
            temp_indices_backward, test_size=0.5, random_state=const.SEED
        )
    else:
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw data path {raw_path} does not exist. Please run data creation first.")
        print(f"Creating splits for raw data in: {raw_path}")
        n_total = len(list(raw_path.glob("*.pt")))
        print(f"Detected  {n_total} individual files")
        indices = list(range(n_total))
        test_size = 1.0 - const.TRAINING_SHARE
        train_indices_forward, temp_indices = train_test_split(
            indices, test_size=test_size, random_state=const.SEED
        )
        val_indices_forward, test_indices_forward = train_test_split(
            temp_indices, test_size=0.5, random_state=const.SEED
        )
        train_indices_backward = train_indices_forward
        val_indices_backward = val_indices_forward
        test_indices_backward = test_indices_forward

    splits = {
        'train_index_forward': train_indices_forward,
        'val_index_forward': val_indices_forward,
        'test_index_forward': test_indices_forward,
        'train_index_backward': train_indices_backward,
        'val_index_backward': val_indices_backward,
        'test_index_backward': test_indices_backward,
    }

    # Save splits to shared location
    torch.save(splits, splits_file)
    print(f"Created splits and saved to {splits_file}")
    return splits
        

def main():
    """
    Main function to create splits.
    """
    print("Starting data splitting process...")
    #check if splits.pt already exist
    splits_path = Path(const.SHARED_DATA_PATH) / "splits" / "splits.pt"
    if splits_path.exists():
        print(f"Splits already exist at {splits_path}, exiting.")
        return
    # if in const.RAW_PATH there are at leas const.N_SAMPLES files, we assume raw data exists
    if Path(const.RAW_PATH).exists() and len(list(Path(const.RAW_PATH).glob("*.pt"))) >= const.N_SAMPLES:
        print("Raw data detected, creating splits from raw data...")
        create_data_splits(use_processed_data=False)
    elif Path(const.PROCESSED_PATH).exists():
        print("Processed data detected, creating splits from processed data...")
        create_data_splits(use_processed_data=True)
    else:
        raise FileNotFoundError(f"No raw data found at {const.RAW_PATH} or processed data found at {const.PROCESSED_PATH}. Please run data creation or processing first.")
    print("Splits created successfully!")
    

if __name__ == "__main__":
    main()
