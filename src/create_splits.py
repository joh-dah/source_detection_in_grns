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
    # if splits_file.exists():
    #     print(f"Splits already exist at {splits_file}, loading existing splits")
    #     return torch.load(splits_file)
    
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
        if const.FIXED_TRAINING_SIZE is not None:
            if const.FIXED_TRAINING_SIZE >= n_forward:
                raise ValueError(f"Fixed training size {const.FIXED_TRAINING_SIZE} is greater than or equal to total forward samples {n_forward}.")
            test_size = int((n_forward - const.FIXED_TRAINING_SIZE)/2)
        else: 
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
        if const.FIXED_TRAINING_SIZE is not None:
            if const.FIXED_TRAINING_SIZE >= n_total:
                raise ValueError(f"Fixed training size {const.FIXED_TRAINING_SIZE} is greater than or equal to total forward samples {n_forward}.")
            test_size = int((n_total - const.FIXED_TRAINING_SIZE)/2)
        else: 
            forward_indices = list(range(n_total))
            test_size = 1.0 - const.TRAINING_SHARE
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
    print(f"Train samples: {len(train_indices_forward)}, Validation samples: {len(val_indices_forward)}, Test samples: {len(test_indices_forward)}")
    return splits
        

def main():
    """
    Main function to create splits.
    """
    print("Starting data splitting process...")
    #check if splits.pt already exist
    splits_path = Path(const.SHARED_DATA_PATH) / "splits" / "splits.pt"
    # if splits_path.exists():
    #     print(f"Splits already exist at {splits_path}, exiting.")
    #     return
    # Try raw data first, then fall back to processed data
    raw_data_usable = False
    
    if Path(const.RAW_PATH).exists():
        file_count = len(list(Path(const.RAW_PATH).glob("*.pt")))
        # If remove_near_duplicates is enabled, any files > 0 is valid
        # If disabled, we need at least N_SAMPLES files
        min_files_needed = 1 if const.REMOVE_NEAR_DUPLICATES else const.N_SAMPLES
        
        if file_count >= min_files_needed:
            print(f"Raw data detected ({file_count} files), creating splits from raw data...")
            create_data_splits(use_processed_data=False)
            raw_data_usable = True
        else:
            expected_desc = "at least 1" if const.REMOVE_NEAR_DUPLICATES else f"at least {const.N_SAMPLES}"
            print(f"Raw data insufficient at {const.RAW_PATH} (found {file_count} files, need {expected_desc})")
            print("Falling back to processed data...")
    
    if not raw_data_usable and Path(const.PROCESSED_PATH).exists():
        # Use processed data as fallback
        print("Processed data detected, creating splits from processed data...")
        create_data_splits(use_processed_data=True)
    elif not raw_data_usable:
        # Only throw error if neither raw nor processed data is usable
        raw_files = len(list(Path(const.RAW_PATH).glob("*.pt"))) if Path(const.RAW_PATH).exists() else 0
        expected_desc = "at least 1" if const.REMOVE_NEAR_DUPLICATES else f"at least {const.N_SAMPLES}"
        error_msg = f"No sufficient raw data found at {const.RAW_PATH} (found {raw_files} files, need {expected_desc})"
        error_msg += f" and no processed data found at {const.PROCESSED_PATH}"
        error_msg += ". Please run data creation or processing first."
        raise FileNotFoundError(error_msg)
    print("Splits created successfully!")
    

if __name__ == "__main__":
    main()
