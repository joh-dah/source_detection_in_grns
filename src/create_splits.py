import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from src import constants as const

def create_data_splits():
    """
    Run the data splitting process.
    This function is intended to be called from the command line.
    """
    data_dir = Path(const.DATA_PATH)
    print(f"Creating splits for data in: {data_dir}")
    outdir = data_dir / "splits"
    os.makedirs(outdir, exist_ok=True)
    
    dataset_size = len(list(Path(const.RAW_PATH).glob("*.pt")))
    print(f"Detected  {dataset_size} individual files")

    indices = list(range(dataset_size))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.4, random_state=const.SEED
    )
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

    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created splits for {dataset_size} individual files and saved to {osp.join(outdir, 'splits.pt')}")
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
