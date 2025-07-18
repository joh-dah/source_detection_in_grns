import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from src import constants as const


# def create_single_split(dataset_forward, dataset_backward, outdir):
#     """
#     Create a single train/val/test split (no cross-validation).
#     Returns a flat dictionary structure for PDGrapher compatibility.
    
#     Args:
#         dataset_forward: List of forward data objects
#         dataset_backward: List of backward data objects  
#         outdir: Output directory for splits.pt file
#     """
#     os.makedirs(outdir, exist_ok=True)

#     assert len(dataset_backward) > 0, "Dataset must not be empty."
#     assert len(dataset_forward) == len(dataset_backward), "Forward and backward datasets must have the same length."

#     indices = list(range(len(dataset_backward)))
#     train, temp = train_test_split(
#         indices, test_size=1-const.TRAINING_SHARE, random_state=42
#     )
#     val, test = train_test_split(
#         temp, test_size=0.5, random_state=42
#     )
    
#     splits = {
#         'train_index': train,
#         'val_index': val,
#         'test_index': test,
#     }

#     torch.save(splits, osp.join(outdir, 'splits.pt'))
#     print(f"Created data split and saved to {osp.join(outdir, 'splits.pt')}")
#     return splits


# def create_pdgrapher_splits(processed_dir, out_dir):
#     """    Load forward and backward datasets from the processed directory.
#     Args:
#         processed_dir: Path to the processed data directory
#     Returns:
#         dataset_forward: List of forward data objects
#         dataset_backward: List of backward data objects
#     """
#     forward_file = processed_dir / "data_forward.pt"
#     backward_file = processed_dir / "data_backward.pt"

#     dataset_forward = torch.load(forward_file, weights_only=False)
#     dataset_backward = torch.load(backward_file, weights_only=False)

#     return create_single_split(dataset_forward, dataset_backward, out_dir)


def create_splits_for_individual_files(dataset_size, processed_dir):
    """
    Create splits for individual file datasets (GNN models).
    
    Args:
        dataset_size: Number of data files
        nfolds: Number of folds
        splits_type: Type of splits (default: 'random')
        processed_dir: Directory to save splits
    """
    outdir = processed_dir / "splits"
    os.makedirs(outdir, exist_ok=True)

    # Single split
    indices = list(range(dataset_size))
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.4, random_state=const.SEED
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=const.SEED
    )

    if const.MODEL == "pdgrapher":
        splits = {
        'train_index_forward': train_indices,
        'val_index_forward': val_indices,
        'test_index_forward': test_indices,
        'train_index_backward': train_indices,
        'val_index_backward': val_indices,
        'test_index_backward': test_indices
    }
    else:
        splits = {
            'train_index': train_indices,
            'val_index': val_indices,
            'test_index': test_indices
        }

    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created splits for {dataset_size} individual files and saved to {osp.join(outdir, 'splits.pt')}")
    return splits


def create_data_splits():
    """
    Run the data splitting process.
    This function is intended to be called from the command line.
    """
    processed_dir = Path(const.PROCESSED_PATH)
    print(f"Creating splits for data in: {processed_dir}")
    outdir = processed_dir / "splits"
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

    if const.MODEL == "pdgrapher":
        splits = {
        'train_index_forward': train_indices,
        'val_index_forward': val_indices,
        'test_index_forward': test_indices,
        'train_index_backward': train_indices,
        'val_index_backward': val_indices,
        'test_index_backward': test_indices
    }
    else:
        splits = {
            'train_index': train_indices,
            'val_index': val_indices,
            'test_index': test_indices
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
