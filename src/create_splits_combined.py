import os
import os.path as osp
from sklearn.model_selection import train_test_split
import torch
from pathlib import Path
from src import constants as const


def create_single_split(dataset_forward, dataset_backward, outdir):
    """
    Create a single train/val/test split (no cross-validation).
    Returns a flat dictionary structure for PDGrapher compatibility.
    
    Args:
        dataset_forward: List of forward data objects
        dataset_backward: List of backward data objects  
        outdir: Output directory for splits.pt file
    """
    os.makedirs(outdir, exist_ok=True)

    assert len(dataset_backward) > 0, "Backward dataset must not be empty."

    indices_backward = list(range(len(dataset_backward)))
    train_backward, temp_backward = train_test_split(
        indices_backward, test_size=1-const.TRAINING_SHARE, random_state=42
    )
    val_backward, test_backward = train_test_split(
        temp_backward, test_size=0.5, random_state=42
    )

    if len(dataset_forward) > 0:
        indices_forward = list(range(len(dataset_forward)))
        train_forward, temp_forward = train_test_split(
            indices_forward, test_size=1-const.TRAINING_SHARE, random_state=42
        )
        val_forward, test_forward = train_test_split(
            temp_forward, test_size=0.5, random_state=42
        )
    else:
        train_forward, val_forward, test_forward = None, None, None
    
    splits = {
        'train_index_forward': train_forward,
        'val_index_forward': val_forward,
        'test_index_forward': test_forward,
        'train_index_backward': train_backward,
        'val_index_backward': val_backward,
        'test_index_backward': test_backward
    }

    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created data split and saved to {osp.join(outdir, 'splits.pt')}")
    return splits


def create_pdgrapher_splits(processed_dir, out_dir):
    """    Load forward and backward datasets from the processed directory.
    Args:
        processed_dir: Path to the processed data directory
    Returns:
        dataset_forward: List of forward data objects
        dataset_backward: List of backward data objects
    """
    forward_file = processed_dir / "data_forward.pt"
    backward_file = processed_dir / "data_backward.pt"

    if forward_file.exists():
        dataset_forward = torch.load(forward_file, weights_only=False)
        print(f"Loaded {len(dataset_forward)} forward data objects")
    else:
        print("No forward data file found")
    
    if backward_file.exists():
        dataset_backward = torch.load(backward_file, weights_only=False)
        print(f"Loaded {len(dataset_backward)} backward data objects")
    else:
        raise FileNotFoundError(f"Backward data file not found: {backward_file}")
    
    return create_single_split(dataset_forward, dataset_backward, out_dir)


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
        indices, test_size=0.4, random_state=42
    )
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, random_state=42
    )

    splits = {
        'train_index': train_indices,
        'val_index': val_indices,
        'test_index': test_indices
    }

    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created splits for {dataset_size} individual files and saved to {osp.join(outdir, 'splits.pt')}")
    return splits



def run_data_splitting():
    """
    Run the data splitting process.
    This function is intended to be called from the command line.
    """
    print(f"Creating splits for data in: {const.PROCESSED_PATH}")
    out_dir = Path(const.PROCESSED_PATH) / "splits"#
    processed_dir = Path(const.PROCESSED_PATH)

    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    if const.MODEL == "pdgrapher":
        print("Detected PDGrapher-style data")
        splits = create_pdgrapher_splits(processed_dir, out_dir)
    elif const.MODEL in ["GAT", "GCNSI"]:
        individual_files = sorted(list(processed_dir.glob("[0-9]*.pt")))
        print(f"Detected GNN-style data with {len(individual_files)} individual files")
        # For GNN models, we just need indices since data is loaded by index
        dataset_size = len(individual_files)
        splits = create_splits_for_individual_files(dataset_size, processed_dir)

    return splits
        



def main():
    """
    Main function to create splits.
    """
    print("Starting data splitting process...")
    run_data_splitting()
    print("Splits created successfully!")
    



if __name__ == "__main__":
    main()
