"""
Create splits for PDGrapher-style data.
Based on PDGrapher's create_standard_splits.py
"""

import os
import os.path as osp
from sklearn.model_selection import KFold, train_test_split
import torch
from pathlib import Path
import argparse
from src import constants as const


def create_splits_n_fold(dataset_forward, dataset_backward, splits_type, nfolds, outdir):
    """
    Create n-fold splits for forward and backward datasets.
    
    Args:
        dataset_forward: List of forward data objects
        dataset_backward: List of backward data objects  
        splits_type: Type of splits (e.g., 'random')
        nfolds: Number of folds
        outdir: Output directory for splits.pt file
    """
    kf = KFold(nfolds, shuffle=True, random_state=42)
    os.makedirs(outdir, exist_ok=True)
    
    splits = {}
    
    if splits_type == 'random':
        i = 1
        
        # Case 1: Both forward and backward data exist
        if len(dataset_forward) > 0:
            for train_test_index_forward, train_test_index_backward in zip(
                kf.split(dataset_forward), kf.split(dataset_backward)
            ):
                # Split forward data
                train_index_forward = train_test_index_forward[0]
                test_index_forward = train_test_index_forward[1]
                train_index_forward, val_index_forward = train_test_split(
                    train_index_forward, test_size=0.2, random_state=42
                )
                
                # Split backward data
                train_index_backward = train_test_index_backward[0]
                test_index_backward = train_test_index_backward[1]
                train_index_backward, val_index_backward = train_test_split(
                    train_index_backward, test_size=0.2, random_state=42
                )
                
                # Assertions to ensure correct splits
                assert len(dataset_forward) == len(train_index_forward) + len(val_index_forward) + len(test_index_forward)
                assert len(dataset_backward) == len(train_index_backward) + len(val_index_backward) + len(test_index_backward)
                assert len(set(test_index_forward).intersection(train_index_forward)) == 0
                assert len(set(test_index_forward).intersection(val_index_forward)) == 0
                assert len(set(train_index_forward).intersection(val_index_forward)) == 0
                assert len(set(test_index_backward).intersection(train_index_backward)) == 0
                assert len(set(test_index_backward).intersection(val_index_backward)) == 0
                assert len(set(train_index_backward).intersection(val_index_backward)) == 0
                
                splits[i] = {
                    'train_index_forward': train_index_forward,
                    'val_index_forward': val_index_forward,
                    'test_index_forward': test_index_forward,
                    'train_index_backward': train_index_backward,
                    'val_index_backward': val_index_backward,
                    'test_index_backward': test_index_backward
                }
                i += 1
        
        # Case 2: Only backward data exists (no forward data)
        else:
            for train_test_index_backward in kf.split(dataset_backward):
                train_index_backward = train_test_index_backward[0]
                test_index_backward = train_test_index_backward[1]
                train_index_backward, val_index_backward = train_test_split(
                    train_index_backward, test_size=0.2, random_state=42
                )
                
                assert len(dataset_backward) == len(train_index_backward) + len(val_index_backward) + len(test_index_backward)
                assert len(set(test_index_backward).intersection(train_index_backward)) == 0
                assert len(set(test_index_backward).intersection(val_index_backward)) == 0
                assert len(set(train_index_backward).intersection(val_index_backward)) == 0
                
                splits[i] = {
                    'train_index_forward': None,
                    'val_index_forward': None, 
                    'test_index_forward': None,
                    'train_index_backward': train_index_backward,
                    'val_index_backward': val_index_backward,
                    'test_index_backward': test_index_backward
                }
                i += 1
    
    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created {nfolds}-fold splits and saved to {osp.join(outdir, 'splits.pt')}")
    return splits


def create_splits_for_data(data_path, nfolds, splits_type='random'):
    """
    Create splits for processed PDGrapher-style data.
    
    Args:
        data_path: Path to the processed data directory
        nfolds: Number of folds
        splits_type: Type of splits (default: 'random')
    """
    data_path = Path(data_path)
    processed_dir = data_path / "processed"
    
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Load forward and backward datasets
    forward_file = processed_dir / "data_forward.pt"
    backward_file = processed_dir / "data_backward.pt"
    
    dataset_forward = []
    dataset_backward = []
    
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
    
    # Create output directory for splits
    splits_setting = f"{nfolds}fold"
    outdir = processed_dir / "splits" / splits_type / splits_setting
    
    # Create splits
    splits = create_splits_n_fold(
        dataset_forward, dataset_backward, splits_type, nfolds, str(outdir)
    )
    
    return splits


def main():
    """
    Main function to create splits.
    """
    parser = argparse.ArgumentParser(description="Create splits for PDGrapher-style data")
    parser.add_argument("--data-path", type=str, default=None,
                       help="Path to data directory (overrides constants)")
    parser.add_argument("--splits-type", type=str, default="random",
                       help="Type of splits (default: random)")
    
    args = parser.parse_args()
    
    # Determine data path
    if args.data_path:
        data_path = args.data_path
    else:
        try:
            data_path = const.DATA_PATH
        except AttributeError:
            data_path = "data"
    
    print(f"Creating splits for data in: {data_path}")
    print(f"Number of folds: {const.N_FOLDS}")
    print(f"Splits type: {args.splits_type}")
    
    # Create splits
    splits = create_splits_for_data(
        data_path=data_path,
        nfolds=const.N_FOLDS,
        splits_type=args.splits_type
    )
    
    print("Splits created successfully!")
    
    # Print summary
    for fold_idx, fold_data in splits.items():
        print(f"\nFold {fold_idx}:")
        if fold_data['train_index_forward'] is not None:
            print(f"  Forward - Train: {len(fold_data['train_index_forward'])}, "
                  f"Val: {len(fold_data['val_index_forward'])}, "
                  f"Test: {len(fold_data['test_index_forward'])}")
        print(f"  Backward - Train: {len(fold_data['train_index_backward'])}, "
              f"Val: {len(fold_data['val_index_backward'])}, "
              f"Test: {len(fold_data['test_index_backward'])}")


if __name__ == "__main__":
    main()
