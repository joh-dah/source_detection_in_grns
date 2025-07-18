"""
Create splits for both PDGrapher-style and GNN-style data.
Supports both forward/backward datasets (PDGrapher) and individual file datasets (GNN models).
Based on PDGrapher's create_standard_splits.py but extended for all model types.
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
        nfolds: Number of folds (use 1 for single split)
        outdir: Output directory for splits.pt file
    """
    # Handle single split case (no cross-validation)
    if nfolds == 1:
        return create_single_split(dataset_forward, dataset_backward, splits_type, outdir)
    
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


def create_single_split(dataset_forward, dataset_backward, splits_type, outdir):
    """
    Create a single train/val/test split (no cross-validation).
    Returns a flat dictionary structure for PDGrapher compatibility.
    
    Args:
        dataset_forward: List of forward data objects
        dataset_backward: List of backward data objects  
        splits_type: Type of splits (e.g., 'random')
        outdir: Output directory for splits.pt file
    """
    os.makedirs(outdir, exist_ok=True)
    
    if splits_type == 'random':
        # Case 1: Both forward and backward data exist
        if len(dataset_forward) > 0:
            # Split forward data: 60% train, 20% val, 20% test
            indices_forward = list(range(len(dataset_forward)))
            train_forward, temp_forward = train_test_split(
                indices_forward, test_size=0.4, random_state=42
            )
            val_forward, test_forward = train_test_split(
                temp_forward, test_size=0.5, random_state=42
            )
            
            # Split backward data: 60% train, 20% val, 20% test
            indices_backward = list(range(len(dataset_backward)))
            train_backward, temp_backward = train_test_split(
                indices_backward, test_size=0.4, random_state=42
            )
            val_backward, test_backward = train_test_split(
                temp_backward, test_size=0.5, random_state=42
            )
            
            # Assertions to ensure correct splits
            assert len(dataset_forward) == len(train_forward) + len(val_forward) + len(test_forward)
            assert len(dataset_backward) == len(train_backward) + len(val_backward) + len(test_backward)
            assert len(set(test_forward).intersection(train_forward)) == 0
            assert len(set(test_forward).intersection(val_forward)) == 0
            assert len(set(train_forward).intersection(val_forward)) == 0
            assert len(set(test_backward).intersection(train_backward)) == 0
            assert len(set(test_backward).intersection(val_backward)) == 0
            assert len(set(train_backward).intersection(val_backward)) == 0
            
            splits = {
                'train_index_forward': train_forward,
                'val_index_forward': val_forward,
                'test_index_forward': test_forward,
                'train_index_backward': train_backward,
                'val_index_backward': val_backward,
                'test_index_backward': test_backward
            }
        
        # Case 2: Only backward data exists (no forward data)
        else:
            # Split backward data: 60% train, 20% val, 20% test
            indices_backward = list(range(len(dataset_backward)))
            train_backward, temp_backward = train_test_split(
                indices_backward, test_size=0.4, random_state=42
            )
            val_backward, test_backward = train_test_split(
                temp_backward, test_size=0.5, random_state=42
            )
            
            assert len(dataset_backward) == len(train_backward) + len(val_backward) + len(test_backward)
            assert len(set(test_backward).intersection(train_backward)) == 0
            assert len(set(test_backward).intersection(val_backward)) == 0
            assert len(set(train_backward).intersection(val_backward)) == 0
            
            splits = {
                'train_index_forward': None,
                'val_index_forward': None, 
                'test_index_forward': None,
                'train_index_backward': train_backward,
                'val_index_backward': val_backward,
                'test_index_backward': test_backward
            }
    
    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created single split and saved to {osp.join(outdir, 'splits.pt')}")
    return splits


def create_splits_for_data(processed_dir, nfolds, splits_type='random'):
    """
    Create splits for processed data (works for both PDGrapher and GNN models).
    
    Args:
        processed_dir: Path to the processed data directory
        nfolds: Number of folds
        splits_type: Type of splits (default: 'random')
    """
    processed_dir = Path(processed_dir)
    if not processed_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_dir}")
    
    # Check what kind of data we have
    forward_file = processed_dir / "data_forward.pt"
    backward_file = processed_dir / "data_backward.pt"
    
    # Check for individual numbered files (GNN-style)
    individual_files = sorted(list(processed_dir.glob("[0-9]*.pt")))
    
    dataset_forward = []
    dataset_backward = []
    dataset_individual = []
    
    # Handle PDGrapher-style data (forward/backward files)
    if forward_file.exists() or backward_file.exists():
        print("Detected PDGrapher-style data")
        
        if forward_file.exists():
            dataset_forward = torch.load(forward_file, weights_only=False)
            print(f"Loaded {len(dataset_forward)} forward data objects")
        else:
            print("No forward data file found")
        
        if backward_file.exists():
            dataset_backward = torch.load(backward_file, weights_only=False)
            print(f"Loaded {len(dataset_backward)} backward data objects")
        else:
            print("No backward data file found")
        
        if not dataset_forward and not dataset_backward:
            raise FileNotFoundError("No valid PDGrapher data files found")
        
        # Create PDGrapher-style splits
        outdir = processed_dir / "splits"
        splits = create_splits_n_fold(
            dataset_forward, dataset_backward, splits_type, nfolds, str(outdir)
        )
        
    # Handle GNN-style data (individual numbered files)
    elif individual_files:
        print(f"Detected GNN-style data with {len(individual_files)} individual files")
        
        # For GNN models, we just need indices since data is loaded by index
        dataset_size = len(individual_files)
        splits = create_splits_for_individual_files(dataset_size, nfolds, splits_type, processed_dir)
        
    else:
        raise FileNotFoundError(f"No valid data files found in {processed_dir}. Expected either 'data_forward.pt'/'data_backward.pt' or numbered files like '0.pt', '1.pt', etc.")
    
    return splits


def create_splits_for_individual_files(dataset_size, nfolds, splits_type, processed_dir):
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
    
    if splits_type == 'random':
        if nfolds == 1:
            # Single split
            indices = list(range(dataset_size))
            train_indices, temp_indices = train_test_split(
                indices, test_size=0.4, random_state=42
            )
            val_indices, test_indices = train_test_split(
                temp_indices, test_size=0.5, random_state=42
            )
            
            # Assertions
            assert len(indices) == len(train_indices) + len(val_indices) + len(test_indices)
            assert len(set(test_indices).intersection(train_indices)) == 0
            assert len(set(test_indices).intersection(val_indices)) == 0
            assert len(set(train_indices).intersection(val_indices)) == 0
            
            splits = {
                'train_index': train_indices,
                'val_index': val_indices,
                'test_index': test_indices
            }
            
        else:
            # N-fold splits
            kf = KFold(nfolds, shuffle=True, random_state=42)
            splits = {}
            
            indices = list(range(dataset_size))
            for i, (train_test_idx, _) in enumerate(kf.split(indices), 1):
                train_idx, test_idx = train_test_split(
                    train_test_idx, test_size=0.25, random_state=42  # 75% train, 25% test from the train_test split
                )
                train_idx, val_idx = train_test_split(
                    train_idx, test_size=0.2, random_state=42  # 80% train, 20% val from the remaining
                )
                
                splits[i] = {
                    'train_index': train_idx.tolist(),
                    'val_index': val_idx.tolist(),
                    'test_index': test_idx.tolist()
                }
    
    # Save splits to file
    torch.save(splits, osp.join(outdir, 'splits.pt'))
    print(f"Created splits for {dataset_size} individual files and saved to {osp.join(outdir, 'splits.pt')}")
    return splits


def main():
    """
    Main function to create splits for any model type.
    """
    parser = argparse.ArgumentParser(description="Create splits for processed data (PDGrapher or GNN models)")
    parser.add_argument("--processed-dir", type=str, default=None,
                       help="Path to processed data directory (e.g., data/processed/GAT)")
    parser.add_argument("--model", type=str, default=None,
                       help="Model type (will use data/processed/{model})")
    parser.add_argument("--nfolds", type=int, default=None,
                       help="Number of folds (use 1 for single split, overrides constants)")
    parser.add_argument("--splits-type", type=str, default="random",
                       help="Type of splits (default: random)")
    
    args = parser.parse_args()
    
    # Determine processed data directory
    if args.processed_dir:
        processed_dir = Path(args.processed_dir)
    elif args.model:
        processed_dir = Path(f"data/processed/{args.model}")
    else:
        try:
            processed_dir = Path(f"data/processed/{const.MODEL}")
        except AttributeError:
            raise ValueError("Must specify either --processed-dir, --model, or have const.MODEL defined")
    
    # Determine number of folds
    if args.nfolds is not None:
        nfolds = args.nfolds
    else:
        try:
            nfolds = const.N_FOLDS
        except AttributeError:
            nfolds = 5  # default to 5-fold
    
    print(f"Creating splits for data in: {processed_dir}")
    print(f"Number of folds: {nfolds}")
    print(f"Splits type: {args.splits_type}")
    
    # Create splits
    splits = create_splits_for_data(
        processed_dir=processed_dir,
        nfolds=nfolds,
        splits_type=args.splits_type
    )
    
    print("Splits created successfully!")
    
    # Print summary
    if isinstance(splits, dict) and 'train_index' in splits:
        # GNN-style splits
        if nfolds == 1:
            print(f"\nSingle Split:")
            print(f"  Train: {len(splits['train_index'])}, "
                  f"Val: {len(splits['val_index'])}, "
                  f"Test: {len(splits['test_index'])}")
        else:
            for fold_idx, fold_data in splits.items():
                if isinstance(fold_idx, int):
                    print(f"\nFold {fold_idx}:")
                    print(f"  Train: {len(fold_data['train_index'])}, "
                          f"Val: {len(fold_data['val_index'])}, "
                          f"Test: {len(fold_data['test_index'])}")
    else:
        # PDGrapher-style splits
        if nfolds == 1:
            print(f"\nSingle Split:")
            if splits.get('train_index_forward') is not None:
                print(f"  Forward - Train: {len(splits['train_index_forward'])}, "
                      f"Val: {len(splits['val_index_forward'])}, "
                      f"Test: {len(splits['test_index_forward'])}")
            if splits.get('train_index_backward') is not None:
                print(f"  Backward - Train: {len(splits['train_index_backward'])}, "
                      f"Val: {len(splits['val_index_backward'])}, "
                      f"Test: {len(splits['test_index_backward'])}")
        else:
            for fold_idx, fold_data in splits.items():
                if isinstance(fold_idx, int):
                    print(f"\nFold {fold_idx}:")
                    if fold_data.get('train_index_forward') is not None:
                        print(f"  Forward - Train: {len(fold_data['train_index_forward'])}, "
                              f"Val: {len(fold_data['val_index_forward'])}, "
                              f"Test: {len(fold_data['test_index_forward'])}")
                    if fold_data.get('train_index_backward') is not None:
                        print(f"  Backward - Train: {len(fold_data['train_index_backward'])}, "
                              f"Val: {len(fold_data['val_index_backward'])}, "
                              f"Test: {len(fold_data['test_index_backward'])}")


if __name__ == "__main__":
    main()
