#!/usr/bin/env python3
"""
Training time estimator for PDGrapher
"""
import torch
from pdgrapher import Dataset
import time

def estimate_training_time():
    """Estimate training time based on data size and hardware"""
    print("=== Training Time Estimator ===")
    
    # Load dataset to get size info
    print("Loading dataset...")
    dataset = Dataset(
        forward_path="external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt",
        backward_path="external/PDGrapher/data/processed/torch_data/real_lognorm/data_backward_A549.pt",
        splits_path="external/PDGrapher/data/splits/genetic/A549/random/1fold/splits.pt"
    )
    
    # Get dataloader info
    train_loader_f, train_loader_b, val_loader_f, val_loader_b, test_loader_f, test_loader_b = dataset.get_dataloaders()
    
    print(f"Training batches: forward={len(train_loader_f)}, backward={len(train_loader_b)}")
    print(f"Validation batches: forward={len(val_loader_f)}, backward={len(val_loader_b)}")
    print(f"Test batches: forward={len(test_loader_f)}, backward={len(test_loader_b)}")
    
    # Load graph info
    edge_index = torch.load("external/PDGrapher/data/processed/torch_data/real_lognorm/edge_index_A549.pt")
    print(f"Graph edges: {edge_index.shape}")
    print(f"Number of variables: {dataset.get_num_vars()}")
    
    # Estimate times per epoch
    # These are rough estimates based on typical GNN performance
    if torch.cuda.is_available():
        device_type = "GPU"
        # GPU estimates (faster)
        train_time_per_batch = 0.1  # seconds
        val_time_per_batch = 0.05   # seconds (no backprop)
        test_time_per_batch = 0.05  # seconds (no backprop)
    else:
        device_type = "CPU"
        # CPU estimates (slower)
        train_time_per_batch = 0.5  # seconds
        val_time_per_batch = 0.3    # seconds
        test_time_per_batch = 0.3   # seconds
    
    # Calculate time per epoch
    train_time = (len(train_loader_f) + len(train_loader_b)) * train_time_per_batch
    val_time = (len(val_loader_f) + len(val_loader_b)) * val_time_per_batch
    test_time = (len(test_loader_f) + len(test_loader_b)) * test_time_per_batch
    
    epoch_time = train_time + val_time + test_time
    
    print(f"\n=== Time Estimates ({device_type}) ===")
    print(f"Training time per epoch: ~{train_time:.1f}s")
    print(f"Validation time per epoch: ~{val_time:.1f}s")
    print(f"Testing time per epoch: ~{test_time:.1f}s")
    print(f"Total time per epoch: ~{epoch_time:.1f}s ({epoch_time/60:.1f} minutes)")
    
    # Estimates for different epoch counts
    print(f"\n=== Total Training Time Estimates ===")
    for epochs in [2, 10, 50, 100]:
        total_time = epochs * epoch_time
        print(f"{epochs:3d} epochs: {total_time/60:6.1f} minutes ({total_time/3600:5.2f} hours)")
    
    print(f"\n=== Memory Estimates ===")
    # Very rough memory estimates
    if torch.cuda.is_available():
        print("GPU memory usage:")
        print("- Model: ~200-500 MB")
        print("- Data batches: ~100-300 MB")
        print("- Gradients & optimizer: ~400-800 MB")
        print("- Total estimated: ~1-2 GB GPU memory")
    
    print("RAM usage:")
    print("- Dataset loading: ~500MB - 2GB")
    print("- Model & training: ~1-3 GB")
    print("- Total estimated: ~2-5 GB RAM")

if __name__ == "__main__":
    estimate_training_time()
