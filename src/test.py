import torch
import torch.optim as optim
import src.constants as const

from pdgrapher import Dataset, PDGrapher, Trainer

import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

pdgrapher_path = "external/pdgrapher"

def main():
    print("=== Starting PDGrapher Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    dataset = Dataset(
        forward_path="external/PDGrapher/data/processed/torch_data/real_lognorm/data_forward_A549.pt",
        backward_path="external/PDGrapher/data/processed/torch_data/real_lognorm/data_backward_A549.pt",
        splits_path="external/PDGrapher/data/splits/genetic/A549/random/1fold/splits.pt"
    )
    
    print(f"Dataset loaded - Number of variables: {dataset.get_num_vars()}")
    
    # Get dataloader info for estimation
    train_loader_f, train_loader_b, val_loader_f, val_loader_b, test_loader_f, test_loader_b = dataset.get_dataloaders()
    print(f"Training batches: forward={len(train_loader_f)}, backward={len(train_loader_b)}")
    print(f"Validation batches: forward={len(val_loader_f)}, backward={len(val_loader_b)}")
    print(f"Test batches: forward={len(test_loader_f)}, backward={len(test_loader_b)}")

    edge_index = torch.load("external/PDGrapher/data/processed/torch_data/real_lognorm/edge_index_A549.pt")
    print(f"Graph edges: {edge_index.shape}")
    
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
        "dim_gnn": 8, "num_vars": dataset.get_num_vars()
        })
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    model.set_optimizers_and_schedulers([optim.Adam(model.response_prediction.parameters(), lr=0.0075),
        optim.Adam(model.perturbation_discovery.parameters(), lr=0.0033)])

    if const.ON_CLUSTER:
        fabric_kwargs = {"accelerator": "cuda", "devices": 1}
    else:
        fabric_kwargs = {"accelerator": "cpu", "devices": 1}

    trainer = Trainer(
        fabric_kwargs=fabric_kwargs,
        log=True, logging_name="tuned",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )
    
    print(f"\n=== Starting Training (2 epochs) ===")
    print("This will show progress for each epoch...")
    import time
    start_time = time.time()

    model_performance = trainer.train(model, dataset, n_epochs = 2)
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n=== Training Complete ===")
    print(f"Total training time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Average time per epoch: {total_time/2:.2f} seconds")

    print(model_performance)
    
    # Create the directory if it doesn't exist
    import os
    output_dir = "external/PDGrapher/examples/PDGrapher"
    os.makedirs(output_dir, exist_ok=True)
    
    with open("external/PDGrapher/examples/PDGrapher/tuned_final.txt", "w") as f:
        f.write(str(model_performance))


if __name__ == "__main__":
    main()
