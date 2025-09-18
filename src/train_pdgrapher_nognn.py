import torch.optim as optim
import torch
import src.constants as const
import src.utils as utils
from pathlib import Path
import os

# Import the new PDGrapher implementation without external dependencies
from architectures.PDGrapherNoGNN import PDGrapherNoGNN

# Try to import PDGrapher dataset and trainer - adapt as needed
try:
    from pdgrapher import Dataset, Trainer
except ImportError:
    print("Warning: PDGrapher package not available, using fallback implementations")
    # You'll need to implement fallback Dataset and Trainer classes
    # or adapt the training loop to work with your existing data processing

torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """ Train PDGrapherNoGNN model without graph neural networks. """

    data_processed_dir = Path(const.PROCESSED_PATH)
    splits_path = const.SPLITS_PATH

    # Check if all required files exist
    required_files = [
        splits_path,
        f"{data_processed_dir}/data_forward.pt",
        f"{data_processed_dir}/data_backward.pt",
        f"{data_processed_dir}/thresholds.pt"
        # Note: No edge_index.pt required for NoGNN version
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}. Please run data processing first.")

    dataset = Dataset(
        forward_path=f"{data_processed_dir}/data_forward.pt",
        backward_path=f"{data_processed_dir}/data_backward.pt",
        splits_path=const.SPLITS_PATH,
    )

    # Load thresholds (required for discretizing expression values)
    thresholds_path = data_processed_dir / "thresholds.pt"
    thresholds = torch.load(thresholds_path, weights_only=False)
    print(f"Loaded thresholds: {list(thresholds.keys())}")
    
    # Validate thresholds
    for direction in ['forward', 'backward']:
        if direction not in thresholds:
            raise ValueError(f"Missing {direction} thresholds")
        if len(thresholds[direction]) != 501:  # 500 bins + 1
            print(f"Warning: {direction} thresholds has {len(thresholds[direction])} values, expected 501")

    # Initialize PDGrapherNoGNN (no edge_index needed)
    model = PDGrapherNoGNN(
        num_nodes=dataset.get_num_vars(),  # Pass number of nodes instead of edge_index
        model_kwargs={
            "n_layers_nn": const.LAYERS, 
            "n_layers_gnn": const.LAYERS,  # Now represents dense layers
            "positional_features_dims": 64,
            "embedding_layer_dim": 64,
            "dim_gnn": 64,
            "num_vars": dataset.get_num_vars()
        }
    )
    
    # Set optimizers (same as PDGrapher)
    model.set_optimizers_and_schedulers([
        optim.Adam(model.response_prediction.parameters(), lr=0.0075),
        optim.Adam(model.perturbation_discovery.parameters(), lr=0.0033)
    ])
    
    # Use CUDA if available
    if torch.cuda.is_available():
        accelerator = "cuda"
    else:
        accelerator = "cpu"

    # Initialize trainer
    trainer = Trainer(
        fabric_kwargs={"accelerator": accelerator, "devices": 1},
        log=True, logging_name=f"{const.EXPERIMENT}_tuned_nognn",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )

    # Train the model
    model_performance = trainer.train(model, dataset, n_epochs=const.EPOCHS)

    print(model_performance)
    
    # Save training results
    results_dir = Path("examples/PDGrapher")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    with open(results_dir / f"{const.EXPERIMENT}_multifold_final_nognn.txt", "w") as f:
        f.write(str(model_performance))

    # Save the best perturbation discovery model
    save_best_perturbation_model(model, trainer)


def save_best_perturbation_model(pdgrapher_model, trainer):
    """
    Save the best perturbation discovery model for PDGrapherNoGNN.
    """
    timestamp = utils.get_current_time()
    
    model_save_dir = Path(const.MODEL_PATH) / const.MODEL
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"{const.MODEL_NAME}_{timestamp}.pt"
    model_save_path = model_save_dir / model_filename
    
    perturbation_model = pdgrapher_model.perturbation_discovery
    
    save_dict = {
        'model_state_dict': perturbation_model.state_dict(),
        'num_nodes': const.N_NODES,  # Store num_nodes instead of edge_index
        'model_config': {
            'n_layers_nn': const.LAYERS,
            'n_layers_gnn': const.LAYERS,
            'positional_features_dims': 64,
            'embedding_layer_dim': 64,
            'dim_gnn': 64,
            'num_vars': perturbation_model.num_nodes,
            'n_nodes': const.N_NODES
        },
        'training_config': {
            'experiment': const.EXPERIMENT,
            'epochs': const.EPOCHS,
            'learning_rate_pd': 0.0033,
            'learning_rate_rp': 0.0075,
            'use_supervision': True,
            'supervision_multiplier': 0.05
        },
        'training_performance': trainer.best_performance if hasattr(trainer, 'best_performance') else None,
        'timestamp': timestamp,
        'model_type': 'perturbation_discovery_nognn'
    }

    torch.save(save_dict, model_save_path)
    print(f"Best perturbation discovery model (NoGNN) saved to {model_save_path}")
    
    # Save latest version
    latest_path = model_save_dir / f"{const.MODEL_NAME}_latest.pt"
    torch.save(save_dict, latest_path)
    print(f"Latest model path: {latest_path}")
    
    return model_save_path


if __name__ == "__main__":
    main()
