import torch.optim as optim
import torch
import src.constants as const
import src.utils as utils
from pathlib import Path
import os

from pdgrapher import Dataset, PDGrapher, Trainer

torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """

    data_processed_dir = Path(const.DATA_PATH) / "processed" / const.MODEL
    splits_path = const.SPLITS_PATH

    # Check if all required files exist
    required_files = [
        splits_path,
        f"{data_processed_dir}/data_forward.pt",
        f"{data_processed_dir}/data_backward.pt",
        f"{data_processed_dir}/edge_index.pt",
        f"{data_processed_dir}/thresholds.pt"
    ]
    
    for file_path in required_files:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file not found: {file_path}. Please run data processing first.")

    dataset = Dataset(
        forward_path=f"{data_processed_dir}/data_forward.pt",
        backward_path=f"{data_processed_dir}/data_backward.pt",
        splits_path=const.SPLITS_PATH,
    )

    # Load thresholds (required by PDGrapher for discretizing expression values)
    thresholds_path = data_processed_dir / "thresholds.pt"
    thresholds = torch.load(thresholds_path, weights_only=False)
    print(f"Loaded thresholds: {list(thresholds.keys())}")
    
    # Validate thresholds have the expected structure
    for direction in ['forward', 'backward']:
        if direction not in thresholds:
            raise ValueError(f"Missing {direction} thresholds")
        if len(thresholds[direction]) != 501:  # 500 bins + 1
            print(f"Warning: {direction} thresholds has {len(thresholds[direction])} values, expected 501")

    edge_index = torch.load(f"{data_processed_dir}/edge_index.pt", weights_only=False)

    
    #TODO: CHECK THOSE VALUES:
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": const.LAYERS, 
        "n_layers_gnn": const.LAYERS, 
        "positional_features_dim": 64,  # Should match expression discretization bins (500)
        "embedding_layer_dim": 64,      # Increased from 8 for better representation
        "dim_gnn": 64,                  # Increased from 8 for better capacity
        "num_vars": dataset.get_num_vars()
    })
    
    model.set_optimizers_and_schedulers([
        optim.Adam(model.response_prediction.parameters(), lr=0.0075),
        optim.Adam(model.perturbation_discovery.parameters(), lr=0.0033)
    ])
    
    # If cuda is available, use it, otherwise set accelerator to cpu
    if torch.cuda.is_available():
        accelerator = "cuda"
    else:
        accelerator = "cpu"

    trainer = Trainer(
        fabric_kwargs={"accelerator": accelerator, "devices": 1},
        log=True, logging_name=f"{const.EXPERIMENT}_tuned",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )

    # Iterate over all of the folds and train on each one
    model_performance = trainer.train(model, dataset, n_epochs=const.EPOCHS)

    print(model_performance)
    
    # Save training results
    with open(f"examples/PDGrapher/{const.EXPERIMENT}_multifold_final.txt", "w") as f:
        f.write(str(model_performance))

    # Save the best perturbation discovery model
    save_best_perturbation_model(model, trainer, edge_index)


def save_best_perturbation_model(pdgrapher_model, trainer, edge_index):
    """
    Save the best perturbation discovery model with complete configuration.
    
    Args:
        pdgrapher_model: The trained PDGrapher model
        trainer: The PDGrapher trainer (may contain best model state)
        edge_index: The graph edge index tensor
    """
    timestamp = utils.get_current_time()
    
    model_save_dir = Path(const.MODEL_PATH) / const.MODEL
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"{const.MODEL_NAME}_{timestamp}.pt"
    model_save_path = model_save_dir / model_filename
    
    perturbation_model = pdgrapher_model.perturbation_discovery
    
    save_dict = {
        'model_state_dict': perturbation_model.state_dict(),
        'edge_index': edge_index,
        'model_config': {
            'n_layers_nn': const.LAYERS,
            'n_layers_gnn': const.LAYERS,
            'positional_features_dim': 64,
            'embedding_layer_dim': 64,
            'dim_gnn': 64,
            'num_vars': perturbation_model.num_vars if hasattr(perturbation_model, 'num_vars') else None,
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
        'model_type': 'perturbation_discovery'
    }

    torch.save(save_dict, model_save_path)
    print(f"Best perturbation discovery model saved to {model_save_path}")
    latest_path = model_save_dir / f"{const.MODEL_NAME}_latest.pt"
    print(f"Latest model path: {latest_path}")
    torch.save(save_dict, latest_path)
    
    return model_save_path


if __name__ == "__main__":
    main()