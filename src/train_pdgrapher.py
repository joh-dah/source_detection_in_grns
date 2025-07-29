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

    # Check if files exist
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Splits file not found at {splits_path}. Please run 'src/create_splits.py' first.")

    dataset = Dataset(
        forward_path=f"{data_processed_dir}/data_forward.pt",
        backward_path=f"{data_processed_dir}/data_backward.pt",
        splits_path=const.SPLITS_PATH,
    )

    edge_index = torch.load(f"{data_processed_dir}/edge_index.pt", weights_only=False)
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": const.LAYERS, 
        "n_layers_gnn": const.LAYERS, 
        "positional_features_dim": 64, 
        "embedding_layer_dim": 8,
        "dim_gnn": 8, 
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
        log=True, logging_name="tuned",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )

    # Iterate over all of the folds and train on each one
    model_performance = trainer.train(model, dataset, n_epochs=const.EPOCHS)

    print(model_performance)
    
    # Save training results
    with open(f"examples/PDGrapher/multifold_final.txt", "w") as f:
        f.write(str(model_performance))

    # Save the best perturbation discovery model
    save_best_perturbation_model(model, trainer)


def save_best_perturbation_model(pdgrapher_model, trainer):
    """
    Save the best perturbation discovery model with timestamp.
    
    Args:
        pdgrapher_model: The trained PDGrapher model
        trainer: The PDGrapher trainer (may contain best model state)
    """
    timestamp = utils.get_current_time()
    
    model_save_dir = Path(const.MODEL_PATH) / const.MODEL
    model_save_dir.mkdir(parents=True, exist_ok=True)
    
    model_filename = f"{const.MODEL_NAME}_{timestamp}.pt"
    model_save_path = model_save_dir / model_filename
    
    perturbation_model = pdgrapher_model.perturbation_discovery
    
    save_dict = {
        'model_state_dict': perturbation_model.state_dict(),
        'model_config': {
            'n_layers_nn': const.LAYERS,
            'n_layers_gnn': const.LAYERS,
            'positional_features_dim': 64,
            'embedding_layer_dim': 8,
            'dim_gnn': 8,
            'num_vars': pdgrapher_model.perturbation_discovery.num_vars if hasattr(pdgrapher_model.perturbation_discovery, 'num_vars') else None
        },
        'training_performance': trainer.best_performance if hasattr(trainer, 'best_performance') else None,
        'timestamp': timestamp,
        'model_type': 'perturbation_discovery',
        'epochs': const.EPOCHS
    }

    torch.save(save_dict, model_save_path)
    latest_path = model_save_dir / f"{const.MODEL_NAME}_latest.pt"
    torch.save(save_dict, latest_path)
    
    return model_save_path


if __name__ == "__main__":
    main()