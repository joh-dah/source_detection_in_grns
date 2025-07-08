import torch.optim as optim
import torch
import src.constants as const
from pathlib import Path

from pdgrapher import Dataset, PDGrapher, Trainer

import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """

    data_processed_dir = Path(const.DATA_PATH) / const.MODEL / "processed"

    #check if files exist
    if not os.path.exists(f"{data_processed_dir}/splits/splits.pt"):
        raise FileNotFoundError("Splits file not found. Please run 'src/create_splits.py' first.")

    dataset = Dataset(
        forward_path=f"{data_processed_dir}/data_forward.pt",
        backward_path=f"{data_processed_dir}/data_backward.pt",
        splits_path=f"{data_processed_dir}/splits/splits.pt"
    )

    # debug print the keys in splits
    print(dataset.splits.keys())  


    edge_index = torch.load(f"{data_processed_dir}/edge_index.pt", weights_only=False)
    model = PDGrapher(edge_index, model_kwargs={
        "n_layers_nn": 2, "n_layers_gnn": 2, "positional_features_dim": 64, "embedding_layer_dim": 8,
        "dim_gnn": 8, "num_vars": dataset.get_num_vars()
        })
    model.set_optimizers_and_schedulers([optim.Adam(model.response_prediction.parameters(), lr=0.0075),
        optim.Adam(model.perturbation_discovery.parameters(), lr=0.0033)])
    

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda", "devices": 1},
        log=True, logging_name="tuned",
        use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.05,
        log_train=False, log_test=True
    )

    # Iterate over all of the folds and train on each one
    if const.N_FOLDS == 1:
        model_performance = trainer.train(model, dataset, n_epochs = 2)
    else:
        model_performanc = trainer.train_kfold(model, dataset, n_epochs = 1)

    print(model_performance)
    with open(f"examples/PDGrapher/multifold_final.txt", "w") as f:
        f.write(str(model_performance))


if __name__ == "__main__":
    main()