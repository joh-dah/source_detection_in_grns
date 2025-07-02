import numpy as np
import torch

from pdgrapher import Dataset, PDGrapher, Trainer

import os
torch.set_num_threads(5)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def main():
    """ torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0) """

    dataset = Dataset(
        forward_path="data/processed/data_forward.pt",
        backward_path="data/processed/data_backward.pt",
        splits_path="data/processed/splits/random/5fold/splits.pt"
    )

    edge_index = torch.load("data/processed/edge_index.pt", weights_only=False)
    model = PDGrapher(edge_index, model_kwargs={"n_layers_nn": 1, "n_layers_gnn": 1, "num_vars": dataset.get_num_vars()})

    trainer = Trainer(
        fabric_kwargs={"accelerator": "cuda"},
        log=True, use_forward_data=True, use_backward_data=True, use_supervision=True,
        use_intervention_data=True, supervision_multiplier=0.01,
        log_train=False, log_test=True
    )

    # Iterate over all of the folds and train on each one
    model_performances = trainer.train_kfold(model, dataset, n_epochs = 1)

    print(model_performances)
    with open(f"examples/PDGrapher/multifold_final.txt", "w") as f:
        f.write(str(model_performances))


if __name__ == "__main__":
    main()