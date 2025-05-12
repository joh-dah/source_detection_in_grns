""" Creates a data set of graphs with modeled signal propagation for training and validation."""
import argparse
from pathlib import Path
import shutil
import numpy as np
import src.constants as const
import grins.racipe_run as rr
import pandas as pd



def run_racipe_simulation(network_name, dest_dir, sample_count):
    """
    """
    topo_file = f"{const.TOPO_PATH}/{network_name}.topo"
    
    shutil.rmtree(dest_dir, ignore_errors=True)

    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    # racipe will create a directory named network_name inside of dest_dir

    num_replicates = 1
    num_params = int(np.sqrt(sample_count))
    num_init_conds = int(np.sqrt(sample_count))

    #TODO parallize?
    #TODO: make deterministic with seed

    print("Generating parameter files")

    rr.gen_topo_param_files(
        topo_file,
        dest_dir,
        num_replicates,
        num_params,
        num_init_conds,
        sampling_method="Sobol",
    )

    print("Run steady-state Racipe simulations")

    rr.run_all_replicates(
        topo_file,
        dest_dir,
        batch_size=4000,
    )


def cleanup_racipe_output(network_name, dest_dir):
    """
    Racipe creates multiple files and directorys. 
    This method is for cleanup and removes all unneeded files/directorys.
    """
    solution_file = dest_dir / network_name / "001" / 'trrust_mock_steadystate_solutions_001.parquet'
    shutil.copyfile(solution_file, dest_dir / "steady_states.parquet")
    shutil.rmtree(dest_dir / network_name)

def main():
    """
    Creates a data set of graphs with modeled signal propagation for training and validation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--validation",
        action="store_true",
        help="whether to create validation or training data",
    )
    parser.add_argument(
        "--network", 
        type=str, 
        help="name of the network that should be used"
    )
    args = parser.parse_args()

    print(args.network)

    train_or_val = "validation" if args.validation else "training"
    dest_dir = Path(const.DATA_PATH) / train_or_val / "raw" / "steady_states" / args.network
    print(dest_dir)

    sample_count = const.VALIDATION_SIZE if args.validation else const.TRAINING_SIZE


    print(f"{args.network} {train_or_val} Data:")

    run_racipe_simulation(args.network, dest_dir, sample_count)
    cleanup_racipe_output(args.network, dest_dir)




if __name__ == "__main__":
    main()
