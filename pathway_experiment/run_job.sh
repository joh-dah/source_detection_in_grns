#!/bin/bash
#SBATCH --job-name=pw_data_creation
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pw_data_creation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pw_data_creation_%j.err

source .venv/bin/activate

python -m pathway_experiment.5_raw_data_creation
