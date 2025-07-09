#!/bin/bash
#SBATCH --job-name="pdgrapher_test"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/sc/home/johanna.dahlkemper/source_detection_in_grns/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/pdgrapher_%j.out
#SBATCH --error=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/pdgrapher_%j.err

# Activate virtual environment
source .venv/bin/activate

python -m src.data_creation_pdgrapher
python -m src.data_processing_pdgrapher
python -m src.train_pdgrapher
python -m src.pdgrapher_perturbation_validation