#!/bin/bash
#SBATCH --job-name="pdgrapher_test"
#SBATCH --exclude="ga01,ga02"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=/sc/home/johanna.dahlkemper/source_detection_in_grns/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/pdgrapher_%j.out
#SBATCH --error=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/pdgrapher_%j.err

# Print node and environment info for debugging
echo "=== Job Environment Info ==="
echo "Running on node: $(hostname)"
echo "Architecture: $(uname -m)"
echo "Date: $(date)"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Python version: $(python3 --version)"
echo "==============================="

# Activate virtual environment
source .venv/bin/activate

python -m src.data_creation_pdgrapher --network=tiny
python -m src.data_processing_pdgrapher
python -m src.test_pdgrapher_5fold