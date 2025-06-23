#!/bin/bash
#SBATCH --job-name="data_creation_tp53"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=/sc/home/johanna.dahlkemper/source_detection_in_grns/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/data_creation_%j.out
#SBATCH --error=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/data_creation_%j.err

eval "$(conda shell.bash hook)"
conda activate grn-env

dvc repro  generate-training-data
dvc repro  generate-validation-data
dvc repro  process-training-data
dvc repro  process-validation-data
