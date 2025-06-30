#!/bin/bash
#SBATCH --job-name="validation"
#SBATCH --time=01:00:00         
#SBATCH --partition=gpu             
#SBATCH --gres=gpu:1
#SBATCH --mem=16G                    
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=/sc/home/johanna.dahlkemper/source_detection_in_grns/
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --verbose
#SBATCH --output=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/validation_%j.out
#SBATCH --error=/sc/home/johanna.dahlkemper/source_detection_in_grns/slurm_out/validation_%j.err

# Activate virtual environment
source .venv/bin/activate

dvc repro validation
dvc repro visualization