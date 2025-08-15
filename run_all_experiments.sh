#!/bin/bash

# Script to run all experiments in configs/experiments/
# Usage: ./run_all_experiments.sh

#one liner:
#for yaml_file in configs/experiments/*.yaml; do experiment_name=$(basename "$yaml_file" .yaml); echo "Starting $experiment_name"; EXPERIMENT_NAME="$experiment_name" srun slurm/dual_model_pipeline.sbatch; sleep 2; done
# for yaml_file in configs/experiments/*.yaml; do experiment_name=$(basename "$yaml_file" .yaml); echo "Starting $experiment_name"; EXPERIMENT_NAME="$experiment_name" srun slurm/dual_model_training_pipeline.sbatch; sleep 2; done


EXPERIMENTS_DIR="configs/experiments"
SLURM_SCRIPT="slurm/dual_model_pipeline.sbatch"

echo "Starting batch runs for all experiments..."
echo "Experiments directory: $EXPERIMENTS_DIR"
echo "SLURM script: $SLURM_SCRIPT"
echo ""

# Check if experiments directory exists
if [ ! -d "$EXPERIMENTS_DIR" ]; then
    echo "Error: Experiments directory $EXPERIMENTS_DIR not found!"
    exit 1
fi

# Check if SLURM script exists
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "Error: SLURM script $SLURM_SCRIPT not found!"
    exit 1
fi

# Counter for submitted jobs
job_count=0

# Loop through all .yaml files in the experiments directory
for yaml_file in "$EXPERIMENTS_DIR"/*.yaml; do
    # Check if any .yaml files exist
    if [ ! -f "$yaml_file" ]; then
        echo "No .yaml files found in $EXPERIMENTS_DIR"
        exit 1
    fi
    
    # Extract experiment name (filename without path and extension)
    experiment_name=$(basename "$yaml_file" .yaml)
    
    echo "Submitting job for experiment: $experiment_name"
    
    # Submit the job
    EXPERIMENT_NAME="$experiment_name" srun "$SLURM_SCRIPT"
    
    # Increment counter
    ((job_count++))
    
    # Optional: Add a small delay between submissions to avoid overwhelming the scheduler
    sleep 2
done

echo ""
echo "Batch submission complete!"
echo "Total jobs submitted: $job_count"
echo ""
echo "You can monitor your jobs with: squeue -u $USER"
