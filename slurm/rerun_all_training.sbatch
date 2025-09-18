#!/bin/bash

# Rerun training for ALL experiments
# Usage: bash slurm/rerun_all_training.sbatch

CONFIG_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns/configs/experiments"

echo "=== RERUNNING TRAINING FOR ALL EXPERIMENTS ==="

# Discover all experiments
for config_file in $(find $CONFIG_DIR -maxdepth 1 -name "*.yaml" -type f | sort); do
    experiment_name=$(basename "$config_file" .yaml)
    echo "Submitting retraining pipeline for: $experiment_name"
    
    # Submit retraining pipeline in background
    EXPERIMENT_NAME=$experiment_name bash slurm/rerun_training_pipeline.sbatch &
    
    # Small delay to avoid overwhelming the scheduler
    sleep 2
done

# Wait for all background processes
wait

echo "All retraining pipelines submitted!"
echo "Monitor with: squeue -u \$USER"