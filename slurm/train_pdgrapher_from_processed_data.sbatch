#!/bin/bash

# PDGrapher Training from Downloaded Processed Data
# This script handles the complete pipeline for training PDGrapher when you have processed data but missing thresholds
# Usage: sbatch slurm/train_pdgrapher_from_processed_data.sbatch EXPERIMENT_NAME

#SBATCH --job-name="pdg_complete_pipeline"
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# Get experiment name from command line argument
EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: Experiment name must be provided as argument"
    echo "Usage: sbatch slurm/train_pdgrapher_from_processed_data.sbatch EXPERIMENT_NAME"
    exit 1
fi

# Create output and timing directories
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Change to project directory
cd $BASE_DIR

# Activate virtual environment
source .venv/bin/activate

echo "=== PDGRAPHER COMPLETE PIPELINE FROM PROCESSED DATA ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Base directory: $BASE_DIR"

# Check initial GPU memory
echo ""
echo "🔍 Initial GPU Memory Status:"
python gpu_monitor.py --once
echo ""

# Set environment variables
export EXPERIMENT_NAME=$EXPERIMENT
export MODEL=pdgrapher
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize timing log
TIMING_LOG_FILE="$TIMING_LOG_DIR/${EXPERIMENT}_pdgrapher_complete_timing.csv"
echo "stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_FILE

# Define processed data directory using the same system as the training script
# This ensures consistency with the constants system
EXPERIMENT_CONFIG="configs/experiments/${EXPERIMENT}.yaml"

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config not found: $EXPERIMENT_CONFIG"
    exit 1
fi

# # Calculate the correct path using the same config merging as constants system (avoiding import issues)
# PROCESSED_DIR=$(python3 -c "
# import yaml
# import json
# import hashlib
# from pathlib import Path

# def deep_merge(base, override):
#     result = base.copy()
#     for key, value in override.items():
#         if key in result and isinstance(result[key], dict) and isinstance(value, dict):
#             result[key] = deep_merge(result[key], value)
#         else:
#             result[key] = value
#     return result

# def get_data_fingerprint(data_creation_params, network, seed):
#     fingerprint_data = {
#         'data_creation': data_creation_params,
#         'network': network,
#         'seed': seed
#     }
#     canonical_json = json.dumps(fingerprint_data, sort_keys=True, separators=(',', ':'))
#     hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
#     return hash_object.hexdigest()[:12]

# def get_graph_perturbation_fingerprint(graph_perturbation_params):
#     canonical_json = json.dumps(graph_perturbation_params, sort_keys=True, separators=(',', ':'))
#     hash_object = hashlib.sha256(canonical_json.encode('utf-8'))
#     return hash_object.hexdigest()[:12]

# # Load configs like constants system
# config_dir = Path('configs')

# # 1. Load shared config
# shared_config = yaml.safe_load(open(config_dir / 'shared.yaml'))

# # 2. Load model config
# model_config = yaml.safe_load(open(config_dir / 'models' / 'pdgrapher.yaml'))
# config = deep_merge(shared_config, model_config)

# # 3. Load experiment config
# exp_config = yaml.safe_load(open(config_dir / 'experiments' / 'og_pdgrapher_data.yaml'))
# exp_config['experiment'] = 'og_pdgrapher_data'
# config = deep_merge(config, exp_config)

# # Get parameters from merged config
# data_creation = config['data_creation']
# graph_perturbation = config['graph_perturbation']
# network = data_creation['network']
# seed = config.get('seed', 42)
# experiment_name = 'og_pdgrapher_data'

# # Calculate hashes
# shared_hash = get_data_fingerprint(data_creation, network, seed)
# experiment_hash = get_graph_perturbation_fingerprint(graph_perturbation)

# # Print the correct experiment data path
# print(f'data/experiments/{shared_hash}/{experiment_hash}/{experiment_name}/processed/pdgrapher')
# ")

# SPLITS_PATH="$PROCESSED_DIR/splits.pt"

# echo "Using constants system paths:"
# echo "Processed data directory: $PROCESSED_DIR"
# echo "Splits file path: $SPLITS_PATH"

# # Verify required processed files exist
# REQUIRED_FILES=("data_forward.pt" "data_backward.pt" "edge_index.pt")
# for file in "${REQUIRED_FILES[@]}"; do
#     if [ ! -f "$PROCESSED_DIR/$file" ]; then
#         echo "ERROR: Required file not found: $PROCESSED_DIR/$file"
#         echo "Please ensure you have downloaded the processed data files:"
#         echo "  - data_forward.pt"
#         echo "  - data_backward.pt" 
#         echo "  - edge_index.pt"
#         echo ""
#         echo "Expected directory structure:"
#         echo "  $PROCESSED_DIR/"
#         echo "  ├── data_forward.pt"
#         echo "  ├── data_backward.pt"
#         echo "  ├── edge_index.pt"
#         echo "  └── thresholds.pt  (will be created)"
#         exit 1
#     fi
#     echo "✓ Found: $PROCESSED_DIR/$file"
# done

# =============================================================================
# Stage 1: Data Splitting (if missing)
# =============================================================================
echo ""
echo "=== STAGE 1: DATA SPLITTING ==="

START_TIME=$(date +%s)

if [ -f "$SPLITS_PATH" ]; then
    echo "Splits file already exists: $SPLITS_PATH"
    echo "Skipping data splitting..."
else
    echo "Splits file not found. Creating data splits..."
    echo "This will create train/validation/test splits from your processed data"
    
    # Create the splits directory if it doesn't exist
    mkdir -p "$(dirname "$SPLITS_PATH")"
    
    # Run data splitting for processed data
    python -m src.create_splits --experiment $EXPERIMENT --model pdgrapher
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Data splitting failed!"
        exit 1
    fi
    
    echo "✓ Data splits created successfully"
fi

END_TIME=$(date +%s)
echo "data_splitting,$START_TIME,$END_TIME,$((END_TIME - START_TIME)),$SLURM_JOB_ID" >> $TIMING_LOG_FILE

# =============================================================================
# Stage 2: Compute Thresholds (if missing)
# =============================================================================
echo ""
echo "=== STAGE 2: THRESHOLD COMPUTATION ==="
THRESHOLDS_FILE="$PROCESSED_DIR/thresholds.pt"

START_TIME=$(date +%s)

if [ -f "$THRESHOLDS_FILE" ]; then
    echo "Thresholds file already exists: $THRESHOLDS_FILE"
    echo "Skipping threshold computation..."
else
    echo "Thresholds file not found. Computing thresholds from processed data..."
    echo "This may take several minutes for large datasets..."
    
    # Set environment variables for compute_thresholds.py
    export EXPERIMENT_NAME=$EXPERIMENT
    export MODEL=pdgrapher

    python -m src.compute_thresholds --experiment $EXPERIMENT --model pdgrapher

    if [ $? -ne 0 ]; then
        echo "ERROR: Threshold computation failed!"
        exit 1
    fi
    
    echo "✓ Thresholds computed successfully"
fi

END_TIME=$(date +%s)
echo "threshold_computation,$START_TIME,$END_TIME,$((END_TIME - START_TIME)),$SLURM_JOB_ID" >> $TIMING_LOG_FILE

# =============================================================================
# Stage 3: PDGrapher Training
# =============================================================================
echo ""
echo "=== STAGE 3: PDGRAPHER TRAINING ==="

START_TIME=$(date +%s)

echo "📊 Pre-training GPU memory:"
python gpu_monitor.py --once
echo ""

echo "Starting PDGrapher training with computed thresholds..."

python -m src.train_pdgrapher --experiment $EXPERIMENT --model pdgrapher
training_exit_code=$?

echo ""
echo "📊 Post-training GPU memory:"
python gpu_monitor.py --once

if [ $training_exit_code -ne 0 ]; then
    echo "ERROR: PDGrapher training failed!"
    exit 1
fi

echo "✓ PDGrapher training completed successfully"

END_TIME=$(date +%s)
echo "training,$START_TIME,$END_TIME,$((END_TIME - START_TIME)),$SLURM_JOB_ID" >> $TIMING_LOG_FILE

# =============================================================================
# Stage 4: Model Validation
# =============================================================================
echo ""
echo "=== STAGE 4: MODEL VALIDATION ==="

START_TIME=$(date +%s)

echo "Starting PDGrapher validation..."
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

if [ $? -ne 0 ]; then
    echo "ERROR: PDGrapher validation failed!"
    exit 1
fi

echo "✓ PDGrapher validation completed successfully"

END_TIME=$(date +%s)
echo "validation,$START_TIME,$END_TIME,$((END_TIME - START_TIME)),$SLURM_JOB_ID" >> $TIMING_LOG_FILE

# =============================================================================
# Pipeline Complete
# =============================================================================
echo ""
echo "=== PIPELINE COMPLETE ==="
echo "Experiment: $EXPERIMENT"
echo "Completed at: $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Display timing summary
echo ""
echo "Timing Summary:"
echo "==============="
cat $TIMING_LOG_FILE

# Check if model and results were created
MODEL_FILE="$BASE_DIR/models/pdgrapher/${EXPERIMENT}_latest.pt"
RESULTS_DIR="$BASE_DIR/reports"

echo ""
echo "Output Files:"
echo "============="
if [ -f "$MODEL_FILE" ]; then
    echo "✓ Trained model: $MODEL_FILE"
else
    echo "⚠ Warning: Model file not found at expected location"
fi

if [ -d "$RESULTS_DIR" ]; then
    echo "✓ Results directory: $RESULTS_DIR"
    echo "Latest result files:"
    find "$RESULTS_DIR" -name "*${EXPERIMENT}*" -type f -printf "  %p\n" | head -5
else
    echo "⚠ Warning: Results directory not found"
fi

echo ""
echo "To check your results:"
echo "  ls -la $RESULTS_DIR/*${EXPERIMENT}*"
echo "  ls -la $MODEL_FILE"

echo ""
echo "🎉 PDGrapher pipeline completed successfully!"
echo "You can now analyze your results or run additional experiments."