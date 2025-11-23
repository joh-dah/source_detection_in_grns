#!/bin/bash

# PDGrapher Training from Downloaded Processed Data
# This script handles the complete pipeline for training PDGrapher when you have processed data but missing thresholds
# Usage: sbatch slurm/train_pdgrapher_from_processed_data.sbatch EXPERIMENT_NAME

#SBATCH --job-name="pdg_complete_pipeline"
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
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
    echo "Usage: sbatch slurm/train_pdgraphernognn_from_processed_data.sbatch EXPERIMENT_NAME"
    exit 1
fi

# Create output and timing directories
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Change to project directory
cd $BASE_DIR

# Activate virtual environment
source .venv/bin/activate

echo "=== PDGRAPHERNOGNN COMPLETE PIPELINE FROM PROCESSED DATA ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Base directory: $BASE_DIR"

# Set environment variables
export EXPERIMENT_NAME=$EXPERIMENT
export MODEL=pdgraphernognn
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Initialize timing log
TIMING_LOG_FILE="$TIMING_LOG_DIR/${EXPERIMENT}_pdgraphernognn_complete_timing.csv"
echo "stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_FILE

# Define processed data directory using the same system as the training script
# This ensures consistency with the constants system
EXPERIMENT_CONFIG="configs/experiments/${EXPERIMENT}.yaml"

if [ ! -f "$EXPERIMENT_CONFIG" ]; then
    echo "ERROR: Experiment config not found: $EXPERIMENT_CONFIG"
    exit 1
fi


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
# Stage 3: PDGrapherNoGNN Training
# =============================================================================
echo ""
echo "=== STAGE 3: PDGRAPHERNOGNN TRAINING ==="

START_TIME=$(date +%s)

echo "Starting PDGrapherNoGNN training with computed thresholds..."
python -m src.train_pdgrapher_nognn --experiment $EXPERIMENT --model pdgrapher_nognn
training_exit_code=$?

if [ $training_exit_code -ne 0 ]; then
    echo "ERROR: PDGrapherNoGNN training failed!"
    exit 1
fi

echo "✓ PDGrapherNoGNN training completed successfully"
END_TIME=$(date +%s)
echo "training,$START_TIME,$END_TIME,$((END_TIME - START_TIME)),$SLURM_JOB_ID" >> $TIMING_LOG_FILE

# =============================================================================
# Stage 4: Model Validation
# =============================================================================
echo ""
echo "=== STAGE 4: MODEL VALIDATION ==="

START_TIME=$(date +%s)

echo "Starting PDGrapher validation..."
python -m src.validation --experiment $EXPERIMENT --model pdgrapher_nognn

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