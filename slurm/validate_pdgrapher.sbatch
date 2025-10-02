#!/bin/bash

# PDGrapher Validation Only
# Usage: sbatch slurm/validate_pdgrapher.sbatch EXPERIMENT_NAME

#SBATCH --job-name="pdg_validation"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=slurm_out/%x_%j.out
#SBATCH --error=slurm_out/%x_%j.err

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"

# Get experiment name from command line argument
EXPERIMENT=$1

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: Experiment name must be provided as argument"
    echo "Usage: sbatch slurm/validate_pdgrapher.sbatch EXPERIMENT_NAME"
    exit 1
fi

# Create output directory
mkdir -p $SLURM_OUT_DIR

# Change to project directory
cd $BASE_DIR

# Activate virtual environment
source .venv/bin/activate

echo "=== PDGRAPHER VALIDATION ONLY ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Experiment: $EXPERIMENT"
echo "Base directory: $BASE_DIR"

# Check initial GPU memory
echo ""
echo "🔍 Initial GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits || echo "GPU info not available"
echo ""

# Set environment variables
export EXPERIMENT_NAME=$EXPERIMENT
export MODEL=pdgrapher
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "🔬 Starting PDGrapher validation..."
echo "Debug: Loading Python environment..."
python -c "print('Python is working')"
echo "Debug: Testing imports..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
echo "Debug: Testing CUDA..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
echo "Debug: Starting validation script..."

START_TIME=$(date +%s)

# Run validation with verbose output
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

if [ $? -eq 0 ]; then
    echo "✅ Validation completed successfully!"
else
    echo "❌ Validation failed!"
    exit 1
fi

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

echo ""
echo "📊 Final GPU Memory Status:"
nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv,noheader,nounits || echo "GPU info not available"

echo ""
echo "=== VALIDATION SUMMARY ==="
echo "Duration: ${DURATION} seconds"
echo "End time: $(date)"
echo "✅ Validation job completed"
