#!/bin/bash

# PDGrapher Full Pipeline Runner
# This script submits multiple SLURM jobs with dependencies to run the complete PDGrapher pipeline

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"

# Create output directory if it doesn't exist
mkdir -p $SLURM_OUT_DIR

echo "Starting PDGrapher pipeline..."
echo "Base directory: $BASE_DIR"

# Job 1: Data Creation (CPU-intensive, simulation-heavy)
echo "Submitting PDGrapher data creation job..."
DATA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="pdgrapher_data"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pdgrapher_data_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pdgrapher_data_%j.err

source .venv/bin/activate

# Force JAX to use CPU for data creation (more stable)
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

echo "=== STARTING PDGRAPHER DATA CREATION STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "JAX_PLATFORMS=\$JAX_PLATFORMS"

python -m src.data_creation_pdgrapher
python -m src.data_processing_pdgrapher

echo "PDGrapher data creation completed at: \$(date)"
EOF
)

echo "PDGrapher data creation job submitted with ID: $DATA_JOB_ID"

# Job 2: Training (GPU-intensive, depends on data creation)
echo "Submitting PDGrapher training job..."
TRAINING_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgrapher_training"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pdgrapher_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pdgrapher_training_%j.err

source .venv/bin/activate

# Try GPU first, fallback to CPU if CUDA issues
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "=== STARTING PDGRAPHER TRAINING STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Depends on data job: $DATA_JOB_ID"

# Test JAX backend
python -c "import jax; print(f'JAX backend: {jax.default_backend()}'); print(f'JAX devices: {jax.devices()}')" || {
    echo "GPU initialization failed, falling back to CPU..."
    export JAX_PLATFORMS=cpu
}

python -m src.train_pdgrapher

echo "PDGrapher training completed at: \$(date)"
EOF
)

echo "PDGrapher training job submitted with ID: $TRAINING_JOB_ID"

# Job 3: Validation (GPU for inference, depends on training)
echo "Submitting PDGrapher validation job..."
VALIDATION_JOB_ID=$(sbatch --parsable --dependency=afterok:$TRAINING_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgrapher_validation"
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pdgrapher_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pdgrapher_validation_%j.err

source .venv/bin/activate

# Try GPU first, fallback to CPU if CUDA issues
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8

echo "=== STARTING PDGRAPHER VALIDATION STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Depends on training job: $TRAINING_JOB_ID"

# Test JAX backend
python -c "import jax; print(f'JAX backend: {jax.default_backend()}'); print(f'JAX devices: {jax.devices()}')" || {
    echo "GPU initialization failed, falling back to CPU..."
    export JAX_PLATFORMS=cpu
}

python -m src.pdgrapher_perturbation_validation

echo "PDGrapher validation completed at: \$(date)"
echo "=== PDGRAPHER PIPELINE COMPLETED ==="
EOF
)

echo "PDGrapher validation job submitted with ID: $VALIDATION_JOB_ID"

# Summary
echo ""
echo "=== PDGRAPHER PIPELINE SUBMISSION SUMMARY ==="
echo "Data creation job:  $DATA_JOB_ID"
echo "Training job:       $TRAINING_JOB_ID (depends on $DATA_JOB_ID)"
echo "Validation job:     $VALIDATION_JOB_ID (depends on $TRAINING_JOB_ID)"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $DATA_JOB_ID,$TRAINING_JOB_ID,$VALIDATION_JOB_ID"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel $DATA_JOB_ID $TRAINING_JOB_ID $VALIDATION_JOB_ID"
echo ""
echo "Check logs in: $SLURM_OUT_DIR"
echo ""
