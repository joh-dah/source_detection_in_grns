#!/bin/bash

# Full DVC Pipeline Runner
# This script submits multiple SLURM jobs with dependencies to run the complete pipeline

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"

# Create output directory if it doesn't exist
mkdir -p $SLURM_OUT_DIR

echo "Starting full DVC pipeline..."
echo "Base directory: $BASE_DIR"

# Job 1: Data Creation (CPU-intensive, long-running)
echo "Submitting data creation job..."
DATA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="pipeline_data"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pipeline_data_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pipeline_data_%j.err

eval "\$(conda shell.bash hook)"
conda activate grn-env

echo "=== STARTING DATA CREATION STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"

dvc repro generate-training-data
dvc repro generate-validation-data
dvc repro generate-test-data
dvc repro process-training-data
dvc repro process-validation-data
dvc repro process-test-data

echo "Data creation completed at: \$(date)"
EOF
)

echo "Data creation job submitted with ID: $DATA_JOB_ID"

# Job 2: Training (GPU-intensive, depends on data creation)
echo "Submitting training job..."
TRAINING_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pipeline_training"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pipeline_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pipeline_training_%j.err

eval "\$(conda shell.bash hook)"
conda activate grn-env

echo "=== STARTING TRAINING STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Depends on data job: $DATA_JOB_ID"

dvc repro training

echo "Training completed at: \$(date)"
EOF
)

echo "Training job submitted with ID: $TRAINING_JOB_ID"

# Job 3: Validation (GPU for inference, depends on training)
echo "Submitting validation job..."
VALIDATION_JOB_ID=$(sbatch --parsable --dependency=afterok:$TRAINING_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pipeline_validation"
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/pipeline_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/pipeline_validation_%j.err

eval "\$(conda shell.bash hook)"
conda activate grn-env

echo "=== STARTING VALIDATION STAGE ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Depends on training job: $TRAINING_JOB_ID"

dvc repro validation
dvc repro visualization

echo "Validation and visualization completed at: \$(date)"
echo "=== FULL PIPELINE COMPLETED ==="
EOF
)

echo "Validation job submitted with ID: $VALIDATION_JOB_ID"

# Summary
echo ""
echo "=== PIPELINE SUBMISSION SUMMARY ==="
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
