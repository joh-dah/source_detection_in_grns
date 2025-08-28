#!/bin/bash

# Shared Data Creation Script
# Creates ONLY data creation and splits that can be shared across multiple experiments
# Data processing is experiment-specific and will be done in the experiment runner

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"

# Default experiment name for data configuration (override with EXPERIMENT_NAME)
EXPERIMENT=${EXPERIMENT_NAME:-shared_baseline}

echo "Creating shared data for experiment configuration: $EXPERIMENT"
echo "This data will be reused by all experiments with the same data_creation parameters"

# Create output directory if it doesn't exist
mkdir -p $SLURM_OUT_DIR

# Submit the shared data creation job
SHARED_DATA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="shared_data_creation"
#SBATCH --mem=256G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/shared_data_creation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/shared_data_creation_%j.err

source .venv/bin/activate

echo "=== STARTING SHARED DATA CREATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Configuration: $EXPERIMENT"
echo "This data will be shared across multiple experiments"

export EXPERIMENT_NAME=$EXPERIMENT

# === 1. Data Creation ===
echo "--- Step 1: Data Creation ---"
START_TIME_DATA=\$(date +%s)

python -m src.data_creation --experiment $EXPERIMENT --model gat

END_TIME_DATA=\$(date +%s)
DATA_DURATION=\$((\$END_TIME_DATA - \$START_TIME_DATA))
echo "Data creation completed in \$DATA_DURATION seconds"

# === 2. Data Splitting ===
echo "--- Step 2: Data Splitting ---"
START_TIME_SPLIT=\$(date +%s)

python -m src.create_splits --experiment $EXPERIMENT --model gat

END_TIME_SPLIT=\$(date +%s)
SPLIT_DURATION=\$((\$END_TIME_SPLIT - \$START_TIME_SPLIT))
echo "Data splitting completed in \$SPLIT_DURATION seconds"

# === Summary ===
TOTAL_DURATION=\$((\$END_TIME_SPLIT - \$START_TIME_DATA))
echo ""
echo "=== SHARED DATA CREATION COMPLETE ==="
echo "Configuration: $EXPERIMENT"
echo "Total time: \$TOTAL_DURATION seconds"
echo "Breakdown:"
echo "  - Data Creation: \$DATA_DURATION seconds"
echo "  - Data Splitting: \$SPLIT_DURATION seconds"
echo ""
echo "This shared data is now ready for parallel experiment execution!"
echo "Experiments will handle their own graph perturbation and data processing."

EOF
)

echo ""
echo "=== SHARED DATA CREATION SUBMITTED ==="
echo "Job ID: $SHARED_DATA_JOB_ID"
echo "Configuration: $EXPERIMENT"
echo ""
echo "Monitor with:"
echo "  squeue -j $SHARED_DATA_JOB_ID"
echo "  tail -f $SLURM_OUT_DIR/shared_data_creation_*.out"
echo ""
echo "Once complete, you can run multiple experiments in parallel using:"
echo "  EXPERIMENT_NAME=exp1 SHARED_DATA_JOB_ID=$SHARED_DATA_JOB_ID sbatch slurm/experiment_runner.sbatch"
echo "  EXPERIMENT_NAME=exp2 SHARED_DATA_JOB_ID=$SHARED_DATA_JOB_ID sbatch slurm/experiment_runner.sbatch"
echo "  ..."
echo ""
echo "Cancel if needed:"
echo "  scancel $SHARED_DATA_JOB_ID"