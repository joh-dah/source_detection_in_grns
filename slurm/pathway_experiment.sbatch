#!/bin/bash

# Dual Model Single Experiment Runner Script
# Runs PDGrapher and PDGrapherNoGNN for a single experiment
# Usage: EXPERIMENT_NAME=pathway bash slurm/pathway_experiment.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx25,gx27,gx28,gx29"

# Get experiment name and data fingerprint
EXPERIMENT="${EXPERIMENT_NAME}"
FINGERPRINT=656ca5f25b8e

# K-Fold configuration
K_FOLDS=${K_FOLDS:-5}  # Default to 5 folds
echo "K-Fold configuration: $K_FOLDS folds"

# TRAIN_ONLY mode: if TRAIN_ONLY=1, skip shared data / perturbation / processing / baseline
TRAIN_ONLY=${TRAIN_ONLY:-0}

if [ "$TRAIN_ONLY" = "1" ] || [ "$TRAIN_ONLY" = "true" ]; then
    echo "TRAIN_ONLY mode: will skip graph perturbation, data processing and baseline. Only training+validation will be submitted."
    # Ensure EXPERIMENT is provided
    if [ -z "$EXPERIMENT" ]; then
        echo "ERROR: EXPERIMENT_NAME must be provided for TRAIN_ONLY mode"
        exit 1
    fi

    # Submit training jobs directly (no dependencies on processing) for PDGrapher models
    echo "Submitting PDGrapher training job (train-only)..."
    PDG_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_train_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING (TRAIN_ONLY) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Train PDGrapher model
python -m src.train_pdgrapher --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher training completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    echo "PDGrapher training job submitted with ID: $PDG_TRAIN_JOB_ID"

    echo "Submitting PDGrapherNoGNN training job (train-only)..."
    PDGNN_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_train_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN TRAINING (TRAIN_ONLY) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Train PDGrapherNoGNN model
python -m src.train_pdgrapher_nognn --experiment $EXPERIMENT --model pdgrapher_nognn

END_TIME=\$(date +%s)
echo "PDGrapherNoGNN training completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher_nognn,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    echo "PDGrapherNoGNN training job submitted with ID: $PDGNN_TRAIN_JOB_ID"

    # Submit validation jobs depending on their corresponding training jobs
    echo "Submitting validation jobs (train-only)..."
    PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_val_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION (TRAIN_ONLY) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
python -m src.validation --experiment $EXPERIMENT --model pdgrapher
EOF
)
    echo "PDGrapher validation job submitted with ID: $PDG_VAL_JOB_ID"

    PDGNN_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDGNN_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_val_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN VALIDATION (TRAIN_ONLY) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
python -m src.validation --experiment $EXPERIMENT --model pdgrapher_nognn
EOF
)
    echo "PDGrapherNoGNN validation job submitted with ID: $PDGNN_VAL_JOB_ID"

    echo ""
    echo "=== TRAIN_ONLY MODE PIPELINE SUBMITTED ==="
    echo "Experiment: $EXPERIMENT"
    echo "PDGrapher training: $PDG_TRAIN_JOB_ID"
    echo "PDGrapher validation: $PDG_VAL_JOB_ID"
    echo "PDGrapherNoGNN training: $PDGNN_TRAIN_JOB_ID"
    echo "PDGrapherNoGNN validation: $PDGNN_VAL_JOB_ID"
    
    # Return the first training job id for tracking
    echo "Job ID: $PDG_TRAIN_JOB_ID"
    exit 0
fi

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT_NAME must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

if [ -z "$FINGERPRINT" ]; then
    echo "ERROR: DATA_FINGERPRINT is not set"
    exit 1
fi

# Verify shared data exists
SHARED_DATA_DIR="$BASE_DIR/data/shared/$FINGERPRINT"
if [ ! -d "$SHARED_DATA_DIR" ]; then
    echo "ERROR: Shared data directory $SHARED_DATA_DIR does not exist for fingerprint $FINGERPRINT"
    exit 1
fi

SHARED_DATA_DEP_MSG="pre-created shared data at $SHARED_DATA_DIR"
SPLIT_DEP_FLAG=""

echo "Starting dual-model experiment pipeline..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"
echo "Models: PDGrapher, PDGrapherNoGNN"
echo "Shared data: $SHARED_DATA_DEP_MSG"

# Create output directories if they don't exist
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Initialize timing log with header
echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

# =============================================================================
# Job 1: K-Fold Data Splitting (depends on shared data creation)
# =============================================================================
echo "Submitting k-fold data splitting job..."
SPLIT_JOB_ID=$(sbatch --parsable $SPLIT_DEP_FLAG <<EOF
#!/bin/bash
#SBATCH --job-name="kfold_split_${EXPERIMENT}"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_kfold_split_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_kfold_split_%j.err

source .venv/bin/activate

echo "=== STARTING K-FOLD DATA SPLITTING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "K-Folds: $K_FOLDS"
echo "Depends on shared data: $SHARED_DATA_DEP_MSG"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export K_FOLDS=$K_FOLDS

# Create k-fold splits for this experiment
python -m src.create_kfold_splits --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "K-fold data splitting completed at: \$(date)"
echo "$EXPERIMENT,kfold,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "K-fold data splitting job submitted with ID: $SPLIT_JOB_ID"



# =============================================================================
# Job 2: Parallel K-Fold Training for PDGrapher models
# =============================================================================

# Arrays to store job IDs for dependency tracking
PDG_TRAIN_JOB_IDS=()
PDGNN_TRAIN_JOB_IDS=()
PDG_VAL_JOB_IDS=()
PDGNN_VAL_JOB_IDS=()

# Submit training jobs for each fold (all in parallel)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapher training job for fold $fold..."
    PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_train_fold${fold}_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_fold${fold}_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_fold${fold}_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING (FOLD $fold of $K_FOLDS) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Fold: $fold / $K_FOLDS"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS

# Train PDGrapher model for this fold
python -m src.train_pdgrapher --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher training (fold $fold) completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,training,fold_$fold,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    PDG_TRAIN_JOB_IDS+=($PDG_TRAIN_JOB_ID)
    echo "PDGrapher training job (fold $fold) submitted with ID: $PDG_TRAIN_JOB_ID"
done

# Submit PDGrapherNoGNN training jobs for each fold (all in parallel)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapherNoGNN training job for fold $fold..."
    PDGNN_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_train_fold${fold}_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_fold${fold}_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_fold${fold}_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN TRAINING (FOLD $fold of $K_FOLDS) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Fold: $fold / $K_FOLDS"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS

# Train PDGrapherNoGNN model for this fold
python -m src.train_pdgrapher_nognn --experiment $EXPERIMENT --model pdgrapher_nognn

END_TIME=\$(date +%s)
echo "PDGrapherNoGNN training (fold $fold) completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher_nognn,training,fold_$fold,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    PDGNN_TRAIN_JOB_IDS+=($PDGNN_TRAIN_JOB_ID)
    echo "PDGrapherNoGNN training job (fold $fold) submitted with ID: $PDGNN_TRAIN_JOB_ID"
done

# =============================================================================
# Job 3: Parallel K-Fold Validation for PDGrapher models
# =============================================================================

# Submit validation jobs for each fold (after corresponding training completes)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapher validation job for fold $fold..."
    PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:${PDG_TRAIN_JOB_IDS[$fold]} <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_val_fold${fold}_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_fold${fold}_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_fold${fold}_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION (FOLD $fold of $K_FOLDS) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Fold: $fold / $K_FOLDS"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS

# Validate PDGrapher model for this fold
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher validation (fold $fold) completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,validation,fold_$fold,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    PDG_VAL_JOB_IDS+=($PDG_VAL_JOB_ID)
    echo "PDGrapher validation job (fold $fold) submitted with ID: $PDG_VAL_JOB_ID"
done

# Submit PDGrapherNoGNN validation jobs for each fold (after corresponding training completes)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapherNoGNN validation job for fold $fold..."
    PDGNN_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:${PDGNN_TRAIN_JOB_IDS[$fold]} <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_val_fold${fold}_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_fold${fold}_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_fold${fold}_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN VALIDATION (FOLD $fold of $K_FOLDS) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Fold: $fold / $K_FOLDS"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS

# Validate PDGrapherNoGNN model for this fold
python -m src.validation --experiment $EXPERIMENT --model pdgrapher_nognn

END_TIME=\$(date +%s)
echo "PDGrapherNoGNN validation (fold $fold) completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher_nognn,validation,fold_$fold,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    PDGNN_VAL_JOB_IDS+=($PDGNN_VAL_JOB_ID)
    echo "PDGrapherNoGNN validation job (fold $fold) submitted with ID: $PDGNN_VAL_JOB_ID"
done

# =============================================================================
# Job 4: K-Fold Results Aggregation (depends on all validations)
# =============================================================================

# Create dependency strings for all validation jobs
PDG_VAL_DEPS=$(IFS=, ; echo "${PDG_VAL_JOB_IDS[*]}")
PDGNN_VAL_DEPS=$(IFS=, ; echo "${PDGNN_VAL_JOB_IDS[*]}")

echo "Submitting k-fold aggregation job..."
AGGREGATE_JOB_ID=$(sbatch --parsable --dependency=afterok:${PDG_VAL_DEPS},${PDGNN_VAL_DEPS} <<EOF
#!/bin/bash
#SBATCH --job-name="aggregate_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_aggregate_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_aggregate_%j.err

source .venv/bin/activate

echo "=== AGGREGATING K-FOLD RESULTS ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "K-Folds: $K_FOLDS"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export K_FOLDS=$K_FOLDS
export DATA_FINGERPRINT=$FINGERPRINT

# Run k-fold aggregation script to combine results across folds
python -m src.aggregate_fold_results --experiment $EXPERIMENT --k_folds $K_FOLDS

END_TIME=\$(date +%s)
echo "K-fold aggregation completed at: \$(date)"
echo "$EXPERIMENT,kfold,aggregation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "K-fold aggregation job submitted with ID: $AGGREGATE_JOB_ID"

# =============================================================================
# Pipeline Summary
# =============================================================================

echo ""
echo "=== K-FOLD EXPERIMENT PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo "K-Folds: $K_FOLDS"
echo "Shared data dependency: $SHARED_DATA_DEP_MSG"
echo ""
echo "Pipeline structure:"
echo "  1. K-Fold Data Splitting:  $SPLIT_JOB_ID (after $SHARED_DATA_DEP_MSG)"
echo "  2. Parallel Training (all folds):"
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "     - PDGrapher Fold $fold:     ${PDG_TRAIN_JOB_IDS[$fold]}"
    echo "     - PDGrapherNoGNN Fold $fold: ${PDGNN_TRAIN_JOB_IDS[$fold]}"
done
echo "  3. Parallel Validation (all folds):"
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "     - PDGrapher Fold $fold:     ${PDG_VAL_JOB_IDS[$fold]}"
    echo "     - PDGrapherNoGNN Fold $fold: ${PDGNN_VAL_JOB_IDS[$fold]}"
done
echo "  4. K-Fold Aggregation:     $AGGREGATE_JOB_ID (after all validations)"
echo ""
echo "Monitor progress:"
echo "  squeue -j $SPLIT_JOB_ID"
echo "  squeue -j $(IFS=, ; echo \"${PDG_TRAIN_JOB_IDS[*]}\")"
echo "  squeue -j $(IFS=, ; echo \"${PDG_VAL_JOB_IDS[*]}\")"
echo "  squeue -j $AGGREGATE_JOB_ID"
echo ""
echo "Cancel all jobs if needed:"
ALL_JOB_IDS="$SPLIT_JOB_ID,$(IFS=, ; echo \"${PDG_TRAIN_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDGNN_TRAIN_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDG_VAL_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDGNN_VAL_JOB_IDS[*]}\"),$AGGREGATE_JOB_ID"
echo "  scancel $ALL_JOB_IDS"

# Store first job ID for master pipeline tracking
echo "Job ID: $SPLIT_JOB_ID"
