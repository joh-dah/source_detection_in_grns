#!/bin/bash

# Dual Model Single Experiment Runner Script (With Shared Data Dependency)
# Runs PDGrapher and PDGrapherNoGNN for a single experiment
# Usage: SHARED_DATA_JOB_ID=12345 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx27,gx28,gx29"

# Get experiment name and shared data job dependency
EXPERIMENT=${EXPERIMENT_NAME}
SHARED_DATA_DEPENDENCY=${SHARED_DATA_JOB_ID}
FINGERPRINT=${DATA_FINGERPRINT}
SKIP_SHARED_DATA_DEPENDENCY=${SKIP_SHARED_DATA_DEPENDENCY:-0}

# K-Fold configuration
K_FOLDS=${K_FOLDS:-5}  # Default to 5 folds

# TRAIN_ONLY mode: if TRAIN_ONLY=1, skip shared data / perturbation / processing / baseline
TRAIN_ONLY=${TRAIN_ONLY:-0}

if [ "$TRAIN_ONLY" = "1" ] || [ "$TRAIN_ONLY" = "true" ]; then
    echo "TRAIN_ONLY mode: will skip graph perturbation, data processing and baseline. Only training+validation will be submitted."
    echo "K-Fold configuration: $K_FOLDS folds"
    # Ensure EXPERIMENT is provided
    if [ -z "$EXPERIMENT" ]; then
        echo "ERROR: EXPERIMENT_NAME must be provided for TRAIN_ONLY mode"
        exit 1
    fi

    # Arrays to store job IDs for k-fold dependency tracking
    PDG_TRAIN_JOB_IDS=()
    PDGNN_TRAIN_JOB_IDS=()
    PDG_VAL_JOB_IDS=()
    PDGNN_VAL_JOB_IDS=()

    # Submit training jobs for each fold (all in parallel)
    for fold in $(seq 0 $((K_FOLDS - 1))); do
        echo "Submitting PDGrapher training job for fold $fold (train-only)..."
        PDG_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
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

echo "=== STARTING PDGRAPHER TRAINING (TRAIN_ONLY, FOLD $fold of $K_FOLDS) ==="
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
        echo "Submitting PDGrapherNoGNN training job for fold $fold (train-only)..."
        PDGNN_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
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

echo "=== STARTING PDGRAPHER NOGNN TRAINING (TRAIN_ONLY, FOLD $fold of $K_FOLDS) ==="
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

    # Submit validation jobs for each fold (after corresponding training completes)
    echo "Submitting validation jobs for all folds (train-only)..."
    for fold in $(seq 0 $((K_FOLDS - 1))); do
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

echo "=== STARTING PDGRAPHER VALIDATION (TRAIN_ONLY, FOLD $fold of $K_FOLDS) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS
python -m src.validation --experiment $EXPERIMENT --model pdgrapher
EOF
)
        PDG_VAL_JOB_IDS+=($PDG_VAL_JOB_ID)
        echo "PDGrapher validation job (fold $fold) submitted with ID: $PDG_VAL_JOB_ID"
    done

    for fold in $(seq 0 $((K_FOLDS - 1))); do
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

echo "=== STARTING PDGRAPHER NOGNN VALIDATION (TRAIN_ONLY, FOLD $fold of $K_FOLDS) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS
python -m src.validation --experiment $EXPERIMENT --model pdgrapher_nognn
EOF
)
        PDGNN_VAL_JOB_IDS+=($PDGNN_VAL_JOB_ID)
        echo "PDGrapherNoGNN validation job (fold $fold) submitted with ID: $PDGNN_VAL_JOB_ID"
    done

    # Create dependency strings for all validation jobs
    PDG_VAL_DEPS=$(IFS=, ; echo "${PDG_VAL_JOB_IDS[*]}")
    PDGNN_VAL_DEPS=$(IFS=, ; echo "${PDGNN_VAL_JOB_IDS[*]}")

    # K-fold aggregation job (depends on all validations)
    echo "Submitting k-fold aggregation job (train-only)..."
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

echo "=== AGGREGATING K-FOLD RESULTS (TRAIN_ONLY) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export K_FOLDS=$K_FOLDS
python -m src.aggregate_fold_results --experiment $EXPERIMENT --k_folds $K_FOLDS --model pdgrapher
EOF
)
    echo "K-fold aggregation job submitted with ID: $AGGREGATE_JOB_ID"

    # Return the first training job id for tracking
    echo "Job ID: ${PDG_TRAIN_JOB_IDS[0]}"
    exit 0
fi

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT_NAME must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

if [ -z "$FINGERPRINT" ]; then
    echo "ERROR: DATA_FINGERPRINT must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

if [ "$SKIP_SHARED_DATA_DEPENDENCY" = "1" ] || [ "$SKIP_SHARED_DATA_DEPENDENCY" = "true" ]; then
    SHARED_DATA_DEPENDENCY=""
    SHARED_DATA_DIR="$BASE_DIR/data/shared/$FINGERPRINT"
    if [ ! -d "$SHARED_DATA_DIR" ]; then
        echo "ERROR: Shared data directory $SHARED_DATA_DIR does not exist for fingerprint $FINGERPRINT"
        exit 1
    fi
    SHARED_DATA_DEP_MSG="pre-created shared data at $SHARED_DATA_DIR"
    SPLIT_DEP_FLAG=""
else
    if [ -z "$SHARED_DATA_DEPENDENCY" ]; then
        echo "ERROR: SHARED_DATA_JOB_ID must be provided"
        echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
        exit 1
    fi
    SHARED_DATA_DEP_MSG="$SHARED_DATA_DEPENDENCY"
    SPLIT_DEP_FLAG="--dependency=afterok:$SHARED_DATA_DEPENDENCY"
fi

echo "Starting dual-model experiment pipeline with shared data dependency..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"
echo "Models: PDGrapher, PDGrapherNoGNN"
echo "Shared data dependency: $SHARED_DATA_DEP_MSG"

# Create output directories if they don't exist
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Initialize timing log with header
echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

# =============================================================================
# Job 1: K-Fold Data Splitting (depends on shared data creation)
# =============================================================================
echo "Submitting k-fold data splitting job..."
echo "K-Fold configuration: $K_FOLDS folds"
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
# Job 2: Graph Perturbation (depends on per-experiment splitting)
# =============================================================================
echo "Submitting graph perturbation job..."
GRAPH_PERT_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="graph_pert_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_graph_pert_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_graph_pert_%j.err

source .venv/bin/activate

echo "=== STARTING GRAPH PERTURBATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on split job: $SPLIT_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Apply graph perturbations for this specific experiment
python -m src.graph_perturbation --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "Graph perturbation completed at: \$(date)"

# Log timing for all methods that will use this perturbed graph
echo "$EXPERIMENT,gat,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,pdgrapher,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,pdgrapher_nognn,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "Graph perturbation job submitted with ID: $GRAPH_PERT_JOB_ID"

# =============================================================================
# Job 3: Parallel Data Processing for all three models
# =============================================================================

# Job 2a: GAT Data Processing
echo "Submitting GAT data processing job..."
GAT_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_proc_${EXPERIMENT}"
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_gat_proc_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_gat_proc_%j.err

source .venv/bin/activate

echo "=== STARTING GAT DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on graph perturbation job: $GRAPH_PERT_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Process data for GAT
python -m src.data_processing --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "GAT data processing completed at: \$(date)"
echo "$EXPERIMENT,gat,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "GAT data processing job submitted with ID: $GAT_PROC_JOB_ID"

# Job 2b: PDGrapher Data Processing
echo "Submitting PDGrapher data processing job..."
PDG_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_proc_${EXPERIMENT}"
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_proc_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_proc_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on graph perturbation job: $GRAPH_PERT_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Process data for PDGrapher
python -m src.data_processing --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher data processing completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "PDGrapher data processing job submitted with ID: $PDG_PROC_JOB_ID"

# Job 2c: PDGrapherNoGNN Data Processing
echo "Submitting PDGrapherNoGNN data processing job..."
PDGNN_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_proc_${EXPERIMENT}"
#SBATCH --mem=128G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_proc_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_proc_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER NOGNN DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on graph perturbation job: $GRAPH_PERT_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export MODEL=pdgrapher_nognn

# Process data for PDGrapherNoGNN
python -m src.data_processing --experiment $EXPERIMENT --model pdgrapher_nognn

END_TIME=\$(date +%s)
echo "PDGrapherNoGNN data processing completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher_nognn,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "PDGrapherNoGNN data processing job submitted with ID: $PDGNN_PROC_JOB_ID"

# =============================================================================
# Job 4: Parallel K-Fold Training for PDGrapher models
# =============================================================================

# Arrays to store job IDs for dependency tracking
PDG_TRAIN_JOB_IDS=()
PDGNN_TRAIN_JOB_IDS=()
PDG_VAL_JOB_IDS=()
PDGNN_VAL_JOB_IDS=()
BASELINE_JOB_IDS=()

# Submit training jobs for each fold (all in parallel)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapher training job for fold $fold..."
    PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_PROC_JOB_ID <<EOF
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

echo "=== STARTING PDGRAPHER TRAINING (FOLD ${fold} of ${K_FOLDS}) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: ${EXPERIMENT}"
echo "Fold: ${fold} / ${K_FOLDS}"
echo "Depends on PDGrapher processing job: ${PDG_PROC_JOB_ID}"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=${EXPERIMENT}
export DATA_FINGERPRINT=${FINGERPRINT}
export FOLD_INDEX=${fold}
export K_FOLDS=${K_FOLDS}

# Train PDGrapher model for this fold
python -m src.train_pdgrapher --experiment ${EXPERIMENT} --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher training (fold ${fold}) completed at: \$(date)"
echo "${EXPERIMENT},pdgrapher,training,fold_${fold},\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> ${TIMING_LOG_DIR}/${EXPERIMENT}_timing.csv
EOF
)
    PDG_TRAIN_JOB_IDS+=($PDG_TRAIN_JOB_ID)
    echo "PDGrapher training job (fold $fold) submitted with ID: $PDG_TRAIN_JOB_ID"
done

# Submit PDGrapherNoGNN training jobs for each fold (all in parallel)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "Submitting PDGrapherNoGNN training job for fold $fold..."
    PDGNN_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDGNN_PROC_JOB_ID <<EOF
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
echo "Depends on PDGrapherNoGNN processing job: $PDGNN_PROC_JOB_ID"

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
# Job 5: Baseline Computation for each fold (after k-fold splitting)
# =============================================================================
echo "Submitting baseline jobs for all folds..."
# Baseline should depend on k-fold split job (baseline is computed per fold)
for fold in $(seq 0 $((K_FOLDS - 1))); do
    BASELINE_JOB_ID=$(sbatch --parsable --dependency=afterok:$SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="baseline_fold${fold}_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu-batch
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_fold${fold}_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_fold${fold}_%j.err

source .venv/bin/activate

echo "=== STARTING BASELINE COMPUTATION (FOLD $fold of $K_FOLDS) ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Fold: $fold / $K_FOLDS"
echo "Depends on k-fold split job: $SPLIT_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
export FOLD_INDEX=$fold
export K_FOLDS=$K_FOLDS

# Compute baseline metrics for this fold (uses GAT baseline)
python -m src.baseline --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "Baseline computation (fold $fold) completed at: \$(date)"
echo "$EXPERIMENT,baseline,baseline,fold_$fold,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
    BASELINE_JOB_IDS+=($BASELINE_JOB_ID)
    echo "Baseline job (fold $fold) submitted with ID: $BASELINE_JOB_ID"
done

# =============================================================================
# Job 6: Parallel K-Fold Validation for PDGrapher models
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
echo "Depends on PDGrapher training job: ${PDG_TRAIN_JOB_IDS[$fold]}"

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
echo "Depends on PDGrapherNoGNN training job: ${PDGNN_TRAIN_JOB_IDS[$fold]}"

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
# Job 7: K-Fold Results Aggregation (depends on all validations)
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
export DATA_FINGERPRINT=$FINGERPRINT
export K_FOLDS=$K_FOLDS

# Run k-fold aggregation script to combine results across folds
python -m src.aggregate_fold_results --experiment $EXPERIMENT --k_folds $K_FOLDS --model pdgrapher

END_TIME=\$(date +%s)
echo "K-fold aggregation completed at: \$(date)"
echo "$EXPERIMENT,kfold,aggregation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "K-fold aggregation job submitted with ID: $AGGREGATE_JOB_ID"

echo ""
echo "=== K-FOLD EXPERIMENT PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo "K-Folds: $K_FOLDS"
echo "Shared data dependency: $SHARED_DATA_DEP_MSG"
echo ""
echo "Pipeline structure:"
echo "  1. K-Fold Data Splitting:  $SPLIT_JOB_ID (after $SHARED_DATA_DEP_MSG)"
echo "  2. Graph Perturbation:     $GRAPH_PERT_JOB_ID (after split $SPLIT_JOB_ID)"
echo "  3a. GAT Data Processing:   $GAT_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  3b. PDGrapher Data Proc:   $PDG_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  3c. PDGrapherNoGNN Proc:   $PDGNN_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  4. Parallel Training (all folds):"
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "     - PDGrapher Fold $fold:      ${PDG_TRAIN_JOB_IDS[$fold]}"
    echo "     - PDGrapherNoGNN Fold $fold: ${PDGNN_TRAIN_JOB_IDS[$fold]}"
done
echo "  5. Baseline (all folds):"
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "     - Baseline Fold $fold:       ${BASELINE_JOB_IDS[$fold]}"
done
echo "  6. Parallel Validation (all folds):"
for fold in $(seq 0 $((K_FOLDS - 1))); do
    echo "     - PDGrapher Fold $fold:      ${PDG_VAL_JOB_IDS[$fold]}"
    echo "     - PDGrapherNoGNN Fold $fold: ${PDGNN_VAL_JOB_IDS[$fold]}"
done
echo "  7. K-Fold Aggregation:     $AGGREGATE_JOB_ID (after all validations)"
echo ""
echo "Monitor progress:"
echo "  squeue -j $SPLIT_JOB_ID"
echo "  squeue -j $(IFS=, ; echo \"${PDG_TRAIN_JOB_IDS[*]}\")"
echo "  squeue -j $(IFS=, ; echo \"${BASELINE_JOB_IDS[*]}\")"
echo "  squeue -j $(IFS=, ; echo \"${PDG_VAL_JOB_IDS[*]}\")"
echo "  squeue -j $AGGREGATE_JOB_ID"
echo ""
echo "Cancel all jobs if needed:"
ALL_JOB_IDS="$SPLIT_JOB_ID,$GRAPH_PERT_JOB_ID,$GAT_PROC_JOB_ID,$PDG_PROC_JOB_ID,$PDGNN_PROC_JOB_ID,$(IFS=, ; echo \"${PDG_TRAIN_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDGNN_TRAIN_JOB_IDS[*]}\"),$(IFS=, ; echo \"${BASELINE_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDG_VAL_JOB_IDS[*]}\"),$(IFS=, ; echo \"${PDGNN_VAL_JOB_IDS[*]}\"),$AGGREGATE_JOB_ID"
echo "  scancel $ALL_JOB_IDS"

# Store first job ID for master pipeline tracking
echo "Job ID: $SPLIT_JOB_ID"
