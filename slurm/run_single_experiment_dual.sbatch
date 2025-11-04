#!/bin/bash

# Dual Model Single Experiment Runner Script (With Shared Data Dependency)
# Runs PDGrapher and PDGrapherNoGNN for a single experiment
# Usage: SHARED_DATA_JOB_ID=12345 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx21,gx25,gx27,gx28,gx29"

# Get experiment name and shared data job dependency
EXPERIMENT=${EXPERIMENT_NAME}
SHARED_DATA_DEPENDENCY=${SHARED_DATA_JOB_ID}
FINGERPRINT=${DATA_FINGERPRINT}

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
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
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
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
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
#SBATCH --partition=gpu
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
#SBATCH --partition=gpu
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

    # Final comparison job depending on validations
    DUAL_COMPARE_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_VAL_JOB_ID,$PDGNN_VAL_JOB_ID,$BASELINE_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_compare_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_dual_compare_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_dual_compare_%j.err

source .venv/bin/activate

echo "=== STARTING DUAL MODEL RESULT COMPARISON (TRAIN_ONLY) ==="
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT
python -m src.compare_results --model pdgrapher --exp_id $EXPERIMENT
EOF
)
    echo "Dual compare job submitted with ID: $DUAL_COMPARE_JOB_ID"

    # Return the first training job id for tracking
    echo "Job ID: $PDG_TRAIN_JOB_ID"
    exit 0
fi

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT_NAME must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

if [ -z "$SHARED_DATA_DEPENDENCY" ]; then
    echo "ERROR: SHARED_DATA_JOB_ID must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

if [ -z "$FINGERPRINT" ]; then
    echo "ERROR: DATA_FINGERPRINT must be provided"
    echo "Usage: SHARED_DATA_JOB_ID=12345 DATA_FINGERPRINT=abc123 EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment_dual.sbatch"
    exit 1
fi

echo "Starting dual-model experiment pipeline with shared data dependency..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"
echo "Models: PDGrapher, PDGrapherNoGNN"
echo "Shared data dependency: $SHARED_DATA_DEPENDENCY"

# Create output directories if they don't exist
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Initialize timing log with header
echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

# =============================================================================
# Job 1: Graph Perturbation (depends on shared data creation)
# =============================================================================
echo "Submitting graph perturbation job..."
GRAPH_PERT_JOB_ID=$(sbatch --parsable --dependency=afterok:$SHARED_DATA_DEPENDENCY <<EOF
#!/bin/bash
#SBATCH --job-name="graph_pert_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
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
echo "Depends on shared data job: $SHARED_DATA_DEPENDENCY"

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
# Job 2: Parallel Data Processing for all three models
# =============================================================================

# Job 2a: GAT Data Processing
echo "Submitting GAT data processing job..."
GAT_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_proc_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
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
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
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
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
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
# Job 3: Parallel Training for all three models
# =============================================================================

# (GAT training removed — only PDGrapher models will be trained)

# Job 3b: PDGrapher Training
echo "Submitting PDGrapher training job..."
PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_train_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_train_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher processing job: $PDG_PROC_JOB_ID"

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

# Job 3c: PDGrapherNoGNN Training
echo "Submitting PDGrapherNoGNN training job..."
PDGNN_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDGNN_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_train_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_train_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapherNoGNN processing job: $PDGNN_PROC_JOB_ID"

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

# =============================================================================
# Job 4: Baseline Computation (after data processing is complete for any model)
# =============================================================================
echo "Submitting baseline job..."
# Baseline should depend on GAT processing (GAT data used for baseline)
BASELINE_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="baseline_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_%j.err

source .venv/bin/activate

echo "=== STARTING BASELINE COMPUTATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT processing job: $GAT_PROC_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Compute baseline metrics
python -m src.baseline --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "Baseline computation completed at: \$(date)"
echo "$EXPERIMENT,baseline,baseline,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "Baseline job submitted with ID: $BASELINE_JOB_ID"

# =============================================================================
# Job 5: Parallel Validation for all three models
# =============================================================================

# (GAT validation removed — only PDGrapher models will be validated)

# Job 5b: PDGrapher Validation
echo "Submitting PDGrapher validation job..."
PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_val_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_val_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher training job: $PDG_TRAIN_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Validate PDGrapher model
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher validation completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,validation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "PDGrapher validation job submitted with ID: $PDG_VAL_JOB_ID"

# Job 5c: PDGrapherNoGNN Validation
echo "Submitting PDGrapherNoGNN validation job..."
PDGNN_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDGNN_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdgnn_val_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdgnn_val_%j.err

source .venv/bin/activate
export MODEL=pdgrapher_nognn

echo "=== STARTING PDGRAPHER NOGNN VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapherNoGNN training job: $PDGNN_TRAIN_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Validate PDGrapherNoGNN model
python -m src.validation --experiment $EXPERIMENT --model pdgrapher_nognn

END_TIME=\$(date +%s)
echo "PDGrapherNoGNN validation completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher_nognn,validation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "PDGrapherNoGNN validation job submitted with ID: $PDGNN_VAL_JOB_ID"

# =============================================================================
# Job 6: Dual Model Result Comparison (depends on validations + baseline)
# =============================================================================
echo "Submitting dual-model result comparison job..."
DUAL_COMPARE_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_VAL_JOB_ID,$PDGNN_VAL_JOB_ID,$BASELINE_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_compare_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_dual_compare_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_dual_compare_%j.err

source .venv/bin/activate

echo "=== STARTING DUAL MODEL RESULT COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher validation job: $PDG_VAL_JOB_ID"
echo "Depends on PDGrapherNoGNN validation job: $PDGNN_VAL_JOB_ID"
echo "Depends on baseline job: $BASELINE_JOB_ID"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT
export DATA_FINGERPRINT=$FINGERPRINT

# Run dual model comparison for this experiment (use PDGrapher as reference)
python -m src.compare_results --model pdgrapher --exp_id $EXPERIMENT

END_TIME=\$(date +%s)
echo "Dual model result comparison completed at: \$(date)"
echo "$EXPERIMENT,dual_comparison,comparison,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

echo ""
echo "=== DUAL MODEL EXPERIMENT PIPELINE COMPLETE ==="
echo "Experiment: $EXPERIMENT"
echo "PDGrapher and PDGrapherNoGNN have been trained, validated, and compared"
echo "Results are available in the reports/ directory"
echo "Total pipeline time: \$((\$END_TIME - $GRAPH_PERT_JOB_ID)) seconds"
EOF
)
echo "Dual model result comparison job submitted with ID: $DUAL_COMPARE_JOB_ID"

echo ""
echo "=== DUAL MODEL EXPERIMENT PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo "Shared data dependency: $SHARED_DATA_DEPENDENCY"
echo ""
echo "Pipeline structure:"
echo "  1. Graph Perturbation:          $GRAPH_PERT_JOB_ID (after $SHARED_DATA_DEPENDENCY)"
echo "  2a. GAT Data Processing:        $GAT_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  2b. PDGrapher Data Processing:  $PDG_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  2c. PDGrapherNoGNN Data Proc:   $PDGNN_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  3a. PDGrapher Training:         $PDG_TRAIN_JOB_ID (after $PDG_PROC_JOB_ID)"
echo "  3b. PDGrapherNoGNN Training:    $PDGNN_TRAIN_JOB_ID (after $PDGNN_PROC_JOB_ID)"
echo "  4. Baseline:                    $BASELINE_JOB_ID (after $GAT_PROC_JOB_ID)"
echo "  5a. PDGrapher Validation:       $PDG_VAL_JOB_ID (after $PDG_TRAIN_JOB_ID)"
echo "  5b. PDGrapherNoGNN Validation:  $PDGNN_VAL_JOB_ID (after $PDGNN_TRAIN_JOB_ID)"
echo "  6. Dual Model Comparison:       $DUAL_COMPARE_JOB_ID (after validations + baseline)"
echo ""
echo "Monitor progress:"
echo "  squeue -j $GRAPH_PERT_JOB_ID,$GAT_PROC_JOB_ID,$PDG_PROC_JOB_ID,$PDGNN_PROC_JOB_ID,$PDG_TRAIN_JOB_ID,$PDGNN_TRAIN_JOB_ID,$BASELINE_JOB_ID,$PDG_VAL_JOB_ID,$PDGNN_VAL_JOB_ID,$DUAL_COMPARE_JOB_ID"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel $GRAPH_PERT_JOB_ID,$GAT_PROC_JOB_ID,$PDG_PROC_JOB_ID,$PDGNN_PROC_JOB_ID,$PDG_TRAIN_JOB_ID,$PDGNN_TRAIN_JOB_ID,$BASELINE_JOB_ID,$PDG_VAL_JOB_ID,$PDGNN_VAL_JOB_ID,$DUAL_COMPARE_JOB_ID"

# Store first job ID for master pipeline tracking
echo "Job ID: $GRAPH_PERT_JOB_ID"
