#!/bin/bash

# Dual Model Pipeline Runner (Data Already Present)
# Runs training and validation for both GAT and PDGrapher models
# Assumes data creation and processing have already been completed

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# Experiment configuration (can be overridden via environment)
EXPERIMENT=${EXPERIMENT_NAME:-testrun}

# Create timing log directory
mkdir -p $TIMING_LOG_DIR

# gpu configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx21,gx25,gx27,gx28,gx29"

# Create output directory if it doesn't exist
mkdir -p $SLURM_OUT_DIR

echo "Starting dual model pipeline..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"

# Function to log timing information
log_timing() {
    local stage=$1
    local method=$2
    local start_time=$3
    local end_time=$4
    local job_id=$5
    
    local duration=$((end_time - start_time))
    echo "$EXPERIMENT,$method,$stage,$start_time,$end_time,$duration,$job_id" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
}

# Initialize timing log with header
echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

# =============================================================================
# Job 1: Data Splitting
# =============================================================================
echo "Submitting data splitting job..."
DATA_SPLIT_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="data_splitting"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_data_splitting_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_data_splitting_%j.err

source .venv/bin/activate

echo "=== STARTING DATA SPLITTING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

# Record start time
START_TIME=\$(date +%s)

# Set config
export EXPERIMENT_NAME=$EXPERIMENT

# Create data splits (shared between models)
python -m src.create_splits --experiment $EXPERIMENT --model gat

# Record end time and log
END_TIME=\$(date +%s)
echo "Data splitting completed at: \$(date)"

# Log timing for all methods that use this data splitting
echo "$EXPERIMENT,gat,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_rumor,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_random,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_advanced_random,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,pdgrapher,data_splitting,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "Data splitting job submitted with ID: $DATA_SPLIT_JOB_ID"

# =============================================================================
# Job 2a: GAT Training (depends on data splitting)
# =============================================================================
echo "Submitting GAT training job..."
GAT_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_training"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_gat_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_gat_training_%j.err

source .venv/bin/activate

echo "=== STARTING GAT TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on data splitting job: $DATA_SPLIT_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Set GAT config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Train GAT model
python -m src.training --model gat --experiment $EXPERIMENT

# Record end time and log
END_TIME=\$(date +%s)
echo "GAT training completed at: \$(date)"

# Log timing for GAT
echo "$EXPERIMENT,gat,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "GAT training job submitted with ID: $GAT_TRAIN_JOB_ID"

# =============================================================================
# Job 2b: PDGrapher Training (depends on data splitting)
# =============================================================================
echo "Submitting PDGrapher training job..."
PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_training"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=3-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_pdg_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_pdg_training_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on data splitting job: $DATA_SPLIT_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Set PDGrapher config
export MODEL_TYPE=pdgrapher
export EXPERIMENT_NAME=$EXPERIMENT

# Train PDGrapher model
python -m src.train_pdgrapher --model pdgrapher --experiment $EXPERIMENT

# Record end time and log
END_TIME=\$(date +%s)
echo "PDGrapher training completed at: \$(date)"

# Log timing for PDGrapher
echo "$EXPERIMENT,pdgrapher,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "PDGrapher training job submitted with ID: $PDG_TRAIN_JOB_ID"


# =============================================================================
# Job 2c: Baseline (depends on data splitting)
# =============================================================================
echo "Submitting baseline job..."
BASELINE_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_SPLIT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="baseline"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --chdir=$BASE_DIR   
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_baseline_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_baseline_%j.err

source .venv/bin/activate
echo "=== STARTING BASELINE SCRIPT ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on data splitting job: $DATA_SPLIT_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Set baseline config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Run the baseline script
python -m src.baseline --experiment $EXPERIMENT

# Record end time and log for all baseline methods
END_TIME=\$(date +%s)
echo "Baseline script completed at: \$(date)"

# Log timing for all baseline methods (they all run in this single job)
echo "$EXPERIMENT,baseline_rumor,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_random,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_advanced_random,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)
echo "Baseline job submitted with ID: $BASELINE_JOB_ID"

# =============================================================================
# Job 3a: GAT Validation (depends on GAT training)
# =============================================================================
echo "Submitting GAT validation job..."
GAT_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_validation"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_gat_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_gat_validation_%j.err

source .venv/bin/activate

echo "=== STARTING GAT VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT training job: $GAT_TRAIN_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Set GAT config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Validate GAT model
python -m src.validation --model gat --experiment $EXPERIMENT

# Record end time and log
END_TIME=\$(date +%s)
echo "GAT validation completed at: \$(date)"

# Log timing for GAT
echo "$EXPERIMENT,gat,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "GAT validation job submitted with ID: $GAT_VAL_JOB_ID"

# =============================================================================
# Job 3b: PDGrapher Validation (depends on PDGrapher training)
# =============================================================================
echo "Submitting PDGrapher validation job..."
PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_validation"
#SBATCH --time=3-00:00:00
#SBATCH --partition=gpu
#SBATCH --nodelist=$COMPATIBLE_GPU_NODES
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_pdg_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_pdg_validation_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher training job: $PDG_TRAIN_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Set PDGrapher config
export MODEL_TYPE=pdgrapher
export EXPERIMENT_NAME=$EXPERIMENT

# Validate PDGrapher model
python -m src.validation --model pdgrapher --experiment $EXPERIMENT

# Record end time and log
END_TIME=\$(date +%s)
echo "PDGrapher validation completed at: \$(date)"

# Log timing for PDGrapher
echo "$EXPERIMENT,pdgrapher,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "PDGrapher validation job submitted with ID: $PDG_VAL_JOB_ID"

# =============================================================================
# Job 4: Result Comparison
# =============================================================================
echo "Submitting result comparison job..."
COMPARE_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$BASELINE_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="result_comparison"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_result_comparison_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_result_comparison_%j.err

source .venv/bin/activate

echo "=== STARTING RESULT COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT validation job: $GAT_VAL_JOB_ID"
echo "Depends on PDGrapher validation job: $PDG_VAL_JOB_ID"
echo "Depends on Baseline job: $BASELINE_JOB_ID"

# Set config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Compare results of all methods
python -m src.compare_results --model gat --experiment $EXPERIMENT

echo "Result comparison completed at: \$(date)"
EOF
)

echo "Result Comparison job submitted with ID: $COMPARE_JOB_ID"

# =============================================================================
# Job 5: Cross Run Comparison
# =============================================================================
echo "Submitting cross run comparison job..."
CROSS_RUN_ID=$(sbatch --parsable --dependency=afterok:$COMPARE_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="cross_run_comparison"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/$EXPERIMENT\_cross_run_comparison_%j.out
#SBATCH --error=$SLURM_OUT_DIR/$EXPERIMENT\_cross_run_comparison_%j.err

source .venv/bin/activate

echo "=== STARTING CROSS RUN COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on result comparison job: $COMPARE_JOB_ID"


# Compare results of all methods across all runs
python cross_run_comparison.py --methods all

echo "Cross run comparison completed at: \$(date)"
EOF
)

echo "Cross Run Comparison job submitted with ID: $CROSS_RUN_ID"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=== DUAL MODEL PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo ""
echo "Pipeline structure:"
echo "  1. Data Splitting:       $DATA_SPLIT_JOB_ID"
echo "  2a. GAT Training:        $GAT_TRAIN_JOB_ID (after $DATA_SPLIT_JOB_ID)"
echo "  2b. PDGrapher Training:  $PDG_TRAIN_JOB_ID (after $DATA_SPLIT_JOB_ID)"
echo "  2c. Baseline:            $BASELINE_JOB_ID (after $DATA_SPLIT_JOB_ID)"
echo "  3a. GAT Validation:      $GAT_VAL_JOB_ID (after $GAT_TRAIN_JOB_ID)"
echo "  3b. PDGrapher Validation:$PDG_VAL_JOB_ID (after $PDG_TRAIN_JOB_ID)"
echo "  4. Result Comparison:    $COMPARE_JOB_ID (after $GAT_VAL_JOB_ID, $PDG_VAL_JOB_ID, $BASELINE_JOB_ID)"
echo "  5. Cross Run Comparison: $CROSS_RUN_ID (after $COMPARE_JOB_ID)"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $DATA_SPLIT_JOB_ID,$GAT_TRAIN_JOB_ID,$PDG_TRAIN_JOB_ID,$BASELINE_JOB_ID,$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$COMPARE_JOB_ID,$CROSS_RUN_ID"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel $DATA_SPLIT_JOB_ID $GAT_TRAIN_JOB_ID $PDG_TRAIN_JOB_ID $BASELINE_JOB_ID $GAT_VAL_JOB_ID $PDG_VAL_JOB_ID $COMPARE_JOB_ID $CROSS_RUN_ID"
echo ""
echo "Check logs in: $SLURM_OUT_DIR"
