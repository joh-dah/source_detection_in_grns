#!/bin/bash

# Retraining Pipeline - Only runs training, validation, and comparison stages
# Assumes all data processing (shared data, graph perturbation, data processing) is already complete
# Usage: EXPERIMENT_NAME=my_experiment bash slurm/rerun_training_pipeline.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# GPU configuration
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx21,gx25,gx27,gx28,gx29"

# Get experiment name
EXPERIMENT=${EXPERIMENT_NAME}

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT_NAME must be provided"
    echo "Usage: EXPERIMENT_NAME=my_experiment bash slurm/rerun_training_pipeline.sbatch"
    exit 1
fi

echo "Starting retraining pipeline for experiment: $EXPERIMENT"
echo "Assumes data processing is already complete"

# Append to existing timing log (don't overwrite)
if [ ! -f "$TIMING_LOG_DIR/${EXPERIMENT}_timing.csv" ]; then
    echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
fi

# =============================================================================
# Job 1: GAT Training (no dependencies - starts immediately)
# =============================================================================
echo "Submitting GAT training job..."
GAT_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="retrain_gat_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_retrain_gat_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_retrain_gat_%j.err

source .venv/bin/activate

echo "=== RETRAINING GAT ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT

# Train GAT model
python -m src.training --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "GAT retraining completed at: \$(date)"
echo "$EXPERIMENT,gat,retraining,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

# =============================================================================
# Job 2: PDGrapher Training (no dependencies - starts immediately)
# =============================================================================
echo "Submitting PDGrapher training job..."
PDG_TRAIN_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="retrain_pdg_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_retrain_pdg_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_retrain_pdg_%j.err

source .venv/bin/activate

echo "=== RETRAINING PDGRAPHER ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT

# Train PDGrapher model
python -m src.train_pdgrapher --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher retraining completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,retraining,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

# =============================================================================
# Job 3: GAT Validation (depends on GAT training)
# =============================================================================
echo "Submitting GAT validation job..."
GAT_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="reval_gat_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_reval_gat_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_reval_gat_%j.err

source .venv/bin/activate

echo "=== RE-VALIDATING GAT ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT

# Validate GAT model
python -m src.validation --experiment $EXPERIMENT --model gat

END_TIME=\$(date +%s)
echo "GAT re-validation completed at: \$(date)"
echo "$EXPERIMENT,gat,revalidation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

# =============================================================================
# Job 4: PDGrapher Validation (depends on PDGrapher training)
# =============================================================================
echo "Submitting PDGrapher validation job..."
PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="reval_pdg_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_reval_pdg_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_reval_pdg_%j.err

source .venv/bin/activate

echo "=== RE-VALIDATING PDGRAPHER ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT

# Validate PDGrapher model
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

END_TIME=\$(date +%s)
echo "PDGrapher re-validation completed at: \$(date)"
echo "$EXPERIMENT,pdgrapher,revalidation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

# =============================================================================
# Job 5: Result Comparison (depends on both validations)
# =============================================================================
echo "Submitting result comparison job..."
RESULT_COMPARISON_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="recompare_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_recompare_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_recompare_%j.err

source .venv/bin/activate

echo "=== RE-RUNNING RESULT COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

START_TIME=\$(date +%s)
export EXPERIMENT_NAME=$EXPERIMENT

# Run result comparison for this experiment
python -m src.compare_results --model gat --exp_id $EXPERIMENT

END_TIME=\$(date +%s)
echo "Result re-comparison completed at: \$(date)"
echo "$EXPERIMENT,result_comparison,recomparison,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo ""
echo "=== RETRAINING PIPELINE SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo "Pipeline:"
echo "  1. GAT Training:         $GAT_TRAIN_JOB_ID (starts immediately)"
echo "  2. PDGrapher Training:   $PDG_TRAIN_JOB_ID (starts immediately)" 
echo "  3. GAT Validation:       $GAT_VAL_JOB_ID (after $GAT_TRAIN_JOB_ID)"
echo "  4. PDGrapher Validation: $PDG_VAL_JOB_ID (after $PDG_TRAIN_JOB_ID)"
echo "  5. Result Comparison:    $RESULT_COMPARISON_JOB_ID (after both validations)"
echo ""
echo "Monitor with: squeue -j $GAT_TRAIN_JOB_ID,$PDG_TRAIN_JOB_ID,$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$RESULT_COMPARISON_JOB_ID"