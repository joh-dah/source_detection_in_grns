#!/bin/bash

# Single Experiment Runner Script (No Dependencies)
# Runs a single experiment when shared data is already available
# Usage: EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx21,gx25,gx27,gx28,gx29"

# Get experiment name
EXPERIMENT=${EXPERIMENT_NAME}

if [ -z "$EXPERIMENT" ]; then
    echo "ERROR: EXPERIMENT_NAME must be provided"
    echo "Usage: EXPERIMENT_NAME=my_experiment bash slurm/run_single_experiment.sbatch"
    exit 1
fi

echo "Starting single experiment pipeline..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"
echo "Assuming shared data is already available"

# Create output directories if they don't exist
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Initialize timing log with header
echo "experiment,method,stage,start_time,end_time,duration_seconds,job_id" > $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv

# =============================================================================
# Job 1: Graph Perturbation (assumes shared data exists)
# =============================================================================
echo "Submitting graph perturbation job..."
GRAPH_PERT_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="graph_perturbation_${EXPERIMENT}"
#SBATCH --mem=32G
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

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Perturb the graph
python -m src.graph_perturbation --experiment $EXPERIMENT --model gat

# Record end time and log
END_TIME=\$(date +%s)
echo "Graph perturbation completed at: \$(date)"

# Log timing for GAT, PDGrapher and baseline methods (they all share this processing step)
echo "$EXPERIMENT,gat,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_rumor,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_random,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_advanced_random,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,pdgrapher,graph_perturbation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "Graph perturbation job submitted with ID: $GRAPH_PERT_JOB_ID"

# =============================================================================
# Job 2a: GAT Data Processing (depends on graph perturbation)
# =============================================================================
echo "Submitting GAT data processing job..."
GAT_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_processing_${EXPERIMENT}"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_gat_processing_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_gat_processing_%j.err

source .venv/bin/activate

echo "=== STARTING GAT DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on graph perturbation job: $GRAPH_PERT_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Process data for GAT
python -m src.data_processing --experiment $EXPERIMENT --model gat

# Record end time and log (shared with baseline methods)
END_TIME=\$(date +%s)
echo "GAT data processing completed at: \$(date)"

# Log timing for GAT and baseline methods (they share this processing step)
echo "$EXPERIMENT,gat,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_rumor,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_random,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_advanced_random,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "GAT data processing job submitted with ID: $GAT_PROC_JOB_ID"

# =============================================================================
# Job 2b: PDGrapher Data Processing (depends on graph perturbation)
# =============================================================================
echo "Submitting PDGrapher data processing job..."
PDG_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$GRAPH_PERT_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_processing_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_processing_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_processing_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on graph perturbation job: $GRAPH_PERT_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Process data for PDGrapher
python -m src.data_processing --experiment $EXPERIMENT --model pdgrapher

# Record end time and log
END_TIME=\$(date +%s)
echo "PDGrapher data processing completed at: \$(date)"

# Log timing for PDGrapher
echo "$EXPERIMENT,pdgrapher,data_processing,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "PDGrapher data processing job submitted with ID: $PDG_PROC_JOB_ID"

# =============================================================================
# Job 3: Baseline (depends on GAT processing)
# =============================================================================
echo "Submitting baseline job..."
BASELINE_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="baseline_${EXPERIMENT}"
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_baseline_%j.err

source .venv/bin/activate

echo "=== STARTING BASELINE EVALUATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT processing job: $GAT_PROC_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Run baseline evaluation
python -m src.baseline --experiment $EXPERIMENT --model gat

# Record end time and log for all baseline methods
END_TIME=\$(date +%s)
echo "Baseline evaluation completed at: \$(date)"

# Log timing for all baseline methods (they all run in this single job)
echo "$EXPERIMENT,baseline_rumor,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_random,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
echo "$EXPERIMENT,baseline_advanced_random,evaluation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "Baseline job submitted with ID: $BASELINE_JOB_ID"

# =============================================================================
# Job 4a: GAT Training (depends on GAT processing)
# =============================================================================
echo "Submitting GAT training job..."
GAT_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_training_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_gat_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_gat_training_%j.err

source .venv/bin/activate

echo "=== STARTING GAT TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT processing job: $GAT_PROC_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Train GAT model
python -m src.training --experiment $EXPERIMENT --model gat

# Record end time and log
END_TIME=\$(date +%s)
echo "GAT training completed at: \$(date)"

# Log timing for GAT
echo "$EXPERIMENT,gat,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "GAT training job submitted with ID: $GAT_TRAIN_JOB_ID"

# =============================================================================
# Job 4b: PDGrapher Training (depends on PDGrapher processing)
# =============================================================================
echo "Submitting PDGrapher training job..."
PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_training_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_training_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher processing job: $PDG_PROC_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Train PDGrapher model
python -m src.train_pdgrapher --experiment $EXPERIMENT --model pdgrapher

# Record end time and log
END_TIME=\$(date +%s)
echo "PDGrapher training completed at: \$(date)"

# Log timing for PDGrapher
echo "$EXPERIMENT,pdgrapher,training,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "PDGrapher training job submitted with ID: $PDG_TRAIN_JOB_ID"

# =============================================================================
# Job 5a: GAT Validation (depends on GAT training)
# =============================================================================
echo "Submitting GAT validation job..."
GAT_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="gat_validation_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_gat_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_gat_validation_%j.err

source .venv/bin/activate

echo "=== STARTING GAT VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT training job: $GAT_TRAIN_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Validate GAT model
python -m src.validation --experiment $EXPERIMENT --model gat

# Record end time and log
END_TIME=\$(date +%s)
echo "GAT validation completed at: \$(date)"

# Log timing for GAT
echo "$EXPERIMENT,gat,validation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "GAT validation job submitted with ID: $GAT_VAL_JOB_ID"

# =============================================================================
# Job 5b: PDGrapher Validation (depends on PDGrapher training)
# =============================================================================
echo "Submitting PDGrapher validation job..."
PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="pdg_validation_${EXPERIMENT}"
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
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_pdg_validation_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher training job: $PDG_TRAIN_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Validate PDGrapher model
python -m src.validation --experiment $EXPERIMENT --model pdgrapher

# Record end time and log
END_TIME=\$(date +%s)
echo "PDGrapher validation completed at: \$(date)"

# Log timing for PDGrapher
echo "$EXPERIMENT,pdgrapher,validation,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "PDGrapher validation job submitted with ID: $PDG_VAL_JOB_ID"

# =============================================================================
# Job 6: Result Comparison (depends on all validation and baseline)
# =============================================================================
echo "Submitting result comparison job..."
RESULT_COMPARISON_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$BASELINE_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="result_comparison_${EXPERIMENT}"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_result_comparison_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_result_comparison_%j.err

source .venv/bin/activate

echo "=== STARTING RESULT COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT validation job: $GAT_VAL_JOB_ID"
echo "Depends on PDGrapher validation job: $PDG_VAL_JOB_ID"
echo "Depends on baseline job: $BASELINE_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

export EXPERIMENT_NAME=$EXPERIMENT

# Run result comparison for this experiment
python -m src.compare_results --model gat --experiment $EXPERIMENT

# Record end time and log
END_TIME=\$(date +%s)
echo "Result comparison completed at: \$(date)"

# Log timing for result comparison
echo "$EXPERIMENT,result_comparison,comparison,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "Result comparison job submitted with ID: $RESULT_COMPARISON_JOB_ID"

# =============================================================================
# Job 7: Cross-Run Comparison (depends on result comparison)
# =============================================================================
echo "Submitting cross-run comparison job..."
CROSS_RUN_JOB_ID=$(sbatch --parsable --dependency=afterok:$RESULT_COMPARISON_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="cross_run_${EXPERIMENT}"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=4:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/${EXPERIMENT}_cross_run_comparison_%j.out
#SBATCH --error=$SLURM_OUT_DIR/${EXPERIMENT}_cross_run_comparison_%j.err

source .venv/bin/activate

echo "=== STARTING CROSS-RUN COMPARISON ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on result comparison job: $RESULT_COMPARISON_JOB_ID"

# Record start time
START_TIME=\$(date +%s)

# Run cross-run comparison
python cross_run_comparison.py

# Record end time and log
END_TIME=\$(date +%s)
echo "Cross-run comparison completed at: \$(date)"

# Log timing for cross-run comparison
echo "$EXPERIMENT,cross_run_comparison,comparison,\$START_TIME,\$END_TIME,\$((\$END_TIME - \$START_TIME)),\$SLURM_JOB_ID" >> $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv
EOF
)

echo "Cross-run comparison job submitted with ID: $CROSS_RUN_JOB_ID"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=== SINGLE EXPERIMENT PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo ""
echo "Pipeline structure:"
echo "  1. Graph Perturbation:        $GRAPH_PERT_JOB_ID"
echo "  2a. GAT Processing:           $GAT_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  2b. PDGrapher Processing:     $PDG_PROC_JOB_ID (after $GRAPH_PERT_JOB_ID)"
echo "  3. Baseline:                  $BASELINE_JOB_ID (after $GAT_PROC_JOB_ID)"
echo "  4a. GAT Training:             $GAT_TRAIN_JOB_ID (after $GAT_PROC_JOB_ID)"
echo "  4b. PDGrapher Training:       $PDG_TRAIN_JOB_ID (after $PDG_PROC_JOB_ID)"
echo "  5a. GAT Validation:           $GAT_VAL_JOB_ID (after $GAT_TRAIN_JOB_ID)"
echo "  5b. PDGrapher Validation:     $PDG_VAL_JOB_ID (after $PDG_TRAIN_JOB_ID)"
echo "  6. Result Comparison:         $RESULT_COMPARISON_JOB_ID (after $GAT_VAL_JOB_ID, $PDG_VAL_JOB_ID, $BASELINE_JOB_ID)"
echo "  7. Cross-Run Comparison:      $CROSS_RUN_JOB_ID (after $RESULT_COMPARISON_JOB_ID)"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $GRAPH_PERT_JOB_ID,$GAT_PROC_JOB_ID,$PDG_PROC_JOB_ID,$BASELINE_JOB_ID,$GAT_TRAIN_JOB_ID,$PDG_TRAIN_JOB_ID,$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$RESULT_COMPARISON_JOB_ID,$CROSS_RUN_JOB_ID"
echo ""
echo "Check timing log:"
echo "  tail -f $TIMING_LOG_DIR/${EXPERIMENT}_timing.csv"
echo ""
echo "Cancel entire pipeline if needed:"
echo "  scancel $GRAPH_PERT_JOB_ID $GAT_PROC_JOB_ID $PDG_PROC_JOB_ID $BASELINE_JOB_ID $GAT_TRAIN_JOB_ID $PDG_TRAIN_JOB_ID $GAT_VAL_JOB_ID $PDG_VAL_JOB_ID $RESULT_COMPARISON_JOB_ID $CROSS_RUN_JOB_ID"
echo "  squeue -u johanna.dahlkemper"
echo ""
echo "Check logs in: $SLURM_OUT_DIR"
