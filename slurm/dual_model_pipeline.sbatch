#!/bin/bash

# Dual Model Pipeline Runner
# Runs complete pipeline for both GAT and PDGrapher models
# Data creation is shared, but processing/training/validation are separate

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"

# Experiment configuration (can be overridden via environment)
EXPERIMENT=${EXPERIMENT_NAME:-testrun}

# Create output directory if it doesn't exist
mkdir -p $SLURM_OUT_DIR

echo "Starting dual model pipeline..."
echo "Base directory: $BASE_DIR"
echo "Experiment: $EXPERIMENT"

# =============================================================================
# Job 1: Data Creation (shared for both models)
# =============================================================================
echo "Submitting shared data creation job..."
DATA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="dual_data_creation"
#SBATCH --mem=32G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=3:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_data_creation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_data_creation_%j.err

source .venv/bin/activate

echo "=== STARTING SHARED DATA CREATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"

# Use GAT config for data creation (both models use same data params)
export EXPERIMENT_NAME=$EXPERIMENT

# Create raw data (shared between models)
python -m src.data_creation --experiment $EXPERIMENT

echo "Data creation completed at: \$(date)"
EOF
)

echo "Data creation job submitted with ID: $DATA_JOB_ID"

# =============================================================================
# Job 2a: GAT Data Processing (depends on data creation)
# =============================================================================
echo "Submitting GAT data processing job..."
GAT_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_gat_processing"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_gat_processing_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_gat_processing_%j.err

source .venv/bin/activate

echo "=== STARTING GAT DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on data job: $DATA_JOB_ID"

# Set GAT config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Process data for GAT
python -m src.data_processing --model gat --experiment $EXPERIMENT

echo "GAT data processing completed at: \$(date)"
EOF
)

echo "GAT data processing job submitted with ID: $GAT_PROC_JOB_ID"

# =============================================================================
# Job 2b: PDGrapher Data Processing (depends on data creation)
# =============================================================================
echo "Submitting PDGrapher data processing job..."
PDG_PROC_JOB_ID=$(sbatch --parsable --dependency=afterok:$DATA_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_pdg_processing"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=1:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_pdg_processing_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_pdg_processing_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER DATA PROCESSING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on data job: $DATA_JOB_ID"

# Set PDGrapher config
export MODEL_TYPE=pdgrapher
export EXPERIMENT_NAME=$EXPERIMENT

# Process data for PDGrapher
python -m src.data_processing --model pdgrapher --experiment $EXPERIMENT

echo "PDGrapher data processing completed at: \$(date)"
EOF
)

echo "PDGrapher data processing job submitted with ID: $PDG_PROC_JOB_ID"

# =============================================================================
# Job 3a: GAT Training (depends on GAT processing)
# =============================================================================
echo "Submitting GAT training job..."
GAT_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_gat_training"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_gat_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_gat_training_%j.err

source .venv/bin/activate

echo "=== STARTING GAT TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT processing job: $GAT_PROC_JOB_ID"

# Set GAT config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Train GAT model
python -m src.training --model gat --experiment $EXPERIMENT

echo "GAT training completed at: \$(date)"
EOF
)

echo "GAT training job submitted with ID: $GAT_TRAIN_JOB_ID"

# =============================================================================
# Job 3b: PDGrapher Training (depends on PDGrapher processing)
# =============================================================================
echo "Submitting PDGrapher training job..."
PDG_TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_PROC_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_pdg_training"
#SBATCH --mem=16G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --account=sci-renard
#SBATCH --partition=gpu
#SBATCH --time=4:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_pdg_training_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_pdg_training_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER TRAINING ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher processing job: $PDG_PROC_JOB_ID"

# Set PDGrapher config
export MODEL_TYPE=pdgrapher
export EXPERIMENT_NAME=$EXPERIMENT

# Train PDGrapher model
python -m src.train_pdgrapher --model pdgrapher --experiment $EXPERIMENT

echo "PDGrapher training completed at: \$(date)"
EOF
)

echo "PDGrapher training job submitted with ID: $PDG_TRAIN_JOB_ID"

# =============================================================================
# Job 4a: GAT Validation (depends on GAT training)
# =============================================================================
echo "Submitting GAT validation job..."
GAT_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_gat_validation"
#SBATCH --time=30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_gat_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_gat_validation_%j.err

source .venv/bin/activate

echo "=== STARTING GAT VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT training job: $GAT_TRAIN_JOB_ID"

# Set GAT config
export MODEL_TYPE=gat
export EXPERIMENT_NAME=$EXPERIMENT

# Validate GAT model
python -m src.validation --model gat --experiment $EXPERIMENT

echo "GAT validation completed at: \$(date)"
EOF
)

echo "GAT validation job submitted with ID: $GAT_VAL_JOB_ID"

# =============================================================================
# Job 4b: PDGrapher Validation (depends on PDGrapher training)
# =============================================================================
echo "Submitting PDGrapher validation job..."
PDG_VAL_JOB_ID=$(sbatch --parsable --dependency=afterok:$PDG_TRAIN_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_pdg_validation"
#SBATCH --time=30:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_pdg_validation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_pdg_validation_%j.err

source .venv/bin/activate

echo "=== STARTING PDGRAPHER VALIDATION ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on PDGrapher training job: $PDG_TRAIN_JOB_ID"

# Set PDGrapher config
export MODEL_TYPE=pdgrapher
export EXPERIMENT_NAME=$EXPERIMENT

# Validate PDGrapher model
python -m src.validation --model pdgrapher --experiment $EXPERIMENT

echo "PDGrapher validation completed at: \$(date)"
EOF
)

echo "PDGrapher validation job submitted with ID: $PDG_VAL_JOB_ID"

# =============================================================================
# Job 5: Summary Report (depends on both validations)
# =============================================================================
echo "Submitting summary report job..."
SUMMARY_JOB_ID=$(sbatch --parsable --dependency=afterok:$GAT_VAL_JOB_ID:$PDG_VAL_JOB_ID <<EOF
#!/bin/bash
#SBATCH --job-name="dual_summary"
#SBATCH --time=10:00
#SBATCH --partition=cpu
#SBATCH --mem=4G
#SBATCH --ntasks=1
#SBATCH --account=sci-renard
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/dual_summary_%j.out
#SBATCH --error=$SLURM_OUT_DIR/dual_summary_%j.err

source .venv/bin/activate

echo "=== GENERATING DUAL MODEL SUMMARY ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Experiment: $EXPERIMENT"
echo "Depends on GAT validation: $GAT_VAL_JOB_ID"
echo "Depends on PDGrapher validation: $PDG_VAL_JOB_ID"

# Create summary of results from both models
echo "Creating comparison summary for experiment: $EXPERIMENT"
echo "Results available in:"
echo "  - reports/GAT/$EXPERIMENT"
echo "  - reports/PDGrapher/$EXPERIMENT"

echo "=== DUAL MODEL PIPELINE COMPLETED ==="
echo "Completion time: \$(date)"
EOF
)

echo "Summary job submitted with ID: $SUMMARY_JOB_ID"

# =============================================================================
# Final Summary
# =============================================================================
echo ""
echo "=== DUAL MODEL PIPELINE SUBMISSION SUMMARY ==="
echo "Experiment: $EXPERIMENT"
echo ""
echo "Pipeline structure:"
echo "  1. Data Creation:        $DATA_JOB_ID"
echo "  2a. GAT Processing:      $GAT_PROC_JOB_ID (after $DATA_JOB_ID)"
echo "  2b. PDGrapher Processing:$PDG_PROC_JOB_ID (after $DATA_JOB_ID)"
echo "  3a. GAT Training:        $GAT_TRAIN_JOB_ID (after $GAT_PROC_JOB_ID)"
echo "  3b. PDGrapher Training:  $PDG_TRAIN_JOB_ID (after $PDG_PROC_JOB_ID)"
echo "  4a. GAT Validation:      $GAT_VAL_JOB_ID (after $GAT_TRAIN_JOB_ID)"
echo "  4b. PDGrapher Validation:$PDG_VAL_JOB_ID (after $PDG_TRAIN_JOB_ID)"
echo "  5. Summary Report:       $SUMMARY_JOB_ID (after both validations)"
echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"
echo "  squeue -j $DATA_JOB_ID,$GAT_PROC_JOB_ID,$PDG_PROC_JOB_ID,$GAT_TRAIN_JOB_ID,$PDG_TRAIN_JOB_ID,$GAT_VAL_JOB_ID,$PDG_VAL_JOB_ID,$SUMMARY_JOB_ID"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel $DATA_JOB_ID $GAT_PROC_JOB_ID $PDG_PROC_JOB_ID $GAT_TRAIN_JOB_ID $PDG_TRAIN_JOB_ID $GAT_VAL_JOB_ID $PDG_VAL_JOB_ID $SUMMARY_JOB_ID"
echo ""
echo "Check logs in: $SLURM_OUT_DIR"
echo ""
echo "Expected total runtime: ~6-8 hours"
