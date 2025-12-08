#!/bin/bash

# Dual Model Pipeline Runner (reuse existing shared data)
# Same as run_all_experiments_dual.sbatch but skips shared data_creation
# Assumes data/shared/{fingerprint} directories already exist
# Usage: bash slurm/run_all_experiments_dual_skip_data_creation.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"
CONFIG_DIR="$BASE_DIR/configs/experiments"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx25,gx27,gx28,gx29"

echo "=== DUAL MODEL PIPELINE RUNNER (SKIP DATA CREATION) ==="
echo "Models: PDGrapher, PDGrapherNoGNN"
echo "Base directory: $BASE_DIR"
echo "Config directory: $CONFIG_DIR"
echo "Start time: $(date)"
echo ""

# Create output directories
mkdir -p $SLURM_OUT_DIR
mkdir -p $TIMING_LOG_DIR

# Check if config directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "ERROR: Config directory $CONFIG_DIR does not exist!"
    exit 1
fi

# Discover all experiment configs (first depth only)
echo "Discovering experiment configurations..."
EXPERIMENT_CONFIGS=$(find $CONFIG_DIR -maxdepth 1 -name "*.yaml" -type f | sort)

if [ -z "$EXPERIMENT_CONFIGS" ]; then
    echo "ERROR: No experiment configurations found in $CONFIG_DIR"
    exit 1
fi

echo "Found experiment configurations:"
for config in $EXPERIMENT_CONFIGS; do
    experiment_name=$(basename "$config" .yaml)
    echo "  - $experiment_name"
done
echo ""

# Create temporary files for tracking
EXPERIMENT_JOBS_FILE=$(mktemp)
FINGERPRINT_MAP_FILE=$(mktemp)

# Clean up temp files on exit
trap "rm -f $EXPERIMENT_JOBS_FILE $FINGERPRINT_MAP_FILE" EXIT

echo "Analyzing data sharing requirements and verifying shared data presence..."

# Step 1: Calculate fingerprints for all experiments and group by shared data
python3 << PYTHON_EOF
import sys
import os
import yaml
from pathlib import Path

# Set a dummy experiment to avoid constants.py errors
os.environ['EXPERIMENT_NAME'] = 'dummy'

# Add src to path
sys.path.append('/sc/home/johanna.dahlkemper/source_detection_in_grns/src')
from data_utils import get_data_fingerprint

config_dir = "/sc/home/johanna.dahlkemper/source_detection_in_grns/configs/experiments"
fingerprint_map_file = "$FINGERPRINT_MAP_FILE"

# Group experiments by data fingerprint
fingerprint_groups = {}

for config_file in sorted(Path(config_dir).glob("*.yaml")):
    experiment_name = config_file.stem
    os.environ['EXPERIMENT_NAME'] = experiment_name

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    data_creation_params = config['data_creation']
    network = data_creation_params['network']
    seed = config.get('seed', 42)

    fingerprint = get_data_fingerprint(data_creation_params, network, seed)

    if fingerprint not in fingerprint_groups:
        fingerprint_groups[fingerprint] = []
    fingerprint_groups[fingerprint].append(experiment_name)

    print(f"  {experiment_name}: {fingerprint[:12]}...")

with open(fingerprint_map_file, 'w') as f:
    for fingerprint, experiments in fingerprint_groups.items():
        f.write(f"{fingerprint}:{','.join(experiments)}\n")

print("\nData sharing analysis complete:")
print(f"  - {len(fingerprint_groups)} unique data scenarios")
print(f"  - {sum(len(exps) for exps in fingerprint_groups.values())} total experiments")
for fingerprint, experiments in fingerprint_groups.items():
    if len(experiments) > 1:
        print(f"  - {fingerprint[:12]}... shared by: {', '.join(experiments)}")
PYTHON_EOF

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to analyze experiment configurations"
    exit 1
fi

echo ""
echo "Verifying shared data directories exist (skipping data creation)..."
MISSING_SHARED_DATA=0
while IFS=':' read -r fingerprint experiments; do
    SHARED_DIR="$BASE_DIR/data/shared/$fingerprint"
    if [ ! -d "$SHARED_DIR" ]; then
        echo "ERROR: Missing shared data directory $SHARED_DIR (needed for: $experiments)"
        MISSING_SHARED_DATA=1
    else
        echo "  Found shared data: $SHARED_DIR (for $experiments)"
    fi
done < $FINGERPRINT_MAP_FILE

echo ""
if [ $MISSING_SHARED_DATA -ne 0 ]; then
    echo "ERROR: One or more shared data directories are missing. Aborting submissions."
    exit 1
fi

echo "Submitting dual-model experiment jobs using existing shared data..."

# Submit individual experiments that rely on existing shared data
while IFS=':' read -r fingerprint experiments; do
    IFS=',' read -ra EXP_ARRAY <<< "$experiments"
    for experiment in "${EXP_ARRAY[@]}"; do
        echo "  Submitting dual-model experiment: $experiment (shared data fingerprint: ${fingerprint:0:12}...)"

        EXP_JOB_ID=$(SKIP_SHARED_DATA_DEPENDENCY=1 DATA_FINGERPRINT=$fingerprint EXPERIMENT_NAME=$experiment bash slurm/run_single_experiment_dual.sbatch 2>/dev/null | grep "Job ID:" | grep -o '[0-9]\+' | head -1 || echo "ERROR")

        if [ "$EXP_JOB_ID" = "ERROR" ]; then
            echo "    ERROR: Failed to submit dual-model experiment $experiment"
            continue
        fi

        echo "    Experiment $experiment dual-model pipeline started with first job ID: $EXP_JOB_ID"
        echo "$experiment:$EXP_JOB_ID" >> $EXPERIMENT_JOBS_FILE
    done
done < $FINGERPRINT_MAP_FILE

echo ""
echo "=== DUAL MODEL PIPELINE SUBMISSION SUMMARY (NO DATA CREATION) ==="

echo "Dual Model Experiment Jobs:"
if [ -s "$EXPERIMENT_JOBS_FILE" ]; then
    while IFS=':' read -r experiment exp_job_id; do
        echo "  - Experiment $experiment: Job $exp_job_id (using pre-created shared data)"
    done < $EXPERIMENT_JOBS_FILE
else
    echo "  No experiment jobs were submitted."
fi

echo ""
echo "Monitor all jobs with:"
echo "  squeue -u $USER"

echo ""
echo "Check experiment logs:"
echo "  ls -la $SLURM_OUT_DIR/*_graph_pert_*"

echo ""
echo "Cancel all jobs if needed:"
experiment_jobs=$(awk -F':' '{print $2}' $EXPERIMENT_JOBS_FILE 2>/dev/null | tr '\n' ' ' || echo "")
echo "  scancel $experiment_jobs"
echo ""
