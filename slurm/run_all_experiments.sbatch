#!/bin/bash

# Master Pipeline Runner
# Automatically discovers all experiments in configs/experiments/ and runs them
# Groups experiments by shared data requirements and creates optimal execution plan
# Usage: bash slurm/run_all_experiments.sbatch

# Base directory
BASE_DIR="/sc/home/johanna.dahlkemper/source_detection_in_grns"
SLURM_OUT_DIR="$BASE_DIR/slurm_out"
TIMING_LOG_DIR="$BASE_DIR/timing_logs"
CONFIG_DIR="$BASE_DIR/configs/experiments"

# GPU configuration (excluding gx26 due to GPU errors)
COMPATIBLE_GPU_NODES="gx01,gx02,gx03,gx04,gx05,gx06,gx07,gx08,gx09,gx10,gx12,gx13,gx21,gx25,gx27,gx28,gx29"

echo "=== MASTER PIPELINE RUNNER ==="
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
SHARED_DATA_JOBS_FILE=$(mktemp)
EXPERIMENT_JOBS_FILE=$(mktemp)
FINGERPRINT_MAP_FILE=$(mktemp)

# Clean up temp files on exit
trap "rm -f $SHARED_DATA_JOBS_FILE $EXPERIMENT_JOBS_FILE $FINGERPRINT_MAP_FILE" EXIT

echo "Analyzing data sharing requirements..."

# Step 1: Calculate fingerprints for all experiments and group by shared data
python3 << PYTHON_EOF
import sys
import os
import yaml
import hashlib
import json
from pathlib import Path

# Add src to path
sys.path.append('/sc/home/johanna.dahlkemper/source_detection_in_grns/src')
from data_utils import get_data_fingerprint

config_dir = "/sc/home/johanna.dahlkemper/source_detection_in_grns/configs/experiments"
fingerprint_map_file = "$FINGERPRINT_MAP_FILE"

# Group experiments by data fingerprint
fingerprint_groups = {}

for config_file in sorted(Path(config_dir).glob("*.yaml")):
    experiment_name = config_file.stem
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        # Extract required parameters for fingerprint
        data_creation_params = config['data_creation']
        network = data_creation_params['network']
        seed = config.get('seed', 42)  # Default seed if not specified
        
        # Calculate data fingerprint
        fingerprint = get_data_fingerprint(data_creation_params, network, seed)
        
        if fingerprint not in fingerprint_groups:
            fingerprint_groups[fingerprint] = []
        fingerprint_groups[fingerprint].append(experiment_name)
        
        print(f"  {experiment_name}: {fingerprint[:12]}...")
        
    except Exception as e:
        print(f"ERROR: Failed to process {experiment_name}: {e}")
        sys.exit(1)

# Write fingerprint mapping to file
with open(fingerprint_map_file, 'w') as f:
    for fingerprint, experiments in fingerprint_groups.items():
        f.write(f"{fingerprint}:{','.join(experiments)}\n")

print(f"\nData sharing analysis complete:")
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
echo "Submitting shared data creation jobs..."

# Step 2: Submit shared data creation jobs
while IFS=':' read -r fingerprint experiments; do
    # Use first experiment in group as representative for data creation
    representative_experiment=$(echo $experiments | cut -d',' -f1)
    
    echo "  Creating shared data for fingerprint ${fingerprint:0:12}... (using $representative_experiment)"
    
    # Submit shared data creation job
    DATA_JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name="shared_data_${fingerprint:0:12}"
#SBATCH --mem=512G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --account=sci-renard
#SBATCH --partition=cpu
#SBATCH --time=5-00:00:00
#SBATCH --chdir=$BASE_DIR
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=johanna.dahlkemper@student.hpi.de
#SBATCH --output=$SLURM_OUT_DIR/shared_data_creation_%j.out
#SBATCH --error=$SLURM_OUT_DIR/shared_data_creation_%j.err

source .venv/bin/activate

# JAX/XLA Performance Optimizations
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.8
export JAX_PLATFORM_NAME=cpu

echo "=== CREATING SHARED DATA ==="
echo "Job ID: \$SLURM_JOB_ID"
echo "Start time: \$(date)"
echo "Fingerprint: $fingerprint"
echo "Representative experiment: $representative_experiment"
echo "Will be used by: $experiments"

export EXPERIMENT_NAME=$representative_experiment

# === 1. Data Creation ===
echo "--- Step 1: Data Creation ---"
START_TIME_DATA=\$(date +%s)

# Create shared data (this will be stored in data/shared/{fingerprint}/)
python -m src.data_creation --experiment $representative_experiment --model gat

END_TIME_DATA=\$(date +%s)
DATA_DURATION=\$((\$END_TIME_DATA - \$START_TIME_DATA))
echo "Data creation completed in \$DATA_DURATION seconds"

# === 2. Data Splitting ===
echo "--- Step 2: Data Splitting ---"
START_TIME_SPLIT=\$(date +%s)

python -m src.create_splits --experiment $representative_experiment --model gat

END_TIME_SPLIT=\$(date +%s)
SPLIT_DURATION=\$((\$END_TIME_SPLIT - \$START_TIME_SPLIT))
echo "Data splitting completed in \$SPLIT_DURATION seconds"

# === Summary ===
TOTAL_DURATION=\$((\$END_TIME_SPLIT - \$START_TIME_DATA))
echo ""
echo "=== SHARED DATA CREATION COMPLETE ==="
echo "Configuration: $representative_experiment"
echo "Total time: \$TOTAL_DURATION seconds"
echo "Breakdown:"
echo "  - Data Creation: \$DATA_DURATION seconds"
echo "  - Data Splitting: \$SPLIT_DURATION seconds"
echo ""
echo "This shared data is now ready for parallel experiment execution!"
echo "Experiments will handle their own graph perturbation and data processing."

echo "Shared data creation completed at: \$(date)"
echo "Data stored in: data/shared/$fingerprint/"
EOF
)
    
    # Store job ID and fingerprint mapping
    echo "$fingerprint:$DATA_JOB_ID:$experiments" >> $SHARED_DATA_JOBS_FILE
    echo "    Job ID: $DATA_JOB_ID"
    
done < $FINGERPRINT_MAP_FILE

echo ""
echo "Submitting individual experiment jobs..."

# Step 3: Submit individual experiments that depend on their shared data
while IFS=':' read -r fingerprint data_job_id experiments; do
    # Submit job for each experiment that uses this shared data
    IFS=',' read -ra EXP_ARRAY <<< "$experiments"
    for experiment in "${EXP_ARRAY[@]}"; do
        echo "  Submitting experiment: $experiment (depends on shared data job: $data_job_id)"
        
        # Submit single experiment job with dependency on shared data creation
        EXP_JOB_ID=$(SHARED_DATA_JOB_ID=$data_job_id EXPERIMENT_NAME=$experiment bash slurm/run_single_experiment_with_dependency.sbatch 2>/dev/null | grep "Job ID:" | grep -o '[0-9]\+' | head -1 || echo "ERROR")
        
        if [ "$EXP_JOB_ID" = "ERROR" ]; then
            echo "    ERROR: Failed to submit experiment $experiment"
            continue
        fi
        
        echo "    Experiment $experiment pipeline started with first job ID: $EXP_JOB_ID"
        echo "$experiment:$EXP_JOB_ID:$data_job_id" >> $EXPERIMENT_JOBS_FILE
        
    done
done < $SHARED_DATA_JOBS_FILE

echo ""
echo "=== MASTER PIPELINE SUBMISSION SUMMARY ==="

echo ""
echo "Shared Data Creation Jobs:"
while IFS=':' read -r fingerprint data_job_id experiments; do
    echo "  - Fingerprint ${fingerprint:0:12}...: Job $data_job_id (for: $experiments)"
done < $SHARED_DATA_JOBS_FILE

echo ""
echo "Individual Experiment Jobs:"
while IFS=':' read -r experiment exp_job_id data_job_id; do
    echo "  - Experiment $experiment: Job $exp_job_id (depends on shared data: $data_job_id)"
done < $EXPERIMENT_JOBS_FILE 2>/dev/null || echo "  (Individual jobs will be submitted after shared data creation)"

echo ""
echo "Monitor all jobs with:"
echo "  squeue -u \$USER"

echo ""
echo "Check shared data creation logs:"
echo "  ls -la $SLURM_OUT_DIR/shared_data_*"

echo ""
echo "Check experiment logs:"
echo "  ls -la $SLURM_OUT_DIR/*_graph_pert_*"

echo ""
echo "Cancel all jobs if needed:"
shared_jobs=$(awk -F':' '{print $2}' $SHARED_DATA_JOBS_FILE | tr '\n' ' ')
experiment_jobs=$(awk -F':' '{print $2}' $EXPERIMENT_JOBS_FILE 2>/dev/null | tr '\n' ' ' || echo "")
all_jobs="$shared_jobs $experiment_jobs"
echo "  scancel $all_jobs"
echo ""
