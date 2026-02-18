#!/usr/bin/env bash
set -euo pipefail

# Batch runner for pathway experiments.
# For each .yaml (and .yalm) file in configs/experiments, run:
#   EXPERIMENT_NAME=<basename> K_FOLD=10 srun slurm/pathway_experiment

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_DIR="${SCRIPT_DIR}/../configs/experiments"
RUNNER="${SCRIPT_DIR}/pathway_experiment.sbatch"

# Pre-flight checks
if ! command -v sbatch >/dev/null 2>&1; then
  echo "Error: 'sbatch' command not found in PATH." >&2
  echo "Ensure Slurm is available on this system and PATH is set." >&2
  exit 1
fi

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "Error: Config directory not found: ${CONFIG_DIR}" >&2
  exit 1
fi

if [[ ! -f "${RUNNER}" ]]; then
  echo "Error: Runner not found: ${RUNNER}" >&2
  echo "Place pathway_experiment.sbatch in ${SCRIPT_DIR} or update RUNNER accordingly." >&2
  exit 1
fi

shopt -s nullglob
files=("${CONFIG_DIR}"/*.yaml "${CONFIG_DIR}"/*.yalm)
shopt -u nullglob

if (( ${#files[@]} == 0 )); then
  echo "Error: No .yaml or .yalm files found in ${CONFIG_DIR}" >&2
  exit 1
fi

for config in "${files[@]}"; do
  filename="${config##*/}"
  experiment="${filename%.*}"
  echo "Submitting: EXPERIMENT_NAME=${experiment} K_FOLD=10 sbatch ${RUNNER}"
  sbatch --partition=cpu-batch --export=EXPERIMENT_NAME="${experiment}",K_FOLD=10 "${RUNNER}"
done
