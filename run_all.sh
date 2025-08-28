#!/bin/bash

# Quick Start Script for Running All Experiments
# This script makes it super easy to run all your experiments

echo "=== QUICK START: RUN ALL EXPERIMENTS ==="
echo ""

# Check if we're in the right directory
if [ ! -f "slurm/run_all_experiments.sbatch" ]; then
    echo "ERROR: Please run this script from the project root directory"
    echo "Current directory: $(pwd)"
    exit 1
fi

# Check if configs/experiments directory exists and has files
if [ ! -d "configs/experiments" ]; then
    echo "ERROR: configs/experiments directory not found!"
    exit 1
fi

experiment_count=$(find configs/experiments -name "*.yaml" -type f | wc -l)
if [ $experiment_count -eq 0 ]; then
    echo "ERROR: No experiment configurations found in configs/experiments/"
    echo "Please create YAML files in configs/experiments/ first"
    exit 1
fi

echo "Found $experiment_count experiment configuration(s) to run"
echo ""

# Show what will be executed
echo "This will:"
echo "  1. Analyze all experiment configs in configs/experiments/"
echo "  2. Group experiments that share the same data_creation parameters"
echo "  3. Submit shared data creation jobs for each unique data scenario"
echo "  4. Submit individual experiment pipelines that depend on shared data"
echo "  5. Each experiment runs: graph perturbation → data processing → baseline + training → validation → result comparison"
echo ""

# Confirm execution
read -p "Do you want to proceed? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo ""
echo "Starting master pipeline..."

# Run the master pipeline
bash slurm/run_all_experiments.sbatch

echo ""
echo "=== QUICK START COMPLETE ==="
echo ""
echo "Your experiments are now submitted to the SLURM queue!"
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER"
echo ""
echo "Check logs:"
echo "  ls -la slurm_out/"
echo ""
echo "Check timing results:"
echo "  ls -la timing_logs/"
echo ""
echo "Cancel all jobs if needed:"
echo "  scancel -u \$USER"
echo ""
