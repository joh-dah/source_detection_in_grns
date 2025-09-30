#!/usr/bin/env python3
"""
Script to compare specific methods across different runs.
Creates a combined bar chart showing accuracy for each method in each run.

Usage Examples:
    # Compare default methods (pdgrapher and overlap)
    python cross_run_comparison.py
    
    # Compare specific methods
    python cross_run_comparison.py --methods pdgrapher overlap gat
    
    # Compare all available methods
    python cross_run_comparison.py --methods all
    
    # Save to custom directory
    python cross_run_comparison.py --methods all --output-dir custom_output/
    
    # Create network scaling line chart for 500-node experiments
    python cross_run_comparison.py --network-scaling
    
    # Create runtime scaling line chart for 500-node experiments
    python cross_run_comparison.py --runtime-scaling
    
    # Create both network and runtime scaling charts
    python cross_run_comparison.py --network-scaling --runtime-scaling
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.compare_results import compare_methods_across_runs, create_network_scaling_line_chart, create_runtime_scaling_line_chart


def main():
    parser = argparse.ArgumentParser(description='Compare methods across different runs')
    parser.add_argument('--methods', nargs='+', default=['all'],
                       help='Methods to compare. Use "all" to compare all available methods, '
                            'or specify individual methods like "pdgrapher overlap gat" '
                            '(default: all)')
    parser.add_argument('--output-dir', default='reports',
                       help='Output directory for plots (default: reports)')
    parser.add_argument('--network-scaling', action='store_true',
                       help='Create network scaling line chart for 500-node experiments (PDGrapher & PDGrapherNoGNN)')
    parser.add_argument('--runtime-scaling', action='store_true',
                       help='Create runtime scaling line chart for 500-node experiments (PDGrapher & PDGrapherNoGNN)')
    
    args = parser.parse_args()
    
    if args.network_scaling:
        print("Creating network scaling line chart...")
        create_network_scaling_line_chart(args.output_dir)
    
    if args.runtime_scaling:
        print("Creating runtime scaling line chart...")
        create_runtime_scaling_line_chart(args.output_dir)
    
    if not args.network_scaling and not args.runtime_scaling:
        print(f"Comparing methods: {args.methods}")
        print(f"Output directory: {args.output_dir}")
        
        # Run the comparison
        compare_methods_across_runs(methods=args.methods, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
