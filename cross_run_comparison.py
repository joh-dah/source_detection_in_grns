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
"""

import argparse
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.compare_results import compare_methods_across_runs


def main():
    parser = argparse.ArgumentParser(description='Compare methods across different runs')
    parser.add_argument('--methods', nargs='+', default=['all'],
                       help='Methods to compare. Use "all" to compare all available methods, '
                            'or specify individual methods like "pdgrapher overlap gat" '
                            '(default: all)')
    parser.add_argument('--output-dir', default='reports',
                       help='Output directory for plots (default: reports)')
    
    args = parser.parse_args()
    
    print(f"Comparing methods: {args.methods}")
    print(f"Output directory: {args.output_dir}")
    
    # Run the comparison
    compare_methods_across_runs(methods=args.methods, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
