#!/usr/bin/env python3
"""
Resource monitoring utility for PDGrapher training
"""
import psutil
import torch
import time
import sys
from datetime import datetime

def get_gpu_memory():
    if torch.cuda.is_available():
        return {
            'allocated': torch.cuda.memory_allocated() / 1e9,
            'cached': torch.cuda.memory_reserved() / 1e9,
            'total': torch.cuda.get_device_properties(0).total_memory / 1e9
        }
    return None

def monitor_resources(interval=5, duration=None):
    """Monitor system resources"""
    print("=== Resource Monitor ===")
    print(f"Started at: {datetime.now()}")
    print(f"Monitoring every {interval} seconds")
    if duration:
        print(f"Will run for {duration} seconds")
    print()
    
    start_time = time.time()
    
    try:
        while True:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"[{elapsed:6.1f}s] CPU: {cpu_percent:5.1f}% | "
                  f"RAM: {memory.percent:5.1f}% ({memory.used/1e9:.1f}/{memory.total/1e9:.1f} GB)")
            
            # GPU if available
            gpu_mem = get_gpu_memory()
            if gpu_mem:
                gpu_util = gpu_mem['allocated'] / gpu_mem['total'] * 100
                print(f"         GPU: {gpu_util:5.1f}% | "
                      f"VRAM: {gpu_mem['allocated']:.1f}/{gpu_mem['total']:.1f} GB")
            
            # Check if duration exceeded
            if duration and elapsed >= duration:
                break
                
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user")
    
    print(f"\nTotal monitoring time: {elapsed:.1f} seconds")

def estimate_resources():
    """Estimate resource requirements for the training"""
    print("=== Resource Estimation ===")
    
    # System info
    cpu_count = psutil.cpu_count()
    memory = psutil.virtual_memory()
    
    print(f"Available CPUs: {cpu_count}")
    print(f"Available RAM: {memory.total/1e9:.1f} GB")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
    else:
        print("No GPU available")
    
    print("\n=== Recommendations ===")
    print("For Graph Neural Network training:")
    print("- Minimum RAM: 8 GB")
    print("- Recommended RAM: 16+ GB")
    print("- GPU: Highly recommended (10x+ speedup)")
    print("- GPU Memory: 4+ GB for small graphs, 8+ GB for larger graphs")
    
    print("\n=== SLURM Resource Request Template ===")
    print("For your current system, consider:")
    print("  #SBATCH --nodes=1")
    print("  #SBATCH --ntasks-per-node=1")
    print("  #SBATCH --cpus-per-task=4")
    print("  #SBATCH --mem=16G")
    print("  #SBATCH --gres=gpu:1")
    print("  #SBATCH --time=02:00:00  # Adjust based on your training time")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        duration = int(sys.argv[2]) if len(sys.argv) > 2 else None
        monitor_resources(duration=duration)
    else:
        estimate_resources()
