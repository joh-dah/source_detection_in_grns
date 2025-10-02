#!/usr/bin/env python3
"""
GPU Memory Monitor - Check GPU memory usage during training
"""
import subprocess
import time
import sys

def get_gpu_memory():
    """Get GPU memory usage information."""
    try:
        # Run nvidia-smi and parse output
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            
            for i, line in enumerate(lines):
                used, total, free = map(int, line.split(', '))
                used_gb = used / 1024
                total_gb = total / 1024
                free_gb = free / 1024
                usage_percent = (used / total) * 100
                
                gpu_info.append({
                    'gpu_id': i,
                    'used_gb': used_gb,
                    'total_gb': total_gb,
                    'free_gb': free_gb,
                    'usage_percent': usage_percent
                })
            
            return gpu_info
        else:
            return None
            
    except Exception as e:
        print(f"Error getting GPU info: {e}")
        return None

def monitor_gpu(interval=5, duration=None):
    """Monitor GPU memory usage."""
    print("🔍 GPU Memory Monitor")
    print("=" * 50)
    
    start_time = time.time()
    
    while True:
        gpu_info = get_gpu_memory()
        
        if gpu_info:
            current_time = time.strftime("%H:%M:%S")
            print(f"\n⏰ {current_time}")
            
            for gpu in gpu_info:
                print(f"📊 GPU {gpu['gpu_id']}:")
                print(f"   Used:  {gpu['used_gb']:.1f} GB ({gpu['usage_percent']:.1f}%)")
                print(f"   Free:  {gpu['free_gb']:.1f} GB")
                print(f"   Total: {gpu['total_gb']:.1f} GB")
                
                # Memory warnings
                if gpu['usage_percent'] > 90:
                    print(f"   ⚠️  WARNING: High memory usage!")
                elif gpu['usage_percent'] > 80:
                    print(f"   ⚡ High memory usage")
        else:
            print("❌ No GPU information available")
        
        # Check duration
        if duration and (time.time() - start_time) >= duration:
            break
            
        time.sleep(interval)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "--once":
            # Single check
            gpu_info = get_gpu_memory()
            if gpu_info:
                for gpu in gpu_info:
                    print(f"GPU {gpu['gpu_id']}: {gpu['used_gb']:.1f}/{gpu['total_gb']:.1f} GB ({gpu['usage_percent']:.1f}% used)")
            else:
                print("No GPU available")
        else:
            print("Usage: python gpu_monitor.py [--once]")
    else:
        # Continuous monitoring
        try:
            monitor_gpu(interval=2)
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")
