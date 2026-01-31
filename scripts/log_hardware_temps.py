#!/usr/bin/env python3
"""Continuous hardware temperature logger for training runs.

Logs GPU, CPU, and NVMe temperatures every minute to a file.
Run in background during training to track thermal performance.

Usage:
    python log_hardware_temps.py <output_file> [--interval 60]

Example:
    python log_hardware_temps.py temps.log --interval 60 &
"""

import sys
import time
import argparse
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.hw_monitor import get_all_temps, check_temp_health

def log_temperatures(
    output_file: Path,
    interval: int = 60,
    nvme_device: str = "/dev/nvme0"
):
    """Continuously log hardware temperatures.
    
    Args:
        output_file: Path to log file
        interval: Seconds between samples (default: 60)
        nvme_device: NVMe device path (default: /dev/nvme0)
    """
    print(f"🌡️  Temperature Logger Started")
    print(f"   Output: {output_file}")
    print(f"   Interval: {interval}s")
    print(f"   Press Ctrl+C to stop\n")
    
    # Write header
    with open(output_file, 'w') as f:
        f.write("timestamp,gpu_c,cpu_c,nvme_c,gpu_status,cpu_status,nvme_status\n")
    
    try:
        while True:
            # Get temperatures
            temps = get_all_temps(nvme_device)
            statuses = check_temp_health(temps)
            
            # Current timestamp
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Format values (use 'N/A' for missing)
            gpu_str = f"{temps['gpu_temp_c']:.1f}" if temps['gpu_temp_c'] else "N/A"
            cpu_str = f"{temps['cpu_temp_c']:.1f}" if temps['cpu_temp_c'] else "N/A"
            nvme_str = f"{temps['nvme_temp_c']:.1f}" if temps['nvme_temp_c'] else "N/A"
            
            # Write to file
            with open(output_file, 'a') as f:
                f.write(f"{ts},{gpu_str},{cpu_str},{nvme_str},"
                       f"{statuses['gpu_status']},{statuses['cpu_status']},{statuses['nvme_status']}\n")
            
            # Print to console
            print(f"[{ts}] GPU: {gpu_str}°C ({statuses['gpu_status']}) | "
                  f"CPU: {cpu_str}°C ({statuses['cpu_status']}) | "
                  f"NVMe: {nvme_str}°C ({statuses['nvme_status']})")
            
            # Check for warnings
            if any(s in ['hot', 'critical'] for s in statuses.values()):
                print(f"  ⚠️  WARNING: High temperatures detected!")
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Temperature logging stopped")
        print(f"   Log saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Continuously log hardware temperatures during training"
    )
    parser.add_argument(
        "output_file",
        help="Output log file path"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Sampling interval in seconds (default: 60)"
    )
    parser.add_argument(
        "--nvme-device",
        default="/dev/nvme0",
        help="NVMe device path (default: /dev/nvme0)"
    )
    
    args = parser.parse_args()
    
    output_file = Path(args.output_file)
    
    # Create parent directory if needed
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    log_temperatures(output_file, args.interval, args.nvme_device)

if __name__ == "__main__":
    main()
