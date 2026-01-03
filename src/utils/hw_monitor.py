#!/usr/bin/env python3
"""Hardware temperature monitoring utilities.

Reads temperatures from:
- NVIDIA GPU (nvidia-smi) - Core and HBM Memory
- AMD Ryzen CPU (sensors)
- Samsung 9100 Pro NVMe SSD (smartctl/nvme)
"""

import subprocess
import re
from typing import Optional, Dict

# ANSI color codes for terminal output
class Colors:
    RED = '\033[91m'
    YELLOW = '\033[93m'
    GREEN = '\033[92m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def get_gpu_temps() -> Dict[str, Optional[float]]:
    """Get GPU core temperature in Celsius using nvidia-smi.
    
    Returns:
        Dict with 'core' temp
        
    Note: GDDR7 VRAM temperature is NOT exposed by NVIDIA drivers for
    consumer GeForce cards (like RTX 5090D). Only enterprise cards with
    HBM (H100, A100) expose memory temps via nvidia-smi.
    """
    temps = {'core': None}
    
    # Get GPU core temperature (the primary thermal bottleneck)
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            temps['core'] = float(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    
    return temps

def get_cpu_temp() -> Optional[float]:
    """Get CPU temperature in Celsius using sensors (lm-sensors).
    
    Looks for Ryzen/k10temp sensor readings.
    """
    try:
        result = subprocess.run(
            ["sensors"],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            # Look for Tctl or Tdie (AMD Ryzen temp)
            for line in result.stdout.split('\n'):
                if 'Tctl:' in line or 'Tdie:' in line:
                    match = re.search(r'[+]?([\d.]+)°C', line)
                    if match:
                        return float(match.group(1))
                # Fallback: look for any "temp1" reading
                if 'temp1:' in line:
                    match = re.search(r'[+]?([\d.]+)°C', line)
                    if match:
                        return float(match.group(1))
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None

def get_nvme_temp(device: str = "/dev/nvme0", use_sudo: bool = True) -> Optional[float]:
    """Get NVMe SSD temperature in Celsius.
    
    Args:
        device: NVMe device path (e.g., /dev/nvme0)
        use_sudo: Whether to use sudo (required for most systems)
    """
    # Try nvme-cli with sudo
    if use_sudo:
        try:
            result = subprocess.run(
                ["sudo", "-n", "nvme", "smart-log", device],  # -n = non-interactive
                capture_output=True,
                text=True,
                timeout=3
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'temperature' in line.lower():
                        match = re.search(r':\s*([\d.]+)\s*C', line)
                        if match:
                            return float(match.group(1))
        except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
            pass
    
    return None

def get_all_temps(nvme_device: str = "/dev/nvme0") -> Dict[str, Optional[float]]:
    """Get all hardware temperatures.
    
    Returns:
        Dict with keys: gpu_core_temp_c, cpu_temp_c, nvme_temp_c
        
    Note: GDDR7 VRAM temps are NOT available for RTX 5090D (GeForce limitation)
    """
    gpu_temps = get_gpu_temps()
    
    return {
        'gpu_core_temp_c': gpu_temps['core'],
        'cpu_temp_c': get_cpu_temp(),
        'nvme_temp_c': get_nvme_temp(nvme_device),
    }

def check_temp_health(temps: Dict[str, Optional[float]]) -> Dict[str, str]:
    """Check if temperatures are within safe ranges.
    
    Temperature limits for hardware:
    - RTX 5090D: GPU core <85°C (GDDR7 temp not available via nvidia-smi)
    - Ryzen 9950X3D: <85°C (3D V-Cache sensitive)
    - Samsung 9100 Pro: <70°C optimal, <80°C max
    
    Returns:
        Dict with status for each component
        Values: 'ok', 'warm', 'hot', 'critical', 'unknown'
    """
    def classify_temp(temp: Optional[float], warm: float, hot: float, critical: float) -> str:
        if temp is None:
            return 'unknown'
        if temp < warm:
            return 'ok'
        elif temp < hot:
            return 'warm'
        elif temp < critical:
            return 'hot'
        else:
            return 'critical'
    
    return {
        'gpu_core_status': classify_temp(temps['gpu_core_temp_c'], 70, 80, 85),
        'cpu_status': classify_temp(temps['cpu_temp_c'], 70, 80, 90),
        'nvme_status': classify_temp(temps['nvme_temp_c'], 60, 70, 80),
    }

def format_temp_status(temp: Optional[float], status: str, label: str) -> str:
    """Format temperature with color coding.
    
    Args:
        temp: Temperature in Celsius (or None)
        status: Health status ('ok', 'warm', 'hot', 'critical', 'unknown')
        label: Component label
    
    Returns:
        Formatted string with ANSI colors
    """
    if temp is None:
        return f"{label}: N/A"
    
    temp_str = f"{temp:.1f}°C"
    
    if status == 'ok':
        color = Colors.GREEN
    elif status == 'warm':
        color = Colors.YELLOW
    elif status in ['hot', 'critical']:
        color = Colors.RED + Colors.BOLD
    else:
        color = Colors.RESET
    
    return f"{label}: {color}{temp_str} ({status}){Colors.RESET}"

if __name__ == "__main__":
    """Test temperature reading with color output."""
    temps = get_all_temps()
    statuses = check_temp_health(temps)
    
    print("\n" + Colors.BOLD + "Hardware Temperatures:" + Colors.RESET)
    print("  " + format_temp_status(temps['gpu_core_temp_c'], statuses['gpu_core_status'], "GPU Core"))
    print("  " + Colors.RESET + "GPU VRAM: N/A (GDDR7 temps not exposed by NVIDIA for GeForce)" + Colors.RESET)
    print("  " + format_temp_status(temps['cpu_temp_c'], statuses['cpu_status'], "CPU"))
    print("  " + format_temp_status(temps['nvme_temp_c'], statuses['nvme_status'], "NVMe SSD"))
    
    # Check for critical temps
    if any(s in ['hot', 'critical'] for s in statuses.values()):
        print("\n" + Colors.RED + Colors.BOLD + "⚠️  WARNING: HIGH TEMPERATURES DETECTED!" + Colors.RESET)
