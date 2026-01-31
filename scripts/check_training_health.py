#!/usr/bin/env python3
"""Training health checker - outputs structured status for coding agents.

Returns JSON with:
- Training progress (steps, epochs)
- Loss trends (current, start, change, trend)
- Performance (throughput, memory)
- Hardware temps (GPU, CPU, NVMe)
- Health status (OK, WARNING, ERROR)
- Recommendations (continue, investigate, cancel)

Usage:
    python check_training_health.py <log_file>
    
Example:
    python check_training_health.py /tmp/claude/-home-derekszen-Projects-SALT-VLA/tasks/ba62510.output

Output: JSON to stdout (easy for agents to parse)
"""

import sys
import json
import re
import math
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import subprocess

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    from src.utils.hw_monitor import get_all_temps, check_temp_health
    HW_MONITOR_AVAILABLE = True
except ImportError:
    HW_MONITOR_AVAILABLE = False

def parse_training_log(log_file: Path) -> List[Dict]:
    """Parse training log and extract all training steps."""
    steps = []
    pattern = r"\[train\] step=(\d+) loss=([\d.]+) clips/s=([\d.]+) .*gpu_mem_gb=([\d.]+) ram=([\d.]+)%"
    
    if not log_file.exists():
        return steps
    
    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                steps.append({
                    'step': int(match.group(1)),
                    'loss': float(match.group(2)),
                    'throughput': float(match.group(3)),
                    'gpu_mem_gb': float(match.group(4)),
                    'ram_pct': float(match.group(5)),
                })
    
    return steps

def _parse_startup_int(log_file: Path, key: str) -> Optional[int]:
    pattern = re.compile(rf"\[startup\].*\b{re.escape(key)}=(\d+)\b")
    with open(log_file, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                return int(match.group(1))
    return None

def _infer_epochs_from_filename(log_file: Path) -> Optional[int]:
    match = re.search(r"(\d+)\s*epoch", log_file.name)
    if match:
        return int(match.group(1))
    return None

def _infer_pid_from_filename(log_file: Path) -> Optional[int]:
    match = re.search(r"_pid(\d+)\.log$", log_file.name)
    if match:
        return int(match.group(1))
    return None

def _parse_ps_etime_seconds(pid: int) -> Optional[int]:
    """Return process elapsed wall time in seconds using `ps`, if available."""
    try:
        result = subprocess.run(
            ["ps", "-p", str(pid), "-o", "etime="],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None

    if result.returncode != 0:
        return None

    etime = result.stdout.strip()
    if not etime:
        return None

    # Formats: [[dd-]hh:]mm:ss
    # Examples: "02:27:36", "13:05", "1-02:03:04"
    days = 0
    if "-" in etime:
        days_str, etime = etime.split("-", 1)
        try:
            days = int(days_str)
        except ValueError:
            return None

    parts = etime.split(":")
    try:
        if len(parts) == 3:
            hours, minutes, seconds = (int(x) for x in parts)
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = (int(x) for x in parts)
        else:
            return None
    except ValueError:
        return None

    return days * 86400 + hours * 3600 + minutes * 60 + seconds

def infer_total_steps(log_file: Path) -> Dict[str, object]:
    """Best-effort inference of total steps and related config from log contents."""
    if not log_file.exists():
        return {"total_steps": None}

    batch_size = _parse_startup_int(log_file, "batch_size")
    samples = _parse_startup_int(log_file, "samples")
    grad_accum_steps = _parse_startup_int(log_file, "grad_accum_steps") or 1
    epochs = _parse_startup_int(log_file, "epochs") or _infer_epochs_from_filename(log_file)

    steps_per_epoch = None
    total_steps = None

    # NOTE: src/train.py uses drop_last=True, so steps_per_epoch is floor(samples / batch_size).
    if samples is not None and batch_size is not None and batch_size > 0:
        steps_per_epoch = samples // batch_size

    if steps_per_epoch is not None and epochs is not None and grad_accum_steps > 0:
        total_steps = math.ceil((steps_per_epoch * epochs) / grad_accum_steps)

    return {
        "total_steps": total_steps,
        "epochs": epochs,
        "samples": samples,
        "batch_size": batch_size,
        "grad_accum_steps": grad_accum_steps,
        "steps_per_epoch": steps_per_epoch,
    }

def calculate_loss_trend(steps: List[Dict], window: int = 100) -> Dict:
    """Calculate loss trend over recent steps."""
    if len(steps) < 2:
        return {
            'status': 'insufficient_data',
            'direction': 'unknown',
            'percent_change': 0.0,
        }
    
    # Get recent losses
    recent_losses = [s['loss'] for s in steps[-window:]]
    older_losses = [s['loss'] for s in steps[-2*window:-window]] if len(steps) >= 2*window else recent_losses
    
    recent_avg = sum(recent_losses) / len(recent_losses)
    older_avg = sum(older_losses) / len(older_losses)
    
    change = recent_avg - older_avg
    pct_change = (change / older_avg) * 100 if older_avg > 0 else 0.0
    
    # Determine direction
    if pct_change < -1.0:
        direction = 'decreasing'
        status = 'good'
    elif pct_change > 0.5:
        direction = 'increasing'
        status = 'warning'
    else:
        direction = 'plateau'
        status = 'caution'
    
    return {
        'status': status,
        'direction': direction,
        'percent_change': pct_change,
        'recent_avg': recent_avg,
        'older_avg': older_avg,
    }

def calculate_progress(steps: List[Dict], total_steps: Optional[int] = None) -> Dict:
    """Calculate training progress."""
    if not steps:
        return {
            'current_step': 0,
            'progress_pct': 0.0,
            'estimated_steps': 0,
        }
    
    current = steps[-1]['step']
    
    # Estimate total steps if not provided (assume 10 epochs, 10557 steps/epoch)
    if total_steps is None:
        total_steps = 105570  # Default: 10 epochs for full dataset
    
    progress_pct = (current / total_steps) * 100 if total_steps > 0 else 0.0
    
    return {
        'current_step': current,
        'estimated_total_steps': total_steps,
        'progress_pct': min(progress_pct, 100.0),
    }

def calculate_performance(steps: List[Dict]) -> Dict:
    """Calculate performance metrics."""
    if not steps:
        return {
            'avg_throughput': 0.0,
            'current_throughput': 0.0,
            'gpu_mem_gb': 0.0,
            'ram_pct': 0.0,
        }
    
    recent = steps[-10:]  # Last 10 steps
    
    return {
        'avg_throughput': sum(s['throughput'] for s in recent) / len(recent),
        'current_throughput': steps[-1]['throughput'],
        'gpu_mem_gb': steps[-1]['gpu_mem_gb'],
        'ram_pct': steps[-1]['ram_pct'],
    }

def determine_health_status(
    loss_trend: Dict,
    performance: Dict,
    steps: List[Dict]
) -> Tuple[str, str, List[str]]:
    """Determine overall health status and recommendation.
    
    Returns:
        (status, recommendation, issues)
        status: 'healthy', 'warning', 'error'
        recommendation: 'continue', 'investigate', 'cancel'
        issues: List of detected issues
    """
    issues = []
    
    # Check loss trend
    if loss_trend['direction'] == 'increasing':
        issues.append(f"Loss increasing by {loss_trend['percent_change']:.1f}%")
    elif loss_trend['direction'] == 'plateau' and len(steps) > 1000:
        issues.append(f"Loss plateau detected ({loss_trend['percent_change']:+.1f}%)")
    
    # Check memory
    if performance['ram_pct'] > 90:
        issues.append(f"High RAM usage: {performance['ram_pct']:.1f}%")
    if performance['ram_pct'] > 95:
        issues.append("CRITICAL: RAM usage above 95% - system freeze risk!")
    
    # Check throughput
    if performance['current_throughput'] < 50:
        issues.append(f"Low throughput: {performance['current_throughput']:.1f} v/s")
    
    # Check temperatures (if available)
    if HW_MONITOR_AVAILABLE:
        temps = get_all_temps()
        temp_statuses = check_temp_health(temps)
        
        # Check for thermal issues
        if temps['gpu_core_temp_c'] and temps['gpu_core_temp_c'] > 85:
            issues.append(f"High GPU temperature: {temps['gpu_core_temp_c']:.0f}°C")
        if temps['cpu_temp_c'] and temps['cpu_temp_c'] > 90:
            issues.append(f"High CPU temperature: {temps['cpu_temp_c']:.0f}°C")
        if temps['nvme_temp_c'] and temps['nvme_temp_c'] > 75:
            issues.append(f"High NVMe temperature: {temps['nvme_temp_c']:.0f}°C")
    
    # Determine status
    if performance['ram_pct'] > 95 or (loss_trend['direction'] == 'increasing' and len(steps) > 1000):
        status = 'error'
        recommendation = 'cancel'
    elif issues:
        status = 'warning'
        recommendation = 'investigate'
    else:
        status = 'healthy'
        recommendation = 'continue'
    
    return status, recommendation, issues

def check_training_health(log_file: Path) -> Dict:
    """Main health check function."""
    steps = parse_training_log(log_file)
    
    if not steps:
        return {
            'status': 'no_data',
            'message': 'No training data found in log file',
            'log_file': str(log_file),
        }
    
    inferred = infer_total_steps(log_file)
    inferred_total_steps = inferred.get("total_steps")

    # Calculate metrics
    loss_trend = calculate_loss_trend(steps)
    progress = calculate_progress(steps, total_steps=inferred_total_steps if isinstance(inferred_total_steps, int) else None)
    performance = calculate_performance(steps)
    status, recommendation, issues = determine_health_status(loss_trend, performance, steps)
    
    # Get hardware temperatures (if available)
    temps_data = {}
    if HW_MONITOR_AVAILABLE:
        temps = get_all_temps()
        temp_statuses = check_temp_health(temps)
        temps_data = {
            'gpu_core_temp_c': temps['gpu_core_temp_c'],
            'cpu_temp_c': temps['cpu_temp_c'],
            'nvme_temp_c': temps['nvme_temp_c'],
            'gpu_core_status': temp_statuses['gpu_core_status'],
            'cpu_status': temp_statuses['cpu_status'],
            'nvme_status': temp_statuses['nvme_status'],
        }
    
    # ETA (best-effort)
    eta_data: Dict[str, object] = {}
    if isinstance(inferred_total_steps, int) and inferred_total_steps > 0:
        current_step = steps[-1]["step"]
        remaining_steps = max(inferred_total_steps - current_step, 0)
        batch_size = inferred.get("batch_size")
        grad_accum_steps = inferred.get("grad_accum_steps") or 1

        steps_per_sec = None
        if isinstance(batch_size, int) and batch_size > 0:
            # `clips/s` is measured per micro-batch; adjust if grad accumulation is enabled.
            micro_steps_per_sec = performance["avg_throughput"] / batch_size
            if grad_accum_steps > 0:
                steps_per_sec = micro_steps_per_sec / grad_accum_steps

        if steps_per_sec and steps_per_sec > 0:
            eta_seconds = int(remaining_steps / steps_per_sec)
            eta_data = {
                "remaining_steps": remaining_steps,
                "steps_per_sec": round(steps_per_sec, 3),
                "eta_seconds": eta_seconds,
                "eta_hms": str(timedelta(seconds=eta_seconds)),
            }

            pid = _infer_pid_from_filename(log_file)
            if pid is not None:
                etime_seconds = _parse_ps_etime_seconds(pid)
                if etime_seconds is not None and etime_seconds > 0:
                    # Prefer wall-clock estimate using process elapsed time if available.
                    wall_steps_per_sec = current_step / etime_seconds
                    if wall_steps_per_sec > 0:
                        wall_eta_seconds = int(remaining_steps / wall_steps_per_sec)
                        eta_data.update(
                            {
                                "pid": pid,
                                "elapsed_seconds": etime_seconds,
                                "wall_steps_per_sec": round(wall_steps_per_sec, 3),
                                "wall_eta_seconds": wall_eta_seconds,
                                "wall_eta_hms": str(timedelta(seconds=wall_eta_seconds)),
                                "estimated_finish_time": (
                                    datetime.now() + timedelta(seconds=wall_eta_seconds)
                                ).isoformat(timespec="seconds"),
                            }
                        )

    # Build response
    response = {
        'status': status,
        'recommendation': recommendation,
        'issues': issues,
        'progress': {
            'current_step': progress['current_step'],
            'total_steps': progress['estimated_total_steps'],
            'percent_complete': round(progress['progress_pct'], 1),
        },
        'inferred': {
            'epochs': inferred.get('epochs'),
            'samples': inferred.get('samples'),
            'batch_size': inferred.get('batch_size'),
            'grad_accum_steps': inferred.get('grad_accum_steps'),
            'steps_per_epoch': inferred.get('steps_per_epoch'),
        },
        'loss': {
            'current': steps[-1]['loss'],
            'start': steps[0]['loss'],
            'total_change': round(steps[-1]['loss'] - steps[0]['loss'], 4),
            'trend': loss_trend['direction'],
            'trend_percent': round(loss_trend['percent_change'], 2),
        },
        'performance': {
            'throughput_videos_per_sec': round(performance['current_throughput'], 1),
            'gpu_memory_gb': performance['gpu_mem_gb'],
            'ram_percent': performance['ram_pct'],
        },
        'eta': eta_data,
        'log_file': str(log_file),
        'total_steps_logged': len(steps),
    }
    
    # Add temperatures if available
    if temps_data:
        response['temperatures'] = temps_data
    
    return response

def main():
    if len(sys.argv) < 2:
        print(json.dumps({
            'error': 'Missing log file argument',
            'usage': 'python check_training_health.py <log_file>',
        }), file=sys.stderr)
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    
    if not log_file.exists():
        print(json.dumps({
            'error': 'Log file not found',
            'log_file': str(log_file),
        }), file=sys.stderr)
        sys.exit(1)
    
    # Get health status
    health = check_training_health(log_file)
    
    # Output JSON to stdout
    print(json.dumps(health, indent=2))
    
    # Exit code based on status
    if health.get('status') == 'error':
        sys.exit(2)  # Error status
    elif health.get('status') == 'warning':
        sys.exit(1)  # Warning status
    else:
        sys.exit(0)  # Healthy

if __name__ == "__main__":
    main()
