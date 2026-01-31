#!/usr/bin/env python3
"""Real-time training monitor with auto-cancel on plateau detection.

Usage:
    python monitor_training.py <task_output_file>
    
Example:
    python monitor_training.py /tmp/claude/-home-derekszen-Projects-SALT-VLA/tasks/ba62510.output
"""

import sys
import time
import re
from pathlib import Path
from collections import deque

def parse_loss_from_line(line: str) -> tuple[int, float] | None:
    """Extract step and loss from training log line."""
    match = re.search(r"\[train\] step=(\d+) loss=([\d.]+)", line)
    if match:
        return int(match.group(1)), float(match.group(2))
    return None

def calculate_trend(losses: deque, window: int = 100) -> str:
    """Calculate if loss is decreasing, flat, or increasing."""
    if len(losses) < window:
        return "WARMING UP"
    
    recent = list(losses)[-window:]
    older = list(losses)[-2*window:-window] if len(losses) >= 2*window else recent
    
    recent_avg = sum(recent) / len(recent)
    older_avg = sum(older) / len(older)
    
    change = recent_avg - older_avg
    pct_change = (change / older_avg) * 100
    
    if pct_change < -1.0:  # Decreasing by >1%
        return f"✅ DECREASING ({pct_change:.1f}%)"
    elif pct_change > 0.5:  # Increasing by >0.5%
        return f"⚠️  INCREASING ({pct_change:+.1f}%)"
    else:
        return f"⏸️  PLATEAU ({pct_change:+.1f}%)"

def monitor_training(log_file: Path, check_interval: float = 5.0):
    """Monitor training progress in real-time."""
    print(f"📊 Monitoring: {log_file}")
    print(f"Press Ctrl+C to stop monitoring\n")
    
    losses = deque(maxlen=1000)  # Keep last 1000 losses
    last_step = -1
    
    try:
        while True:
            if not log_file.exists():
                print(f"Waiting for {log_file}...")
                time.sleep(check_interval)
                continue
            
            # Read new lines
            with open(log_file, 'r') as f:
                for line in f:
                    result = parse_loss_from_line(line)
                    if result:
                        step, loss = result
                        if step > last_step:
                            losses.append(loss)
                            last_step = step
            
            # Display status
            if losses:
                current_loss = losses[-1]
                start_loss = losses[0]
                trend = calculate_trend(losses)
                
                print(f"\r[Step {last_step:6d}] Loss: {current_loss:.4f} | "
                      f"Start: {start_loss:.4f} | Change: {current_loss - start_loss:+.4f} | "
                      f"Trend: {trend}     ", end='', flush=True)
            
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")
        if losses:
            print(f"\nFinal Stats:")
            print(f"  Start Loss:   {losses[0]:.4f}")
            print(f"  Current Loss: {losses[-1]:.4f}")
            print(f"  Total Change: {losses[-1] - losses[0]:+.4f}")
            print(f"  Steps:        {last_step}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python monitor_training.py <log_file>")
        print("\nExample:")
        print("  python monitor_training.py /tmp/claude/-home-derekszen-Projects-SALT-VLA/tasks/ba62510.output")
        sys.exit(1)
    
    log_file = Path(sys.argv[1])
    monitor_training(log_file)
