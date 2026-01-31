#!/usr/bin/env python3
"""Automated hyperparameter tuning based on training results.

This script:
1. Analyzes training logs to extract loss curves
2. Diagnoses issues (plateau, divergence, etc.)
3. Suggests and applies hyperparameter adjustments
4. Queues the next training run

Usage:
    python auto_tune.py <log_file> [--apply] [--run-next]
"""

import argparse
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass


@dataclass
class TrainingMetrics:
    """Extracted metrics from training log."""
    initial_loss: float
    final_loss: float
    min_loss: float
    avg_loss_first_500: float
    avg_loss_last_500: float
    total_steps: int
    loss_reduction_pct: float
    plateau_detected: bool
    divergence_detected: bool


def parse_training_log(log_file: str) -> list[tuple[int, float]]:
    """Extract (step, loss) pairs from training log."""
    pattern = r'\[train\] step=(\d+) loss=([\d.]+)'
    data = []

    with open(log_file, 'r') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                step = int(match.group(1))
                loss = float(match.group(2))
                data.append((step, loss))

    return data


def analyze_metrics(data: list[tuple[int, float]]) -> TrainingMetrics:
    """Analyze training data and compute metrics."""
    if not data:
        raise ValueError("No training data found")

    steps = [d[0] for d in data]
    losses = [d[1] for d in data]

    initial_loss = losses[0]
    final_loss = losses[-1]
    min_loss = min(losses)

    # Average of first and last 500 steps (or all if fewer)
    n_samples = min(25, len(losses) // 2)  # ~500 steps at log_interval=20
    avg_first = sum(losses[:n_samples]) / n_samples if n_samples > 0 else initial_loss
    avg_last = sum(losses[-n_samples:]) / n_samples if n_samples > 0 else final_loss

    loss_reduction = (initial_loss - final_loss) / initial_loss * 100

    # Plateau detection: loss change < 5% over last half of training
    mid_idx = len(losses) // 2
    if mid_idx > 0:
        avg_mid = sum(losses[mid_idx:mid_idx+n_samples]) / min(n_samples, len(losses) - mid_idx)
        plateau = abs(avg_mid - avg_last) / avg_mid < 0.05
    else:
        plateau = False

    # Divergence detection: loss increasing over time
    divergence = avg_last > avg_first * 1.1

    return TrainingMetrics(
        initial_loss=initial_loss,
        final_loss=final_loss,
        min_loss=min_loss,
        avg_loss_first_500=avg_first,
        avg_loss_last_500=avg_last,
        total_steps=steps[-1] if steps else 0,
        loss_reduction_pct=loss_reduction,
        plateau_detected=plateau,
        divergence_detected=divergence,
    )


def read_current_config(config_file: str = "train_extreme.py") -> dict:
    """Read current hyperparameters from train_extreme.py."""
    config = {}

    with open(config_file, 'r') as f:
        content = f.read()

    # Extract key parameters
    patterns = {
        'lr': r'lr\s*=\s*([\d.e-]+)',
        'min_lr': r'min_lr\s*=\s*([\d.e-]+)',
        'warmup_steps': r'warmup_steps\s*=\s*(\d+)',
        'batch_size': r'batch_size\s*=\s*(\d+)',
        'epochs': r'epochs\s*=\s*(\d+)',
        'weight_decay': r'weight_decay\s*=\s*([\d.]+)',
        'grad_clip': r'grad_clip\s*=\s*([\d.]+)',
        'mask_ratio': r'mask_ratio\s*=\s*([\d.]+)',
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            val = match.group(1)
            try:
                config[key] = float(val) if '.' in val or 'e' in val.lower() else int(val)
            except ValueError:
                config[key] = val

    return config


def suggest_adjustments(metrics: TrainingMetrics, config: dict) -> dict:
    """Suggest hyperparameter adjustments based on metrics."""
    suggestions = {}
    reasoning = []

    # Current values
    lr = config.get('lr', 3e-4)
    min_lr = config.get('min_lr', 3e-5)
    warmup_steps = config.get('warmup_steps', 200)
    weight_decay = config.get('weight_decay', 0.05)

    # Analyze and suggest

    # 1. Check if loss is still high (>11) - may need higher LR or more capacity
    if metrics.avg_loss_last_500 > 11.0:
        if metrics.plateau_detected:
            # Plateau at high loss - increase LR
            new_lr = min(lr * 1.5, 1e-3)  # Cap at 1e-3
            if new_lr != lr:
                suggestions['lr'] = new_lr
                suggestions['min_lr'] = new_lr * 0.1
                reasoning.append(f"Plateau at loss={metrics.avg_loss_last_500:.2f} - increasing LR: {lr:.1e} -> {new_lr:.1e}")
        elif not metrics.divergence_detected:
            # Still learning but slow - try slight LR increase
            new_lr = min(lr * 1.2, 8e-4)
            if new_lr != lr:
                suggestions['lr'] = new_lr
                suggestions['min_lr'] = new_lr * 0.1
                reasoning.append(f"Slow learning - slight LR increase: {lr:.1e} -> {new_lr:.1e}")

    # 2. Check for divergence - need lower LR
    if metrics.divergence_detected:
        new_lr = lr * 0.5
        suggestions['lr'] = new_lr
        suggestions['min_lr'] = new_lr * 0.1
        suggestions['warmup_steps'] = int(warmup_steps * 1.5)
        reasoning.append(f"Divergence detected - reducing LR: {lr:.1e} -> {new_lr:.1e}")

    # 3. Good progress - minor tweaks
    if metrics.loss_reduction_pct > 25 and metrics.avg_loss_last_500 < 11.0:
        # Good learning - maybe try less weight decay for better fitting
        if weight_decay > 0.02:
            suggestions['weight_decay'] = weight_decay * 0.8
            reasoning.append(f"Good progress - reducing weight_decay: {weight_decay} -> {weight_decay * 0.8:.3f}")

    # 4. Warmup adjustment based on stability
    if metrics.initial_loss > 16:  # High initial loss suggests warmup issues
        suggestions['warmup_steps'] = min(warmup_steps + 100, 500)
        reasoning.append(f"High initial loss - extending warmup: {warmup_steps} -> {suggestions['warmup_steps']}")

    return {'suggestions': suggestions, 'reasoning': reasoning}


def apply_adjustments(config_file: str, adjustments: dict) -> None:
    """Apply hyperparameter adjustments to train_extreme.py."""
    suggestions = adjustments.get('suggestions', {})
    if not suggestions:
        print("No adjustments to apply")
        return

    with open(config_file, 'r') as f:
        content = f.read()

    for key, new_val in suggestions.items():
        # Format value appropriately
        if isinstance(new_val, float) and new_val < 0.01:
            val_str = f"{new_val:.1e}"
        elif isinstance(new_val, float):
            val_str = f"{new_val:.4f}".rstrip('0').rstrip('.')
        else:
            val_str = str(new_val)

        # Replace in file
        pattern = rf'({key}\s*=\s*)([\d.e-]+)'
        content = re.sub(pattern, rf'\g<1>{val_str}', content)

    with open(config_file, 'w') as f:
        f.write(content)

    print(f"Applied adjustments to {config_file}:")
    for key, val in suggestions.items():
        print(f"  {key} = {val}")


def log_experiment(metrics: TrainingMetrics, config: dict, adjustments: dict) -> None:
    """Append experiment results to EXPERIMENTS.md."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    entry = f"""
---

## Auto-Tuning Run ({timestamp})

### Configuration Used
```python
lr = {config.get('lr', 'N/A')}
min_lr = {config.get('min_lr', 'N/A')}
warmup_steps = {config.get('warmup_steps', 'N/A')}
weight_decay = {config.get('weight_decay', 'N/A')}
```

### Results
- Initial loss: {metrics.initial_loss:.2f}
- Final loss: {metrics.final_loss:.2f}
- Min loss: {metrics.min_loss:.2f}
- Loss reduction: {metrics.loss_reduction_pct:.1f}%
- Total steps: {metrics.total_steps}
- Plateau detected: {metrics.plateau_detected}
- Divergence detected: {metrics.divergence_detected}

### Analysis
"""

    for reason in adjustments.get('reasoning', ['No adjustments needed']):
        entry += f"- {reason}\n"

    if adjustments.get('suggestions'):
        entry += "\n### Applied Adjustments\n"
        for key, val in adjustments['suggestions'].items():
            entry += f"- `{key}`: {val}\n"

    with open("EXPERIMENTS.md", 'a') as f:
        f.write(entry)

    print(f"Logged results to EXPERIMENTS.md")


def run_next_training() -> str:
    """Queue the next training run and return the task output file."""
    print("Queueing next train_extreme run...")

    result = subprocess.run(
        ["bash", "-c", "PYTHONPATH=. uv run python train_extreme.py"],
        capture_output=False,
        text=True,
    )

    return result.returncode


def main():
    parser = argparse.ArgumentParser(description="Automated hyperparameter tuning")
    parser.add_argument("log_file", help="Training log file to analyze")
    parser.add_argument("--apply", action="store_true", help="Apply suggested adjustments")
    parser.add_argument("--run-next", action="store_true", help="Run next training after applying")
    parser.add_argument("--config", default="train_extreme.py", help="Config file to modify")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SALT-VLA Auto-Tuning Analysis")
    print(f"{'='*60}\n")

    # Parse and analyze
    print(f"Analyzing: {args.log_file}")
    data = parse_training_log(args.log_file)
    print(f"Found {len(data)} training steps\n")

    if not data:
        print("ERROR: No training data found in log file")
        return 1

    metrics = analyze_metrics(data)
    print("Metrics:")
    print(f"  Initial loss: {metrics.initial_loss:.2f}")
    print(f"  Final loss:   {metrics.final_loss:.2f}")
    print(f"  Min loss:     {metrics.min_loss:.2f}")
    print(f"  Reduction:    {metrics.loss_reduction_pct:.1f}%")
    print(f"  Plateau:      {metrics.plateau_detected}")
    print(f"  Divergence:   {metrics.divergence_detected}")
    print()

    # Read current config
    config = read_current_config(args.config)
    print(f"Current config ({args.config}):")
    for k, v in config.items():
        print(f"  {k} = {v}")
    print()

    # Suggest adjustments
    adjustments = suggest_adjustments(metrics, config)

    if adjustments['reasoning']:
        print("Analysis & Suggestions:")
        for reason in adjustments['reasoning']:
            print(f"  - {reason}")
        print()

    if adjustments['suggestions']:
        print("Proposed changes:")
        for k, v in adjustments['suggestions'].items():
            print(f"  {k}: {config.get(k, 'N/A')} -> {v}")
        print()
    else:
        print("No changes suggested - current config looks good!\n")

    # Log experiment
    log_experiment(metrics, config, adjustments)

    # Apply if requested
    if args.apply and adjustments['suggestions']:
        apply_adjustments(args.config, adjustments)
        print()

    # Run next if requested
    if args.run_next:
        return run_next_training()

    return 0


if __name__ == "__main__":
    exit(main())
