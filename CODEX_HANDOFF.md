# SALT-VLA Project Handoff for Codex

## Quick Start Prompt

Copy this prompt to start Codex:

---

**PROMPT FOR CODEX:**

```
You are continuing work on the SALT-VLA project - a V-JEPA self-supervised learning system for training vision encoders.

## Project Location
/home/derekszen/Projects/SALT-VLA

## First Steps
1. Read `progress.txt` for complete training history and current status
2. Read `CLAUDE.md` for project rules and validated hyperparameters
3. Check current training status:
   ```bash
   ps aux | grep python.*train
   tail -20 wandb/latest-run/files/output.log
   ```

## Current State
- Training ViT-Base (86M params) encoder via V-JEPA on SSv2 dataset
- Experiment 1 (higher LR) is running and showing promising results
- Min loss achieved: 10.40 (target: <10.0 for publication)
- All paper-aligned hyperparameters are implemented

## Your Task
Continue the hyperparameter sweep to find optimal config for SoTA ViT-B encoder:
1. Monitor Exp1 completion, record final min loss
2. If Exp1 min loss < 10.76: Queue Exp3 (train_vitb_exp3_combined.py)
3. If Exp1 min loss >= 10.76: Queue Exp2 (train_vitb_exp2_highmask.py)
4. After sweep: run best config for 10-20 epochs

## Key Commands
# Run tests
PYTHONPATH=. uv run pytest tests/ -v

# Start training
PYTHONPATH=. nohup /home/derekszen/.local/bin/uv run python <script>.py > /tmp/<name>.log 2>&1 &

# Monitor training
tail -f wandb/latest-run/files/output.log

# Check min loss
cat wandb/latest-run/files/output.log | grep "\[train\]" | awk -F'loss=' '{print $2}' | awk '{print $1}' | sort -n | head -5

## Important Files
- src/train.py - Main training function
- src/models/salt.py - V-JEPA model architecture
- train_vitb_exp*.py - Experiment configs
- progress.txt - Full history (READ THIS FIRST)

## Do NOT
- Modify src/train.py or src/models/salt.py without running tests first
- Use batch_size > 32 (causes bus errors)
- Increase epochs for hyperparameter search (3 epochs is sufficient)
```

---

## Detailed Context

### What This Project Does
Trains a ViT-Base vision encoder using V-JEPA (Video Joint Embedding Predictive Architecture):
- Teacher: Frozen VideoMAEv2-Base provides target latents
- Student: ViT-Base learns to predict masked patch representations
- Predictor: 12-layer transformer predicts masked tokens from visible ones
- Loss: MSE between predicted and teacher latents

### Architecture Overview
```
Video → Student Encoder → Visible Patches → Predictor → Predicted Masked Patches
                                                              ↓
Video → Teacher Encoder → Full Patches → Masked Patches → MSE Loss
```

### Current Best Configuration
```python
train(
    student_model_name="vit_base_patch16_224",
    batch_size=32,
    epochs=3,
    lr=3e-4,        # KEY FINDING: 2x higher than initial
    min_lr=3e-5,
    warmup_steps=500,
    grad_clip=0.02,
    betas=(0.9, 0.95),
    weight_decay_start=0.04,
    weight_decay_end=0.4,
    mask_ratio=0.75,
    masking_strategy="multiblock",
    predictor_dim=384,
    predictor_depth=12,
    grad_checkpointing=False,
    cudnn_benchmark=True,
)
```

### Training Results Summary
| Run | LR | Min Loss | Finding |
|-----|-----|----------|---------|
| Baseline | 1.5e-4 | 10.76 | Plateau at 11.3-11.7 |
| Exp1 (current) | 3e-4 | 10.40 | BETTER - higher LR helps |

### Experiment Queue
1. **Exp1** (RUNNING): `train_vitb_exp1_higherlr.py` - LR=3e-4
2. **Exp2** (QUEUED): `train_vitb_exp2_highmask.py` - mask_ratio=0.9
3. **Exp3** (QUEUED): `train_vitb_exp3_combined.py` - LR=3e-4 + mask=0.9

### Decision Tree After Exp1
```
Exp1 completes
    │
    ├─ min_loss < 10.76 → Run Exp3 (test if higher mask helps too)
    │
    └─ min_loss >= 10.76 → Run Exp2 (try different approach)

After sweep:
    → Take best config
    → Run 10-20 epochs for final model
    → Target: loss < 10.0
```

### Environment Setup
```bash
cd /home/derekszen/Projects/SALT-VLA
source .venv/bin/activate  # or use uv run

# Verify environment
PYTHONPATH=. uv run pytest tests/ -v  # Should pass 64 tests
```

### Hardware
- GPU: NVIDIA RTX 5090 D (31GB VRAM, using 25GB limit)
- Storage: Samsung 9100 Pro NVMe at /mnt/ssv2
- Dataset: 168K SSv2 videos, pre-cached latents

### Key Learnings to Remember
1. **LR=3e-4 works better than 1.5e-4** for ViT-Base
2. **More epochs don't break plateau** - need hyperparameter changes
3. **3-epoch runs are sufficient** for hyperparameter search
4. **grad_checkpointing=False** gives 2x throughput
5. **Loss interpretation**: ~12-15=random, ~11-12=learning, <10=good

### Files to Read
1. `progress.txt` - Complete training history (START HERE)
2. `CLAUDE.md` - Project rules and constraints
3. `src/train.py` - Training function signature
4. `train_vitb_exp*.py` - Current experiment configs

### What NOT to Change
- Predictor depth (keep at 12, paper-aligned)
- Batch size (keep at 32, higher causes crashes)
- Multi-block masking (already implemented correctly)
- Weight decay schedule (already cosine 0.04→0.4)

### Wandb
Project: https://wandb.ai/zengzhuoxi-peking-university/salt-vla
Local logs: `wandb/latest-run/files/output.log`

---

## Verification Checklist

Before making changes, verify:
- [ ] Read progress.txt completely
- [ ] Tests pass: `PYTHONPATH=. uv run pytest tests/ -v`
- [ ] Current experiment status checked
- [ ] Understand the decision tree for next experiment

## Contact/Notes
- Goal: SoTA ViT-B encoder for VLA robotics
- Target loss: <10.0 for publication
- IntPhys2 evaluation pipeline is scaffolded but not implemented
