# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**WorDepth** — Variational Language Prior for Monocular Depth Estimation on NYU-Depth-v2.

Three model variants:
- **WorDepth**: image + text prior (CLIP features → language VAE, `alter_prob` controls branch switching)
- **Baseline B**: image-only Swin-L + depth decoder (`baseline_mode: true` → `baseline_arch=True`)
- **+Relational**: either variant + `RelationalDepthLoss` on object pairs (uses `src/relational_train.py`)

## Architecture

**Model (`src/networks/wordepth.py`):**
```
Input: RGB (B,3,480,640) + text_feature (B,1024, precomputed CLIP)
  ↓
Swin-L encoder → multi-scale features x (1/32), x2 (1/16), x3 (1/8), x4 (1/4)
  ↓
Two branches (selected by alter_prob during training):
  [Text VAE]  text_feature → Text_Encoder → mean, std → sample ε
  [Image]     encoder feats → EpsLayer → ε (learned from image)
  ↓
Decoder: d_feat = mean + std * ε → Refine blocks → Up → OutConv → exp() → depth_pred
```
`baseline_arch=True` skips Text_Encoder and EpsLayer entirely (image-only decoder path).

**Losses:**
- `SILogLoss` — pixel-wise scale-invariant log loss (`src/networks/loss.py`)
- `RelationalDepthLoss` — hinge loss on object-level depth ordering: `max(0, d_front - d_back + margin)` (`src/networks/relational_depth_loss.py`)
- Total: `L_depth + λ_rel × L_rel` (`rel_weight` in config)

**Relational representative depth per object mask:**
- `repr_mode: median` → median of valid pixels
- `repr_mode: statistical` → `μ + α × σ` (`rel_statistical_alpha`)

## Key Entry Points

| Task | Script |
|------|--------|
| Standard training | `src/train.py` |
| Relational training | `src/relational_train.py` |
| Depth metric eval | `src/eval.py` |
| RSR relational eval | `src/relational_eval_rsr.py` |
| Rel margin tuning | `scripts/analyze_relation_gt_gaps.py` |
| End-to-end pipeline | `scripts/run_train_then_eval.py` |

## Commands

### Training

```bash
# WorDepth (text-guided), multi-GPU
torchrun --nproc_per_node=2 src/train.py configs/arguments_run_nyu_wordepth_paper_exp2_effbatch24.yaml --use_ddp

# Baseline B (image-only), multi-GPU
torchrun --nproc_per_node=2 src/train.py configs/arguments_run_nyu_baseline_exp1_effbatch24.yaml --use_ddp

# WorDepth + Relational loss, multi-GPU
torchrun --nproc_per_node=2 src/relational_train.py configs/arguments_run_nyu_wordepth_relational_exp3_effbatch24_stat_alpha10_lambda005_margin05.yaml --use_ddp
```

For Baseline + Relational: use `src/relational_train.py` with `baseline_mode: true`, `use_relational_loss: true`.

### Evaluation

```bash
# Standard depth metrics (AbsRel, RMSE, δ<1.25, etc.)
python src/eval.py configs/arguments_eval_nyu_paper.yaml

# RSR (Relation Satisfaction Rate) — note @ prefix for argparse file
python src/relational_eval_rsr.py @configs/arguments_eval_nyu_rsr.yaml

# Relational GT gap distribution (for tuning rel_margin)
python scripts/analyze_relation_gt_gaps.py \
  --relations_dir_train ./data/nyu_relational/statistical_train \
  --filenames_file ./data_splits/nyudepthv2_train_split.txt \
  --gt_path ./data/nyu_v2_sync --rel_statistical_alpha 1.0 --use_gpu
```

Set `baseline_arch: true` in eval configs **only** when evaluating Baseline B checkpoints.

## Datasets / Paths

- **RGB & depth root**: `./data/nyu_v2_sync`
- **Splits**: `./data_splits/nyudepthv2_{train,val,test}_split.txt` (21K/2.4K/654 samples)
- **Text features**: `./data/text_feat/nyu/{train,test}/...pt` (precomputed CLIP)
- **Relational annotations**: `./data/nyu_relational/statistical_{train,test}/` (masks + relations JSON)
- **Pretrained backbone**: `./swin_large_patch4_window12_384_22k.pth`

## Experiment Structure

Runs saved under `./runs/`, named e.g.:
- `nyu_train_wordepth_paper_exp2_effbatch24/`
- `nyu_train_baseline_exp1_effbatch24/`
- `nyu_train_wordepth_relational_exp3_.../`

Each run dir contains: frozen config YAML, `model-steps-*.pt` checkpoints, `metrics.json`, `best_metrics.json`, `eval_results_*.json`, `rsr_results_*.json`.

## Key Config Parameters

| Parameter | Meaning |
|-----------|---------|
| `alter_prob` | Probability of training text VAE branch vs image branch |
| `baseline_mode` | `true` → Baseline B (no text path) |
| `accumulation_steps` | Gradient accumulation steps (effective_batch = batch_size × steps × n_gpus) |
| `weight_kld` | KLD loss weight for VAE |
| `use_relational_loss` | Enable RelationalDepthLoss |
| `rel_weight` | λ_rel (relational loss weight, typically 0.005) |
| `rel_margin` | Hinge margin m in depth units (typically 0.1–0.5 m) |
| `rel_repr` | `median` or `statistical` |
| `rel_statistical_alpha` | α for μ + ασ representative depth |
| `relations_dir_train` | Path to relational annotation dir |
| `eigen_crop` | `true` for standard NYU evaluation crop |

## Code Conventions

- CLI args use YAML configs; `relational_eval_rsr.py` uses `@config_file` argparse syntax.
- Keep CLI arg names stable — all `arguments_*.yaml` configs depend on them.
- Use `strict=False` in checkpoint loading only when necessary for `baseline_arch` compatibility.
- `legacy=False` (no skip connections) is the current model default.

## Paths to De-prioritize

- `runs/` — large logs & checkpoints (open specific files only on request)
- `data/` — datasets
- `*.pt`, `*.pth`, `*.npz`, `__pycache__/`, `.venv/`, `.git/`
