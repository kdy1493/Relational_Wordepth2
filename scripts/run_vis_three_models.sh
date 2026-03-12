#!/usr/bin/env bash
# Visualize Baseline vs WorDepth vs WorDepth+Rel (same 3 models as requested).
# Run from repo root:  bash scripts/run_vis_three_models.sh
# (Activate your venv first if needed: source .venv_bw/bin/activate)

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
[ -f .venv_bw/bin/activate ] && source .venv_bw/bin/activate

CONFIG="$REPO_ROOT/configs/arguments_eval_nyu_all_wordepth.yaml"
OUT_DIR="$REPO_ROOT/runs/vis_three_models"
NUM_SAMPLES=4

# Checkpoints (best_abs_rel)
CKPT_BASELINE="$REPO_ROOT/runs/nyu_train_baseline_exp3_effbatch24_adameps0001_nolegacy/model-45000-best_abs_rel_0.05853"
CKPT_WORDEPTH="$REPO_ROOT/runs/nyu_train_wordepth_paper_exp2_effbatch24/model-45000-best_abs_rel_0.03969"
CKPT_RELATIONAL="$REPO_ROOT/runs/nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin00/model-45000-best_abs_rel_0.03893"

python3 src/visualize_three_models.py "$CONFIG" \
  --ckpt_baseline "$CKPT_BASELINE" \
  --ckpt_wordepth "$CKPT_WORDEPTH" \
  --ckpt_relational "$CKPT_RELATIONAL" \
  --output_dir "$OUT_DIR" \
  --num_samples "$NUM_SAMPLES"

echo "Done. Open: $OUT_DIR/comparison_three_models.png"
