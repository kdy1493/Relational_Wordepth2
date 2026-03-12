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
REL_DIR="$REPO_ROOT/data/nyu_relational/statistical_test"
NUM_SAMPLES=4

# Checkpoints (best_abs_rel)
# --- (1) Baseline vs WorDepth vs WorDepth+Rel: 첫 번째는 반드시 baseline으로 학습한 ckpt (baseline_arch=True 로드됨)
# CKPT_BASELINE="$REPO_ROOT/runs/nyu_train_baseline_exp3_effbatch24_adameps0001_nolegacy/model-45000-best_abs_rel_0.05853"
# CKPT_WORDEPTH="$REPO_ROOT/runs/nyu_train_wordepth_paper_exp2_effbatch24/model-45000-best_abs_rel_0.03969"
# CKPT_RELATIONAL="$REPO_ROOT/runs/nyu_train_wordepth_relational_exp5_effbatch24_stat_alpha10_lambda001_margin00/model-42000-best_abs_rel_0.03683"
#
# --- (2) WorDepth+Rel 세 개 비교 (λ=0.05 / 0.01 / 0.02): 세 ckpt 모두 WorDepth 구조 → --three_models_same_arch 필수
CKPT_BASELINE="$REPO_ROOT/runs/nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin00/model-45000-best_abs_rel_0.03893"
CKPT_WORDEPTH="$REPO_ROOT/runs/nyu_train_wordepth_relational_exp5_effbatch24_stat_alpha10_lambda001_margin00/model-42000-best_abs_rel_0.03683"
CKPT_RELATIONAL="$REPO_ROOT/runs/nyu_train_wordepth_relational_exp6_effbatch24_stat_alpha10_lambda002_margin00/model-42000-best_abs_rel_0.03491"

python3 src/visualize_three_models.py "$CONFIG" \
  --ckpt_baseline "$CKPT_BASELINE" \
  --ckpt_wordepth "$CKPT_WORDEPTH" \
  --ckpt_relational "$CKPT_RELATIONAL" \
  --output_dir "$OUT_DIR" \
  --num_samples "$NUM_SAMPLES" \
  --show_masks \
  --relations_dir_eval "$REL_DIR" \
  --three_models_same_arch

echo "Done. Open latest: $OUT_DIR/comparison_three_models*.png"
