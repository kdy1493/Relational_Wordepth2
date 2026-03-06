#!/usr/bin/env bash
# Evaluate all best_abs_rel checkpoints with post_process=true, then print a summary table.
# Run from repo root:  bash scripts/eval_all_best_abs_rel_pptrue.sh
#
# Note: This overwrites existing eval_results_*best_abs_rel*.json in each run dir with post_process=true results.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_CONFIG_BASELINE="configs/arguments_eval_nyu_paper.yaml"
EVAL_CONFIG_WORDEPTH="configs/arguments_eval_nyu_all_wordepth.yaml"
SUMMARY_JSON="$REPO_ROOT/runs/eval_all_best_abs_rel_pptrue_summary.json"

echo "=== Evaluating all best_abs_rel checkpoints (post_process=true) ==="

# Build list: run_dir|checkpoint_path
LIST=""
for d in runs/nyu_train_*/; do
  ckpt=$(ls "$d"model-*-best_abs_rel* 2>/dev/null | head -1)
  [ -n "$ckpt" ] && LIST="$LIST${d}|$ckpt"$'\n'
done

# Run eval for each (baseline runs use baseline config, else wordepth config)
idx=0
total=$(echo "$LIST" | grep -c . || true)
while IFS= read -r line; do
  [ -z "$line" ] && continue
  run_dir="${line%%|*}"
  ckpt="${line#*|}"
  name=$(basename "${run_dir%/}")
  idx=$((idx + 1))
  echo "[$idx/$total] $name"
  if [[ "$name" == *"nyu_train_baseline"* ]]; then
    python src/eval.py "$EVAL_CONFIG_BASELINE" --checkpoint_path "$ckpt" --post_process
  else
    python src/eval.py "$EVAL_CONFIG_WORDEPTH" --checkpoint_path "$ckpt" --post_process
  fi
done <<< "$LIST"

echo ""
echo "=== Collecting results and printing table ==="

# Collect results: read each eval_results_*best_abs_rel*.json (now with post_process=true)
# and output a markdown table. Use run name and short param summary.
printf '%s\n' "| 이름 | 주요 파라미터 (변경된 것 위주) | Eval result (test, post_process=true) |"
printf '%s\n' "|------|-------------------------------|---------------------------------------|"

for d in runs/nyu_train_*/; do
  json=$(ls "$d"eval_results_*best_abs_rel*.json 2>/dev/null | head -1)
  [ -z "$json" ] && continue
  name=$(basename "${d%/}")
  # Parse metrics from JSON (portable: no jq required, use grep/sed)
  abs_rel=$(grep -o '"abs_rel": [0-9.]*' "$json" | head -1 | sed 's/"abs_rel": //')
  silog=$(grep -o '"silog": [0-9.]*' "$json" | head -1 | sed 's/"silog": //')
  d1=$(grep -o '"d1": [0-9.]*' "$json" | head -1 | sed 's/"d1": //')
  rms=$(grep -o '"rms": [0-9.]*' "$json" | head -1 | sed 's/"rms": //')
  pp=$(grep -o '"post_process": [a-z]*' "$json" | head -1 | sed 's/"post_process": //')
  [ -z "$abs_rel" ] && abs_rel="-"
  [ -z "$silog" ] && silog="-"
  [ -z "$d1" ] && d1="-"
  [ -z "$rms" ] && rms="-"
  metrics="abs_rel $abs_rel · silog $silog · δ1 $d1 · rms $rms (pp=$pp)"
  # Short param summary per run (change-focused)
  case "$name" in
    nyu_train_baseline_exp0_effbatch96) params="baseline, eff_batch 96 (accum=4), adam_eps=1e-6" ;;
    nyu_train_baseline_exp1_effbatch24) params="baseline, eff_batch 24 (accum=1), adam_eps=1e-6" ;;
    nyu_train_baseline_exp2_effbatch24_adameps0001) params="baseline, eff_batch 24, adam_eps=1e-3, legacy=true" ;;
    nyu_train_wordepth_paper_exp0_alter001) params="WorDepth, alter_prob=0.01" ;;
    nyu_train_wordepth_paper_exp1_alter05) params="WorDepth, alter_prob=0.5" ;;
    nyu_train_wordepth_paper_exp2_effbatch24) params="WorDepth, eff_batch 24, alter_prob=0.5" ;;
    nyu_train_wordepth_paper_exp3_effbatch24_adam_eps0001) params="WorDepth, eff_batch 24, adam_eps=1e-3" ;;
    nyu_train_baseline_relational_exp0_stat_alpha10_lambda01_margin05) params="Baseline+Rel, rel_weight=0.1, margin=0.5" ;;
    nyu_train_baseline_relational_exp1_stat_alpha10_lambda005_margin05) params="Baseline+Rel, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp0_alter001_stat_alpha10_lambda01_margin05) params="WorDepth+Rel, alter_prob=0.01, rel_weight=0.1, margin=0.5" ;;
    nyu_train_wordepth_relational_exp1_alter001_stat_alpha00_lambda01_margin05) params="WorDepth+Rel, rel_statistical_alpha=0, rel_weight=0.1" ;;
    nyu_train_wordepth_relational_exp2_alter05_stat_alpha10_lambda005_margin05) params="WorDepth+Rel, alter_prob=0.5, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp3_effbatch24_stat_alpha10_lambda005_margin05) params="WorDepth+Rel, eff_batch 24, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin00) params="WorDepth+Rel, eff_batch 24, rel_weight=0.05, margin=0.0" ;;
    nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin002) params="WorDepth+Rel, accum=2, margin=0.0, compile=false" ;;
    *) params="(see run config)" ;;
  esac
  printf '| %s | %s | %s |\n' "$name" "$params" "$metrics"
done

echo ""
echo "Done. Eval results (post_process=true) are in each run dir as eval_results_*best_abs_rel*.json"
