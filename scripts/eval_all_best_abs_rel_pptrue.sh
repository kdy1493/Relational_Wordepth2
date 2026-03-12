#!/usr/bin/env bash
# Evaluate all best_abs_rel checkpoints with post_process=false and post_process=true, then print a summary table.
# Run from repo root:  bash scripts/eval_all_best_abs_rel_pptrue.sh
#
# Writes two result files per run: eval_results_*_ppfalse.json and eval_results_*_pptrue.json.

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

EVAL_CONFIG_BASELINE="configs/arguments_eval_nyu_all_baseline.yaml"
EVAL_CONFIG_WORDEPTH="configs/arguments_eval_nyu_all_wordepth.yaml"
SUMMARY_JSON="$REPO_ROOT/runs/eval_all_best_abs_rel_pptrue_summary.json"

echo "=== Evaluating all best_abs_rel checkpoints (post_process=false and post_process=true) ==="

# Build list: run_dir|checkpoint_path
LIST=""
for d in runs/nyu_train_*/; do
  ckpt=$(ls "$d"model-*-best_abs_rel* 2>/dev/null | head -1)
  [ -n "$ckpt" ] && LIST="$LIST${d}|$ckpt"$'\n'
done

# Run eval for each: first pp=false, then pp=true (baseline runs use baseline config, else wordepth config)
# On eval failure we continue so other runs still run; failed runs are listed at the end.
idx=0
total=$(echo "$LIST" | grep -c . || true)
FAILED=""
while IFS= read -r line; do
  [ -z "$line" ] && continue
  run_dir="${line%%|*}"
  ckpt="${line#*|}"
  name=$(basename "${run_dir%/}")
  idx=$((idx + 1))
  echo "[$idx/$total] $name (pp=false, pp=true)"
  python3 -c "import torch; torch.cuda.empty_cache(); torch.cuda.synchronize()" 2>/dev/null || true
  run_eval() {
    if [[ "$name" == *"nyu_train_baseline"* ]]; then
      python3 src/eval.py "$EVAL_CONFIG_BASELINE" --checkpoint_path "$ckpt" "$@"
    else
      python3 src/eval.py "$EVAL_CONFIG_WORDEPTH" --checkpoint_path "$ckpt" "$@"
    fi
  }
  run_eval || { FAILED="$FAILED $name"; echo "[WARN] Eval failed: $name"; }
  run_eval --post_process || { FAILED="$FAILED $name"; echo "[WARN] Eval failed: $name"; }
done <<< "$LIST"
[ -n "$FAILED" ] && echo "=== Failed runs (re-run manually if needed):$FAILED"

echo ""
echo "=== Collecting results and printing table ==="

# Result files: eval_results_*_ppfalse.json and eval_results_*_pptrue.json
echo "| 이름 | 주요 파라미터 (변경된 것 위주) | Eval (pp=false) | Eval (pp=true) |"
echo "|------|-------------------------------|-----------------|----------------|"

for d in $(ls -d runs/nyu_train_*/ 2>/dev/null | sort); do
  json_ppfalse=$(ls "$d"eval_results_*best_abs_rel*_ppfalse.json 2>/dev/null | head -1)
  json_pptrue=$(ls "$d"eval_results_*best_abs_rel*_pptrue.json 2>/dev/null | head -1)
  [ -z "$json_ppfalse" ] && [ -z "$json_pptrue" ] && continue
  name=$(basename "${d%/}")

  parse_metrics() {
    local j="$1"
    if [ -z "$j" ] || [ ! -f "$j" ]; then
      echo "-"
      return
    fi
    local a s g r
    a=$(grep -o '"abs_rel": [0-9.]*' "$j" | head -1 | sed 's/"abs_rel": //')
    s=$(grep -o '"silog": [0-9.]*' "$j" | head -1 | sed 's/"silog": //')
    g=$(grep -o '"d1": [0-9.]*' "$j" | head -1 | sed 's/"d1": //')
    r=$(grep -o '"rms": [0-9.]*' "$j" | head -1 | sed 's/"rms": //')
    [ -z "$a" ] && a="-"; [ -z "$s" ] && s="-"; [ -z "$g" ] && g="-"; [ -z "$r" ] && r="-"
    [ "$a" != "-" ] && a=$(printf '%.4f' "$a")
    [ "$s" != "-" ] && s=$(printf '%.4f' "$s")
    [ "$g" != "-" ] && g=$(printf '%.4f' "$g")
    [ "$r" != "-" ] && r=$(printf '%.4f' "$r")
    echo "abs_rel=$a silog=$s δ1=$g rms=$r"
  }
  metrics_ppfalse=$(parse_metrics "$json_ppfalse")
  metrics_pptrue=$(parse_metrics "$json_pptrue")

  # Short param summary per run (change-focused)
  case "$name" in
    nyu_train_baseline_exp0_effbatch96) params="baseline, eff_batch 96 (accum=4), adam_eps=1e-6" ;;
    nyu_train_baseline_exp1_effbatch24) params="baseline, eff_batch 24 (accum=1), adam_eps=1e-6" ;;
    nyu_train_baseline_exp2_effbatch24_adameps0001) params="baseline, eff_batch 24, adam_eps=1e-3, legacy=true" ;;
    nyu_train_baseline_exp3_effbatch24_adameps0001_nolegacy) params="baseline, eff_batch 24, adam_eps=1e-3, legacy=false" ;;
    nyu_train_wordepth_paper_exp0_alter001) params="WorDepth, alter_prob=0.01" ;;
    nyu_train_wordepth_paper_exp1_alter05) params="WorDepth, alter_prob=0.5" ;;
    nyu_train_wordepth_paper_exp2_effbatch24) params="WorDepth, eff_batch 24, alter_prob=0.5" ;;
    nyu_train_wordepth_paper_exp3_effbatch24_adam_eps0001) params="WorDepth, eff_batch 24, adam_eps=1e-3" ;;
    nyu_train_wordepth_paper_exp4_adameps0001_weightkld00025) params="WorDepth, eff_batch 24, weight_kld=2.5e-3, alter_prob=0.5" ;;
    nyu_train_baseline_relational_exp0_stat_alpha10_lambda01_margin05) params="Baseline+Rel, rel_weight=0.1, margin=0.5" ;;
    nyu_train_baseline_relational_exp1_stat_alpha10_lambda005_margin05) params="Baseline+Rel, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp0_alter001_stat_alpha10_lambda01_margin05) params="WorDepth+Rel, alter_prob=0.01, rel_weight=0.1, margin=0.5" ;;
    nyu_train_wordepth_relational_exp1_alter001_stat_alpha00_lambda01_margin05) params="WorDepth+Rel, rel_statistical_alpha=0, rel_weight=0.1" ;;
    nyu_train_wordepth_relational_exp2_alter05_stat_alpha10_lambda005_margin05) params="WorDepth+Rel, alter_prob=0.5, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp3_effbatch24_stat_alpha10_lambda005_margin05) params="WorDepth+Rel, eff_batch 24, rel_weight=0.05, margin=0.5" ;;
    nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin00) params="WorDepth+Rel, eff_batch 24, rel_weight=0.05, margin=0.0" ;;
    nyu_train_wordepth_relational_exp4_effbatch24_stat_alpha10_lambda005_margin002) params="WorDepth+Rel, accum=2, margin=0.0, compile=false" ;;
    nyu_train_wordepth_relational_exp5_effbatch24_stat_alpha10_lambda001_margin00) params="WorDepth+Rel, eff_batch 24, rel_weight=0.01, margin=0.0" ;;
    nyu_train_wordepth_relational_exp6_effbatch24_stat_alpha10_lambda002_margin00) params="WorDepth+Rel, eff_batch 24, rel_weight=0.02, margin=0.0" ;;
    *) params="(see run config)" ;;
  esac
  printf '| %s | %s | %s | %s |\n' "$name" "$params" "$metrics_ppfalse" "$metrics_pptrue"
done

echo ""
echo "Done. Eval results are in each run dir: eval_results_*_ppfalse.json and eval_results_*_pptrue.json"
