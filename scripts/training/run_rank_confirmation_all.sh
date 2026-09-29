#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
base='results_rank_factorial_controlled/confirmation_5000'
mkdir -p "$base"
for rank in 1 2 4 8; do
  for seed in 42 123 456; do
    for method in lora symmetric_quadratic; do
      .venv/bin/python scripts/training/run_rank_confirmation_cell.py --method "$method" --rank "$rank" --seed "$seed"
      nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
    done
  done
  complete=$(find "$base" -maxdepth 1 -name "*_r${rank}_seed*_steps5000" -exec test -f '{}/summary.json' \; -print | wc -l)
  printf '# Progress\n\nCompleted rank blocks are validated by required summary/config/metrics/final checkpoint/test artifacts.\n\n- rank %s: %s / 6 complete\n' "$rank" "$complete" > "$base/PROGRESS.md"
done
