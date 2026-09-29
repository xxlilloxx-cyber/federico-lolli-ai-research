#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for rank in 1 2 4 8; do
 for method in lora symmetric_quadratic; do
  expected="results_rank_scaling_ablation/${method}_r${rank}_a${rank}_seed42_steps1000/summary.json"
  [[ -f "$expected" ]] && continue
  .venv/bin/python scripts/run_rank_factorial_controlled.py --method "$method" --rank "$rank" --alpha "$rank" --seed 42 --steps 1000 --output results_rank_scaling_ablation
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
 done
done
