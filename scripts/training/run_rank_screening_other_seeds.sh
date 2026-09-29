#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for seed in 123 456; do
  for rank in 1 2 4 8; do
    for method in lora symmetric_quadratic; do
      destination="results_rank_factorial_controlled/screening/${method}_r${rank}_seed${seed}_steps1000/summary.json"
      [[ -f "$destination" ]] && continue
      nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
      .venv/bin/python scripts/training/run_rank_factorial_controlled.py --method "$method" --rank "$rank" --seed "$seed" --steps 1000 --output results_rank_factorial_controlled/screening
      python3 - "$destination" <<'PY'
import json, math, sys
s=json.load(open(sys.argv[1])); assert s['steps']==1000 and math.isfinite(s['validation_loss'])
PY
      sync
      nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
    done
  done
done
