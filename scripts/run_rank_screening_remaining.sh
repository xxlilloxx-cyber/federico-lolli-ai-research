#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for spec in 'symmetric_quadratic 2' 'lora 4' 'symmetric_quadratic 4' 'lora 8' 'symmetric_quadratic 8'; do
  read -r method rank <<<"$spec"
  destination="results_rank_factorial_controlled/screening/${method}_r${rank}_seed42_steps1000/summary.json"
  [[ -f "$destination" ]] && continue
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
  .venv/bin/python scripts/run_rank_factorial_controlled.py --method "$method" --rank "$rank" --seed 42 --steps 1000 --output results_rank_factorial_controlled/screening
  python3 - "$destination" <<'PY'
import json,math,sys
s=json.load(open(sys.argv[1])); assert s['steps']==1000 and math.isfinite(s['validation_loss'])
PY
  sync
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
done
