#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
for spec in 'lora 123' 'symmetric_quadratic 123' 'lora 456' 'symmetric_quadratic 456'; do
  read -r method seed <<<"$spec"
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
  .venv/bin/python scripts/training/run_controlled_downstream.py --method "$method" --rank 4 --seed "$seed" --steps 500 --train-samples 4096 --val-samples 1000 --test-samples 7600 --eval-every 100 --output results_downstream_controlled/full
  sync
  nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null
done
