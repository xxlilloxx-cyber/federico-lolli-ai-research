#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for cfg in config/historical/baseline.yaml config/historical/lora.yaml config/historical/quadratic.yaml config/historical/signed_quadratic.yaml config/historical/feature_interaction.yaml; do
  .venv/bin/python src/train.py "$cfg" || echo "failed config=$cfg" >> results/failures.log
done
.venv/bin/python scripts/aggregation/generate_report.py
