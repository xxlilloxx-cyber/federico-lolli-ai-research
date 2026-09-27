#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
for cfg in configs/baseline.yaml configs/lora.yaml configs/quadratic.yaml configs/signed_quadratic.yaml configs/feature_interaction.yaml; do
  .venv/bin/python src/train.py "$cfg" || echo "failed config=$cfg" >> results/failures.log
done
.venv/bin/python scripts/generate_report.py
