# Controlled language-modeling rank study: staged protocol

The full factorial design is 4 local ranks (1, 2, 4, 8) × 2 methods × 3
seeds = 24 fresh runs. The existing 5,000-step GPT-2/WikiText-2 runs required
approximately 766–853 seconds per run under comparable settings. A full
factorial therefore requires about 5.1–5.7 GPU-hours before evaluation and
retries, which is unsafe to launch as a single unattended batch on the 4 GB
GPU given the documented CUDA lockups.

## Stage 1: controlled screening

Run all 8 method/rank cells with seed 42 for 1,000 fresh optimizer steps,
using the historical WikiText-2 pipeline unchanged except for the explicit
new output directory `results_rank_factorial_controlled/screening/`.
This is a screening study, not a replacement for the full factorial and is
never combined with the historical 5,000-step data.

## Stage 2: confirmation

For every rank/method pattern selected *before test evaluation*, run seeds
42, 123, and 456 for 5,000 fresh steps. The exact full command template is:

```bash
.venv/bin/python scripts/run_rank_factorial_controlled.py \
  --method lora --rank 1 --seed 42 --steps 5000 \
  --output results_rank_factorial_controlled/full
```

Repeat for methods `lora`, `symmetric_quadratic`, ranks `1 2 4 8`, and seeds
`42 123 456`. The aggregator must require all three seeds per cell and compute
per-seed PPL as `exp(token_cross_entropy)` before aggregation.
