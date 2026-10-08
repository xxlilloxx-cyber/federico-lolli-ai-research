# Factual learning with LoRA and Symmetric adaptation

This isolated experiment asks whether a parallel LoRA + Symmetric adapter learns arbitrary new factual associations differently from either branch alone. It does not implement temporal noise, parameter drift, or consolidation.

## Conditions and placement

The frozen GPT-2 Small backbone is adapted at zero-based `transformer.h[0].attn.c_proj`, matching the primary controlled Project 01 placement. For a row activation `x`, the four conditions are:

- frozen: `W0(x)`;
- LoRA: `W0(x) + (alpha_L/r_L)(xA)B`;
- Symmetric: `W0(x) + (alpha_S/r_S)((xU) ⊙ (xU))P`;
- combined: the frozen output plus both independent branches.

The implementation reuses the validated `LowRankAdapter(kind="symmetric_quadratic")` for the single Symmetric condition. The combined module uses the same formula and exposes its LoRA (`A`, `B`) and Symmetric (`U`, `P`) factors independently.

For GPT-2 `c_proj` (`768 -> 768`), one rank-4 branch has 6,144 trainable parameters. The principal matched-budget comparison uses rank 4 for LoRA and Symmetric and ranks 2+2 for combined, so each adapted condition has 6,144 parameters. The same-rank configuration retains rank 4 in both combined branches and therefore has 12,288 parameters. The frozen baseline has zero trainable parameters.

## Data and measurements

The deterministic generator creates disjoint fact sets A and B from synthetic identifiers. Training, validation, and paraphrase prompts use distinct templates. Evaluation scores the complete target token sequence and records target NLL, geometric mean correct-token probability, sequence probability, and greedy exact match.

Phase 1 learns A. Phase 2 learns B, after which A is evaluated again to measure interference. A final evaluation after disabling all further optimization is labelled **frozen-adapter retention**. It is not described as long-term retention.

`best_validation_loss` and `best_validation_step` are stored separately from `final_validation_loss` and `final_step`. The latter refers to set-B validation at the final Phase-2 checkpoint.

## Commands

Run tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q
```

Run the short smoke protocol for all conditions:

```bash
for method in frozen lora symmetric combined; do
  .venv/bin/python experiments/factual_learning_lora_symmetric/run_experiment.py \
    --config experiments/factual_learning_lora_symmetric/configs/principal_matched_budget.json \
    --condition "$method" --seed 42 --facts-per-set 2 \
    --phase1-steps 4 --phase2-steps 4 --evaluation-every-steps 2 \
    --output-root results_factual_learning/smoke
done
```

Run the complete matched-budget campaign without rerunning completed cells:

```bash
for seed in 42 123 456; do
  for method in frozen lora symmetric combined; do
    .venv/bin/python experiments/factual_learning_lora_symmetric/run_experiment.py \
      --config experiments/factual_learning_lora_symmetric/configs/principal_matched_budget.json \
      --condition "$method" --seed "$seed"
  done
done
```

Aggregate and plot saved results:

```bash
.venv/bin/python experiments/factual_learning_lora_symmetric/aggregate_results.py \
  --input results_factual_learning/principal_matched_budget --expected-seeds 42 123 456
.venv/bin/python experiments/factual_learning_lora_symmetric/plot_results.py \
  --input results_factual_learning/principal_matched_budget
```

Every run writes `config.json`, `dataset.json`, `metrics.csv`, `summary.json`, logs, and adapter-only checkpoints where adaptation parameters exist. Output lives under the ignored `results_factual_learning/` tree.

The module boundaries leave room for later time-dependent perturbation and plastic/fixed consolidation components. Neither mechanism is active in this baseline.
