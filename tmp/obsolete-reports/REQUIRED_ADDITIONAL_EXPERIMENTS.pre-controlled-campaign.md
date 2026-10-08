# Required additional experiments

No downstream accuracy/F1 benchmark, Hessian interaction metric, checkpoint spectral analysis, or inference-latency measurement is available in the existing artifacts. These must not be inferred from language-model loss.

## Recommended task-level evaluation

Use one small, public text-classification task with a declared GPT-2 classification head and identical LoRA/Symmetric parameter budgets. Record accuracy and macro-F1 across seeds 42, 123, and 456. This is a new experiment, not a reinterpretation of WikiText-2 loss.

## Commands after implementing the task runner

```bash
.venv/bin/python scripts/check_environment.py
PYTHONPATH=. .venv/bin/pytest -q
PYTHONPATH=. .venv/bin/python scripts/run_downstream_classification.py --adapter lora --seed 42
PYTHONPATH=. .venv/bin/python scripts/run_downstream_classification.py --adapter symmetric_quadratic --seed 42
```

## Optional analyses

Adapter checkpoint spectral analysis requires final adapter checkpoints and should report the singular values of LoRA's effective update and a clearly defined proxy for Symmetric. Hessian or activation-interaction analysis requires a separately defined sample and instrumentation. Neither exists in the completed result files.
