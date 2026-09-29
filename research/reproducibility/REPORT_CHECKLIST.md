# Report checklist — final audit

- [x] Historical experiments retained and indexed in `MASTER_RESULTS.csv`.
- [x] 6,144-parameter fixed-budget study: 18/18 valid 2,000-step runs.
- [x] 12,288-parameter fixed-budget study: 18/18 valid 2,000-step runs.
- [x] Long convergence study: 18/18 valid 5,000-step runs.
- [x] Adapter-only checkpoint save at steps 1000, 2000, 3000, 4000, 5000 and final.
- [x] Checkpoint reload smoke test: maximum absolute logit error 0.0.
- [x] Checkpoint metadata mismatch rejection tested.
- [x] Validation curves, AULC, seed mean and standard deviation computed.
- [x] WikiText-2 public test evaluation for all 18 final adapters.
- [x] Frozen GPT-2 test baseline evaluated.
- [x] Validation-to-test generalization gaps computed.
- [x] CUDA lockup history retained separately and excluded from statistics.
- [x] Unit tests: 4 passed with `PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/pytest -q`.

## Not available or deliberately not claimed

- No test-set checkpoint was used for model selection; all test evaluations use final step 5000.
- No broad statistical significance claim is made from three seeds.
- No generalization claim beyond GPT-2 Small and WikiText-2 is made.
