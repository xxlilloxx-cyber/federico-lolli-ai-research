# Controlled rank confirmation: 5,000 steps

- **Objective:** independently confirm the screening pattern and evaluate the
  final checkpoint on held-out WikiText-2 test data.
- **Backbone/data:** frozen GPT-2; WikiText-2.
- **Methods/ranks:** LoRA and Symmetric Quadratic at ranks 1, 2, 4, and 8.
- **Placement:** zero-based block 0, `attn.c_proj`.
- **Budget:** 1,536 × rank adapter parameters.
- **Training:** 5,000 fresh optimizer steps; seeds 42, 123, and 456; 24 runs.
- **Definitions:** best recorded validation is the within-run minimum;
  final validation is a fresh step-5,000 evaluation; final test evaluates that
  same final checkpoint.
- **Private originals/checkpoints:**
  `results_rank_factorial_controlled/confirmation_5000/`.
- **Protocol:** [PROTOCOL.md](PROTOCOL.md) and
  [`config/rank/confirmation-5000.json`](../../config/rank/confirmation-5000.json).
- **Canonical tables/figures:** `results/tables/rank-confirmation-5000/` and
  `results/figures/rank-confirmation-5000/`.

Resume-safe execution is provided by
`scripts/training/run_rank_confirmation_cell.py` and
`scripts/training/run_rank_confirmation_all.sh`.
