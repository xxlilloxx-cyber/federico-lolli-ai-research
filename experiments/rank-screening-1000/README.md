# Controlled rank screening: 1,000 steps

- **Objective:** screen ranks 1, 2, 4, and 8 under one controlled protocol.
- **Backbone/data:** frozen GPT-2; WikiText-2 causal language modeling.
- **Methods:** LoRA and Symmetric Quadratic.
- **Placement:** zero-based block 0, `attn.c_proj`.
- **Budget:** 1,536 × rank trainable adapter parameters.
- **Training:** 1,000 fresh optimizer steps; seeds 42, 123, and 456; 24 runs.
- **Evaluation:** fresh validation every 25 steps. This screen did not provide
  the independent final-checkpoint held-out evidence of the later confirmation.
- **Private originals:** `results_rank_factorial_controlled/screening/`.
- **Protocol:** [`config/rank/screening-1000.json`](../../config/rank/screening-1000.json).
- **Canonical tables/figures:** `results/tables/rank-screening-1000/` and
  `results/figures/rank-screening-1000/`.

Use `scripts/training/run_rank_factorial_controlled.py` for an individual cell.
