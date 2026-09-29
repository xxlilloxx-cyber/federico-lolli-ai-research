# Controlled AG News experiment

- **Objective:** compare LoRA and Symmetric Quadratic on held-out
  classification accuracy and macro-F1.
- **Backbone/data:** frozen GPT-2; AG News.
- **Adapter:** rank 4 at zero-based block 0, `attn.c_proj`.
- **Budget:** 9,216 trainable parameters including the classification head.
- **Training:** 500 optimizer steps; seeds 42, 123, and 456.
- **Evaluation:** deterministic stratified validation subset and the official
  7,600-example test split. Test data did not select the method.
- **Private originals:** `results_downstream_controlled/full/` (ignored).
- **Protocol:** [`config/downstream/ag-news-controlled.json`](../../config/downstream/ag-news-controlled.json).
- **Canonical tables:** [`results/tables/downstream-ag-news/`](../../results/tables/downstream-ag-news/).
- **Canonical figures:** [`results/figures/downstream-ag-news/`](../../results/figures/downstream-ag-news/).
- **Report:** [DOWNSTREAM_CONTROLLED_RESULTS.md](DOWNSTREAM_CONTROLLED_RESULTS.md).

Reproduce with `scripts/training/run_downstream_remaining.sh`; the runner skips
valid completed cells. This is a training command and is not required to
reproduce the published plots from the canonical CSVs.
