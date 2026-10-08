# Quadratic Transformer — Controlled Experiments Progress Report

**Date:** 28 September 2026  
**Repository branch:** `public-release-preparation`  
**Status:** downstream comparison complete; rank study screening complete; multi-seed rank confirmation, spectral analysis, interaction analysis, and controlled cost study have not yet been run.

## Purpose

This phase tests whether the activation-dependent **Symmetric Quadratic Adapter** is better, equivalent, or worse than standard **LoRA** under controlled conditions. It does not assume that a quadratic adapter is superior.

For a frozen projection with row-vector input \(x\), LoRA adds

\[
\Delta y_{\mathrm{LoRA}}=\frac{\alpha}{r}(xA)B.
\]

The Symmetric Quadratic Adapter adds

\[
z=xU, \qquad \Delta y_{\mathrm{Sym}}=\frac{\alpha}{r}(z\odot z)P.
\]

The output is initially unchanged because the output-side factor is initialized to zero. The methods were placed at the same frozen GPT-2 projection, `transformer.h[0].attn.c_proj`.

## 1. Completed controlled downstream experiment

### Design

| Item | Value |
|---|---|
| Dataset | AG News, four-class text classification |
| Backbone | GPT-2 (`gpt2`) sequence-classification model |
| Adapter placement | `transformer.h[0].attn.c_proj`, block 0 |
| Adapter rank | 4 for both methods |
| Trainable parameters | 9,216 for both methods, including the same classification head |
| Training data | 4,096 deterministic training examples |
| Validation data | fixed 1,000-example stratified split from AG News training data |
| Test data | official AG News test split, evaluated only after training |
| Training | 500 optimizer steps; batch size 1; gradient accumulation 4; AdamW; learning rate 3e-4 |
| Seeds | 42, 123, 456 |

All six runs produced finite losses and complete `config.json`, `metrics.csv`, `summary.json`, and final checkpoint artifacts.

### Individual held-out test results

| Method | Seed | Accuracy | Macro-F1 | Test loss |
|---|---:|---:|---:|---:|
| LoRA | 42 | 0.8143 | 0.8099 | 0.6047 |
| LoRA | 123 | 0.8217 | 0.8177 | 0.5294 |
| LoRA | 456 | 0.8242 | 0.8192 | 0.4996 |
| Symmetric Quadratic | 42 | 0.8195 | 0.8152 | 0.5984 |
| Symmetric Quadratic | 123 | 0.7747 | 0.7568 | 0.9046 |
| Symmetric Quadratic | 456 | 0.8122 | 0.8067 | 0.5511 |

### Aggregate held-out results

| Method | Accuracy, mean ± sample SD | Macro-F1, mean ± sample SD | Test loss, mean ± sample SD |
|---|---:|---:|---:|
| LoRA | 0.8201 ± 0.0051 | 0.8156 ± 0.0050 | 0.5445 ± 0.0542 |
| Symmetric Quadratic | 0.8021 ± 0.0240 | 0.7929 ± 0.0315 | 0.6847 ± 0.1919 |

Paired Symmetric-minus-LoRA averages are **−0.0179** accuracy, **−0.0227** macro-F1, and **+0.1401** test loss. Symmetric has higher accuracy and macro-F1 for only one matched seed (42); LoRA is higher for seeds 123 and 456.

### Interpretation

Under this specific controlled AG News protocol, the evidence favors LoRA on average. The observed variability of Symmetric across seeds is also larger. With only three seeds, this does not establish a formal significance claim, but it clearly does **not** support a general downstream advantage for Symmetric Quadratic.

This outcome is compatible with the earlier WikiText-2 result: the language-modeling behavior depended on placement, local rank, depth allocation, and training duration. It should not be interpreted as a contradiction or as evidence that one adapter universally dominates the other.

Sources: `docs/data/downstream_controlled_results.csv`, `docs/data/downstream_controlled_aggregate.csv`, and `docs/data/downstream_controlled_paired_differences.csv`.

## 2. Controlled language-modeling rank screening

### Design

This is a new, intentionally separate **screening** experiment, not a final multi-seed conclusion.

| Item | Value |
|---|---|
| Dataset/task | WikiText-2 causal language modeling |
| Backbone | GPT-2 frozen backbone |
| Placement | `transformer.h[0].attn.c_proj`, block 0 |
| Methods | LoRA and Symmetric Quadratic |
| Ranks | 1, 2, 4, 8 |
| Seed | 42 only |
| Training length | 1,000 fresh optimizer steps |
| Protocol | sequence length 128, gradient accumulation 4, AdamW, LR 3e-4, alpha 4 |

### Final validation loss at 1,000 steps

| Rank | LoRA | Symmetric Quadratic | Symmetric − LoRA |
|---:|---:|---:|---:|
| 1 | 3.6171 | 3.6614 | +0.0443 |
| 2 | 3.5861 | 3.5561 | −0.0300 |
| 4 | 3.5530 | 3.4780 | −0.0750 |
| 8 | 3.5492 | 3.4315 | −0.1177 |

### Interpretation

At seed 42 and 1,000 steps, Symmetric is worse at rank 1 but lower in validation loss at ranks 2, 4, and 8. The gap grows numerically with rank in this screening. This is potentially informative, but it is **not sufficient evidence** of a rank effect because it contains one seed only and a shorter training protocol than the historical 5,000-step study.

The appropriate next experiment is a predeclared multi-seed confirmation: 5,000 fresh steps for ranks 1, 2, 4, and 8, methods LoRA and Symmetric, seeds 42/123/456. This is 24 runs and was estimated at roughly 5–6 GPU-hours on the available RTX A500 before validation/test overhead. It must remain separate from historical experiments and from this screening table.

Sources: `results_rank_factorial_controlled/screening/*/summary.json` and `metrics.csv` (local experimental artifacts; no aggregate public CSV yet).

## 3. Relation to completed historical evidence

The repository’s previous 5,000-step WikiText-2 study compared LoRA and Symmetric Quadratic at a matched 12,288-parameter adapter budget across concentrated, two-layer, and four-layer placements. Its held-out results were placement-dependent: Symmetric had the strongest matched-placement test advantage in the concentrated configuration, was close in the two-layer configuration, and LoRA had lower test loss in the four-layer configuration.

The new AG News result and rank screen add controlled evidence; they do not replace or alter any historical run. A strong scientific claim would require the planned rank confirmation and analyses of adapter spectra, interaction structure, and cost.

## 4. Current conclusion

The data currently support the narrower statement that activation-dependent low-rank quadratic adaptation is trainable under small parameter budgets and can improve language-modeling loss in some placements. They do not support universal superiority over LoRA. In the completed AG News classification comparison, LoRA had better average held-out accuracy, macro-F1, and test loss.

## 5. Next steps

1. Aggregate the rank screening into a reproducible CSV and figure.
2. Run the predeclared multi-seed rank confirmation or explicitly retain the screening as exploratory if compute is unavailable.
3. Use final checkpoints from controlled experiments for a correctly defined spectral analysis.
4. Compute an adapter-level derivative/Hessian interaction analysis, explicitly distinguishing the nonlinear adapter from the surrounding Transformer.
5. Measure training and inference cost under identical conditions.

No remote push, merge, deployment, or publication has been performed.
