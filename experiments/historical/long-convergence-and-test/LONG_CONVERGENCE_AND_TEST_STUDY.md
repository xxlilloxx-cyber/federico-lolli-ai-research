# Long Convergence and Held-out Test Study

**Author:** Federico Lolli  
**Public contact:** xxlilloxx@gmail.com  
**Scope:** final 12,288-parameter, 5,000-step campaign.

## 1. Status and definitions

This is the final state of the long campaign: 18/18 valid runs, covering LoRA and Symmetric Quadratic, concentrated/two-layer/four-layer placement, and seeds 42, 123, and 456. It supersedes earlier interim wording that described missing checkpoints, missing test evaluation, or incomplete runs. CUDA lockups from earlier campaigns remain operational history only; no failed run appears in the tables below.

For every seed, `best_recorded_validation_loss` is the minimum fresh validation value; `final_validation_loss` is the fresh result at step 5,000; `final_test_loss` evaluates that exact final checkpoint. Means and sample SDs aggregate unrounded per-seed values. Test and validation perplexity means are means of per-seed perplexities.

## 2. Checkpointing and protocol

Adapter-only checkpoints were saved at 1,000, 2,000, 3,000, 4,000 and 5,000 steps. They include adapter state and compatibility metadata; frozen GPT-2 is reconstructed from pretrained weights. Reloading a saved adapter yielded maximum absolute logits error 0.0 on the smoke-test batch. Mismatched architecture, rank, layers, or metadata are rejected.

Runs use GPT-2 Small, frozen backbone, WikiText-2 causal language modelling, 128-token blocks, batch 1, accumulation 4, AdamW 3e-4, clipping 1.0, FP16 backbone and FP32 adapters. The public test set was evaluated only after final checkpoint selection; it was not used to choose architecture, placement, rank, or stopping time.

## 3. Exact final-step validation

| Geometry | Model | Final validation loss | Final validation PPL | Best recorded validation loss |
|---|---|---:|---:|---:|
| Concentrated [2], r=8 | LoRA | 3.3703 ± 0.0045 | 29.09 ± 0.13 | 3.3577 ± 0.0041 |
| Concentrated [2], r=8 | Symmetric | 3.2831 ± 0.0080 | 26.66 ± 0.21 | 3.2559 ± 0.0048 |
| Two [0,11], r=4 | LoRA | 3.2186 ± 0.0166 | 25.00 ± 0.42 | 3.2003 ± 0.0178 |
| Two [0,11], r=4 | Symmetric | 3.1876 ± 0.0119 | 24.23 ± 0.29 | 3.1520 ± 0.0126 |
| Four [0,3,7,11], r=2 | LoRA | 3.1743 ± 0.0162 | 23.91 ± 0.39 | 3.1537 ± 0.0204 |
| Four [0,3,7,11], r=2 | Symmetric | 3.1925 ± 0.0157 | 24.35 ± 0.38 | 3.1557 ± 0.0320 |

The corresponding final Symmetric-minus-LoRA differences are -0.0872, -0.0310, and +0.0182. The apparently near-zero +0.0020 four-layer difference belongs to **best-recorded** validation, not final step 5,000.

## 4. Held-out test and generalization description

The frozen base model obtains test loss 4.1279 / PPL 62.05 on 2,213 128-token blocks. Final-checkpoint adapter results are:

| Geometry | Model | Final test loss | Final test PPL | Test − final validation |
|---|---|---:|---:|---:|
| Concentrated | LoRA | 3.7741 ± 0.0022 | 43.56 ± 0.10 | 0.4037 ± 0.0036 |
| Concentrated | Symmetric | 3.7067 ± 0.0060 | 40.72 ± 0.24 | 0.4236 ± 0.0042 |
| Two-layer | LoRA | 3.6361 ± 0.0075 | 37.95 ± 0.29 | 0.4176 ± 0.0092 |
| Two-layer | Symmetric | 3.6306 ± 0.0025 | 37.74 ± 0.09 | 0.4431 ± 0.0142 |
| Four-layer | LoRA | 3.6192 ± 0.0115 | 37.31 ± 0.43 | 0.4449 ± 0.0086 |
| Four-layer | Symmetric | 3.6452 ± 0.0149 | 38.29 ± 0.57 | 0.4527 ± 0.0044 |

**Observation.** The concentrated test PPL decreases from 43.5567 to 40.7165 (6.52%) under the same 12,288-parameter budget. Two-layer test results are close. Four-layer LoRA has lower held-out loss. The paired test–validation difference is larger for Symmetric in all geometries. **Interpretation.** This is consistent with a placement-dependent generalization difference, but cannot identify its cause or establish overfitting.

## 5. Trajectories and AULC

The canonical public trajectory data contain 3,618 rows: a genuine pre-update value (3.8293571472) and 200 scheduled fresh validations per run. Step 1 is post-update and deliberately excluded. In raw metrics each run contains 201 fresh post-update measurements (step 1 plus steps 25…5000) and 4,799 cached rows. The shared step-0 value is a repeated deterministic frozen-model evaluation, not 18 independent baselines.

Legacy AULC is an arithmetic mean of all 5,000 logged per-step validation columns, including cached values and excluding step 0; it is not a numerical integral. It is retained only for comparisons made under the same schedule. Current figures use only fresh public validation points and never smooth endpoints.

## 6. Conclusion and limits

Symmetric Quadratic is trainable and beneficial for the concentrated matched placement. Its margin narrows as the fixed rank budget is distributed across depth and the four-layer held-out result favours LoRA. The study is limited to one model, one corpus, three seeds, one optimizer protocol, short context, and the `attn.c_proj` long-run placement. It does not establish a universal winner.

## 7. Data, figures, and references

The authoritative publication-safe data are `docs/data/long_final_per_seed.csv`, `long_final_aggregate.csv`, and `long_validation_trajectories.csv`; figure provenance is `docs/data/FIGURE_MANIFEST.md`. See `TECHNICAL_REPORT.md` for the mathematical derivation, full methods, historical studies, and bibliography. Related work includes Hu et al. (2022) LoRA, Li et al. (2024) LoRAN, and Zhang et al. (2026) PERA; LoRAN and PERA are not experimental baselines in this study.
