# LoRA, Symmetric, and Combined Adaptation: Final Comparative Report

**Status:** complete  
**Campaign completion:** 134/134 cells  
**Generated:** 2026-10-04  
**Scope:** new Combined-extension study; historical and previously published results are preserved separately.

## Executive summary

This study evaluates a parallel **Combined** adapter containing independent LoRA and Symmetric branches. The primary Combined scaling is `alpha / (r_lora + r_symmetric)`, applied to the sum of both unscaled branch outputs. The principal question is whether the combination improves on either constituent method at matched adapter parameter budgets.

The controlled 5,000-step WikiText-2 results do **not** show a general Combined advantage. Symmetric has the lowest mean final validation and held-out test loss at every matched budget. Combined is worse than both single methods at 3,072 parameters, approximately tied with LoRA but worse than Symmetric at 6,144 parameters, and lies between Symmetric and LoRA at 12,288 parameters. At the largest budget, Combined beats LoRA in all three paired seeds but remains worse than Symmetric in all three.

AG News gives a different ordering. Combined has the highest mean accuracy and macro-F1, exceeding Symmetric in all three seeds, but LoRA has the lowest mean test loss. Combined exceeds LoRA accuracy and macro-F1 in two of three seeds. With only three seeds, these results support a task-dependent descriptive difference and do not establish statistical superiority.

Mechanism analysis shows that both Combined branches contribute measurably. At larger budgets, removing the Symmetric branch causes substantially more loss degradation than removing the LoRA branch. This helps explain why Combined approaches Symmetric, but it does not demonstrate synergy: the full Combined model still does not outperform pure Symmetric on controlled WikiText-2.

## 1. Experiment inventory and validation

| Campaign | Cells | Status | Scientific role |
|---|---|---|---|
| 5,000-step rank confirmation | 33 | Complete | Controlled |
| Scaling ablation | 22 | Complete | Exploratory; seed 42 |
| 12-block placement | 36 | Complete | Exploratory; 20 steps |
| Attention/MLP placement | 12 | Complete | Exploratory; unequal module dimensions |
| Fixed-budget multilayer | 15 | Complete | Exploratory; seed 42 |
| AG News | 9 | Complete | Controlled |
| Computational benchmark | 6 | Complete | Matched benchmark |
| Mechanism analysis | 1 analysis over 33 checkpoints | Complete | Post-training analysis |

All 134 planned cells have `complete: true`. No completed summary contains NaN or Inf, and all reported parameter counts match the configured budgets. The initial AG News attempt failed before its first optimizer update because the FP16 backbone and FP32 classification head were called without autocast. The runner was corrected to match the validated historical mixed-precision protocol, a separate one-step diagnostic passed, and all nine scientific AG News cells were then executed from fresh run directories. The failed attempt contributes no result.

## 2. Mathematical definitions and budgets

For a row activation `x`, the three branches are

- LoRA: `delta_L = (xA)B`
- Symmetric: `delta_S = ((xU) ⊙ (xU))P`
- Combined primary: `delta_C = [alpha/(r_L+r_S)] (delta_L + delta_S)`

The matrices `A/B` and `U/P` are independent. For GPT-2 Small `attn.c_proj`, each unit of rank contributes 1,536 parameters. The controlled matched budgets are:

| Adapter budget | LoRA | Symmetric | Combined |
|---|---|---|---|
| 3,072 | r=2 | r=2 | 1+1 |
| 6,144 | r=4 | r=4 | 2+2 |
| 12,288 | r=8 | r=8 | 4+4 |

The optional Combined branch-wise policy uses `alpha/r_L` and `alpha/r_S` separately and is reported only in the scaling ablation. It is not pooled with the primary comparison. The historical `Linear+Quadratic` implementation is not identical to Combined: it includes independent `U` and `V` interaction factors and different branch scaling. See [HISTORICAL_LINEAR_QUADRATIC_VS_COMBINED.md](HISTORICAL_LINEAR_QUADRATIC_VS_COMBINED.md).

## 3. Controlled WikiText-2 rank confirmation

Protocol: frozen GPT-2 Small; WikiText-2 causal language modeling; sequence length 128; zero-based block 0 `attn.c_proj`; AdamW; learning rate `3e-4`; batch size 1; gradient accumulation 4; clipping 1.0; no scheduler; `alpha=4`; seeds 42, 123, and 456; 5,000 fresh optimizer steps. Final validation is measured at step 5,000 and held-out test loss uses the same final checkpoint. Lower loss is better.

The run artifacts keep `best_validation_loss`, `best_validation_step`, and `final_validation_loss` separate. `best_validation_loss` is the minimum fresh validation measurement observed during training; it is not relabelled as the final result. The main comparison below uses final step-5,000 validation and the held-out test evaluation of that same final checkpoint.

### 3.1 All configurations

| Method | Total nominal rank | Final validation loss | Held-out test loss |
|---|---|---|---|
| LoRA | 1 | 3.5740 ± 0.0060 | 3.9260 ± 0.0009 |
| LoRA | 2 | 3.4782 ± 0.0078 | 3.8496 ± 0.0023 |
| LoRA | 4 | 3.3483 ± 0.0093 | 3.7657 ± 0.0055 |
| LoRA | 8 | 3.3051 ± 0.0117 | 3.7315 ± 0.0031 |
| Symmetric | 1 | 3.5325 ± 0.0168 | 3.9138 ± 0.0064 |
| Symmetric | 2 | 3.4356 ± 0.0245 | 3.8310 ± 0.0147 |
| Symmetric | 4 | 3.3121 ± 0.0031 | 3.7429 ± 0.0021 |
| Symmetric | 8 | 3.2452 ± 0.0086 | 3.6938 ± 0.0042 |
| Combined | 2 | 3.4846 ± 0.0216 | 3.8647 ± 0.0082 |
| Combined | 4 | 3.3486 ± 0.0252 | 3.7690 ± 0.0068 |
| Combined | 8 | 3.2720 ± 0.0107 | 3.7155 ± 0.0037 |

For completeness, the independently recorded best-validation summaries are:

| Method | Total nominal rank | Best recorded validation loss | Best steps by seed 42/123/456 |
|---|---|---|---|
| LoRA | 1 | 3.5320 ± 0.0044 | 4625, 4675, 4675 |
| LoRA | 2 | 3.4485 ± 0.0058 | 3950, 4675, 4675 |
| LoRA | 4 | 3.3264 ± 0.0074 | 4675, 3975, 4675 |
| LoRA | 8 | 3.2843 ± 0.0106 | 4675, 4675, 4675 |
| Symmetric | 1 | 3.5036 ± 0.0105 | 4625, 4800, 4675 |
| Symmetric | 2 | 3.4127 ± 0.0278 | 4975, 4675, 4675 |
| Symmetric | 4 | 3.2560 ± 0.0066 | 3950, 4675, 3950 |
| Symmetric | 8 | 3.2055 ± 0.0046 | 4675, 4675, 4675 |
| Combined | 2 | 3.4329 ± 0.0228 | 4625, 4675, 4675 |
| Combined | 4 | 3.3203 ± 0.0262 | 4175, 4975, 4675 |
| Combined | 8 | 3.2346 ± 0.0119 | 3975, 3950, 4675 |

### 3.2 Matched adapter budgets

| Budget | Method | Rank allocation | Final validation loss | Held-out test loss |
|---|---|---|---|---|
| 3,072 | LoRA | 2 | 3.4782 ± 0.0078 | 3.8496 ± 0.0023 |
| 3,072 | Symmetric | 2 | 3.4356 ± 0.0245 | 3.8310 ± 0.0147 |
| 3,072 | Combined | 1+1 | 3.4846 ± 0.0216 | 3.8647 ± 0.0082 |
| 6,144 | LoRA | 4 | 3.3483 ± 0.0093 | 3.7657 ± 0.0055 |
| 6,144 | Symmetric | 4 | 3.3121 ± 0.0031 | 3.7429 ± 0.0021 |
| 6,144 | Combined | 2+2 | 3.3486 ± 0.0252 | 3.7690 ± 0.0068 |
| 12,288 | LoRA | 8 | 3.3051 ± 0.0117 | 3.7315 ± 0.0031 |
| 12,288 | Symmetric | 8 | 3.2452 ± 0.0086 | 3.6938 ± 0.0042 |
| 12,288 | Combined | 4+4 | 3.2720 ± 0.0107 | 3.7155 ± 0.0037 |

### 3.3 Paired held-out differences

Differences are first method minus second method; negative values favor the first method.

| Budget | Comparison | Mean paired difference ± sample SD | First method lower in |
|---|---|---|---|
| 3,072 | Symmetric − LoRA | -0.0185 ± 0.0124 | 3/3 |
| 3,072 | Combined − LoRA | 0.0151 ± 0.0105 | 0/3 |
| 3,072 | Combined − Symmetric | 0.0337 ± 0.0228 | 0/3 |
| 6,144 | Symmetric − LoRA | -0.0229 ± 0.0041 | 3/3 |
| 6,144 | Combined − LoRA | 0.0033 ± 0.0049 | 1/3 |
| 6,144 | Combined − Symmetric | 0.0261 ± 0.0070 | 0/3 |
| 12,288 | Symmetric − LoRA | -0.0377 ± 0.0067 | 3/3 |
| 12,288 | Combined − LoRA | -0.0160 ± 0.0052 | 3/3 |
| 12,288 | Combined − Symmetric | 0.0217 ± 0.0022 | 0/3 |

**Interpretation.** Symmetric is lower than LoRA and Combined for every matched-budget mean. Combined shows no held-out advantage over Symmetric at any budget. Its comparison with LoRA changes with capacity: it is worse at 3,072 parameters, nearly tied at 6,144, and better at 12,288. This is evidence of a budget-dependent interaction, not evidence that summing the branches is generally beneficial.

## 4. Convergence

The table reconstructs fresh validation evaluations from the independent 5,000-step confirmation runs. `C−L` and `C−S` are Combined minus LoRA and Combined minus Symmetric.

| Budget | Step | LoRA | Symmetric | Combined | C−L | C−S |
|---|---|---|---|---|---|---|
| 3,072 | 1000 | 3.5982 | 3.5816 | 3.5941 | -0.0041 | +0.0125 |
| 3,072 | 2000 | 3.5445 | 3.5045 | 3.5537 | +0.0091 | +0.0491 |
| 3,072 | 3000 | 3.4987 | 3.4679 | 3.5114 | +0.0127 | +0.0435 |
| 3,072 | 4000 | 3.4674 | 3.4373 | 3.4691 | +0.0017 | +0.0319 |
| 3,072 | 5000 | 3.4782 | 3.4356 | 3.4846 | +0.0064 | +0.0490 |
| 6,144 | 1000 | 3.5594 | 3.4882 | 3.5348 | -0.0246 | +0.0465 |
| 6,144 | 2000 | 3.4681 | 3.4061 | 3.4695 | +0.0013 | +0.0633 |
| 6,144 | 3000 | 3.4158 | 3.3350 | 3.4106 | -0.0051 | +0.0756 |
| 6,144 | 4000 | 3.3470 | 3.2854 | 3.3506 | +0.0036 | +0.0652 |
| 6,144 | 5000 | 3.3483 | 3.3121 | 3.3486 | +0.0003 | +0.0364 |
| 12,288 | 1000 | 3.5451 | 3.4209 | 3.4552 | -0.0899 | +0.0342 |
| 12,288 | 2000 | 3.4456 | 3.3294 | 3.3589 | -0.0867 | +0.0295 |
| 12,288 | 3000 | 3.3674 | 3.2727 | 3.3056 | -0.0618 | +0.0329 |
| 12,288 | 4000 | 3.3031 | 3.2251 | 3.2531 | -0.0500 | +0.0280 |
| 12,288 | 5000 | 3.3051 | 3.2452 | 3.2720 | -0.0330 | +0.0268 |

Combined is consistently above Symmetric at all recorded checkpoints and budgets. Against LoRA, Combined changes ordering during training at the two smaller budgets and maintains an advantage throughout the recorded trajectory at 12,288 parameters. The size of this high-budget advantage narrows from `−0.0900` at step 1,000 to `−0.0330` at step 5,000 in validation loss. These finite trajectories do not establish asymptotic behavior.

## 5. Controlled AG News classification

Protocol: frozen GPT-2 Small; block 0 `attn.c_proj`; 4,096 training examples; deterministic 1,000-example stratified validation subset; original 7,600-example test split; 500 optimizer steps; rank-4 LoRA, rank-4 Symmetric, or Combined 2+2; 6,144 adapter parameters plus the shared 3,072-parameter classification head; seeds 42, 123, and 456.

| Method | Test accuracy | Test macro-F1 | Test loss |
|---|---|---|---|
| LoRA | 0.8186 ± 0.0129 | 0.8141 ± 0.0135 | 0.5446 ± 0.0637 |
| Symmetric | 0.8068 ± 0.0048 | 0.8002 ± 0.0055 | 0.6399 ± 0.0724 |
| Combined | 0.8241 ± 0.0121 | 0.8199 ± 0.0143 | 0.5521 ± 0.0211 |

### Paired differences

| Comparison | Accuracy difference | Macro-F1 difference | Test-loss difference |
|---|---|---|---|
| Symmetric − LoRA | -0.0118 ± 0.0128 | -0.0139 ± 0.0136 | 0.0953 ± 0.0839 |
| Combined − LoRA | 0.0055 ± 0.0079 | 0.0058 ± 0.0065 | 0.0075 ± 0.0429 |
| Combined − Symmetric | 0.0173 ± 0.0091 | 0.0197 ± 0.0117 | -0.0878 ± 0.0673 |

Combined has the highest mean accuracy and macro-F1, and it exceeds Symmetric on both metrics in all three seeds. It exceeds LoRA accuracy and macro-F1 in two of three seeds. LoRA retains the lowest mean test loss; Combined test loss is lower than LoRA in only one seed. Thus metric choice matters: the classification decisions favor Combined on average, while probabilistic calibration or confidence as summarized by cross-entropy does not. These are descriptive results with `n=3`.

These fresh extension results are independent of, and do not replace, the earlier published AG News campaign. Small numerical differences from that campaign may reflect the independently executed run environment and initialization path.

## 6. Scaling ablation

All scaling cells use seed 42 and 1,000 steps, so they are exploratory. `fixed_alpha` keeps `alpha=4`; `constant_scale` sets `alpha/r=1`; `total_rank` is the primary Combined policy; `branch_wise` scales each Combined branch separately.

| Method | Rank | Policy | Final validation | Final test |
|---|---|---|---|---|
| Combined | 1+1 | branch_wise | 3.6529 | 3.9719 |
| Combined | 2+2 | branch_wise | 3.5592 | 3.9120 |
| Combined | 4+4 | branch_wise | 3.4067 | 3.8021 |
| Combined | 1+1 | total_rank | 3.6148 | 3.9541 |
| Combined | 2+2 | total_rank | 3.5785 | 3.9247 |
| Combined | 4+4 | total_rank | 3.4346 | 3.8274 |
| LoRA | 1 | constant_scale | 3.6422 | 3.9777 |
| LoRA | 2 | constant_scale | 3.6133 | 3.9613 |
| LoRA | 4 | constant_scale | 3.5710 | 3.9240 |
| LoRA | 8 | constant_scale | 3.5161 | 3.8708 |
| LoRA | 1 | fixed_alpha | 3.6430 | 3.9764 |
| LoRA | 2 | fixed_alpha | 3.6120 | 3.9574 |
| LoRA | 4 | fixed_alpha | 3.5710 | 3.9240 |
| LoRA | 8 | fixed_alpha | 3.5563 | 3.9058 |
| Symmetric | 1 | constant_scale | 3.7024 | 4.0254 |
| Symmetric | 2 | constant_scale | 3.5719 | 3.9198 |
| Symmetric | 4 | constant_scale | 3.4932 | 3.8646 |
| Symmetric | 8 | constant_scale | 3.4278 | 3.8063 |
| Symmetric | 1 | fixed_alpha | 3.6944 | 4.0300 |
| Symmetric | 2 | fixed_alpha | 3.5657 | 3.9167 |
| Symmetric | 4 | fixed_alpha | 3.4932 | 3.8646 |
| Symmetric | 8 | fixed_alpha | 3.4461 | 3.8191 |

At the largest tested configuration, Combined 4+4 is slightly below Symmetric r8 under total-rank scaling in final validation (`3.4346` versus `3.4461`), and branch-wise scaling reduces it further to `3.4067`. This short single-seed result does not persist as superiority over Symmetric in the three-seed 5,000-step confirmation. Scaling materially affects magnitude and cannot be treated as a minor implementation detail.

## 7. Exploratory placement and multilayer results

### 7.1 Twelve-block scan

The scan uses only 20 optimizer steps and seed 42.

| Method | Best block | Best final validation | Worst block | Worst final validation | Range |
|---|---|---|---|---|---|
| LoRA | 7 | 3.8222 | 0 | 3.8289 | 0.0067 |
| Symmetric | 2 | 3.8033 | 1 | 3.8320 | 0.0287 |
| Combined | 5 | 3.7691 | 0 | 3.8303 | 0.0612 |

The scan identifies candidates but is too short for a rank ordering of methods or blocks. In particular, Combined block 5 has the lowest short-screen value, but this should not be interpreted as a confirmed placement advantage.

### 7.2 Attention versus MLP

| Method | Projection(s) | Trainable adapter parameters | Final validation |
|---|---|---|---|
| LoRA | mlp.c_fc+mlp.c_proj | 15,360 | 3.8201 |
| LoRA | mlp.c_proj | 15,360 | 3.8232 |
| LoRA | mlp.c_fc | 15,360 | 3.8270 |
| LoRA | attn.c_proj | 6,144 | 3.8289 |
| Symmetric | mlp.c_fc+mlp.c_proj | 15,360 | 3.8055 |
| Symmetric | mlp.c_proj | 15,360 | 3.8216 |
| Symmetric | attn.c_proj | 6,144 | 3.8265 |
| Symmetric | mlp.c_fc | 15,360 | 3.8288 |
| Combined | mlp.c_fc | 15,360 | 3.8165 |
| Combined | attn.c_proj | 6,144 | 3.8303 |
| Combined | mlp.c_proj | 15,360 | 3.8406 |
| Combined | mlp.c_fc+mlp.c_proj | 15,360 | 3.8478 |

Within each projection, the three methods are budget matched. Across attention and MLP projections, the parameter counts differ because GPT-2 MLP dimensions are larger; those rows are not a controlled placement comparison.

### 7.3 Fixed-budget multilayer study

| Budget | Method | Layers | Local rank | Final validation | Final test |
|---|---|---|---|---|---|
| 6,144 | LoRA | [0, 3, 7, 11] | 1 | 3.4137 | 3.7824 |
| 6,144 | Symmetric | [0, 11] | 2 | 3.4248 | 3.8089 |
| 6,144 | LoRA | [0, 11] | 2 | 3.4524 | 3.8123 |
| 6,144 | Symmetric | [2] | 4 | 3.4984 | 3.8512 |
| 6,144 | Symmetric | [0, 3, 7, 11] | 1 | 3.5163 | 3.8576 |
| 6,144 | LoRA | [2] | 4 | 3.5773 | 3.9137 |
| 12,288 | Symmetric | [0, 11] | 4 | 3.3415 | 3.7123 |
| 12,288 | Combined | [0, 3, 7, 11] | 1+1 | 3.3460 | 3.7538 |
| 12,288 | Symmetric | [0, 3, 7, 11] | 2 | 3.3537 | 3.7472 |
| 12,288 | LoRA | [0, 3, 7, 11] | 2 | 3.3644 | 3.7531 |
| 12,288 | Combined | [0, 11] | 2+2 | 3.4071 | 3.7770 |
| 12,288 | LoRA | [0, 11] | 4 | 3.4144 | 3.7727 |
| 12,288 | Symmetric | [2] | 8 | 3.4263 | 3.8008 |
| 12,288 | Combined | [2] | 4+4 | 3.4687 | 3.8260 |
| 12,288 | LoRA | [2] | 8 | 3.5704 | 3.9084 |

At 6,144 parameters, four-layer LoRA has the best final validation and test values among the executed geometries; no exact four-layer Combined geometry exists at that budget with both branches active and integer ranks. At 12,288 parameters, two-layer Symmetric is best on both metrics. Combined does not lead any matched-budget multilayer group. Because these are single-seed exploratory geometries, they motivate confirmation rather than a general placement conclusion.

## 8. Mechanism analysis

The analysis uses all 33 final rank checkpoints. Local Jacobians and output covariance are evaluated on the first 16 token positions of fixed WikiText-2 validation block 0. Combined branch ablations use the first 64 held-out test blocks without retraining.

### 8.1 Local effective rank and second-order response

| Budget | Method | Local Jacobian effective rank | Output-covariance effective rank | Adapter Hessian Frobenius norm |
|---|---|---|---|---|
| 3,072 | LoRA | 1.9214 ± 0.0507 | 1.9338 ± 0.0616 | 0 (analytical) |
| 3,072 | Symmetric | 1.6303 ± 0.0694 | 1.9642 ± 0.0340 | 396.77 ± 23.36 |
| 3,072 | Combined | 1.7894 ± 0.0273 | 1.7511 ± 0.0275 | 357.97 ± 99.52 |
| 6,144 | LoRA | 3.7599 ± 0.0859 | 3.2454 ± 0.1625 | 0 (analytical) |
| 6,144 | Symmetric | 3.0964 ± 0.1477 | 3.1289 ± 0.3077 | 188.93 ± 13.31 |
| 6,144 | Combined | 3.1640 ± 0.0571 | 2.9865 ± 0.2056 | 225.42 ± 8.65 |
| 12,288 | LoRA | 6.5718 ± 0.0851 | 4.4399 ± 0.1135 | 0 (analytical) |
| 12,288 | Symmetric | 5.5839 ± 0.2038 | 4.2847 ± 0.1955 | 90.85 ± 2.97 |
| 12,288 | Combined | 5.7483 ± 0.1539 | 4.4368 ± 0.0916 | 119.16 ± 3.45 |

LoRA has an input-independent Jacobian and an exactly zero adapter-only Hessian. Symmetric and Combined have input-dependent Jacobians. The Combined Jacobian effective rank lies between or near those of the constituent methods at matched budget. Hessian values depend on learned factors, scaling, and the fixed scalar reduction used by the analysis; they are not direct measures of useful interaction strength.

### 8.2 Combined branch behavior and inference-only ablation

Positive degradation means held-out loss increases when the named branch is disabled.

| Budget | Combined rank | LoRA output norm | Symmetric output norm | Combined output norm | Branch cosine | Disable LoRA: Δtest | Disable Symmetric: Δtest |
|---|---|---|---|---|---|---|---|
| 3,072 | 1+1 | 11.0370 ± 1.1118 | 11.4552 ± 3.3900 | 19.8281 ± 1.9823 | 0.1241 ± 0.0147 | 0.1392 ± 0.0602 | 0.0944 ± 0.0764 |
| 6,144 | 2+2 | 6.1155 ± 1.9664 | 14.4325 ± 3.9499 | 18.1381 ± 2.0457 | 0.1850 ± 0.0929 | 0.0876 ± 0.0435 | 0.2549 ± 0.0209 |
| 12,288 | 4+4 | 4.3574 ± 0.7159 | 11.1086 ± 0.9811 | 14.0387 ± 0.7079 | 0.4257 ± 0.0576 | 0.0606 ± 0.0100 | 0.3284 ± 0.0154 |

Both branches contribute non-redundant signal: disabling either branch increases mean held-out loss. At 3,072 parameters the relative ablation importance is variable. At 6,144 and 12,288 parameters, disabling Symmetric causes substantially more degradation than disabling LoRA. The average branch cosine is positive and rises with budget, while remaining far below one. This indicates partially aligned, non-identical branch outputs. It does not establish beneficial synergy because pure Symmetric still has lower controlled test loss.

## 9. Computational benchmark

Protocol: batch size 1, sequence length 128, 10 warm-up iterations, 30 measured iterations, identical hardware, and matched parameter budgets. Forward throughput is derived from the measured forward latency. Backward time measures backward execution after the training forward pass.

| Budget | Method | Forward ms | Backward ms | Tokens/s | Peak allocated VRAM MiB |
|---|---|---|---|---|---|
| 6,144 | LoRA | 14.039 ± 0.357 | 13.010 ± 3.753 | 9,124 ± 233 | 396.846 |
| 6,144 | Symmetric | 14.099 ± 0.495 | 13.044 ± 3.829 | 9,089 ± 294 | 396.848 |
| 6,144 | Combined | 13.433 ± 0.345 | 13.304 ± 3.717 | 9,535 ± 245 | 396.847 |
| 12,288 | LoRA | 14.096 ± 0.379 | 12.840 ± 3.808 | 9,087 ± 245 | 396.872 |
| 12,288 | Symmetric | 14.020 ± 0.386 | 13.006 ± 3.830 | 9,137 ± 250 | 396.875 |
| 12,288 | Combined | 13.231 ± 0.460 | 12.977 ± 3.794 | 9,686 ± 340 | 396.874 |

The measured memory differences are below `0.01 MiB`. Point estimates make Combined forward latency lower than the other methods in this run, while its backward latency is similar or slightly higher. Because the benchmark was a single sequential session rather than randomized repeated sessions across hardware states, the apparent Combined speed advantage should be treated as timing variation, not an architectural speed claim. The defensible conclusion is that no substantial cost difference was observed at these ranks on this hardware.

## 10. Answers to the scientific questions

1. **Does Combined learn more efficiently than LoRA?** At matched budget, only at 12,288 parameters in the 5,000-step WikiText-2 study; it is worse at 3,072 and essentially tied at 6,144.
2. **Does Combined outperform Symmetric?** No in the controlled 5,000-step WikiText-2 study. Symmetric is better at every budget and in all nine matched seed comparisons.
3. **Does Combined help downstream classification?** It has the best mean AG News accuracy and macro-F1, but not the best test loss, and it does not beat LoRA on every seed.
4. **Are both Combined branches used?** Yes. Their output norms are non-zero and disabling either branch increases held-out loss. The Symmetric branch is more influential at the two larger budgets.
5. **Is there evidence of synergy?** Not under the central WikiText-2 protocol: the combination does not exceed the stronger constituent, Symmetric. AG News suggests possible task-specific complementarity in classification decisions, requiring replication.
6. **Does scaling matter?** Yes. Branch-wise and total-rank scaling produce meaningfully different short-run outcomes.
7. **What is the computational overhead?** No substantial latency or allocated-memory penalty is resolved by the matched benchmark; differences are small relative to run variability and benchmark ordering effects.

## 11. Limitations

- Central comparisons use GPT-2 Small, WikiText-2, one adapter placement, four single-adapter ranks, three Combined allocations, and three seeds.
- The scaling, placement, attention/MLP, and multilayer studies are exploratory and mostly single-seed.
- The 20-step placement scans cannot support stable block rankings.
- AG News adds a classification head and uses a different objective and training duration, so task and protocol effects are not separable.
- No formal significance claim is made from three seeds. Effect sizes, paired consistency, and sample SD are reported instead.
- Mechanism quantities describe different mathematical objects; effective ranks must not be interpreted as identical notions of weight rank.
- Branch ablations cover a fixed 64-block held-out subset, not the entire WikiText-2 test split.
- The benchmark uses one hardware/software environment and one sequential measurement session.
- The manifest records base git commit `229035bb529c3b70b0fbf58aa6816d6f58b10d5c`, but the extension implementation was intentionally uncommitted during execution. Therefore that commit alone does not identify the exact experimental code. SHA-256 hashes below preserve the executed working-copy provenance.

## 12. Final conclusion

The Combined adapter is a valid matched-budget parallel composition whose two branches both learn useful corrections. Its behavior is not reducible to either constituent branch, as shown by input-dependent Jacobians, intermediate effective ranks, partially aligned branch outputs, and non-zero loss degradation when either branch is disabled.

The central empirical result is nevertheless conservative: **combining LoRA and Symmetric does not improve on pure Symmetric in the controlled WikiText-2 study**. Combined becomes better than LoRA only at the largest matched budget. On AG News it obtains the highest mean accuracy and macro-F1, but LoRA retains the lowest mean test loss and seed-level consistency is mixed against LoRA.

The evidence therefore supports Combined as a task- and budget-dependent alternative, not as a uniformly superior adapter. The clearest next experiment is a preregistered multi-seed AG News replication at both 6,144 and 12,288 adapter parameters, followed by an additional classification dataset, to determine whether the observed accuracy advantage represents repeatable branch complementarity.

## 13. Reproducibility and provenance

Primary sources:

- [Master configuration](config/full_combined_study.json)
- [Master manifest](results_combined_extension/MANIFEST.json)
- [Master results CSV](results_combined_extension/aggregate/MASTER_RESULTS.csv)
- [Rank aggregates](results_combined_extension/aggregate/rank_confirmation.csv)
- [AG News aggregates](results_combined_extension/aggregate/ag_news.csv)
- [Mechanism measurements](results_combined_extension/mechanism/checkpoint_analysis/metrics.csv)
- [Benchmark aggregates](results_combined_extension/aggregate/benchmark.csv)
- [Generated figures](results_combined_extension/figures/)

Executed-code hashes:

| File | SHA-256 |
|---|---|
| run_full_combined_study.py | 749e37a42046c5ae8be59a1baa503ada5e93a7abd7efc24d3c86937c4760c883 |
| config/full_combined_study.json | b4adf9c72cf934c6ed37d1883359048fd21880ae680c6a107d08830d29764855 |
| experiments/combined_extension/adapters.py | 84a2397a34a3dab477f21cb6671c8fde4364a3e31fde754e9d08f48b04222dd0 |
| experiments/combined_extension/run_wikitext_cell.py | a146a4a8f2a185e9be6aff1996c045bc9a05bb3fe2838dc7a49efea0ec6980b6 |
| experiments/combined_extension/run_agnews_cell.py | 47cb8786037e0459a323c3ed3ca8326553b29e790e0c2ccfab6d8ff6584e4527 |
| experiments/combined_extension/run_mechanism.py | 6252586d2c6efbaee759c6957c73e057b25ef5e72b45166cac3cd137bc30833a |
| experiments/combined_extension/run_benchmark_cell.py | 76c45f7fe87e1ac2583dbb605c1cf757c41082d7770a3d91a3905af6ef32d8d4 |


## Appendix A. Matched-budget per-seed WikiText-2 results

| Budget | Method | Seed | Final validation | Held-out test |
|---|---|---|---|---|
| 3,072 | LoRA | 42 | 3.481344 | 3.847219 |
| 3,072 | LoRA | 123 | 3.483978 | 3.851824 |
| 3,072 | LoRA | 456 | 3.469406 | 3.849625 |
| 3,072 | Symmetric | 42 | 3.410359 | 3.815829 |
| 3,072 | Symmetric | 123 | 3.459203 | 3.845244 |
| 3,072 | Symmetric | 456 | 3.437343 | 3.831965 |
| 3,072 | Combined | 42 | 3.503466 | 3.873851 |
| 3,072 | Combined | 123 | 3.461064 | 3.857961 |
| 3,072 | Combined | 456 | 3.489316 | 3.862199 |
| 6,144 | LoRA | 42 | 3.353692 | 3.764799 |
| 6,144 | LoRA | 123 | 3.353632 | 3.771615 |
| 6,144 | LoRA | 456 | 3.337469 | 3.760781 |
| 6,144 | Symmetric | 42 | 3.315170 | 3.744120 |
| 6,144 | Symmetric | 123 | 3.309046 | 3.744011 |
| 6,144 | Symmetric | 456 | 3.312208 | 3.740500 |
| 6,144 | Combined | 42 | 3.322608 | 3.762623 |
| 6,144 | Combined | 123 | 3.373011 | 3.776229 |
| 6,144 | Combined | 456 | 3.350038 | 3.768160 |
| 12,288 | LoRA | 42 | 3.315405 | 3.733781 |
| 12,288 | LoRA | 123 | 3.307320 | 3.732670 |
| 12,288 | LoRA | 456 | 3.292436 | 3.727968 |
| 12,288 | Symmetric | 42 | 3.247288 | 3.694530 |
| 12,288 | Symmetric | 123 | 3.235735 | 3.689251 |
| 12,288 | Symmetric | 456 | 3.252611 | 3.697618 |
| 12,288 | Combined | 42 | 3.277101 | 3.718241 |
| 12,288 | Combined | 123 | 3.279230 | 3.711287 |
| 12,288 | Combined | 456 | 3.259701 | 3.716905 |
