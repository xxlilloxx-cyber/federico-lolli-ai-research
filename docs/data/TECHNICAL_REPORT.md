# Quadratic Transformer — Technical Scientific Report

**Author:** Federico Lolli  
**Contact:** xxlilloxx@gmail.com  
**Publication scope:** Original report text and original figures are CC BY 4.0; source code is MIT-licensed. Third-party models, datasets, libraries, and cited works retain their own terms.

## Data-definition correction for the final 5,000-step study

The historic `summary.json` field named `validation_loss` is the **best recorded validation loss**, not necessarily the value at optimizer step 5,000. Earlier prose and figures mixed this value with final-step curves. The corrected public pipeline exports `long_final_per_seed.csv` and `long_validation_trajectories.csv`: it labels best-recorded and final-step validation separately, inserts the common pre-update step-0 measurement, and plots only fresh validation passes. The true final-step means are LoRA/Symmetric: concentrated 3.3703/3.2831, two-layer 3.2186/3.1876, four-layer 3.1743/3.1925. Held-out test still uses the final step-5,000 checkpoint.

With row-vector convention, the frozen projection is `y_base=xW+b`; the actual LoRA branch is `(alpha/r)(xA)B` and the Symmetric branch is `(alpha/r)(xU)^(⊙2)P`. Thus `Q_j=(alpha/r) sum_k P_kj u_k u_k^T`; rank(Q_j) is at most r, the same directions are shared across outputs, and signed P coefficients mean Q_j need not be positive semidefinite. The final campaigns use alpha=4, giving local branch scales 0.5, 1, and 2 at ranks 8, 4, and 2 respectively.

## 1. Executive Summary

This project studies whether a frozen Transformer can be adapted more effectively with a low-rank second-order correction than with standard LoRA, when the trainable parameter budget is comparable. The central comparison is deliberately empirical: the experiments do not assume that quadratic adapters are superior. The study progressed from a frozen GPT-2 baseline and short single-layer adapter tests to parameter-matched depth scans, fixed-budget multi-layer placement, local-rank analysis, long 5,000-step training, adapter-only checkpointing, and held-out WikiText-2 evaluation.

The backbone is GPT-2 Small (about 124M parameters). Its weights are frozen in adapter experiments. LoRA adds a linear low-rank correction, while the main quadratic model, Symmetric Quadratic Interaction, adds a low-rank sum of squared projections. All long-run comparisons use 12,288 trainable parameters, the same data order within each seed, the same optimizer protocol, and three seeds (42, 123, 456).

The final 5,000-step results show a placement-dependent pattern. With one concentrated insertion point, Symmetric reaches validation loss 3.2559 ± 0.0048 versus 3.3577 ± 0.0041 for LoRA, and test loss 3.7067 ± 0.0060 versus 3.7741 ± 0.0022. With two distributed insertion points, the validation advantage remains smaller (3.1520 versus 3.2003), while the test results are nearly tied (3.6306 versus 3.6361). With four distributed insertion points, validation is effectively tied (3.1557 versus 3.1537) and the held-out test favors LoRA (3.6452 versus 3.6192). Therefore the evidence does not support universal superiority of either function class.

The same conclusion is visible over optimization time. At 2,000 steps the Symmetric-minus-LoRA validation differences were -0.1622, -0.0767 and -0.0222 for concentrated, two-layer and four-layer placement. At 5,000 steps they were -0.1018, -0.0483 and +0.0020. Symmetric retains lower area-under-the-learning-curve (AULC) in all three geometries, but LoRA closes the endpoint gap as training continues, especially when the budget is distributed. This separates convergence behavior from final endpoint performance.

The scientifically supported statement is therefore: **explicit quadratic adaptation showed a strong advantage when capacity was concentrated, but that advantage decreased as capacity was distributed across depth and as optimization continued. At four distributed insertion points, the 5,000-step validation endpoints were effectively tied and held-out WikiText-2 favored LoRA. The results support an interaction among function class, local rank, depth distribution, optimization duration and generalization, rather than a universal winner.**

## 2. Motivation and research question

A Transformer projection normally computes
\[
y=xW,
\]
where the input feature vector \(x\) is multiplied by a frozen weight matrix \(W\). Adapter methods learn a small correction while leaving \(W\) unchanged. LoRA uses
\[
y=xW+\alpha xUV,
\]
with low-rank factors \(U\in\mathbb{R}^{d\times r}\) and \(V\in\mathbb{R}^{r\times d_{out}}\). The correction is linear in \(x\), although it can still be useful because it changes the effective projection matrix.

The experimental question is whether a similarly compact correction can use second-order information. The progression studied is:

\[
x_i\;\rightarrow\;x_i^2\;\rightarrow\;x_i|x_i|\;\rightarrow\;x_ix_j\;\rightarrow\;x^TQx\;\rightarrow\;a^Tx+x^TQx.
\]

The important comparison is not simply “nonlinear versus linear.” A fair comparison must keep the backbone, data, optimization and trainable parameter budget controlled. The study therefore includes same-rank comparisons, matched-budget comparisons, placement studies and multiple seeds.

## 3. Mathematical models

### 3.1 Frozen Transformer baseline

\[
y=xW.
\]
No adapter parameters are trained. This model establishes the frozen-backbone validation, test, memory and throughput reference.

### 3.2 LoRA

\[
\Delta y=\alpha(xA)B=\alpha xAB,
\qquad y=xW+\Delta y.
\]
For rank \(r\), the trainable parameter count is
\[
N_{LoRA}=r(d+d_{out}).
\]
The output factor is zero-initialized, so the initial adapter contribution is zero while the input factor remains trainable.

### 3.3 Element-wise Quadratic Adapter

\[
\Delta y=\alpha(x\odot x)UV.
\]
The square is element-wise: \((x\odot x)_i=x_i^2\), never \(x@x\). It can represent squared individual features followed by a low-rank output map. Its parameter count is \(r(d+d_{out})\). One output factor is zero-initialized.

### 3.4 Signed Quadratic Adapter

\[
\Delta y=\alpha(x\odot|x|)UV.
\]
For positive input values this is \(x^2\); for negative values it is \(-x^2\). The transformation preserves the sign of the original activation while retaining quadratic magnitude. Its parameter count and zero-preserving initialization match the element-wise quadratic model.

### 3.5 Feature-Interaction Adapter

\[
\Delta y=\alpha[(xU)\odot(xV)]P.
\]
For latent coordinate \(k\),
\[
(xU)_k=\sum_i x_iU_{ik},\qquad (xV)_k=\sum_jx_jV_{jk},
\]
so
\[
[(xU)\odot(xV)]_k=\sum_i\sum_j U_{ik}V_{jk}x_ix_j.
\]
The latent product contains both squares and cross-feature terms without materializing a full \(d\times d\) quadratic matrix. The parameter count is \(r(2d+d_{out})\), larger than LoRA at the same rank; matched-budget comparisons therefore use adjusted ranks.

### 3.6 Symmetric Quadratic Interaction

\[
\Delta y=\alpha(xU)^{\odot2}P,
\qquad U\in\mathbb{R}^{d\times r},\ P\in\mathbb{R}^{r\times d_{out}}.
\]
For output \(l\),
\[
\Delta y_l=\sum_kP_{kl}(x^Tu_k)^2=x^TQ_lx,
\]
where
\[
Q_l=\sum_kP_{kl}u_ku_k^T.
\]
Each \(u_ku_k^T\) is rank one and symmetric, so \(Q_l\) is a symmetric low-rank quadratic form represented implicitly. The parameter count is exactly
\[
N_{Sym}=r(d+d_{out}),
\]
which equals LoRA at the same rank. No complete quadratic matrix is stored. The output factor \(P\) is zero-initialized, preserving the original model at step zero.

This is the key analogy:
\[
\text{LoRA: }\Delta W=AB
\qquad\text{versus}\qquad
\text{Symmetric: }Q_l=\sum_kP_{kl}u_ku_k^T.
\]
The first is low-rank linear adaptation; the second is low-rank quadratic adaptation.

### 3.7 Linear + Quadratic Adapter

\[
\Delta y=\alpha_LxAB+\alpha_Q[(xU)\odot(xV)]P.
\]
The linear branch can represent first-order corrections, while the interaction branch represents second-order terms. Their output norms were instrumented separately in the earlier experiments. This model was included to test complementarity rather than to claim that either branch is always necessary.

## 4. Architecture and implementation

All experiments use GPT-2 Small: 12 Transformer blocks, hidden size 768, 12 attention heads, and feed-forward width 3072, with the GPT-2 tokenizer. Candidate modules were inspected programmatically. `attn.c_proj` is 768→768; `mlp.c_fc` is 768→3072; `mlp.c_proj` is 3072→768. The first single-layer experiments targeted `transformer.h[0].attn.c_proj`; later depth and placement studies tested attention and MLP modules explicitly.

The frozen backbone is loaded from the pretrained checkpoint. Only adapter parameters have `requires_grad=True`; optimizer state is therefore not allocated for the backbone. The backbone uses FP16 autocast where supported, while adapter parameters are retained in FP32. The training code reports total, trainable and frozen parameters, peak CUDA allocation, gradient norms, adapter output norms and NaN/Inf counters.

Initialization is identity-preserving: the final factor of each adapter branch starts at zero. Automated tests verify dimensional correctness, finite gradients, frozen-backbone gradients and equality of the original and modified outputs at initialization within numerical tolerance. The checkpoint implementation stores adapter weights plus metadata (architecture, rank, layers, scaling, seed and training step), without duplicating GPT-2 weights. Reload validation produced maximum absolute logits error 0.0 and rejects incompatible architecture, rank, layer or metadata declarations.

## 5. Dataset and task

The task is causal language modeling on WikiText-2. Text is concatenated after removing empty entries, tokenized with the GPT-2 tokenizer and split into consecutive blocks of 128 tokens. Training and validation use the same preprocessing in every comparison. The public test split uses the identical block construction and is evaluated only after model selection; no gradients are computed during test evaluation.

The test evaluation contains 2,213 consecutive 128-token blocks. The frozen GPT-2 reference has test loss 4.1279 and perplexity 62.05. WikiText-2 is useful for a controlled language-modeling prototype, but it is small and does not establish behavior on other corpora, model families or downstream tasks.

## 6. Experimental protocol

The shared protocol is: batch size 1, gradient accumulation 4, sequence length 128, AdamW, learning rate 3e-4, gradient clipping 1.0, frozen GPT-2 backbone, FP16 backbone/autocast and FP32 adapters. Within each comparison, seed, example order, optimizer steps, tokenizer, split and evaluation schedule are identical. Long runs are fresh 0→5,000-step runs; old 2,000-step adapters were not resumed because complete optimizer, scheduler, iterator and RNG state was unavailable.

The study includes short 20/100-step placement screens, 500/1,000/2,000-step controlled comparisons, 2,000-step fixed-budget confirmations at 6,144 and 12,288 parameters, and the final 5,000-step campaign. All final long-run aggregates use seeds 42, 123 and 456.

## 7. Parameter matching and traceability

At a 768→768 projection, one rank unit costs 1,536 parameters. Thus rank 4 equals 6,144 parameters and rank 8 equals 12,288 parameters. For multi-layer experiments the total rank budget is distributed across independent insertion points. A 12,288-parameter comparison therefore uses 1×rank 8, 2×rank 4 or 4×rank 2. Same rank and matched total budget are reported separately; equal rank alone does not imply equal capacity.

`report/MASTER_RESULTS.csv` is the central machine-readable index. Each row records experiment family, architecture, budget, layers, local rank, total rank, trainable parameters, steps, seed, validation loss, perplexity, AULC, wall time, throughput, peak VRAM, validity and result directory. Raw metrics and summaries remain in their original directories.

## 8. Initial adapter comparison

The initial experiments compared the frozen baseline, LoRA, element-wise quadratic, signed quadratic, feature interaction, Symmetric Quadratic and Linear + Quadratic. They verified that all branches can run with a frozen GPT-2 backbone, that zero-preserving initialization reproduces the base output, and that parameter counts are explicit. These runs are the historical foundation for the later Symmetric-versus-LoRA focus; their detailed tables and synthetic benchmark remain in the generated figures and earlier result directories.

The synthetic benchmark tested linear, quadratic and mixed targets. Its purpose was representational diagnosis, not evidence that a synthetic regression advantage must transfer to language modeling. The synthetic and early adapter reports therefore remain separate from the final held-out language-modeling conclusion.

## 9. Long single-layer convergence

The 500/1,000/2,000-step series showed that the relative ordering depends on both architecture and optimization time. The formerly incomplete Linear + Quadratic seed-456 attempt failed during an NVIDIA runtime lockup, but the configuration was rerun successfully; all three seeds are now represented in the valid aggregate. The failed artifact remains documented as operational history and is excluded from statistics.

This phase established the need for longer curves rather than relying on a single endpoint. It also motivated adapter-only checkpointing so that later held-out evaluation could be performed without retraining.

## 10. Depth and placement study

The placement study tested attention output, MLP input/output and combined MLP modules, followed by a full 12-position attention scan. Short screening results are not interpreted as universal depth laws. They identify candidate placements and document the actual module shapes. The full attention scan is recorded in `report/DEPTH_AND_PLACEMENT/DEPTH_AND_PLACEMENT_STUDY.md` and its figures; later fixed-budget multi-layer runs provide the stronger evidence about distributed capacity.

The distinction is important: a short single-layer scan measures sensitivity to insertion location, while a fixed-budget multi-layer campaign measures how a limited parameter budget is allocated across depth.

## 11. Full attention depth scan

The complete attention scan evaluated all 12 `attn.c_proj` positions under the short controlled screen. Symmetric was lower at 11 of 12 positions in that horizon, with one positive signed difference. Because the scan used one seed and a short training horizon, it is evidence of placement sensitivity, not proof of a stable layer-specific advantage. The figures `full_depth_validation.png` and `full_depth_delta_loss.png` preserve the observed signed differences.

## 12. Fixed-budget multi-layer study

At 6,144 parameters, the comparison used concentrated 1×rank 4, two-layer 2×rank 2 and four-layer 4×rank 1 at the same attention projection family. At 12,288 parameters the later confirmation used concentrated 1×rank 8, two-layer 2×rank 4 and four-layer 4×rank 2. LoRA and Symmetric always used the same layer sets within a pair.

The 2,000-step 12,288-parameter validation means were:

| Geometry | LoRA | Symmetric | Symmetric−LoRA |
|---|---:|---:|---:|
| Concentrated | 3.5013 | 3.3391 | -0.1622 |
| Two-layer | 3.3157 | 3.2390 | -0.0767 |
| Four-layer | 3.2578 | 3.2356 | -0.0222 |

At this stage the quadratic advantage decreased as the same budget was spread across more locations. The 6,144 and 12,288 raw aggregates are retained under `report/LOCAL_RANK_DEPTH_INTERACTION/` and `report/MULTILAYER_FIXED_BUDGET/`.

## 13. Local-rank × depth interaction

The rank-depth study asks whether the four-layer behavior is caused by distribution itself or by reducing the local quadratic rank. At fixed total budget, rank 1, rank 2 and rank 4 provide different numbers of rank-one quadratic components in each adapted layer. This is a capacity distinction, not a causal explanation by itself.

The valid 6,144- and 12,288-parameter three-seed confirmations, together with their convergence curves and per-seed scatter plots, are documented in `report/LOCAL_RANK_DEPTH_INTERACTION/LOCAL_RANK_DEPTH_INTERACTION_STUDY.md`. The final long campaign extends this question to 5,000 steps and held-out data.

## 14. 5,000-step long-convergence study

The final campaign contains 18/18 valid runs: two architectures × three geometries × three seeds, all at 12,288 trainable parameters and 5,000 optimizer steps. Checkpoints were saved at 1,000, 2,000, 3,000, 4,000 and 5,000 steps. The final-step aggregates are:

| Geometry | Model | Validation loss | Validation PPL | AULC | Time (s) | Tokens/s |
|---|---|---:|---:|---:|---:|---:|
| Concentrated | LoRA | 3.3577 ± 0.0041 | 28.72 ± 0.12 | 3.4898 | 766.0 | 3342 |
| Concentrated | Symmetric | 3.2559 ± 0.0048 | 25.94 ± 0.12 | 3.3691 | 769.4 | 3327 |
| Two-layer | LoRA | 3.2003 ± 0.0178 | 24.54 ± 0.44 | 3.3340 | 791.4 | 3235 |
| Two-layer | Symmetric | 3.1520 ± 0.0126 | 23.38 ± 0.30 | 3.2822 | 800.3 | 3199 |
| Four-layer | LoRA | 3.1537 ± 0.0204 | 23.43 ± 0.48 | 3.2812 | 835.1 | 3065 |
| Four-layer | Symmetric | 3.1557 ± 0.0320 | 23.48 ± 0.74 | 3.2735 | 852.5 | 3003 |

All runs used about 348.5 MiB peak allocation in the recorded summaries. The validation differences Symmetric−LoRA are -0.1018, -0.0483 and +0.0020 in the same order. The four-layer endpoint difference is much smaller than seed dispersion and must be described as practically tied.

## 15. Held-out test evaluation

The final step-5,000 checkpoint was selected in advance for test reporting. The test results are:

| Geometry | LoRA test loss | Symmetric test loss | Symmetric−LoRA |
|---|---:|---:|---:|
| Concentrated | 3.7741 ± 0.0022 | 3.7067 ± 0.0060 | -0.0674 |
| Two-layer | 3.6361 ± 0.0075 | 3.6306 ± 0.0025 | -0.0055 |
| Four-layer | 3.6192 ± 0.0115 | 3.6452 ± 0.0149 | +0.0260 |

Perplexities are respectively 43.56 ± 0.10 versus 40.72 ± 0.24, 37.95 ± 0.29 versus 37.74 ± 0.09, and 37.31 ± 0.43 versus 38.29 ± 0.57. The frozen baseline is 4.1279 loss and 62.05 perplexity on the same 2,213 blocks.

**Observation.** The concentrated Symmetric advantage transfers to the held-out split; the two-layer result is nearly equivalent; the four-layer ordering reverses in favor of LoRA. **Interpretation.** This is consistent with a placement-dependent generalization effect. **Limitation.** One dataset and three seeds cannot isolate whether the difference is due to function class, optimization, or data-specific generalization.

## 16. Optimization and AULC analysis

AULC is the mean recorded validation loss over the available trajectory, so a lower value means lower average loss during optimization rather than merely a lower final point. Symmetric AULC is lower in all geometries: 3.3691 versus 3.4898, 3.2822 versus 3.3340 and 3.2735 versus 3.2812.

This distinction matters. In the four-layer case, Symmetric has a slightly lower average trajectory loss but reaches a virtually tied validation endpoint and a worse test mean. Thus convergence efficiency and generalization are not interchangeable with endpoint performance. The signed ΔLoss curves in `report/LONG_CONVERGENCE_AND_TEST/figures/delta_loss_5000.png` should be read with zero as the reference: negative favors Symmetric, positive favors LoRA.

## 17. Generalization analysis

The validation-to-test gap is defined as `test loss − validation loss`. The means are:

| Geometry | LoRA gap | Symmetric gap |
|---|---:|---:|
| Concentrated | 0.4164 | 0.4508 |
| Two-layer | 0.4358 | 0.4787 |
| Four-layer | 0.4655 | 0.4894 |

Symmetric has the larger gap in all three tested geometries. This is consistent with a difference in generalization between validation and test distributions, but it is not by itself proof of overfitting: dataset difficulty, split composition and optimization trajectory are not independently manipulated here.

## 18. Computational cost

At the same 12,288-parameter budget, Symmetric is slightly slower. Concentrated training takes 769.4 s versus 766.0 s for LoRA; two-layer takes 800.3 s versus 791.4 s; four-layer takes 852.5 s versus 835.1 s. Throughput is 3,327 versus 3,342, 3,199 versus 3,235, and 3,003 versus 3,065 tokens/s. Parameter efficiency and computational efficiency are therefore separate questions: a model may use the same number of trainable weights while requiring additional element-wise and projection operations.

## 19. Interpretation: observation, interpretation, hypothesis

**Observation.** Symmetric is clearly better in the concentrated 5,000-step validation and test comparisons, remains modestly better in validation for two layers, and is tied or worse in the four-layer endpoint comparisons.

**Interpretation.** The benefit of an explicit quadratic correction depends strongly on where the fixed budget is placed and on optimization duration. When LoRA is distributed across several Transformer blocks, composing multiple linear corrections through an already nonlinear Transformer may recover part of the advantage supplied locally by an explicit quadratic branch.

**Hypothesis.** Distributed low-rank linear corrections may substitute for some local second-order capacity. This is a plausible mechanism, not a demonstrated causal result. The study does not prove that the quadratic representation is intrinsically less useful at depth; it shows that the tested optimization and placement protocol changes the observed trade-off.

## 20. Limitations

The evidence is limited to GPT-2 Small, WikiText-2, context length 128, one optimizer and learning rate, three seeds, one main long-run budget, attention-focused long runs, and a small number of placement geometries. There is no second independent dataset, no second Transformer family, no downstream task, no broad hyperparameter search, and no fully factorial rank × depth design. Inference latency and FLOPs were not measured with a dedicated benchmark. Earlier GPU lockups affected execution logistics, although all 18 final 5,000-step runs completed and are valid. The held-out test was evaluated only after the training protocol and final-step rule were fixed.

The former limitations “no adapter checkpoints” and “no test set” describe the historical state before the long-convergence campaign. They are not current limitations: adapter-only checkpointing and test evaluation are now complete.

## 21. Future work

The most informative next experiments are: (1) repeat the long comparison on an independent corpus; (2) test a second Transformer family; (3) run a factorial design that separates local rank from number and location of layers; (4) repeat the validation-to-test gap analysis to determine whether the larger Symmetric gap is reproducible; (5) extend long runs to MLP and attention+MLP placements; (6) measure inference latency and FLOPs directly; (7) test RMSNorm or other explicitly labeled stability variants; and (8) evaluate downstream adaptation tasks. These follow-ups are motivated by observed placement and generalization interactions, not by an assumption that the quadratic model should win.

## 22. Conclusions

The experiments answer the central question conditionally. Low-rank quadratic adaptation can provide useful capacity at a matched parameter budget, especially when the budget is concentrated in one insertion point. That advantage is not universal: distributing the same budget across depth reduces it, longer optimization allows LoRA to close the gap, and the four-layer held-out test favors LoRA.

The supported conclusion is therefore an interaction statement, not a winner statement. Function class, local rank, depth distribution, optimization duration and generalization jointly determine the observed result. Three seeds and one corpus are sufficient to make this pattern worth further study, but not sufficient to claim a generally superior adapter family.

## 23. Reproducibility and traceability

Primary artifacts are:

- `report/MASTER_RESULTS.csv` and `report/MASTER_RESULTS.md` — central index across experiment families.
- `report/LONG_CONVERGENCE_AND_TEST/aggregate_long.csv` and `long_results.csv` — row-level 5,000-step aggregates.
- `report/LONG_CONVERGENCE_AND_TEST/LONG_CONVERGENCE_AND_TEST_STUDY.md` — detailed long-run and test methodology.
- `report/LOCAL_RANK_DEPTH_INTERACTION/LOCAL_RANK_DEPTH_INTERACTION_STUDY.md` — 6,144/12,288 local-rank and depth results.
- `report/MULTILAYER_FIXED_BUDGET/MULTILAYER_FIXED_BUDGET.md` — fixed-budget placement history.
- `report/DEPTH_AND_PLACEMENT/DEPTH_AND_PLACEMENT_STUDY.md` — module and depth scan.
- `results_long_convergence_5000/` — immutable run directories, checkpoints, metrics and test JSON files.
- `report/LONG_CONVERGENCE_AND_TEST/figures/` — long validation, ΔLoss, test, generalization and adapter-norm figures.

Every numerical table in this report is derived from the CSV/JSON artifacts above. Failed CUDA attempts remain diagnostic artifacts and are excluded through the `valid` field.

## 24. Figure reading guide

For every learning curve, the horizontal axis is optimizer step and lower loss is better. A persistent vertical separation indicates a performance difference at the same step; convergence of curves means the endpoint difference is shrinking. In signed ΔLoss plots, negative values favor Symmetric and positive values favor LoRA. AULC plots summarize the average trajectory, not just the final checkpoint. Parameter-versus-performance plots must be read together with parameter matching; lower loss with more trainable parameters is not automatically a fair win.

The long-run figures are: `validation_four_layer_5000.png`, `validation_two_layer_5000.png`, `validation_concentrated_5000.png`, `delta_loss_5000.png`, `test_loss_5000.png`, `generalization_gap_5000.png` and `adapter_output_norm_5000.png`. Historical placement figures remain under `report/DEPTH_AND_PLACEMENT/figures/` and should be read as short-screen evidence unless their captions explicitly state otherwise. No figure is interpreted as causal evidence by itself; each separates direct observation from compatible interpretation.
