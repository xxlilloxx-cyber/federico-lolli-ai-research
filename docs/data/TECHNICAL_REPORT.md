# Low-Rank Quadratic Adaptation of GPT-2

**Author:** Federico Lolli  
**Public contact:** xxlilloxx@gmail.com  
**Status:** exploratory, reproducible research record; no claim of universal superiority.

## 1. Executive summary

This project asks whether a small, activation-dependent quadratic correction can adapt a frozen Transformer projection usefully at a parameter budget matched to LoRA. The backbone is GPT-2 Small. Its pretrained weights and biases remain frozen; only the matrices of an adapter inserted beside selected attention-output projections are trained.

The final controlled campaign has 18 valid fresh runs: two adapter classes, three placements, three seeds, 12,288 trainable parameters per run, and 5,000 optimizer steps. It separates three quantities that older reports had occasionally conflated:

* **best_recorded_validation_loss** is the minimum *fresh* validation loss recorded within a run;
* **final_validation_loss** is the fresh validation loss at optimizer step 5,000;
* **final_test_loss** is held-out WikiText-2 loss from that same final checkpoint.

At the final checkpoint, Symmetric Quadratic has lower validation and test loss in the concentrated placement. Its mean test perplexity is 40.7165 compared with LoRA's 43.5567, a 6.52% decrease calculated from unrounded per-seed perplexities. The difference narrows in the two-layer placement and reverses on four-layer held-out test, where LoRA has lower mean test loss. The evidence therefore supports useful quadratic adaptation under some tested placements, not a general claim that it replaces LoRA.

## 2. Motivation and model family

A frozen projection maps an activation row vector $x\in\mathbb{R}^{1\times d}$ to $y_\mathrm{base}=xW+b$, where $W\in\mathbb{R}^{d\times d_\mathrm{out}}$ and $b\in\mathbb{R}^{1\times d_\mathrm{out}}$ are frozen pretrained parameters. The adapter adds a trainable correction; it does not train GPT-2 from scratch.

The matched linear baseline is LoRA:

$$\Delta y_\mathrm{LoRA}=\frac{\alpha}{r}(xA)B,\qquad A\in\mathbb{R}^{d\times r},\ B\in\mathbb{R}^{r\times d_\mathrm{out}}.$$

The final quadratic comparison is Symmetric Quadratic:

$$\Delta y_\mathrm{Sym}=\frac{\alpha}{r}(xU)^{\odot2}P,\qquad U\in\mathbb{R}^{d\times r},\ P\in\mathbb{R}^{r\times d_\mathrm{out}}.$$

For output coordinate $j$, writing $u_k$ for column $k$ of $U$,

$$\Delta y_j=\frac{\alpha}{r}\sum_kP_{kj}(xu_k)^2=xQ_jx^T,\qquad Q_j=\frac{\alpha}{r}\sum_kP_{kj}u_ku_k^T.$$

Each $Q_j$ is symmetric with $\operatorname{rank}(Q_j)\le r$. Directions $u_k$ are shared among output coordinates; signed $P_{kj}$ coefficients make $Q_j$ potentially indefinite. The two-feature expansion $(a x_1+b x_2)^2=a^2x_1^2+2abx_1x_2+b^2x_2^2$ shows why a squared projection can contain both squared features and cross-feature interactions.

The implementation sets $\alpha=4$. Local ranks 8, 4, and 2 therefore have effective scales $\alpha/r=0.5,1,2$. Rank and scale change together in this protocol, so depth/rank outcomes are not a factorial isolation of rank alone. The output factors $B$ and $P$ are zero-initialized. Thus both corrections are exactly zero before the first update; the zero output factor receives a gradient first, whereas the input factor has zero initial gradient until its partner is nonzero.

Earlier exploratory branches used the implementation's outer $\alpha/r$ factor: element-wise quadratic $\Delta y=(\alpha/r)(x\odot x)UV$, signed quadratic $\Delta y=(\alpha/r)(x\odot|x|)UV$, and feature interaction $\Delta y=(\alpha/r)[(xU)\odot(xV)]P$. The historical Linear+Quadratic wrapper applies its common outer factor after the branch factors:

$$\Delta y=\frac{\alpha}{r}\left[\frac{\alpha_L}{r_L}(xA)B+\frac{\alpha_Q}{r_Q}((xU)\odot(xV))P\right].$$

## 3. Architecture, data, and protocol

The model is Hugging Face `gpt2` (GPT-2 Small, approximately 124M parameters, 12 zero-based blocks). In the final campaign an adapter receives the **input** to `transformer.h[L].attn.c_proj`: the concatenated causal-attention head output. The frozen `attn.c_proj` and trainable adapter run in parallel, their outputs are summed, GPT-2 applies its attention residual dropout, and the attention residual is added. The second normalization and MLP remain frozen. Block 2 means the third block.

WikiText-2 raw non-empty lines are concatenated with newline separators, tokenized with the GPT-2 tokenizer without added special tokens, and split into consecutive 128-token blocks; a short remainder is discarded. The task is causal language modelling. The recorded dataset contains 36,718 train blocks, 3,760 validation blocks, and 4,358 test blocks before the evaluation limit; the final held-out test used 2,213 blocks under the same block construction. Each validation pass uses the recorded eight-block limit. This small validation sample is a material limitation.

The long campaign used batch size 1, gradient accumulation 4, AdamW at learning rate $3\cdot10^{-4}$, default PyTorch AdamW betas (0.9, 0.999), epsilon $10^{-8}$, and weight decay 0.01, gradient clipping 1.0, and no scheduler or warmup. The backbone ran with FP16 autocast and adapter parameters were FP32. GPT-2 is put in train mode for optimization and eval mode for validation/test. There is no adapter dropout. The pretrained model's exact revision and its configured backbone dropout were not captured in the run configuration and are therefore unavailable. Recorded package versions are PyTorch 2.3.1+cu121, Transformers 4.57.6, Datasets 5.0.1, and Accelerate 1.15.0.

The checkpoint rule was fixed in advance: evaluate **the final step-5,000 adapter checkpoint** on test, rather than selecting a checkpoint using test data. Adapter-only checkpoints at 1,000, 2,000, 3,000, 4,000, and 5,000 steps store adapter tensors and compatibility metadata. A reload smoke test produced maximum absolute logits error 0.0 and rejects incompatible architecture, rank, layers, or metadata.

## 4. Historical experiments

Historical entries are kept as evidence of the project progression, but they are not combined into a single learning curve with the fresh 5,000-step runs. The historical `final_val_loss`/`validation_loss` summaries below mean **best recorded validation** unless an exact step is explicitly named.

| Study | Steps | Main verified observation |
|---|---:|---|
| Adapter-family screen | 20, 50, 500 | LoRA, element-wise quadratic, signed quadratic, feature interaction, Symmetric, and Linear+Quadratic all completed smoke/small-run evaluations; the 500-step three-seed best summaries were LoRA 3.5732, element-wise 3.6351, signed 3.6257, feature interaction 3.5581, Symmetric 3.5526, and Linear+Quadratic 3.5302. |
| Single-point three-seed comparison | 1,000 | At layer 0 rank 4, best summaries were LoRA 3.5491, Symmetric 3.4770, Linear+Quadratic 3.4928. |
| Single-point three-seed comparison | 2,000 | At layer 0 rank 4, best summaries were LoRA 3.4598, Symmetric 3.3784, Linear+Quadratic 3.4360. |
| Depth/placement screens | 20 and 100 | Attention and MLP candidates were screened at short duration; these screens motivate placement questions but are too short for final performance claims. |

The fixed-budget confirmation at 2,000 steps used independent historical runs and should not be mistaken for intermediate points of the fresh 5,000-step trajectories:

| Budget | Geometry (local rank) | LoRA best validation | Symmetric best validation |
|---:|---|---:|---:|
| 6,144 | concentrated [2] (4) | 3.5039 | 3.3745 |
| 6,144 | two [0,11] (2) | 3.3661 | 3.3730 |
| 6,144 | four [0,3,7,11] (1) | 3.3199 | 3.3794 |
| 12,288 | concentrated [2] (8) | 3.5013 | 3.3391 |
| 12,288 | two [0,11] (4) | 3.3157 | 3.2390 |
| 12,288 | four [0,3,7,11] (2) | 3.2578 | 3.2356 |

These tables use the original aggregate files cited in the reproducibility map. They show why the long campaign focused on the 12,288-parameter geometries, but do not prove an endpoint trend from 2,000 to 5,000 steps.

## 5. Final 5,000-step validation results

Every row below aggregates seeds 42, 123, and 456. Means and sample SDs are computed from unrounded per-seed values. Final validation perplexity is the mean of per-seed $\exp(\mathrm{final\ validation\ loss})$, not $\exp(\mathrm{mean\ loss})$.

| Geometry, matched 12,288 parameters | Architecture | Final validation loss | Final validation PPL | Best recorded validation loss |
|---|---|---:|---:|---:|
| Concentrated [2], r=8 | LoRA | 3.3703 ± 0.0045 | 29.09 ± 0.13 | 3.3577 ± 0.0041 |
| Concentrated [2], r=8 | Symmetric | 3.2831 ± 0.0080 | 26.66 ± 0.21 | 3.2559 ± 0.0048 |
| Two [0,11], r=4 each | LoRA | 3.2186 ± 0.0166 | 25.00 ± 0.42 | 3.2003 ± 0.0178 |
| Two [0,11], r=4 each | Symmetric | 3.1876 ± 0.0119 | 24.23 ± 0.29 | 3.1520 ± 0.0126 |
| Four [0,3,7,11], r=2 each | LoRA | 3.1743 ± 0.0162 | 23.91 ± 0.39 | 3.1537 ± 0.0204 |
| Four [0,3,7,11], r=2 each | Symmetric | 3.1925 ± 0.0157 | 24.35 ± 0.38 | 3.1557 ± 0.0320 |

Final Symmetric-minus-LoRA loss differences are -0.0872, -0.0310, and +0.0182 for concentrated, two-layer, and four-layer placement. In contrast, +0.0020 is the **best-recorded** four-layer difference; it is not the final-checkpoint difference. Within this protocol, LoRA benefits more visibly from distributing the fixed budget: its final loss changes from 3.3703 to 3.1743, whereas Symmetric changes from 3.2831 to 3.1925 and appears to saturate between two and four insertion points. This is an observation, not a general law.

## 6. Held-out test and test–validation differences

The frozen base model scored test loss 4.1279 and perplexity 62.05 on 2,213 test blocks. The following uses the same final step-5,000 checkpoints. Test perplexity is again the mean of per-seed test perplexities.

| Geometry | Architecture | Final test loss | Final test PPL | Test − final validation loss |
|---|---|---:|---:|---:|
| Concentrated | LoRA | 3.7741 ± 0.0022 | 43.56 ± 0.10 | 0.4037 ± 0.0036 |
| Concentrated | Symmetric | 3.7067 ± 0.0060 | 40.72 ± 0.24 | 0.4236 ± 0.0042 |
| Two-layer | LoRA | 3.6361 ± 0.0075 | 37.95 ± 0.29 | 0.4176 ± 0.0092 |
| Two-layer | Symmetric | 3.6306 ± 0.0025 | 37.74 ± 0.09 | 0.4431 ± 0.0142 |
| Four-layer | LoRA | 3.6192 ± 0.0115 | 37.31 ± 0.43 | 0.4449 ± 0.0086 |
| Four-layer | Symmetric | 3.6452 ± 0.0149 | 38.29 ± 0.57 | 0.4527 ± 0.0044 |

The test-minus-validation quantities are paired within each seed before their mean and sample SD are calculated; they are descriptive, and are not called an overfitting diagnosis. Concentrated placement retains Symmetric's test advantage. Two layers are nearly tied. Four layers favour LoRA on held-out text. The largest matched-placement advantage is concentrated Symmetric; the best Symmetric mean test configuration is two-layer; the best overall observed mean test configuration is four-layer LoRA.

## 7. Convergence and AULC

The public fresh trajectory export has 3,618 rows: 18 genuine pre-update summary values at step 0 plus 18×200 fresh scheduled evaluations at steps 25,50,…,5,000. It intentionally excludes step-1 values. The raw `metrics.csv` has 5,000 rows per run: 201 fresh post-update evaluations (step 1 and 200 scheduled evaluations, including step 5,000 exactly once) and 4,799 cached non-evaluation rows. Thus the common 3.8293571472 step-0 value is a separately recorded pre-update frozen-model measurement in each summary, not 18 independent baseline experiments. Old plots that began at step 1 appeared to show an initial adapter difference because that point was already post-update.

Historical AULC fields are retained with their original definition: for each run, the arithmetic mean of all 5,000 per-step validation columns, including cached repeats and excluding the summary step 0; then the per-run values are averaged across seeds. This is not an integrated area. A normalized trapezoidal area over fresh points would be $\frac{1}{5000}\sum_i\frac{L_i+L_{i+1}}2(t_{i+1}-t_i)$; it is not silently substituted for the legacy value. Consequently AULC is useful only when compared under the same historical definition and schedule.

## 8. Computational cost

The final campaign's mean wall-clock times / throughput are: concentrated LoRA 766.0 s / 3,342 tokens/s and Symmetric 769.4 s / 3,327 tokens/s; two-layer 791.4 / 3,235 and 800.3 / 3,199; four-layer 835.1 / 3,065 and 852.5 / 3,003. Parameter efficiency and compute efficiency are distinct: both methods train the same 12,288 parameters, but extra quadratic arithmetic and more insertion points have a small measured cost.

## 9. Interpretation, limitations, and conclusion

The supported positive result is narrow but meaningful: explicit low-rank quadratic corrections train stably with roughly 0.01% of a frozen 124M-parameter backbone and improve the concentrated matched placement on both validation and held-out WikiText-2. The benefit decreases when capacity is distributed across depth and with longer optimization; four-layer LoRA ultimately has lower held-out loss. One possible interpretation is that distributed linear corrections, composed through an already nonlinear Transformer, recover some expressive benefit of a local explicit quadratic term. That remains a hypothesis, not a demonstrated mechanism.

Limits are GPT-2 Small only, WikiText-2 only, three seeds, context length 128, one optimizer/learning-rate protocol, final long runs focused on `attn.c_proj`, coupled rank/scale/placement changes, no independent dataset, no second Transformer family, no downstream task, and no controlled factorial rank-by-depth experiment. The next informative studies are an independent dataset, a second model family, factorial local-rank/scale experiments, long MLP and attention-plus-MLP placement comparisons, and a test of whether the larger Symmetric test–validation difference reproduces.

## 10. Reproducibility and figure guide

Canonical public final data are `docs/data/long_final_per_seed.csv`, `docs/data/long_final_aggregate.csv`, and `docs/data/long_validation_trajectories.csv`. Figure provenance is in `docs/data/FIGURE_MANIFEST.md`; `scripts/generate_public_long_figures.py` regenerates the PNG figures, and `scripts/validate_public_long_data.py` checks that final trajectory endpoints equal the aggregate table. Historical raw artifacts remain private and ignored; `report/MASTER_RESULTS.csv`, `report/LOCAL_RANK_DEPTH_INTERACTION/aggregate_6144.csv`, and `report/LOCAL_RANK_DEPTH_INTERACTION/aggregate_12288.csv` index their verified historical summaries.

Validation curves are lower-is-better; shaded bands are sample SD, not confidence intervals. The signed delta curve is Symmetric minus LoRA, so a negative value favours Symmetric. Test bars show individual seeds plus mean ± sample SD. The test–validation figure is secondary/descriptive and does not identify a cause.

## 11. References

1. Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. https://arxiv.org/abs/2106.09685
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. https://arxiv.org/abs/1706.03762
3. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI technical report.
4. Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). *Pointer Sentinel Mixture Models.* ICLR. https://arxiv.org/abs/1609.07843
5. Li, Y., Song, L., and Hou, H. (2024). *LoRAN: Improved Low-Rank Adaptation by a Non-Linear Transformation.* Findings of EMNLP 2024, 3134–3143. https://aclanthology.org/2024.findings-emnlp.177/
6. Zhang, W., Mu, L., Ni, L., Jin, P., and Zhang, Y. (2026). *Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions.* Findings of ACL 2026, 13287–13303. https://aclanthology.org/2026.findings-acl.650/

LoRAN applies a nonlinear transformation to a low-rank **weight update**, and PERA expands low-rank **weight factors** polynomially. Neither was an experimental baseline here. This study instead evaluates a quadratic dependence on the input activation at a frozen projection.
