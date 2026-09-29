# Low-Rank Quadratic Adaptation of GPT-2

**Author:** Federico Lolli  
**Public contact:** xxlilloxx@gmail.com  
**Status:** exploratory, reproducible research record; no claim of universal superiority.

## 1. Executive summary

This project asks whether a small, activation-dependent quadratic correction can adapt a frozen Transformer projection usefully at a parameter budget matched to LoRA. The backbone is GPT-2 Small. Its pretrained weights and biases remain frozen; only the matrices of an adapter inserted beside selected attention-output projections are trained.

The completed evidence includes an 18-run fixed-budget placement campaign, a separate 24-run controlled rank confirmation, six controlled AG News runs, and a single-seed scaling ablation. The 24-run confirmation compares LoRA and Symmetric at ranks 1, 2, 4, and 8 over three seeds and 5,000 fresh optimizer steps. All controlled reporting separates three quantities that older summaries had occasionally conflated:

* **best_recorded_validation_loss** is the minimum *fresh* validation loss recorded within a run;
* **final_validation_loss** is the fresh validation loss at optimizer step 5,000;
* **final_test_loss** is held-out WikiText-2 loss from that same final checkpoint.

In the fixed-budget placement campaign, Symmetric has lower concentrated-placement validation and test loss, the two-layer comparison is close, and four-layer held-out test favours LoRA. In the independent controlled rank confirmation, Symmetric has lower mean final validation and held-out test loss at all four ranks, with all twelve paired held-out comparisons favouring Symmetric. AG News provides the counterexample: LoRA has higher mean accuracy and macro-F1 and lower test loss. The evidence therefore supports task- and protocol-dependent quadratic adaptation, not a general claim that it replaces LoRA.

## 2. Motivation and model family

A frozen projection maps an activation row vector $x\in\mathbb{R}^{1\times d}$ to $y_\mathrm{base}=xW+b$, where $W\in\mathbb{R}^{d\times d_\mathrm{out}}$ and $b\in\mathbb{R}^{1\times d_\mathrm{out}}$ are frozen pretrained parameters. The adapter adds a trainable correction; it does not train GPT-2 from scratch.

The matched linear baseline is LoRA:

$$\Delta y_\mathrm{LoRA}=\frac{\alpha}{r}(xA)B,\qquad A\in\mathbb{R}^{d\times r},\ B\in\mathbb{R}^{r\times d_\mathrm{out}}.$$

The final quadratic comparison is Symmetric Quadratic:

$$\Delta y_\mathrm{Sym}=\frac{\alpha}{r}(xU)^{\odot2}P,\qquad U\in\mathbb{R}^{d\times r},\ P\in\mathbb{R}^{r\times d_\mathrm{out}}.$$

For output coordinate $j$, writing $u_k$ for column $k$ of $U$,

$$\Delta y_j=\frac{\alpha}{r}\sum_kP_{kj}(xu_k)^2=xQ_jx^T,\qquad Q_j=\frac{\alpha}{r}\sum_kP_{kj}u_ku_k^T.$$

Each $Q_j$ is symmetric with $\operatorname{rank}(Q_j)\le r$. Directions $u_k$ are shared among output coordinates; signed $P_{kj}$ coefficients make $Q_j$ potentially indefinite. The two-feature expansion $(a x_1+b x_2)^2=a^2x_1^2+2abx_1x_2+b^2x_2^2$ shows why a squared projection can contain both squared features and cross-feature interactions.

Both matched adapters contain $r(d+d_\mathrm{out})$ trainable parameters. The controlled rank implementation sets $\alpha=4$, so ranks 1, 2, 4, and 8 have effective scales $\alpha/r=4,2,1,0.5$. Rank, parameter count, and scale change together in this protocol; this motivated the separate constant-scale ablation. The output factors $B$ and $P$ are zero-initialized. Thus both corrections are exactly zero before the first update; the output factor receives a gradient first, whereas the input factor has zero initial gradient until its partner is nonzero.

Earlier exploratory branches used the implementation's outer $\alpha/r$ factor: element-wise quadratic $\Delta y=(\alpha/r)(x\odot x)UV$, signed quadratic $\Delta y=(\alpha/r)(x\odot|x|)UV$, and feature interaction $\Delta y=(\alpha/r)[(xU)\odot(xV)]P$. The historical Linear+Quadratic wrapper applies its common outer factor after the branch factors:

$$\Delta y=\frac{\alpha}{r}\left[\frac{\alpha_L}{r_L}(xA)B+\frac{\alpha_Q}{r_Q}((xU)\odot(xV))P\right].$$

![Figure 1. Matched LoRA and Symmetric Quadratic adapter branches beside a frozen row-vector projection.](https://github.com/xxlilloxx-cyber/federico-lolli-ai-research/blob/main/results/figures/architecture/adapter_architecture_comparison.svg)

**Figure 1.** Architecture of the matched LoRA and Symmetric Quadratic
adapters. Both branches run beside the frozen $xW+b$ projection. The Symmetric
branch applies the element-wise square $z\odot z$ to $z=xU$ before $P$; it does
not use a matrix square. Each matched adapter contains $r(d+d_{out})$
trainable parameters.

## 3. Architecture, data, and protocol

The model is Hugging Face `gpt2` (GPT-2 Small, approximately 124M parameters, 12 zero-based blocks). In the final campaign an adapter receives the **input** to `transformer.h[L].attn.c_proj`: the concatenated causal-attention head output. The frozen `attn.c_proj` and trainable adapter run in parallel, their outputs are summed, GPT-2 applies its attention residual dropout, and the attention residual is added. The second normalization and MLP remain frozen. Block 2 means the third block.

WikiText-2 raw non-empty lines are concatenated with newline separators, tokenized with the GPT-2 tokenizer without added special tokens, and split into consecutive 128-token blocks; a short remainder is discarded. The task is causal language modelling. The recorded dataset contains 36,718 train blocks, 3,760 validation blocks, and 4,358 test blocks before the evaluation limit; the final held-out test used 2,213 blocks under the same block construction. Each validation pass uses the recorded eight-block limit. This small validation sample is a material limitation.

The long campaign used batch size 1, gradient accumulation 4, AdamW at learning rate $3\cdot10^{-4}$, default PyTorch AdamW betas (0.9, 0.999), epsilon $10^{-8}$, and weight decay 0.01, gradient clipping 1.0, and no scheduler or warmup. The backbone ran with FP16 autocast and adapter parameters were FP32. GPT-2 is put in train mode for optimization and eval mode for validation/test. There is no adapter dropout. The pretrained model's exact revision and its configured backbone dropout were not captured in the run configuration and are therefore unavailable. Recorded package versions are PyTorch 2.3.1+cu121, Transformers 4.57.6, Datasets 5.0.1, and Accelerate 1.15.0.

The checkpoint rule was fixed in advance: evaluate **the final step-5,000 adapter checkpoint** on test, rather than selecting a checkpoint using test data. Adapter-only checkpoints at 1,000, 2,000, 3,000, 4,000, and 5,000 steps store adapter tensors and compatibility metadata. A reload smoke test produced maximum absolute logits error 0.0 and rejects incompatible architecture, rank, layers, or metadata.

The research proceeded through two stages. Historical studies explored adapter families, training duration, attention and MLP placement, a 12-block attention scan, and multilayer fixed-budget designs. Their protocols were not uniform and their numerical results remain separate. Controlled campaigns then matched backbone, adapter placement, rank, parameter budget, optimizer, training duration, and seed set. The independent 1,000-step rank screen and 5,000-step confirmation are not concatenated into a single trajectory.

The controlled rank confirmation uses zero-based block 0 `attn.c_proj`, ranks 1/2/4/8, seeds 42/123/456, and 5,000 fresh steps, for 24 runs. The controlled AG News campaign uses the same frozen backbone and placement with a trainable classification head, 4,096 training examples, a fixed 1,000-example stratified validation subset, the original test split, three paired seeds, and 500 steps. The held-out test sets do not select method, rank, hyperparameters, or checkpoint.

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

Limits are GPT-2 Small only, two datasets, three main seeds, context length 128, one optimizer/learning-rate protocol, controlled long runs focused on block-0 `attn.c_proj`, coupled rank/parameter-count/scale changes, a single-seed constant-scale ablation, no second Transformer family, and no controlled factorial rank-by-depth experiment. AG News supplies an independent task but not an independent language-model corpus. The next informative studies are a second model family and language-model dataset, a fully factorial rank/scale experiment, and long MLP and attention-plus-MLP placement comparisons.

## 10. Reproducibility and figure guide

Canonical public final data are under `results/tables/`; `docs/data/` contains generated Pages mirrors. Figure provenance is in `results/manifests/FIGURE_MANIFEST.md`; `scripts/figures/generate_public_long_figures.py` regenerates the long-study figures, and `scripts/validation/validate_public_long_data.py` checks that final trajectory endpoints equal the aggregate table. Historical raw artifacts remain private and ignored; `results/tables/historical_master_results.csv` and the publication-safe tables under `experiments/historical/` index their verified historical summaries.

Validation curves are lower-is-better; shaded bands are sample SD, not confidence intervals. The signed delta curve is Symmetric minus LoRA, so a negative value favours Symmetric. Test bars show individual seeds plus mean ± sample SD. The test–validation figure is secondary/descriptive and does not identify a cause.

## 11. Controlled downstream evaluation

The matched AG News experiment used GPT-2, block-0 `attn.c_proj`, rank 4,
9,216 trainable parameters including the classifier, 500 steps, and seeds 42,
123, and 456. LoRA obtained test accuracy 0.8201 ± 0.0051 and macro-F1 0.8156
± 0.0050, with test loss 0.5445 ± 0.0542. Symmetric obtained 0.8021 ± 0.0240,
0.7929 ± 0.0315, and test loss 0.6847 ± 0.1919. Symmetric was higher on one of
three paired seeds. This controlled negative result rules out a task-independent
superiority claim.

## 12. Controlled rank confirmation

A separate factorial confirmation trained fresh single-insertion-point adapters
for 5,000 optimizer steps at ranks 1, 2, 4, and 8, with seeds 42, 123, and 456.
It must not be confused with the historical multi-layer campaign or the earlier
1,000-step screen. Symmetric Quadratic obtained lower mean final validation and
held-out WikiText-2 test loss at all four tested ranks. Mean paired final-test
differences (Symmetric minus LoRA) were -0.0125, -0.0187, -0.0227, and -0.0381
for ranks 1, 2, 4, and 8; their sample SDs were 0.0051, 0.0119, 0.0035, and
0.0072. All twelve paired seed comparisons favoured Symmetric. This evidence is specific to frozen GPT-2 block-0
`attn.c_proj`, WikiText-2, and the recorded optimizer/scaling protocol.

The separate controlled AG News experiment gives an essential counterexample
to a universal claim: LoRA averaged 0.8201 ± 0.0051 test accuracy and 0.8156 ±
0.0050 macro-F1, versus 0.8021 ± 0.0240 and 0.7929 ± 0.0315 for Symmetric.
Thus the evidence supports task dependence, not universal superiority.

A seed-42 constant-effective-scale ablation retained the screening sign pattern:
LoRA was lower at rank 1, while Symmetric was lower at ranks 2, 4, and 8. At
rank 8 the paired difference changed from -0.1177 under alpha=4 to -0.0975
under alpha/r=1. Scaling affects magnitude but did not remove the observed
single-seed trend.

Checkpoint analyses preserve the mathematical distinction between methods.
LoRA has a constant effective update and zero adapter-only Hessian. Symmetric
has no constant `DeltaW`; its local Jacobian is `2(alpha/r) U diag(xU) P` and
its fixed-linear-reduction Hessian is `2(alpha/r) U diag(Pc) U^T`. Analytical
derivatives matched autograd. The measured spectra and Hessian summaries are
reported in `research/mathematical-analysis/MECHANISM_ANALYSIS_REPORT.md`; they describe structure and do not
establish a causal mechanism for task performance.

## 13. Scaling ablation

The separate seed-42 1,000-step ablation compared fixed $\alpha=4$ with a
constant effective scale $\alpha/r=1$. Under constant scale, paired validation
differences at ranks 1, 2, 4, and 8 were +0.0410, -0.0292, -0.0750, and
-0.0975. The rank-8 fixed-alpha result was -0.1177. The sign pattern persisted
for this seed, while magnitude changed. Because this is a single-seed result,
it is exploratory and does not isolate a population-level scaling interaction.

## 14. Spectral, derivative, and interaction analysis

These analyses characterize the local behavior of trained adapters. They
establish structural differences but do not by themselves explain performance.
LoRA is linear at adapter level and has the input-independent Jacobian

$$J_{\mathrm{LoRA}}=\frac{\alpha}{r}AB.$$

Its local transformation is one fixed low-rank update matrix. Symmetric is
quadratic and has the input-dependent local Jacobian

$$J_S(x)=2\frac{\alpha}{r}U\operatorname{diag}(xU)P.$$

It therefore cannot generally be reduced to one constant $\Delta W$.
Double-precision analytical Jacobian and Hessian implementations matched
PyTorch autograd tests.

Mean entropy effective ranks were:

| Rank | LoRA effective update | Symmetric local Jacobian | Symmetric U | Symmetric P | Output covariance |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2 | 1.92 | 1.67 | 2.00 | 1.98 | 1.92 |
| 4 | 3.76 | 3.09 | 3.99 | 3.89 | 3.00 |
| 8 | 6.57 | 5.59 | 7.96 | 7.28 | 3.59 |

These columns describe different mathematical objects and are not
interchangeable notions of weight rank. In particular, Symmetric local-Jacobian
rank is not a direct analogue of a fixed LoRA weight-update rank.

For a fixed linear output reduction $c$, its adapter Hessian is

$$H_S=2\frac{\alpha}{r}U\operatorname{diag}(Pc)U^T,$$

while LoRA's adapter-only Hessian is exactly zero. Symmetric mean Hessian
Frobenius norms were 155.15, 92.13, 52.68, and 17.61 at ranks 1, 2, 4, and 8.
The decrease combines learned-factor effects with the main protocol's
$\alpha/r$ scaling and is not evidence of weaker or less useful interactions
at higher rank. For fixed $c$, the exact quadratic Hessian is input independent;
repeated token activations are therefore not independent observations. A full
Transformer-block Hessian was not evaluated and is not reported as an
empirical result.

## 15. Matched computational benchmark

The rank-4 benchmark uses batch size 1, sequence length 128, 10 warm-up
iterations, and 30 measured iterations. Both methods contain 6,144 trainable
adapter parameters.

| Method | Forward latency | Backward latency | Peak VRAM | Forward throughput |
|---|---:|---:|---:|---:|
| LoRA | 12.157 ± 0.243 ms | 14.805 ± 0.236 ms | 429.05 MiB | 10,529 tokens/s |
| Symmetric Quadratic | 12.184 ± 0.236 ms | 14.766 ± 0.188 ms | 430.31 MiB | 10,505 tokens/s |

Relative to LoRA, Symmetric changes forward latency by approximately +0.22%,
backward latency by -0.27%, peak allocated VRAM by +1.25 MiB, and throughput by
approximately -0.22%. The two methods have nearly identical measured cost in
this implementation and hardware configuration. This focused benchmark should
not be generalized to other hardware or model scales.

## 16. Integrated conclusion

This study shows that the Symmetric Quadratic Adapter is a practical,
computationally lightweight way to introduce explicit second-order interactions
beside a frozen Transformer projection while retaining LoRA-matched parameter
structure. Across 24 independent 5,000-step WikiText-2 runs, Symmetric achieved
lower mean final validation and held-out test loss at ranks 1, 2, 4, and 8. At
rank 8, mean test loss was 3.7319 ± 0.0035 for LoRA and 3.6938 ± 0.0044 for
Symmetric. Rank-1 trajectories crossed between recorded steps 2,000 and 3,000,
showing that optimization duration matters.

The seed-42 constant-scale ablation retained the 1,000-step sign pattern while
changing magnitude, so scale is relevant but does not alone explain that one
seed. AG News provides the essential counterexample: LoRA achieved higher mean
accuracy (0.8201 ± 0.0051 versus 0.8021 ± 0.0240) and macro-F1. The evidence
therefore supports a viable, task-dependent alternative rather than universal
superiority.

LoRA's fixed Jacobian and zero adapter-only Hessian differ structurally from
Symmetric's input-dependent Jacobian and explicit second derivative. These
facts do not establish a causal explanation for performance. The evidence is
limited to GPT-2 Small, restricted placements, two datasets, three main seeds,
and one scaling-ablation seed. It establishes neither state of the art,
population-level statistical significance, universal transfer, nor universal
mathematical novelty. Broader models, tasks, placements, and independently
replicated scale controls are the next required tests.

Canonical per-seed tables, figure provenance, and reproduction commands are in
`results/`, `research/reproducibility/`, and `experiments/`. Related-work and
equivalence qualifications are maintained in `comparisons/`.

## 17. Controlled per-seed measurements

The following values come directly from
`results/tables/rank-confirmation-5000/rank_5000_per_seed.csv`. They are final
fresh validation and held-out test loss from the same step-5,000 checkpoint.

| Method | Rank | Seed | Final validation | Final test |
|---|---:|---:|---:|---:|
| LoRA | 1 | 42 | 3.5676 | 3.9266 |
| LoRA | 1 | 123 | 3.5769 | 3.9247 |
| LoRA | 1 | 456 | 3.5757 | 3.9255 |
| Symmetric | 1 | 42 | 3.5489 | 3.9100 |
| Symmetric | 1 | 123 | 3.5086 | 3.9179 |
| Symmetric | 1 | 456 | 3.5368 | 3.9113 |
| LoRA | 2 | 42 | 3.4796 | 3.8465 |
| LoRA | 2 | 123 | 3.4815 | 3.8514 |
| LoRA | 2 | 456 | 3.4709 | 3.8497 |
| Symmetric | 2 | 42 | 3.4081 | 3.8157 |
| Symmetric | 2 | 123 | 3.4578 | 3.8444 |
| Symmetric | 2 | 456 | 3.4361 | 3.8315 |
| LoRA | 4 | 42 | 3.3533 | 3.7647 |
| LoRA | 4 | 123 | 3.3541 | 3.7714 |
| LoRA | 4 | 456 | 3.3375 | 3.7607 |
| Symmetric | 4 | 42 | 3.3112 | 3.7439 |
| Symmetric | 4 | 123 | 3.3089 | 3.7447 |
| Symmetric | 4 | 456 | 3.3101 | 3.7402 |
| LoRA | 8 | 42 | 3.3165 | 3.7348 |
| LoRA | 8 | 123 | 3.3067 | 3.7329 |
| LoRA | 8 | 456 | 3.2931 | 3.7280 |
| Symmetric | 8 | 42 | 3.2469 | 3.6939 |
| Symmetric | 8 | 123 | 3.2340 | 3.6894 |
| Symmetric | 8 | 456 | 3.2529 | 3.6982 |

AG News per-seed values come from
`results/tables/downstream-ag-news/downstream_controlled_results.csv`:

| Method | Seed | Test loss | Accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| LoRA | 42 | 0.6047 | 0.8143 | 0.8099 |
| LoRA | 123 | 0.5294 | 0.8217 | 0.8177 |
| LoRA | 456 | 0.4996 | 0.8242 | 0.8192 |
| Symmetric | 42 | 0.5984 | 0.8195 | 0.8152 |
| Symmetric | 123 | 0.9046 | 0.7747 | 0.7568 |
| Symmetric | 456 | 0.5511 | 0.8122 | 0.8067 |

The individual observations show why mean differences must be interpreted with
their seed variability and why no significance claim is made from three seeds.

## 18. Related work and mathematical positioning

Published methods are compared at mathematical and architectural levels. Their
benchmark values use different backbones, datasets, budgets, and evaluation
protocols and are not ranked numerically against this study.

| Method | Core mechanism | Relation to this study |
|---|---|---|
| LoRA | low-rank linear correction $(\alpha/r)(xA)B$ | direct experimental baseline |
| DoRA | weight magnitude/direction decomposition | different weight parameterization |
| MoRA | square trainable matrix with nonparametric transforms | weight-space method designed to increase effective update rank |
| HiRA | Hadamard-product high-rank update | weight-space higher-rank adaptation |
| LoRAN | nonlinear transformation of the low-rank weight update | nonlinear weight-space adaptation, not activation-space squaring |
| PERA | polynomial expansion of low-rank factors | polynomial parameter/factor-space structure |
| QuadraNet V2 | factorized quadratic neural transformation | closely related quadratic precedent; full multi-output equivalence is not established |
| This study | $(\alpha/r)((xU)\odot(xU))P$ beside a frozen projection | activation-dependent symmetric quadratic correction |

PERA expands parameters and low-rank factors before composition; this study
applies the nonlinearity to projected activations,
$x\rightarrow xU\rightarrow(xU)\odot(xU)\rightarrow P$. Both concern
higher-order PEFT, but act on different mathematical objects. PERA was not an
experimental baseline.

For one Symmetric output,

$$\Delta y_j=xQ_jx^T,\qquad
Q_j=\frac{\alpha}{r}U\operatorname{diag}(P_{:,j})U^T.$$

This is a factorized quadratic form related to QuadraNet V2. Across the full
multi-output adapter, all outputs share directions in $U$, while $P$ supplies
signed output-specific coefficients. Scalar-form similarity therefore does not
establish complete architectural equivalence. QuadraNet V2 is treated as a
closely related precedent, not evidence of exact equivalence or universal
novelty.

The defining feature of Symmetric is the activation-space sequence
$x\rightarrow xU\rightarrow(xU)\odot(xU)\rightarrow P$, which yields an
input-dependent local Jacobian at a LoRA-comparable parameter count. Only LoRA
was reproduced as a controlled baseline; DoRA, MoRA, HiRA, LoRAN, PERA, and
QuadraNet V2 provide literature context.

## 19. References

1. Hu, E. J., et al. (2022). *LoRA: Low-Rank Adaptation of Large Language Models.* ICLR. https://openreview.net/forum?id=nZeVKeeFYf9
2. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS. https://arxiv.org/abs/1706.03762
3. Radford, A., et al. (2019). *Language Models are Unsupervised Multitask Learners.* OpenAI technical report.
4. Merity, S., Xiong, C., Bradbury, J., and Socher, R. (2016). *Pointer Sentinel Mixture Models.* ICLR. https://arxiv.org/abs/1609.07843
5. Li, Y., Song, L., and Hou, H. (2024). *LoRAN: Improved Low-Rank Adaptation by a Non-Linear Transformation.* Findings of EMNLP 2024. https://aclanthology.org/2024.findings-emnlp.177/
6. Zhang, W., Mu, L., Ni, L., Jin, P., and Zhang, Y. (2026). *Polynomial Expansion Rank Adaptation: Enhancing Low-Rank Fine-Tuning with High-Order Interactions.* Findings of ACL 2026. https://aclanthology.org/2026.findings-acl.650/
7. Liu, S.-Y., et al. (2024). *DoRA: Weight-Decomposed Low-Rank Adaptation.* ICML. https://proceedings.mlr.press/v235/liu24bn.html
8. Jiang, T., et al. (2024). *MoRA: High-Rank Updating for Parameter-Efficient Fine-Tuning.* https://arxiv.org/abs/2405.12130
9. Huang, Y., et al. (2025). *HiRA: Parameter-Efficient Hadamard High-Rank Adaptation.* ICLR. https://proceedings.iclr.cc/paper_files/paper/2025/hash/48c368f105e8145b945227b73255635a-Abstract-Conference.html
10. Xu, Y., et al. (2026). *QuadraNet V2: Efficient and Sustainable Training of High-Order Neural Networks.* WACV. https://openaccess.thecvf.com/content/WACV2026/html/Xu_QuadraNet_V2_Efficient_and_Sustainable_Training_of_High-Order_Neural_Networks_WACV_2026_paper.html

LoRAN applies a nonlinear transformation to a low-rank weight update, and PERA
expands low-rank weight factors polynomially. Neither was an experimental
baseline here. This study evaluates a quadratic dependence on the input
activation at a frozen projection. Further verified metadata is maintained in
`comparisons/papers/REFERENCES.md`.
