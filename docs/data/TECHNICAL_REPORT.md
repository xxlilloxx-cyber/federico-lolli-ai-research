# Low-Rank Quadratic Adaptation of GPT-2

**Author:** Federico Lolli  
**Contact:** xxlilloxx@gmail.com

## 1. Purpose and canonical definitions

This exploratory study compares small trainable corrections to frozen GPT-2 Small projections. The final campaign uses 12,288 adapter parameters, WikiText-2, three seeds, and 5,000 optimizer steps. `best_recorded_validation_loss` is the minimum fresh validation result per seed; `final_validation_loss` is the step-5,000 validation result; `final_test_loss` is test loss from that same final checkpoint. Means and sample SDs aggregate per-seed quantities.

The historical summary field `validation_loss` is the first quantity, not final-step loss. Public canonical exports are `docs/data/long_final_per_seed.csv` and `docs/data/long_validation_trajectories.csv`.

## 2. Mathematics and implementation

Rows are vectors: \(x\in\mathbb{R}^{1\times d}\), \(W\in\mathbb{R}^{d\times d_{out}}\), and \(y_{base}=xW+b\), with frozen \(W,b\). Final-study LoRA is \(\Delta y=(\alpha/r)(xA)B\). Symmetric Quadratic is \(\Delta y=(\alpha/r)(xU)^{\odot2}P\).

For output \(j\), \(\Delta y_j=(\alpha/r)\sum_kP_{kj}(xu_k)^2=xQ_jx^T\), where \(Q_j=(\alpha/r)\sum_kP_{kj}u_ku_k^T\). Thus rank\((Q_j)\le r\); directions are shared across outputs and signed \(P_{kj}\) makes \(Q_j\) potentially indefinite. The expansion \((a x_1+b x_2)^2=a^2x_1^2+2abx_1x_2+b^2x_2^2\) shows the cross term.

The code uses \(\alpha=4\), yielding scales 0.5, 1, and 2 at ranks 8, 4, and 2. Changing local rank therefore also changes scaling. Output factors B, V, and P are zero-initialized: output corrections are initially zero, output factors receive gradients, and input factors have zero initial gradient until output factors change. Earlier element-wise, signed, and interaction branches also apply `alpha/r`; Linear+Quadratic applies separate `alpha_L/r_L` and `alpha_Q/r_Q`.

## 3. Protocol and step-0 provenance

GPT-2 Small (`gpt2`) has 12 zero-based blocks. The final adapters wrap `attn.c_proj`: they receive the concatenated attention output in parallel with frozen `attn.c_proj`, sum their outputs, then GPT-2 applies residual dropout and the attention residual. The MLP path is frozen. Block 2 is the third block.

WikiText-2 raw non-empty lines are joined with newlines, tokenized without added special tokens, and partitioned into 128-token blocks; a short remainder is discarded. Training uses batch 1, accumulation 4, AdamW at 3e-4 with default betas, epsilon, and weight decay, clipping 1.0, no scheduler or warmup, FP16 autocast backbone and FP32 adapters. Validation and test are eval mode; no adapter dropout exists. Eight validation blocks are evaluated each pass. Recorded versions: PyTorch 2.3.1+cu121, Transformers 4.57.6, Datasets 5.0.1, Accelerate 1.15.0. Model/dataset revisions were not recorded.

`initial_validation_loss=3.8293571472` is an original pre-update evaluation in every run summary. It is identical because zero adapters preserve the same frozen model; it is not 18 independent measurements. Step 1 is post-update. The public trajectory contains one labeled step-0 row and 200 fresh evaluations (25..5000) per run: 18×201=3,618 rows. Cached non-evaluation rows are excluded.

## 4. Final 5,000-step results

| Geometry | Architecture | Final validation | Best recorded validation | Final test loss | Test PPL |
|---|---|---:|---:|---:|---:|
| Concentrated [2], r=8 | LoRA | 3.3703 ± 0.0045 | 3.3577 ± 0.0041 | 3.7741 ± 0.0022 | 43.56 ± 0.10 |
| Concentrated [2], r=8 | Symmetric | 3.2831 ± 0.0080 | 3.2559 ± 0.0048 | 3.7067 ± 0.0060 | 40.72 ± 0.24 |
| Two-layer [0,11], r=4 | LoRA | 3.2186 ± 0.0166 | 3.2003 ± 0.0178 | 3.6361 ± 0.0075 | 37.95 ± 0.29 |
| Two-layer [0,11], r=4 | Symmetric | 3.1876 ± 0.0119 | 3.1520 ± 0.0126 | 3.6306 ± 0.0025 | 37.74 ± 0.09 |
| Four-layer [0,3,7,11], r=2 | LoRA | 3.1743 ± 0.0162 | 3.1537 ± 0.0204 | 3.6192 ± 0.0115 | 37.31 ± 0.43 |
| Four-layer [0,3,7,11], r=2 | Symmetric | 3.1925 ± 0.0157 | 3.1557 ± 0.0320 | 3.6452 ± 0.0149 | 38.29 ± 0.57 |

Final Symmetric-minus-LoRA validation differences are -0.0872, -0.0310, and +0.0182. The best-recorded four-layer difference is +0.0020 and is not a final-checkpoint difference. Concentrated test perplexity decreases from 43.5567 to 40.7165, 6.52% from unrounded means. Best Symmetric test is two-layer; best overall mean test is four-layer LoRA.

Paired test-minus-final-validation differences are approximately 0.4038/0.4236, 0.4175/0.4430, and 0.4449/0.4527 for LoRA/Symmetric. They are descriptive only.

## 5. AULC and historical studies

Legacy AULC is a simple mean of all 5,000 per-step validation rows, including cached repeats and excluding step 0, then averaged over seeds. It is not an integral and is retained only as a legacy metric. A normalized trapezoidal alternative would use only fresh points and actual spacing; it is not silently substituted here.

Early 500-step family comparisons, 1,000-step placement screens, and separate 2,000-step 6,144- and 12,288-parameter studies remain historical best-recorded summaries. They established feasibility and motivated the long campaign, but are not combined with fresh 5,000-step endpoints. Supporting tables are in `MASTER_RESULTS.csv`, `DEPTH_AND_PLACEMENT_STUDY.md`, `MULTILAYER_FIXED_BUDGET.md`, and `LOCAL_RANK_DEPTH_INTERACTION_STUDY.md`.

## 6. Interpretation, limits, and references

Explicit quadratic corrections provide useful adaptation capacity in the concentrated matched placement: only 12,288 parameters, about 0.01% of the 124M backbone, are trained. The advantage narrows for two layers and reverses on four-layer held-out test. This is neither statistical proof nor universal superiority. Limits include one model family, one dataset, three seeds, short context, one optimizer protocol, and coupled rank/scaling/placement changes.

Related work: Hu et al. (2021), *LoRA*, arXiv:2106.09685; Vaswani et al. (2017), *Attention Is All You Need*, arXiv:1706.03762; Radford et al. (2019), *Language Models are Unsupervised Multitask Learners*; Merity et al. (2016), *Pointer Sentinel Mixture Models*, arXiv:1609.07843; Li, Song, and Hou (2024), *LoRAN*, Findings of EMNLP 2024. LoRAN is a nonlinear weight-update parameterization and is not tested here. PERA is related polynomial weight-factor work and is not tested here; its claimed 2026 acceptance status requires verification from the original source before publication.
