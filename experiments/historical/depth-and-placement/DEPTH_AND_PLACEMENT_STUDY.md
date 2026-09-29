# Depth and Placement Study — Symmetric Quadratic vs LoRA

**Author:** Federico Lolli  
**Contact:** xxlilloxx@gmail.com

## Executive summary

This report is a separate extension of the main Quadratic Transformer study. It uses every existing result and adds a controlled placement investigation for the currently most informative quadratic parameterization, `Δy = α(xU)^(⊙2)P`, against LoRA, `Δy = α(xA)B`. The first task was an audit of the three Linear+Quadratic 2,000-step seeds. All three are now valid: each has a configuration, a 2,000-row `metrics.csv`, a `summary.json`, matching rank allocation `r_L=1,r_Q=2`, 6,144 trainable parameters, and no NaN/Inf. This historical placement screen predates adapter-only checkpointing. The later long-convergence campaign implements and validates adapter-only checkpoints; this short screen is retained as historical placement evidence.

The placement screening used GPT-2 Small, WikiText-2, sequence length 128, batch 1, gradient accumulation 4, learning rate 3e-4, seed 42 and 20 optimization steps. Later sections of the project extend this screen with complete 12-layer scans, fixed-budget multi-layer confirmations and 5,000-step runs. Attention output adapters were tested at blocks 0, 3, 6, 9 and 11. MLP input/output and combined MLP placement were also tested. These short runs are a screening instrument; the three-seed confirmation used 100 steps for early, middle, late attention and combined MLP at layer 0.

The screening shows Symmetric below LoRA at every tested attention depth, with the largest short-run gap at the last block. The single-module MLP results are nearly tied, while combined MLP is more promising but uses 15,360 parameters. The confirmation means should be read as evidence about placement under 100 steps, not as a replacement for the 500/1,000/2,000-step main comparison.

## 1. Seed audit of Linear+Quadratic at 2,000 steps

The filesystem audit checks the actual rows rather than trusting report prose. A valid run requires `metrics.csv` with exactly 2,000 rows ending at step 2,000, a readable `summary.json`, zero NaN/Inf, and a configuration matching the intended ranks. The seed 456 directory is the successful rerun after the earlier CUDA lockup; the earlier failed attempt did not contain a valid summary and is not double-counted.

| Seed | Completed | Steps | Val loss | PPL | Params | rL | rQ | Result directory | Valid|
|---|---|---|---|---|---|---|---|---|---|
| 123 | True | 2000 | 3.443798542022705 | 31.30564842171543 | 6144 | 1 | 2 | results_extended/linear_quadratic/r123_steps2000 | True |
| 42 | True | 2000 | 3.4191102981567383 | 30.5422294514775 | 6144 | 1 | 2 | results_extended/linear_quadratic/r42_steps2000 | True |
| 456 | True | 2000 | 3.444941997528076 | 31.341465511474144 | 6144 | 1 | 2 | results_extended/linear_quadratic/r456_steps2000 | True |

Aggregated across the three valid seeds: validation loss mean 3.4360, standard deviation 0.0146; perplexity mean 31.06, standard deviation 0.31; training time and the exact source rows remain in `summary.json` and `metrics.csv`. The main technical report has been corrected so all references consistently describe three seeds. The historical failure remains documented as an operational event, not as an additional observation.

## 2. Verified GPT-2 architecture

Programmatic inspection reports 12 Transformer blocks, hidden size 768, 12 attention heads, and FFN inner size 3072. Candidate modules and their weight shapes are: `attn.c_proj` 768×768, `mlp.c_fc` 768×3072, and `mlp.c_proj` 3072×768. The adapter wrapper reads the actual `Conv1D.weight.shape`; it does not assume that all projections have the same width.

At rank 4, a single attention or single MLP projection has LoRA/Symmetric parameter count `r(d+d_out)`: 6,144 for attention and 15,360 for either MLP projection. For the two MLP projections together, rank 2 per projection gives 15,360 total parameters, which is comparable to a rank-4 single MLP projection but not to the 6,144-parameter attention experiment. This distinction is explicit in every table.

## 3. Screening protocol

The screening varied one factor at a time: placement and depth. Model kind, seed, optimizer, sequence length, effective batch, initial GPT-2 weights and validation protocol were held constant. Output factors were zero initialized, so the adapters preserve the original forward output at initialization. The baseline for each placement is the LoRA run at the same placement and rank.

### Screening results

| Configuration | Model | Val loss | PPL | Params | Time s | Peak MiB|
|---|---|---|---|---|---|---|
| attn_layer0_lora | lora | 3.8281142711639404 | 45.975758624725614 | 6144 | 3.242771034998441 | 348.3662109375 |
| attn_layer0_symmetric_quadratic | symmetric_quadratic | 3.825410842895508 | 45.85163431537736 | 6144 | 3.188283905001299 | 348.3662109375 |
| attn_layer11_lora | lora | 3.8251025676727295 | 45.837501571093405 | 6144 | 3.1718695300005493 | 348.3662109375 |
| attn_layer11_symmetric_quadratic | symmetric_quadratic | 3.813446283340454 | 45.3063085027934 | 6144 | 3.186400753998896 | 348.3662109375 |
| attn_layer3_lora | lora | 3.823338031768799 | 45.7566909713946 | 6144 | 3.1624565620004432 | 348.3662109375 |
| attn_layer3_symmetric_quadratic | symmetric_quadratic | 3.81771183013916 | 45.49997744029822 | 6144 | 3.1801246860013634 | 348.3662109375 |
| attn_layer6_lora | lora | 3.828734874725342 | 46.00430019985177 | 6144 | 3.1664789249989553 | 348.3662109375 |
| attn_layer6_symmetric_quadratic | symmetric_quadratic | 3.8205320835113525 | 45.62848002485866 | 6144 | 3.1797870859991235 | 348.3662109375 |
| attn_layer9_lora | lora | 3.827272415161133 | 45.93706994377786 | 6144 | 3.1645137820014497 | 348.3662109375 |
| attn_layer9_symmetric_quadratic | symmetric_quadratic | 3.818800449371338 | 45.54953656141522 | 6144 | 3.188947542999813 | 348.3662109375 |
| mlp_both_layer0_lora | lora | 3.8192758560180664 | 45.571196262026795 | 15360 | 3.3081165369985683 | 350.5693359375 |
| mlp_both_layer0_symmetric_quadratic | symmetric_quadratic | 3.805405616760254 | 44.943476243579994 | 15360 | 3.3428061720005644 | 350.5693359375 |
| mlp_c_fc_lora | lora | 3.8267998695373535 | 45.91536771045821 | 15360 | 3.2053730490006274 | 348.5068359375 |
| mlp_c_fc_symmetric_quadratic | symmetric_quadratic | 3.827401876449585 | 45.94301740101493 | 15360 | 3.2126692040001217 | 348.5068359375 |
| mlp_c_proj_lora | lora | 3.8217315673828125 | 45.6832434881072 | 15360 | 3.22317942600057 | 350.5693359375 |
| mlp_c_proj_symmetric_quadratic | symmetric_quadratic | 3.8217756748199463 | 45.685258503335675 | 15360 | 3.2529554339998867 | 350.5693359375 |

### How to read the screening figures

In `validation_vs_depth.png`, lower is better. A vertical separation between the two lines is the measured short-run difference at the same block. In `delta_loss_vs_depth.png`, values below zero mean Symmetric has lower validation loss than LoRA at that depth. These are 20-step observations; they do not establish long-run convergence or causal representational superiority.

## 4. Three-seed confirmation

The confirmation selected early attention (block 0), middle attention (block 6), late attention (block 11), and combined MLP at block 0. Each LoRA/Symmetric pair used seeds 42, 123 and 456. The purpose was to test whether the screening ordering survives seed variation, not to optimize hyperparameters separately for either model.

| Configuration | Model | N | Val loss mean | SD | PPL mean | Params | Time mean s|
|---|---|---|---|---|---|---|---|
| attn_layer0 | lora | 3 | 3.7611968517303467 | 0.020918728805221238 | 43.00613816860121 | 6144 | 15.15887626066736 |
| attn_layer0 | symmetric_quadratic | 3 | 3.70955483118693 | 0.057097102574750765 | 40.880406019300914 | 6144 | 15.29619594000178 |
| attn_layer11 | lora | 3 | 3.7606658935546875 | 0.007215771666367232 | 42.977781447476445 | 6144 | 15.173265827665697 |
| attn_layer11 | symmetric_quadratic | 3 | 3.692288557688395 | 0.0370113170495133 | 40.154889497945256 | 6144 | 15.270295185000577 |
| attn_layer6 | lora | 3 | 3.784510374069214 | 0.018147542149304224 | 44.01895166017367 | 6144 | 15.16788478299956 |
| attn_layer6 | symmetric_quadratic | 3 | 3.754786173502604 | 0.03087304642512349 | 42.738672238388936 | 6144 | 15.257604006332258 |
| mlp_both_layer0 | lora | 3 | 3.6780516306559243 | 0.023727181338655885 | 39.576627274553665 | 15360 | 15.813022730667095 |
| mlp_both_layer0 | symmetric_quadratic | 3 | 3.612757126490275 | 0.02176522042554516 | 37.073987221430386 | 15360 | 15.987449252999795 |

The error bars in `confirmation_mean_sd.png` are standard deviations across the three seeds. The attention curves in `convergence_attention.png` show validation checkpoints actually saved by the trainer; no interpolation is used.

## 5. Observations, interpretations and limitations

**Observation.** In the 20-step attention scan, Symmetric has lower validation loss than LoRA at blocks 0, 3, 6, 9 and 11. The gap is small at this horizon and grows toward the late block in the recorded screening.

**Interpretation.** This is consistent with a placement dependence: the quadratic branch may interact differently with later contextual representations. It is also compatible with an optimization effect or seed-42 trajectory effect. The scan cannot distinguish these explanations.

**Observation.** Single-module `mlp.c_fc` and `mlp.c_proj` results are almost equal between models, while combined MLP is lower for Symmetric in the short screen.

**Interpretation.** The combined MLP result motivates confirmation, but it also has 15,360 trainable parameters. It cannot be compared directly with the 6,144-parameter attention result without a fixed-total-budget experiment.

**Observation.** Peak VRAM remained about 348 MiB for single projections and about 351 MiB for the combined MLP screen.

**Interpretation.** In this setup the frozen backbone and activations dominate memory. Similar memory does not mean identical compute cost.

**Limitations.** The original depth screen and confirmation are short and the MLP combinations were limited. These historical limitations were subsequently addressed by the full 12-layer scan, fixed-total-budget multi-layer campaigns, local-rank confirmations and long-convergence study documented in the sibling reports. No inference latency or FLOPs measurement was collected. Therefore this report does not claim that a particular depth or module is universally optimal.

## 6. Figures and files

- `figures/validation_vs_depth.png`: attention validation loss versus block depth.
- `figures/perplexity_vs_depth.png`: corresponding perplexity view.
- `figures/delta_loss_vs_depth.png`: Symmetric minus LoRA loss, with zero reference.
- `figures/module_placement.png`: MLP placement screening.
- `figures/confirmation_mean_sd.png`: three-seed confirmation means and standard deviations.
- `figures/convergence_attention.png`: three-seed mean validation trajectories for early, middle and late attention.
- `linear_quadratic_seed_audit.csv`: filesystem audit table.
- `confirmation_summary.csv`: aggregated confirmation statistics.

## 7. Next justified experiments

Those historical screening observations motivated the fixed-total-budget comparison, full-depth scan and long-convergence/test campaigns that are now documented in the sibling reports. Remaining follow-up work is dedicated inference-latency/FLOPs measurement, longer MLP and attention+MLP comparisons, and replication on an independent dataset or model family.

## 8. Full attention depth scan

Because the coarse screen and three-seed confirmation both favored Symmetric at early, middle and late attention locations, a complete single-seed scan was run for all 12 `attn.c_proj` positions. Each run used 20 steps, rank 4, 6,144 trainable parameters, seed 42 and the same protocol. The resulting machine-readable table is `full_depth_scan.csv`.

The observed differences were:

| Layer | LoRA loss | Symmetric loss | ΔLoss (Symmetric − LoRA) |
|---:|---:|---:|---:|
| 0 | 3.8281 | 3.8254 | -0.0027 |
| 1 | 3.8288 | 3.8296 | +0.0009 |
| 2 | 3.8243 | 3.8030 | -0.0213 |
| 3 | 3.8233 | 3.8177 | -0.0056 |
| 4 | 3.8251 | 3.8129 | -0.0122 |
| 5 | 3.8223 | 3.8115 | -0.0108 |
| 6 | 3.8287 | 3.8205 | -0.0082 |
| 7 | 3.8201 | 3.8107 | -0.0094 |
| 8 | 3.8264 | 3.8134 | -0.0130 |
| 9 | 3.8273 | 3.8188 | -0.0085 |
| 10 | 3.8250 | 3.8139 | -0.0111 |
| 11 | 3.8251 | 3.8134 | -0.0117 |

**Observation.** Symmetric is lower at 11 of 12 layers; layer 1 is the only positive difference. The largest short-run gap occurs at layer 2, not at the last layer. **Interpretation.** The effect is not confined to a single endpoint of the network, but the layer-2 result may be a short-run or seed-specific fluctuation because this scan has one seed and 20 steps. **Limitation.** These data do not establish a stable depth profile without longer runs and multi-seed confirmation at the most separated layers.

`full_depth_validation.png` plots both losses; `full_depth_delta_loss.png` plots the signed difference with zero reference. Values below zero mean Symmetric has lower loss. No interpolation or unmeasured checkpoint was used.
