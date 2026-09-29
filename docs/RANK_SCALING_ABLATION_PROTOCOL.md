# Rank-scaling ablation protocol

This seed-42, 1,000-step ablation is separate from both the primary 5,000-step
confirmation and the existing current-scaling screening. It holds the effective
adapter multiplier constant: `alpha / rank = 1`.

| Rank | Alpha | Effective scale |
|---:|---:|---:|
| 1 | 1 | 1 |
| 2 | 2 | 1 |
| 4 | 4 | 1 |
| 8 | 8 | 1 |

Methods are LoRA and Symmetric Quadratic, both inserted at frozen GPT-2
`transformer.h[0].attn.c_proj` in block 0. Dataset, sequence length 128,
gradient accumulation 4, AdamW learning rate 3e-4, validation protocol, and
seed are identical to the controlled screening. The existing alpha=4 runs are
used as Policy A and are never rerun.
