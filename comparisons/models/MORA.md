# MoRA

- **Publication:** Jiang et al., arXiv:2405.12130 (2024).
- **Core mechanism:** learns a square matrix and uses fixed, non-parametric
  input compression and output expansion operators to obtain a high-rank
  update at a LoRA-like trainable budget.
- **Rank:** its central goal is to avoid the rank-at-most-`r` limitation of the
  direct LoRA matrix product.
- **Inference:** the original work describes a mergeable weight update.
- **Evaluation:** instruction tuning, mathematical reasoning, continual
  pretraining, memory, and pretraining; reported gains are strongest on
  memory-intensive tasks.
- **Relationship:** high-rank linear weight update rather than an
  activation-dependent quadratic map.
- **Status here:** related work only; not experimentally evaluated.
