# LoRA

- **Publication:** Hu et al., ICLR 2022; see `../papers/REFERENCES.md`.
- **Formulation:** for a frozen `W in R^(d×d_out)`, LoRA learns
  `DeltaW=(alpha/r)AB`, with `A in R^(d×r)` and
  `B in R^(r×d_out)`, so `Delta y=x DeltaW`.
- **Architecture/location:** additive low-rank weight update on selected
  Transformer projections; the original paper studies attention projections
  and multiple Transformer backbones.
- **Parameters/rank:** `r(d+d_out)` for one projection; matrix rank at most
  `r`.
- **Scaling/initialization:** the common protocol uses `alpha/r` and initializes
  one factor to zero so the initial correction is zero. Exact choices depend
  on implementation.
- **Inference:** the constant update can be merged into `W`.
- **Evidence in this repository:** direct baseline on GPT-2/WikiText-2 and
  GPT-2/AG News, including matched parameters, ranks, seeds, and placement.
- **Relationship:** linear baseline. The Symmetric adapter uses the same
  factor shapes at one projection but squares projected activations, so it is
  not a constant low-rank weight update.
