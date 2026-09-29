# QuadraNet V2

- **Publication:** Xu et al., WACV 2026, pp. 1365–1373.
- **Core formulation:** a frozen/pretrained first-order term is augmented by a
  quadratic term written in the paper as `X^T W_a^T W_b X` after low-rank
  decomposition of the quadratic coefficient.
- **Architecture:** staged quadratic adaptation of convolutional networks,
  combined with selective placement, atrous/sparse design, and a dedicated
  quadratic computation library.
- **Parameters/rank:** the quadratic term is factorized to avoid explicitly
  storing a dense `n×n` matrix; exact budgets depend on the convolutional
  construction and selected neurons.
- **Backbones/tasks:** computer-vision architectures and datasets in the WACV
  protocol, rather than GPT-2 language modeling.
- **Relationship:** both add a quadratic form to a frozen first-order mapping.
  QuadraNet V2 is therefore the closest verified architectural precedent in
  this comparison. The complete multi-output parameter sharing differs: this
  study uses one shared `U` and output-specific signed coefficients in `P`.
  The published scalar/neuron factorization does not, by itself, establish
  equality to that whole vector-valued map.
- **Status here:** not reproduced; no numerical comparison is valid across the
  vision and language protocols.
