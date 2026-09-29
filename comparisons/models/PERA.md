# PERA

- **Publication:** Zhang et al., Findings of ACL 2026,
  DOI 10.18653/v1/2026.findings-acl.650.
- **Core formulation:** structured polynomial expansion is applied to the
  low-rank **parameter factors** before they are composed into a weight update.
  The paper studies original, square, and cross components.
- **Nonlinearity location:** parameter/factor space. Once the adapted weight is
  formed, the layer remains linear in the current activation.
- **Rank/cost claim:** the paper motivates richer factor interactions without
  increasing the nominal rank or inference cost under its construction.
- **Evaluation:** LLaMA-family commonsense reasoning and RoBERTa/GLUE, with rank
  robustness, placement, square/cross ablations, feature-interaction, and
  efficiency analyses.
- **Relationship:** close polynomial motivation, different mathematical
  object. This repository computes `(xU)⊙(xU)` for each activation; it does not
  implement PERA's factor polynomial.
- **Status here:** not an evaluated baseline. No cross-paper score is treated
  as a head-to-head result.
