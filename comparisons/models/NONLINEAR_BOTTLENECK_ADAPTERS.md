# Nonlinear bottleneck adapters and LoRAN

## Houlsby adapters

Houlsby et al. (ICML 2019) insert trainable bottleneck modules around frozen
Transformer sublayers. A typical path is down-projection, nonlinearity,
up-projection, and residual addition. Their BERT study covers 26 classification
tasks including GLUE. The nonlinear activation is applied inside an added
bottleneck; it is not a quadratic form tied across outputs as in this study.

## LoRAN

Li, Song, and Hou (Findings of EMNLP 2024) apply a nonlinear transformation to
the low-rank **weight update**, often summarized as `DeltaW=f(BA)`. The paper
reports SAMSum and 20 Newsgroups experiments. This differs from an
activation-dependent correction `f(x)` whose Jacobian changes with `x`.

Neither method was evaluated as a baseline in this repository. They are
architecturally related because they introduce nonlinearity into PEFT, but
neither is mathematically equivalent to the Symmetric adapter.
