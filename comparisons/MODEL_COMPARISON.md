# Model comparison

This is the canonical high-level comparison. Equivalence details are in
[`MATHEMATICAL_EQUIVALENCE.md`](MATHEMATICAL_EQUIVALENCE.md); experimental
protocols are in [`EXPERIMENTAL_METHODOLOGY.md`](EXPERIMENTAL_METHODOLOGY.md).

| Method | Adaptation object | Linear in activation? | Explicit square | Cross-feature terms | Directly tested here? |
|---|---|---:|---:|---:|---:|
| LoRA | low-rank weight update | yes | no | no at adapter level | yes |
| DoRA | weight magnitude/direction | yes | no | no explicit activation term | no |
| MoRA | high-rank linear weight update via operators | yes | no | no explicit activation term | no |
| HiRA | Hadamard weight modulation | yes | no | no explicit activation term | no |
| PERA | polynomial expansion of low-rank factors | yes after weight construction | factor-space | factor-space | no |
| QuadraNet V2 | factored quadratic form added to pretrained term | no | yes | yes | no |
| Houlsby adapter | nonlinear bottleneck module | generally no | activation-dependent but not specifically square | implicit through network | no |
| LoRAN | nonlinear transform of weight update | depends on constructed update, not current `x` in the same way | weight-space | weight-space | no |
| Symmetric Quadratic | `s((xU)⊙(xU))P` | no | yes | yes after expansion | yes |

Detailed per-model records are under `models/`. Published outcomes use different
backbones and tasks and are never combined numerically with this study.
