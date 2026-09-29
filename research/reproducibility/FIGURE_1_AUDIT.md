# Figure 1 Mathematical Audit

**Date:** 2026-09-29

**Audited candidate location:** `docs/assets/publication/adapter_architecture_comparison.png`

**Candidate SHA-256:** `f9974e04ba54d3f8b7577065823612cc396e108e3c7cf2098c1629b765f54090`

## Decision

The supplied PNG was **not approved for publication** and remains unreferenced.
It conflicts with the canonical row-vector formulation used by the code,
manuscript, technical report, and website.

## Inconsistencies

1. It uses a column-vector path (`W₀x`) instead of `xW+b`.
2. Its LoRA branch is `Vx → U`, rather than canonical `xA → B`.
3. Its Symmetric branch is `Px → square → U`, rather than `xU → square → P`.
4. `U diag(x_U²) P x` is not the implemented adapter and introduces both an
   unsupported diagonal operation and an extra multiplication by `x`.
5. `2rd` assumes `d=d_out`; the general parameter count is `r(d+d_out)`.
6. The frozen bias `b` is omitted.

## Publication action

The public site uses the corrected canonical SVG. The rejected PNG candidate
was not edited, staged, or referenced. It remains a local working-tree file
for manual comparison and is not part of this publication-polish commit.
