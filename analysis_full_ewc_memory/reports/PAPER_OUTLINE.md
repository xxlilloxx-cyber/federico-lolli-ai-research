# Paper Outline

## Abstract
- Key message: EWC strength produces an architecture-dependent stability-plasticity trade-off.
- Evidence: Table 2; Figures 1–3.

## 1. Introduction
- Key message: continual factual adaptation requires balancing retention and acquisition.
- Evidence: ABC baseline context.

## 2. Related motivation
- Key message: parameter efficiency does not itself prevent forgetting.
- Evidence: Figure S2 and parameter budgets.

## 3. Continual factual-learning setup
- Key message: deterministic A→B→C phases without replay.
- Evidence: Table 1.

## 4. Low-rank adaptation geometries
- Key message: LoRA, Symmetric, and Combined provide distinct geometries at 6,144 parameters.
- Evidence: protocol table.

## 5. EWC formulation
- Key message: diagonal Fisher penalizes movement from consolidated parameters.
- Evidence: equation and Fisher tables.

## 6. Exploratory regularization sweep
- Key message: seven λ values reveal the transition shape and under-learning regime.
- Evidence: Figures 1 and 6; Table 5.

## 7. Multi-seed confirmation
- Key message: candidate regions are assessed across seeds 42/123/456.
- Evidence: Table 2; Figures 2–5.

## 8. Stability-plasticity analysis
- Key message: stability, plasticity, balanced memory, probability, and exact match have distinct optima.
- Evidence: Tables 2–4.

## 9. Mechanistic analysis
- Key message: movement and Fisher concentration track constraints descriptively.
- Evidence: Figure 6 and supplementary tables.

## 10. Discussion
- Key message: useful λ depends on geometry; no universal winner is required.

## 11. Limitations
- Key message: three seeds, one fact generator, one model/placement, single phase order.

## 12. Conclusion
- Key message: EWC strength is a controllable design variable for low-rank continual factual learning.
