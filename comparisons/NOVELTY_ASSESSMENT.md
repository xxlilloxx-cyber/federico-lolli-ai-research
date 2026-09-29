# Originality and novelty assessment

## Supported contributions

1. A controlled empirical evaluation of the activation-space map
   `s((xU)⊙(xU))P` against parameter-matched LoRA on frozen GPT-2.
2. Distinct three-seed WikiText-2 screening and 5,000-step confirmation, plus
   held-out final-checkpoint tests and a separate AG News evaluation.
3. A depth/budget study and a single-seed constant-effective-scale ablation.
4. Derivative-aware mechanism analysis that does not misrepresent the nonlinear
   adapter as a constant weight update.

## Claims not established

- The general concept of adding a factorized quadratic form to a frozen
  pretrained mapping is not established as new: QuadraNet V2 is a verified,
  closely related precedent.
- Polynomial/nonlinear PEFT is not new: PERA, LoRAN, and bottleneck adapters
  provide related parameter-space or module-space approaches.
- The experiments do not establish state of the art, universal superiority,
  transfer to other model families, or causal attribution of performance to a
  measured Hessian/spectral quantity.

## Careful characterization

The project is best described as an **independent parameterization and
controlled experimental study of activation-dependent symmetric quadratic
low-rank adaptation**. Its empirical evidence and shared-factor formulation
are useful contributions even though broad mathematical novelty is not claimed.
