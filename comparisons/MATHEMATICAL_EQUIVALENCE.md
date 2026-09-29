# Mathematical equivalence assessment

## Symmetric Quadratic Adapter

For row-vector `x in R^(1×d)`, `U in R^(d×r)`,
`P in R^(r×d_out)`, and `s=alpha/r`, the implementation computes

`f(x) = s ((xU) ⊙ (xU)) P`.

For output coordinate `j`,

`f_j(x) = s sum_k P[k,j] (x u_k)^2 = x Q_j x^T`,

where

`Q_j = s U diag(P[:,j]) U^T`.

Each `Q_j` is symmetric and has rank at most `r`. The directions `u_k` are
shared by all output coordinates, while each output uses its own signed
coefficients `P[k,j]`. This shared-factor vector-valued constraint is part of
the parameterization.

## Classification

| Method | Assessment relative to this formulation | Reason |
|---|---|---|
| LoRA | mathematically different | constant linear map `x(sAB)`; zero adapter-only Hessian |
| DoRA | mathematically different | reparameterizes magnitude/direction of a weight |
| MoRA | mathematically different | high-rank but still linear weight update |
| HiRA | mathematically different | Hadamard modulation is in weight space |
| PERA | related polynomial family, not equivalent | expands parameter factors before composition, rather than projected activations at runtime |
| Houlsby adapter | more general nonlinear bottleneck in one sense, structurally different | arbitrary bottleneck activation does not impose the shared symmetric quadratic forms above |
| LoRAN | related nonlinearity, not equivalent | nonlinear transform applies to the weight update |
| QuadraNet V2 | closest quadratic-form precedent; partial scalar-level correspondence | both use a frozen first-order term plus factored `x^TQx`, but factor constraints and multi-output sharing are not established as identical |

## QuadraNet V2 in detail

The published QuadraNet V2 neuron uses a factorized term of the form
`x^T W_a^T W_b x`. A scalar quadratic function only depends on the symmetric
part of its coefficient matrix. Our output `j` uses the stronger explicit
factorization `U diag(p_j) U^T`, with the same `U` reused for every `j`.

For one output, a QuadraNet factorization can represent a general low-rank
bilinear coefficient before symmetrization, whereas this study restricts both
sides to the same learned directions and supplies signed diagonal weights.
Conversely, matching one scalar form does not establish equality of a complete
multi-output layer: one must also match how factors are shared across outputs,
the number of parameters, sparsity/atrous structure, and placement. The
available evidence supports **architectural and mathematical relatedness**, not
complete equivalence or an unrestricted containment claim.

## Parameter implications

At one `d→d_out` projection, this implementation uses `r(d+d_out)` parameters,
the same count as LoRA. It avoids materializing `d_out` dense `d×d` matrices.
That equality does not extend automatically to QuadraNet V2, PERA, or
bottleneck adapters because their sharing and placement rules differ.
