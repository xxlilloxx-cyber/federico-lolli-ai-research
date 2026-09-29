# Symmetric Quadratic derivative analysis

Let `x` be a row vector in R^(1×d), `U` in R^(d×r), `P` in
R^(r×d_out), and `s=alpha/r`. The implemented adapter is

`f(x) = s ((xU) ⊙ (xU)) P`.

Writing `z=xU`, its Jacobian in input-by-output orientation is

`J(x) = 2s U diag(z) P`.

Thus the local Jacobian depends on the activation. It has rank at most `r`,
but it is not a constant weight update. For a fixed output reduction `c`,

`g(x)=f(x)c`,

the Hessian is

`H_g = 2s U diag(Pc) U^T`.

It is independent of `x` when `c` is fixed, symmetric, and has rank at most
`r`. Its off-diagonal entries encode explicit pairwise input-feature
interactions. The learned coefficients in `P c` can be positive or negative.

For LoRA, `f_L(x)=s(xA)B`, hence `J_L=sAB` is constant and the adapter-only
Hessian is exactly zero. This mathematical distinction does not by itself
imply better task performance.

The numerical tests in `tests/test_symmetric_derivatives.py` compare both
formulas with PyTorch autograd in double precision. The empirical analysis uses
the first public WikiText-2 test block and token positions 0–31 at the input to
block-0 `attn.c_proj`. For the Hessian scalar, `c` is the normalized mean frozen
`c_proj` output direction on that fixed sample. One Hessian is calculated per
checkpoint; token positions are not treated as repeated independent Hessian
observations.
