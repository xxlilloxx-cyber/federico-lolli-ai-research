# Historical Linear+Quadratic versus the new Combined adapter

## Evidence inspected

The historical implementation is `LowRankAdapter(kind="linear_quadratic")` in `src/adapters.py`. Its recorded configurations are `config/historical/linear_quadratic_l1q2.yaml`, `linear_quadratic_l2q1.yaml`, and `linear_quadratic_l3q1.yaml`.

## Historical Linear+Quadratic

For row-vector input `x`, the historical code computes

```text
L(x) = (xA)B
Q(x) = ((xU) ⊙ (xV))P
adapter(x) = (alpha/r) [(alpha_l/r_l)L(x) + (alpha_q/r_q)Q(x)]
```

The experiment configurations omit the outer `rank` and `alpha`, so the training entry point supplies `rank=4` and `alpha=4`; the outer multiplier is therefore one. The effective historical expression is `(4/r_l)L(x) + (4/r_q)Q(x)`.

- Independent matrices: `A`, `B`, `U`, `V`, and `P`.
- Initialization: `A`, `U`, and `V` use Kaiming uniform initialization; `B` and `P` are zero initialized.
- Rank allocations examined: `(r_l,r_q)=(1,2),(2,1),(3,1)`.
- Primary placement: zero-based block 0, `attn.c_proj`.
- At a 768→768 projection the parameter count is `1536 r_l + 2304 r_q`: `(1,2)` has 6,144 parameters, `(2,1)` has 5,376, and `(3,1)` has 6,912.

## New Combined

The new adapter uses independent `A`, `B`, `U`, and `P`:

```text
L(x) = (xA)B
S(x) = ((xU) ⊙ (xU))P
Combined(x) = alpha/(r_l+r_s) [L(x) + S(x)]
```

The optional scaling ablation instead applies `alpha/r_l` and `alpha/r_s` to the respective branches. `A` and `U` use Kaiming uniform initialization; `B` and `P` are zero initialized. At a 768→768 projection its parameter count is `1536(r_l+r_s)`.

## Equivalence verdict

The methods are **not mathematically identical**. The historical quadratic branch is a general bilinear interaction with independent `U` and `V`; the new Symmetric branch reuses `U` on both sides. They also use different parameter formulas, rank allocations, and primary scaling conventions. Historical results labelled Linear+Quadratic must not be relabelled as results for the new Combined adapter.
