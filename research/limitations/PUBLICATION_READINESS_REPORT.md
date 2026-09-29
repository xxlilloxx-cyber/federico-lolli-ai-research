# Publication readiness report

**Review date:** 2026-09-29  
**Author:** Federico Lolli  
**Release state:** local author-review build; no push, merge, Pages deployment,
release, or history rewrite was performed.

## Completed evidence

- Historical GPT-2 adapter-family, duration, placement, and fixed-budget work
  is retained as historical evidence.
- Six controlled AG News runs report held-out accuracy, macro-F1, and loss.
- The controlled 1,000-step rank screen contains 24 valid runs.
- The independent 5,000-step confirmation contains 24 valid runs, final
  validation, and final-checkpoint held-out WikiText-2 test evaluation.
- The seed-42 scaling ablation is complete under its predeclared single-seed
  scope and is labelled exploratory.
- Convergence, LoRA update spectra, Symmetric factor spectra, output covariance,
  local-Jacobian spectra, adapter-only Hessians, and a matched rank-4 cost
  benchmark are complete.
- Analytical Jacobian and Hessian expressions are covered by autograd tests.

## Canonical publication artifacts

- Manuscript sources: `research/paper/SCIENTIFIC_MANUSCRIPT.md` and `.tex`
- Compiled manuscript: `research/paper/SCIENTIFIC_MANUSCRIPT.pdf`
- LaTeX source: `research/paper/SCIENTIFIC_MANUSCRIPT.tex`
- Extended report: `research/paper/TECHNICAL_REPORT.md`
- Evidence review: `research/paper/FINAL_RESEARCH_REVIEW.md`
- Reproducibility: `research/reproducibility/REPRODUCIBILITY.md`
- Comparisons and equivalence qualifications: `comparisons/`
- Sanitized data and figures: `results/`
- Generated public mirrors and website: `docs/`

The manuscript PDF is built from the canonical LaTeX source with the
user-local Tectonic toolchain and mirrored into `docs/data/` for the website.

## Claim and unit audit

The public narrative reports both central outcomes: Symmetric has lower mean
final validation and held-out WikiText-2 loss at ranks 1, 2, 4, and 8 under the
controlled 5,000-step protocol, while LoRA has higher mean AG News accuracy and
macro-F1. It does not claim universal superiority, statistical significance,
state of the art, full mathematical novelty, or causal attribution to measured
derivative structure. Published results from other methods are literature
context rather than direct comparisons.

The throughput unit was checked from the raw cost table. At batch 1 and length
128, about 12 ms per forward pass corresponds to approximately 10.5 thousand
tokens/s. Earlier shorthand stating 10.5 tokens/s omitted a factor of one
thousand and is not used in current publication material.

## Remaining scientific limitations

- GPT-2 Small is the only backbone and model family.
- WikiText-2 and AG News are the only controlled datasets/tasks.
- The main comparisons use three seeds.
- The constant-scale ablation uses seed 42 only.
- Controlled rank confirmation focuses on block-0 `attn.c_proj`.
- No full-block/model Hessian analysis was completed; only adapter-level
  analytical and measured results are published.
- No broad hyperparameter search or second optimizer protocol was performed.

## Privacy and release blocker

Tracked publication files and the `docs/` payload were scanned for local home
paths, unapproved email addresses, private-key markers, common credential
assignment patterns, raw checkpoints, and raw result directories. None were
found in the current tracked public payload. Private raw runs and checkpoints
remain local and ignored.

A separate reachable-history scan found no category hits in branch or tag
history. The all-ref scan found local-path/private experiment metadata only in
the stash-reachable historical result artifacts already recorded by the prior
audit; no secret value is reproduced here.

A local stash ref remains:
`stash@{0}: release-site-revision-before-sanitized-history`. Previous review
identified private experiment metadata in stash-reachable history. A normal
branch push does not ordinarily transfer local stash refs, whereas mirroring
all refs or packaging `.git` could expose them. This distinction must be
checked against the eventual publication procedure. The stash was preserved
and no destructive history action was taken.

## Assessment

The working tree is ready for **carefully qualified author review**. Public
release remains blocked until Federico approves the content and selects a
publication procedure that excludes or safely resolves the documented local
stash-history risk. No deployment has occurred.
