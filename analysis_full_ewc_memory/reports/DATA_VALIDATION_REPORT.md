# Full EWC memory study: data validation

- Canonical scientific runs: 72
- Physical completed runs: 75
- Physical duplicates collapsed: 3
- Confirmation references reused: 9
- Dataset SHA-256: `cc505ec681369c5f9aa4c7123cfff11da878bce17c0669890d416b5eb07f1923`
- No-replay validation: protocols use the validated phase-specific `ewc_like`/ABC runner; A is absent from phase B training and A/B are absent from phase C training.
- Parameter budgets: sequential/EWC 6,144; grow/hard 18,432.
- Lambda-zero: same EWC path with Fisher/reference state and exact zero objective multiplier.

## Result

**VALID** — all checks passed.
