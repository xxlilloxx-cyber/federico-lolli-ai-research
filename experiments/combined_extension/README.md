# LoRA / Symmetric / Combined GPT-2 extension

This isolated study is launched by `run_full_combined_study.py`. It does not rewrite historical source code, results, or manuscripts.

The primary Combined definition uses independent `A,B,U,P`, total rank `R=r_lora+r_symmetric`, and

```text
y = W0(x) + alpha/R [(xA)B + ((xU)⊙(xU))P].
```

The separate scaling ablation applies `alpha/r_lora` and `alpha/r_symmetric` to the two branches. The conventions are stored in separate run cells and never pooled silently.

Controlled campaigns are the 5,000-step WikiText-2 rank confirmation, matched-budget tables, and AG News comparison. Scaling, short placement, attention/MLP, and short multilayer campaigns are labelled exploratory.

With two non-zero integer-ranked branches, four active Combined layers require at least four rank-`1+1` adapters, or 12,288 parameters. Consequently, an exact 6,144-parameter four-layer Combined geometry is impossible without zero or fractional ranks and is not created.

One-command entry points:

```bash
.venv/bin/python run_full_combined_study.py --dry-run
.venv/bin/python run_full_combined_study.py --smoke
.venv/bin/python run_full_combined_study.py --campaign all
```

A cell is skipped only when `summary.json` contains `"complete": true`. Incomplete attempts are archived below that cell's `attempts/` directory before a clean restart. Output and errors stream to the cell's `log.txt`.
