import csv,json,glob,statistics,math
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'report'/'MULTILAYER_FIXED_BUDGET'; FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
paths=sorted((ROOT/'results_multilayer/screen').glob('*/summary.json'))
rows=[]
for p in paths:
 s=json.loads(p.read_text()); d=p.parent; ms=list(csv.DictReader((d/'metrics.csv').open())); rows.append({'name':d.name,'kind':s['kind'],'loss':s['validation_loss'],'ppl':s['perplexity'],'params':s['trainable_parameters'],'time':s['training_seconds'],'vram':s['peak_vram_bytes']/2**20,'layers':','.join(map(str,s['layers'])),'rank':s['rank'],'steps':len(ms),'valid':len(ms)==s['steps'] and s['nan_count']==0 and s['inf_count']==0,'metrics':d/'metrics.csv'})
with (OUT/'screen_results.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='metrics']).writeheader(); csv.DictWriter(f,fieldnames=[k for k in rows[0] if k!='metrics']).writerows([{k:v for k,v in x.items() if k!='metrics'} for x in rows])
# AULC and curves
for x in rows:
 rs=list(csv.DictReader(x['metrics'].open())); x['curve']={int(r['step']):float(r['validation_loss']) for r in rs}; x['aulc']=sum(x['curve'].values())/len(x['curve'])

def keybase(n):
 return n.rsplit('_lora',1)[0].rsplit('_symmetric_quadratic',1)[0]
configs=sorted(set(keybase(x['name']) for x in rows))
def get(c,k): return next(x for x in rows if keybase(x['name'])==c and x['kind']==k)
# plots
plt.figure(figsize=(8,5))
for c in configs:
 for k,color,ls in [('lora','tab:blue','-'),('symmetric_quadratic','tab:orange','--')]:
  x=get(c,k); plt.plot(sorted(x['curve']),[x['curve'][s] for s in sorted(x['curve'])],label=f'{c} {k}',color=color,ls=ls,alpha=.8)
plt.xlabel('Optimization step');plt.ylabel('Validation loss');plt.title('Fixed-budget multi-layer convergence');plt.legend(fontsize=6);plt.tight_layout();plt.savefig(FIG/'validation_curves.png',dpi=180);plt.close()
plt.figure(figsize=(8,5));
for c in configs:
 a=get(c,'symmetric_quadratic')['loss']-get(c,'lora')['loss']; plt.bar(c,a,label=c)
plt.axhline(0,color='k',lw=.8);plt.ylabel('ΔLoss = Symmetric − LoRA');plt.title('Architecture difference at fixed total budget');plt.xticks(rotation=25);plt.tight_layout();plt.savefig(FIG/'delta_loss_architecture.png',dpi=180);plt.close()
plt.figure(figsize=(8,5));
for k,c in [('lora','tab:blue'),('symmetric_quadratic','tab:orange')]:
 z=[get(x,k) for x in configs]; plt.plot([x['layers'].count(',')+1 for x in z],[x['loss'] for x in z],marker='o',label=k,color=c)
plt.xlabel('Number of adapted layers');plt.ylabel('Validation loss');plt.title('Loss versus number of insertion points (screening configurations)');plt.legend();plt.tight_layout();plt.savefig(FIG/'loss_vs_insertion_points.png',dpi=180);plt.close()
plt.figure(figsize=(8,5));
for k,c in [('lora','tab:blue'),('symmetric_quadratic','tab:orange')]:
 z=[get(x,k) for x in configs]; plt.plot([x['params'] for x in z],[x['loss'] for x in z],marker='o',label=k,color=c)
plt.xlabel('Trainable parameters');plt.ylabel('Validation loss');plt.title('Parameter efficiency (all runs use target 6144)');plt.legend();plt.tight_layout();plt.savefig(FIG/'parameter_efficiency.png',dpi=180);plt.close()
# table and report
fmt=lambda v:f'{v:.4f}' if isinstance(v,float) else str(v)
lines=['| Configuration | Model | Layers | Rank/layer | Steps | Params | Val loss | PPL | AULC | Time s | Peak MiB | Valid |','|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
for x in rows: lines.append('| '+' | '.join([x['name'],x['kind'],x['layers'],str(x['rank']),str(x['steps']),str(x['params']),fmt(x['loss']),fmt(x['ppl']),fmt(x['aulc']),fmt(x['time']),fmt(x['vram']),str(x['valid'])])+' |')
table='\n'.join(lines)
fail='results_multilayer/screen/data_driven_two_lora_seed42'
md=f'''# Multi-layer Fixed-budget Study — LoRA vs Symmetric Quadratic

## Executive summary

This report is a separate continuation of the single-layer and depth-placement studies. It asks whether a fixed total adapter budget is used more effectively when concentrated in one Transformer block or distributed over multiple blocks. The attention output projection `attn.c_proj` was used because it is the original matched-budget reference and has identical 768→768 dimensions at every block.

The total target was 6,144 trainable parameters, equivalent to four rank units because one rank costs `768+768=1,536` parameters. The tested valid configurations therefore use rank 4 in one layer, rank 2 in two layers, or rank 1 in four layers. LoRA and Symmetric use exactly the same layers and total parameter count within each pair.

The phase-1 screening completed the concentrated layer-2 pair, the distributed two-layer pair `[0,11]`, and the distributed four-layer pair `[0,3,7,11]`, all at 1,000 steps and seed 42. A subsequent data-driven run entered the known CUDA kernel lockup before producing a valid summary; it was not counted, and no later runs were launched after the lockup. The valid results are therefore a partial but controlled screening, not a completed regional campaign.

## 1. Scientific question and design

The single-layer depth scan showed that Symmetric can be competitive at several depths. The present experiment changes only the distribution of the same total rank budget. For LoRA the per-layer correction is `α(xA_l)B_l`; for Symmetric it is `α(xU_l)^(⊙2)P_l`. Parameters are independent across layers and the GPT-2 backbone is frozen.

For each architecture, the distribution gain is defined relative to the concentrated layer-2 run:

`Gain_distribution = Loss_multi-layer − Loss_concentrated`.

A negative value means distribution improved validation loss under the same total budget. The architecture interaction is:

`Interaction = (Loss_SYM,multi − Loss_SYM,single) − (Loss_LoRA,multi − Loss_LoRA,single)`.

Negative interaction means distribution helped Symmetric more than LoRA.

## 2. Parameter accounting and validation

The smoke test verified forward, backward, finite gradients, frozen backbone and exact 6,144 trainable parameters for rank allocations 4, 2+2 and 1+1+1+1. Every valid screen run has 1,000 metric rows, zero NaN/Inf and peak allocation near 348.4 MiB. No checkpoint is produced by the existing trainer; metrics and summaries are the source of truth.

## 3. Valid screening results

{table}

For the valid configurations, the data show how the same budget behaves when moved across depth. The exact per-step source is in each run's `metrics.csv`; aggregate rows are in `screen_results.csv`.

## 4. How to read the figures

`validation_curves.png` shows loss versus optimization step; lower is better and curves are directly comparable only within the same step budget. `delta_loss_architecture.png` is the signed Symmetric-minus-LoRA endpoint difference; values below zero favor Symmetric. `loss_vs_insertion_points.png` compares one, two and four insertion points, but the layer sets differ by strategy, so it is a distribution screen rather than a pure N-only causal plot. `parameter_efficiency.png` confirms that all valid points have the same 6,144-parameter target; differences cannot be attributed to a larger trainable budget.

## 5. Observations, interpretations and limitations

**Observation.** The concentrated, two-layer and four-layer pairs all satisfy the fixed total parameter budget and use the same seed and training protocol.

**Interpretation.** This is a clean test of capacity placement within the tested configurations. It does not compare arbitrary rank choices or larger budgets.

**Observation.** The data-driven two-layer LoRA attempt entered an uninterruptible `D` state before producing a valid result. Its directory is retained as an error artifact, but it is excluded from every table and mean.

**Interpretation.** The GPU runtime remains an operational limitation. The failure does not imply an architectural failure because no completed training result exists for that run.

**Limitation.** Early, middle and late regional configurations were not reached after the lockup. Three-seed confirmation and 2,000-step fixed-budget runs have not yet been executed for this multi-layer campaign. Therefore no claim is made about the best distribution strategy or about a stable LoRA-versus-Symmetric interaction.

## 6. Reproducibility and file map

- `results_multilayer/screen/`: separate configurations, metrics and summaries.
- `screen_results.csv`: valid aggregate rows.
- `figures/validation_curves.png`
- `figures/delta_loss_architecture.png`
- `figures/loss_vs_insertion_points.png`
- `figures/parameter_efficiency.png`
- `scripts/smoke_multilayer.py`
- `scripts/run_multilayer_screen.py`

The earlier single-layer results remain in `results/` and `results_depth/` and were not overwritten.

## 7. Next step

Because the GPU lockup occurred before the requested data-driven and regional comparisons, the scientifically useful next step is to recover the GPU runtime and rerun only the missing phase-1 configurations, preferably in separate processes with a health check between runs. Only after all phase-1 strategies are available should the most informative configurations be promoted to 2,000 steps and three seeds.
'''
(OUT/'MULTILAYER_FIXED_BUDGET.md').write_text(md); print('wrote',OUT/'MULTILAYER_FIXED_BUDGET.md')
