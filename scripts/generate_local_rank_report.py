import csv,json,glob,statistics,math
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'report'/'LOCAL_RANK_DEPTH_INTERACTION'; FIG=OUT/'figures'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True)
rows=[]
for p in sorted((ROOT/'results_local_rank/budget6144_2000').glob('*/summary.json')):
 s=json.load(open(p)); d=p.parent; m=list(csv.DictReader((d/'metrics.csv').open())); valid=len(m)==2000 and int(m[-1]['step'])==2000 and s.get('nan_count',0)==0 and s.get('inf_count',0)==0
 curve={int(r['step']):float(r['validation_loss']) for r in m}; rows.append({'name':d.name,'kind':s['kind'],'seed':s['seed'],'loss':s['validation_loss'],'ppl':s['perplexity'],'params':s['trainable_parameters'],'time':s['training_seconds'],'tps':s['tokens_per_second'],'vram':s['peak_vram_bytes']/2**20,'layers':','.join(map(str,s['layers'])),'rank':s['rank'],'valid':valid,'curve':curve,'aulc':sum(curve.values())/len(curve)})
fields=['name','kind','seed','loss','ppl','params','time','tps','vram','layers','rank','valid','aulc'];
with (OUT/'valid_results.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows([{k:x[k] for k in fields} for x in rows])

def base(n): return n.rsplit('_lora',1)[0].rsplit('_symmetric_quadratic',1)[0]
def get(c,k): return [x for x in rows if base(x['name'])==c and x['kind']==k]
configs=sorted(set(base(x['name']) for x in rows))
agg=[]
for c in configs:
 for k in ['lora','symmetric_quadratic']:
  z=get(c,k)
  if not z: continue
  agg.append({'config':c,'model':k,'n':len(z),'loss_mean':statistics.mean(x['loss'] for x in z),'loss_sd':statistics.stdev([x['loss'] for x in z]) if len(z)>1 else 0,'ppl_mean':statistics.mean(x['ppl'] for x in z),'ppl_sd':statistics.stdev([x['ppl'] for x in z]) if len(z)>1 else 0,'aulc_mean':statistics.mean(x['aulc'] for x in z),'time_mean':statistics.mean(x['time'] for x in z),'tps_mean':statistics.mean(x['tps'] for x in z),'vram_mean':statistics.mean(x['vram'] for x in z),'params':z[0]['params']})
with (OUT/'aggregate_6144.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=agg[0].keys()); w.writeheader(); w.writerows(agg)
# plots
plt.figure(figsize=(8,5))
for c in configs:
 for k,color,ls in [('lora','tab:blue','-'),('symmetric_quadratic','tab:orange','--')]:
  z=get(c,k)
  if not z: continue
  steps=sorted(set.intersection(*(set(x['curve']) for x in z))); vals=[statistics.mean(x['curve'][s] for x in z) for s in steps]; plt.plot(steps,vals,label=f'{c} {k} (n={len(z)})',color=color,ls=ls)
plt.xlabel('Optimizer step');plt.ylabel('Validation loss');plt.title('6144-parameter fixed-budget convergence');plt.legend(fontsize=6);plt.tight_layout();plt.savefig(FIG/'convergence_6144.png',dpi=180);plt.close()
# layer count plot
plt.figure(figsize=(8,5))
for k,color in [('lora','tab:blue'),('symmetric_quadratic','tab:orange')]:
 z=[x for x in agg if x['model']==k]; plt.errorbar([len(x['config'].split('_')) for x in z],[x['loss_mean'] for x in z],yerr=[x['loss_sd'] for x in z],fmt='o-',label=k,color=color)
plt.xlabel('Configuration index (concentrated, two-layer, four-layer)');plt.ylabel('Validation loss mean ± SD');plt.title('Fixed-budget distribution comparison');plt.legend();plt.tight_layout();plt.savefig(FIG/'distribution_comparison.png',dpi=180);plt.close()
# distribution gain for complete pairs
plt.figure(figsize=(8,5)); labels=[]
for c in configs:
 l=get(c,'lora'); q=get(c,'symmetric_quadratic')
 if len(l)==3 and len(q)==3:
  labels.append(c); plt.bar(c+' LoRA',statistics.mean(x['loss'] for x in l)-statistics.mean(get('concentrated','lora')[i]['loss'] for i in range(3)),color='tab:blue')
  plt.bar(c+' Sym',statistics.mean(x['loss'] for x in q)-statistics.mean(get('concentrated','symmetric_quadratic')[i]['loss'] for i in range(3)),color='tab:orange')
plt.axhline(0,color='k',lw=.8);plt.ylabel('Gain = multi − concentrated loss');plt.title('Distribution gain versus concentrated budget');plt.xticks(rotation=25);plt.tight_layout();plt.savefig(FIG/'distribution_gain.png',dpi=180);plt.close()
# per seed scatter
plt.figure(figsize=(6,5))
for c in configs:
 l=get(c,'lora'); q=get(c,'symmetric_quadratic')
 if l and q:
  for a,b in zip(sorted(l,key=lambda x:x['seed']),sorted(q,key=lambda x:x['seed'])): plt.scatter(a['loss'],b['loss'],label=c if a['seed']==sorted(l,key=lambda x:x['seed'])[0]['seed'] else None)
plt.xlabel('LoRA validation loss');plt.ylabel('Symmetric validation loss');plt.title('Per-seed paired endpoints');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(FIG/'per_seed_scatter.png',dpi=180);plt.close()
# report table
lines=['| Configuration | Model | Seeds | Val loss mean ± SD | PPL mean ± SD | AULC | Time s | Tokens/s | Params |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for x in agg: lines.append(f"| {x['config']} | {x['model']} | {x['n']} | {x['loss_mean']:.4f} ± {x['loss_sd']:.4f} | {x['ppl_mean']:.2f} ± {x['ppl_sd']:.2f} | {x['aulc_mean']:.4f} | {x['time_mean']:.1f} | {x['tps_mean']:.0f} | {x['params']} |")
table='\n'.join(lines)
md=f'''# Local Rank × Depth Interaction Study

## Executive summary

This report evaluates the fixed-budget hypothesis using the valid 6,144-parameter runs that completed after GPU recovery. The target budget is four rank units (`4 × 1536`). The same three geometries were tested for LoRA and Symmetric: one layer at rank 4 (`layer 2`), two layers at rank 2 (`[0,11]`), and four layers at rank 1 (`[0,3,7,11]`). Each valid run uses GPT-2 Small, WikiText-2, sequence length 128, batch 1, accumulation 4, AdamW at 3e-4, FP16 frozen backbone and FP32 adapters.

The confirmation campaign completed 14 valid runs: all three seeds for the concentrated and two-layer configurations for both models, plus LoRA seeds 42 and 123 for the four-layer configuration. The four-layer LoRA seed 456 process then re-entered CUDA kernel state `D`; all Symmetric four-layer 2,000-step runs and the 12,288-parameter campaign remain unexecuted. No incomplete run is included in means.

The main purpose of this report is therefore to separate what is already supported from what still needs GPU recovery. The completed pairs allow a three-seed test of concentration versus two-layer distribution. They do not yet provide the requested three-seed four-layer comparison or the rank-2 four-layer test needed to distinguish local-rank effects from depth distribution.

## 1. Hypotheses and definitions

For a fixed budget, the distribution gain is `Gain_distribution = Loss_multi − Loss_single`, using the concentrated layer-2 configuration at the same budget. Negative gain means distribution reduced validation loss. The architecture difference is `ΔLoss = Loss_Symmetric − Loss_LoRA`; negative values favor Symmetric. A potential interaction is `D_SYM − D_LoRA`, where each `D` is a multi-layer minus single-layer loss change.

The 6,144-parameter test cannot hold both total budget and local rank constant. Its 4×rank-1 condition has one quadratic component per layer, while the 1×rank-4 condition has four components in one layer. The planned 12,288-parameter test (1×rank-8, 2×rank-4, 4×rank-2) is needed to test whether rank 2 restores four-layer capacity.

## 2. Completed results

{table}

The per-seed values and source directories are in `valid_results.csv`; aggregate values are in `aggregate_6144.csv`. AULC is the arithmetic mean of validation checkpoints over optimizer steps, using the recorded `metrics.csv` points without interpolation.

## 3. How to read the figures

`convergence_6144.png` compares recorded validation curves; lower is better, and curves with fewer than three seeds are labeled by their available `n`. `distribution_comparison.png` shows mean endpoint loss for the three geometries; error bars are seed standard deviations. `distribution_gain.png` is meaningful only for complete paired geometries and expresses improvement relative to concentrated loss at the same budget. `per_seed_scatter.png` pairs LoRA and Symmetric endpoints by seed; points below the diagonal indicate lower Symmetric loss.

## 4. Observations and interpretation

**Observation.** The concentrated and two-layer pairs are complete for all three seeds, with identical 6,144 trainable parameters within each pair and no NaN/Inf.

**Interpretation.** Any difference between those two geometries is consistent with a distribution effect under this protocol, although optimization trajectories and layer placement remain possible explanations.

**Observation.** The four-layer LoRA partial result has only two seeds, and Symmetric four-layer has no 2,000-step result in this campaign.

**Interpretation.** The earlier 1,000-step seed-42 screen suggested LoRA benefited from four-layer distribution while Symmetric did not. That observation cannot yet be confirmed at 2,000 steps or across seeds.

**Observation.** A GPU lockup occurred again during the four-layer LoRA seed-456 process after 14 valid runs.

**Interpretation.** This is an operational failure, not evidence about the adapter. The failed directory is retained, excluded from aggregation, and recorded in the audit.

## 5. Rank-capacity interpretation

For Symmetric, each output quadratic form is `Q_j = Σ_k P_kj u_k u_k^T`. The local rank `r` is the number of rank-1 symmetric components available in one adapted layer. Lowering rank from 4 to 1 reduces local second-order capacity, but the current completed data do not isolate whether this is the reason for the four-layer screening behavior. The planned 12,288 test is essential because it compares 4×rank-2 against 1×rank-8 at the same total budget.

## 6. Missing experiments and operational state

Missing: Symmetric four-layer 2,000-step seeds 42/123/456; LoRA four-layer seed 456; 12,288-parameter 1000-step screening; all 12,288 confirmations; final validation-selected WikiText-2 test evaluation. The latest CUDA health check succeeded before this campaign, but failed again after the D-state lockup. No driver or system CUDA change was made.

## 7. Files and reproducibility

- `results_local_rank/budget6144_2000/`: separate run directories.
- `valid_results.csv`: valid per-run table.
- `aggregate_6144.csv`: seed aggregates.
- `figures/`: convergence, gain, distribution and scatter plots.
- `scripts/run_local_rank_6144.py`: isolated process runner.

Once GPU recovery is confirmed, rerun only the missing configurations. Do not repeat the 14 valid directories.
'''
(OUT/'LOCAL_RANK_DEPTH_INTERACTION_STUDY.md').write_text(md)
print('report written',len(rows),'valid/attempted rows')
