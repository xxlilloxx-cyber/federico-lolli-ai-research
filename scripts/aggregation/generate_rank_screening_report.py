#!/usr/bin/env python3
"""Aggregate the completed 24-cell controlled rank screening without changing raw runs."""
import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

root=Path('results_rank_factorial_controlled/screening')
out=Path('experiments/rank-screening-1000'); data=Path('results/tables/rank-screening-1000'); fig=Path('results/figures/rank-screening-1000');out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(root.glob('*/summary.json')):
 s=json.loads(p.read_text())
 rows.append({'method':'LoRA' if s['kind']=='lora' else 'Symmetric Quadratic','architecture_key':s['kind'],'rank':s['rank'],'seed':s['seed'],'steps':s['steps'],'trainable_parameters':s['trainable_parameters'],'final_validation_loss':s['validation_loss'],'final_validation_ppl':math.exp(s['validation_loss']),'best_recorded_validation_loss':min(float(x['validation_loss']) for x in csv.DictReader(open(p.parent/'metrics.csv'))),'training_seconds':s['training_seconds'],'tokens_per_second':s['tokens_per_second'],'peak_vram_mb':s['peak_vram_bytes']/2**20,'result_directory':str(p.parent)})
assert len(rows)==24 and {(r['architecture_key'],r['rank'],r['seed']) for r in rows}=={(m,k,s) for m in ('lora','symmetric_quadratic') for k in (1,2,4,8) for s in (42,123,456)}
fields=list(rows[0]);
with (data/'rank_factorial_screening_per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
agg=[]
for method in ('LoRA','Symmetric Quadratic'):
 for rank in (1,2,4,8):
  x=[r for r in rows if r['method']==method and r['rank']==rank];d={'method':method,'rank':rank,'n':len(x),'trainable_parameters':x[0]['trainable_parameters']}
  for key in ('final_validation_loss','final_validation_ppl','best_recorded_validation_loss','training_seconds','tokens_per_second','peak_vram_mb'):
   v=np.array([r[key] for r in x]);d[key+'_mean']=v.mean();d[key+'_sample_sd']=v.std(ddof=1);d[key+'_min']=v.min();d[key+'_max']=v.max()
  agg.append(d)
with (data/'rank_factorial_screening_aggregate.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(agg[0]));w.writeheader();w.writerows(agg)
paired=[]
for rank in (1,2,4,8):
 for seed in (42,123,456):
  l=next(r for r in rows if r['method']=='LoRA' and r['rank']==rank and r['seed']==seed);q=next(r for r in rows if r['method']=='Symmetric Quadratic' and r['rank']==rank and r['seed']==seed)
  paired.append({'rank':rank,'seed':seed,'symmetric_minus_lora_final_validation_loss':q['final_validation_loss']-l['final_validation_loss'],'symmetric_minus_lora_final_validation_ppl':q['final_validation_ppl']-l['final_validation_ppl']})
with (data/'rank_factorial_screening_paired_differences.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(paired[0]));w.writeheader();w.writerows(paired)
plt.style.use('seaborn-v0_8-whitegrid')
for metric,label,stem in [('final_validation_loss','Final validation loss','final_validation_loss_by_rank'),('final_validation_ppl','Final validation perplexity','final_validation_ppl_by_rank')]:
 f,a=plt.subplots(figsize=(7,4.5))
 for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
  x=np.array([1,2,4,8]); y=[];e=[]
  for rank in x:
   z=[r[metric] for r in rows if r['method']==method and r['rank']==rank];y.append(np.mean(z));e.append(np.std(z,ddof=1))
   a.scatter([rank]*3,z,color=color,alpha=.55,s=30,zorder=3)
  a.errorbar(x,y,yerr=e,marker='o',capsize=4,color=color,label=f'{method}: mean ± sample SD')
 a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Local adapter rank');a.set_ylabel(label);a.set_title('Controlled WikiText-2 rank screening: 1,000 steps, three seeds');a.legend(fontsize=8);f.tight_layout()
 for ext in ('svg','png','pdf'):f.savefig(fig/f'{stem}.{ext}',dpi=240)
 plt.close(f)
f,a=plt.subplots(figsize=(7,4.5))
for rank in (1,2,4,8):
 z=[r['symmetric_minus_lora_final_validation_loss'] for r in paired if r['rank']==rank];a.scatter([rank]*3,z,color='#7c3aed',alpha=.65);a.errorbar(rank,np.mean(z),yerr=np.std(z,ddof=1),fmt='o',color='#7c3aed',capsize=5)
a.axhline(0,color='black',lw=1);a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Local adapter rank');a.set_ylabel('Symmetric − LoRA final validation loss');a.set_title('Paired differences by rank; negative favours Symmetric');f.tight_layout()
for ext in ('svg','png','pdf'):f.savefig(fig/f'paired_loss_difference_by_rank.{ext}',dpi=240)
plt.close(f)
lines=['# Controlled Rank-Factorial Screening Report','','**Scope.** Fresh WikiText-2 causal-language-modeling screening: two methods × ranks 1/2/4/8 × seeds 42/123/456, with 1,000 optimizer steps per run. This is a controlled screening study; it is not the historical 5,000-step campaign and it does not provide held-out test results.','','## Protocol','','All runs use frozen GPT-2 (`gpt2`), adapters at `transformer.h[0].attn.c_proj` in block 0, sequence length 128, gradient accumulation 4, AdamW learning rate 3e-4, alpha 4, and fresh deterministic seeds. LoRA and Symmetric use the same rank within each paired comparison. Since a 768→768 adapter has 1,536 trainable parameters per rank, their adapter budgets match at each rank.','','## Final validation results','','| Rank | LoRA loss (mean ± SD) | Symmetric loss (mean ± SD) | Symmetric − LoRA | LoRA PPL (mean ± SD) | Symmetric PPL (mean ± SD) |']
for rank in (1,2,4,8):
 l=next(x for x in agg if x['method']=='LoRA' and x['rank']==rank);q=next(x for x in agg if x['method']=='Symmetric Quadratic' and x['rank']==rank);d=np.array([x['symmetric_minus_lora_final_validation_loss'] for x in paired if x['rank']==rank]);lines.append(f"| {rank} | {l['final_validation_loss_mean']:.4f} ± {l['final_validation_loss_sample_sd']:.4f} | {q['final_validation_loss_mean']:.4f} ± {q['final_validation_loss_sample_sd']:.4f} | {d.mean():+.4f} ± {d.std(ddof=1):.4f} | {l['final_validation_ppl_mean']:.2f} ± {l['final_validation_ppl_sample_sd']:.2f} | {q['final_validation_ppl_mean']:.2f} ± {q['final_validation_ppl_sample_sd']:.2f} |")
lines += ['','Perplexity is calculated separately as `exp(final token-level cross-entropy loss)` for each seed, then averaged. It is valid here because the stored language-modeling loss is token-level cross entropy.','','## Observations','','- The paired mean loss difference is positive at rank 1 if Symmetric is worse, and negative if Symmetric is lower. Interpret the table rather than a single seed: all three seeds are retained at every rank.','- Rank changes both the number of trainable parameters and the effective scale `alpha/r`; this study does not isolate those factors.','- The result is limited to 1,000 training steps and a small validation sample used by the existing pipeline. It does not establish a 5,000-step or held-out-test rank effect.','','## Files','','- `rank_factorial_screening_per_seed.csv`: one row per valid run.','- `rank_factorial_screening_aggregate.csv`: mean, sample SD, min and max.','- `rank_factorial_screening_paired_differences.csv`: within-seed Symmetric − LoRA comparisons.','- `figures/`: PNG, SVG and PDF figures; dots are seeds and error bars are sample SD.','','## Next controlled step','','Run fresh 5,000-step confirmations for ranks 1/2/4/8 and seeds 42/123/456 before claiming a persistent rank effect. The screening data should not be merged with the historical 5,000-step results.']
(out/'RANK_FACTORIAL_SCREENING_REPORT.md').write_text('\n'.join(lines)+'\n')
print(out)
