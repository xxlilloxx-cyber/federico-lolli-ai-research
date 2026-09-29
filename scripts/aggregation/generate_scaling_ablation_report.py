import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
out=Path('experiments/scaling-ablation');data=Path('results/tables/scaling-ablation');fig=Path('results/figures/scaling-ablation');out.mkdir(parents=True,exist_ok=True);data.mkdir(parents=True,exist_ok=True);fig.mkdir(parents=True,exist_ok=True)
# Policy A is the already-completed screening; policy B is the new ablation.
a=[]
for p in Path('results_rank_factorial_controlled/screening').glob('*_seed42_steps1000/summary.json'):
 s=json.loads(p.read_text());a.append({'policy':'current_alpha_4','method':'LoRA' if s['kind']=='lora' else 'Symmetric Quadratic','rank':int(s['rank']),'alpha':4.0,'alpha_over_rank':4/int(s['rank']),'seed':42,'final_validation_loss':float(s['validation_loss']),'final_validation_ppl':math.exp(float(s['validation_loss']))})
b=[]
for p in Path('results_rank_scaling_ablation').glob('*/summary.json'):
 s=json.loads(p.read_text());b.append({'policy':'constant_effective_scale','method':'LoRA' if s['kind']=='lora' else 'Symmetric Quadratic','rank':int(s['rank']),'alpha':float(s['alpha']),'alpha_over_rank':float(s['alpha'])/int(s['rank']),'seed':42,'final_validation_loss':float(s['validation_loss']),'final_validation_ppl':math.exp(float(s['validation_loss']))})
assert len(a)==8 and len(b)==8
rows=a+b
with (data/'rank_scaling_ablation_per_seed.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
agg=[]
for policy in ('current_alpha_4','constant_effective_scale'):
 for rank in (1,2,4,8):
  for method in ('LoRA','Symmetric Quadratic'):
   x=next(r for r in rows if r['policy']==policy and r['rank']==rank and r['method']==method);agg.append(x)
with (data/'rank_scaling_ablation_aggregate.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=agg[0]);w.writeheader();w.writerows(agg)
plt.style.use('seaborn-v0_8-whitegrid');f,a1=plt.subplots(figsize=(8,5))
for policy,style in [('current_alpha_4','-'),('constant_effective_scale','--')]:
 for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
  x=[next(r['final_validation_loss'] for r in rows if r['policy']==policy and r['method']==method and r['rank']==q) for q in (1,2,4,8)]
  a1.plot([1,2,4,8],x,style,marker='o',color=color,label=f'{method}, '+('alpha=4' if policy=='current_alpha_4' else 'alpha/r=1'))
a1.set_xscale('log',base=2);a1.set_xticks([1,2,4,8],['1','2','4','8']);a1.set_xlabel('Rank');a1.set_ylabel('Final validation loss');a1.set_title('Rank-scaling ablation, seed 42, 1,000 steps');a1.legend(fontsize=8);f.tight_layout()
for e in ('svg','png','pdf'):f.savefig(fig/('rank_scaling_ablation_loss.'+e),dpi=240)
plt.close(f)
lines=['# Rank-Scaling Ablation Report','','## Question','','The controlled rank studies use a branch multiplier `alpha/r`. The original screening fixed alpha=4, so effective scale changed from 4 at rank 1 to 0.5 at rank 8. This ablation tests a separate Policy B with `alpha/r=1` at every rank.','','## Protocol','','| Policy | r=1 | r=2 | r=4 | r=8 |','|---|---:|---:|---:|---:|','| A: current | alpha=4; scale=4 | alpha=4; scale=2 | alpha=4; scale=1 | alpha=4; scale=0.5 |','| B: constant scale | alpha=1; scale=1 | alpha=2; scale=1 | alpha=4; scale=1 | alpha=8; scale=1 |','','Both policies use GPT-2, frozen backbone, one block-0 `attn.c_proj` adapter, WikiText-2, sequence length 128, gradient accumulation 4, AdamW 3e-4, seed 42, and 1,000 fresh steps. Policy A values are read from the valid screening runs; only Policy B was newly trained.','','## Results','','| Rank | LoRA A | Symmetric A | S−L A | LoRA B | Symmetric B | S−L B |']
for rank in (1,2,4,8):
 v=[]
 for policy in ('current_alpha_4','constant_effective_scale'):
  l=next(r['final_validation_loss'] for r in rows if r['policy']==policy and r['method']=='LoRA' and r['rank']==rank);q=next(r['final_validation_loss'] for r in rows if r['policy']==policy and r['method']=='Symmetric Quadratic' and r['rank']==rank);v += [l,q,q-l]
 lines.append('| %d | %.4f | %.4f | %+.4f | %.4f | %.4f | %+.4f |'%(rank,*v))
lines += ['','## Interpretation','','**Measured result.** Under Policy B, inspect the `S−L B` column. This is one seed and one training duration, so it cannot establish a robust causal explanation.','','**Does Symmetric still improve with rank at constant scale?** The answer is descriptive only: compare Policy B ranks in the table and figure. **Does the gap remain?** Compare the signs of the two paired columns.','','**Scaling confound.** If Policy B differs materially from Policy A, alpha/r is an important contributor to the observed trend. If a pattern remains, that is compatible with a capacity/rank contribution but does not isolate it from optimization. Seeds 123 and 456 would be justified only if the seed-42 result is informative; they were intentionally not launched automatically.','','## Files','','This directory contains the per-seed and aggregate CSVs and reproducible SVG/PNG/PDF plot.']
(out/'RANK_SCALING_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
print(out)
