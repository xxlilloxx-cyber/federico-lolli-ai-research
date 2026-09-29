import csv,json,math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
root=Path('results_rank_factorial_controlled/confirmation_5000');out=Path('analysis_rank_confirmation_5000_2026-09-29');fig=out/'figures';fig.mkdir(parents=True,exist_ok=True)
rows=[]
for p in sorted(root.glob('*/summary.json')):
 s=json.loads(p.read_text());rows.append({'method':'LoRA' if s['kind']=='lora' else 'Symmetric Quadratic','rank':int(s['rank']),'seed':int(s['seed']),'adapter_parameters':1536*int(s['rank']),'alpha_over_rank':4/int(s['rank']),'best_recorded_validation_loss':float(s['best_recorded_validation_loss']),'final_validation_loss':float(s['final_validation_loss']),'final_validation_perplexity':float(s['final_validation_perplexity']),'final_test_loss':float(s['final_test_loss']),'final_test_perplexity':float(s['final_test_perplexity']),'result_directory':str(p.parent)})
assert len(rows)==24
def write(n,x):
 with (out/n).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=x[0]);w.writeheader();w.writerows(x)
write('rank_5000_per_seed.csv',rows);agg=[]; paired=[]
for rank in (1,2,4,8):
 for method in ('LoRA','Symmetric Quadratic'):
  z=[r for r in rows if r['rank']==rank and r['method']==method];d={'method':method,'rank':rank,'n':3,'adapter_parameters':z[0]['adapter_parameters'],'alpha_over_rank':z[0]['alpha_over_rank']}
  for k in ('best_recorded_validation_loss','final_validation_loss','final_validation_perplexity','final_test_loss','final_test_perplexity'):
   v=np.array([r[k] for r in z]);d[k+'_mean']=v.mean();d[k+'_sample_sd']=v.std(ddof=1)
  agg.append(d)
 for seed in (42,123,456):
  l=next(r for r in rows if r['rank']==rank and r['seed']==seed and r['method']=='LoRA');q=next(r for r in rows if r['rank']==rank and r['seed']==seed and r['method']=='Symmetric Quadratic');paired.append({'rank':rank,'seed':seed,'symmetric_minus_lora_final_validation_loss':q['final_validation_loss']-l['final_validation_loss'],'symmetric_minus_lora_final_test_loss':q['final_test_loss']-l['final_test_loss']})
write('rank_5000_aggregate.csv',agg);write('rank_5000_paired_differences.csv',paired)
plt.style.use('seaborn-v0_8-whitegrid')
def plot(metric,label,name):
 f,a=plt.subplots(figsize=(7,4.5))
 for method,color in [('LoRA','#2563eb'),('Symmetric Quadratic','#c2410c')]:
  yy=[];ee=[]
  for rank in (1,2,4,8):
   z=np.array([r[metric] for r in rows if r['rank']==rank and r['method']==method]);yy.append(z.mean());ee.append(z.std(ddof=1));a.scatter([rank]*3,z,color=color,alpha=.6)
  a.errorbar([1,2,4,8],yy,yerr=ee,marker='o',capsize=4,color=color,label=method+' mean ± sample SD')
 a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Rank');a.set_ylabel(label);a.set_title('5,000-step controlled confirmation');a.legend();f.tight_layout()
 for e in ('svg','png','pdf'):f.savefig(fig/(name+'.'+e),dpi=220)
plot('final_validation_loss','Final validation loss','rank_5000_final_validation_loss');plot('final_test_loss','Final held-out test loss','rank_5000_final_test_loss');plot('final_validation_perplexity','Final validation perplexity','rank_5000_validation_perplexity')
f,a=plt.subplots(figsize=(7,4.5))
for rank in (1,2,4,8):
 z=np.array([d['symmetric_minus_lora_final_validation_loss'] for d in paired if d['rank']==rank]);a.scatter([rank]*3,z,color='#7c3aed');a.errorbar(rank,z.mean(),yerr=z.std(ddof=1),fmt='o',color='#7c3aed',capsize=5)
a.axhline(0,color='black');a.set_xscale('log',base=2);a.set_xticks([1,2,4,8],['1','2','4','8']);a.set_xlabel('Rank');a.set_ylabel('Symmetric − LoRA final validation loss');a.set_title('Paired final-validation difference');f.tight_layout()
for e in ('svg','png','pdf'):f.savefig(fig/('rank_5000_paired_difference.'+e),dpi=220)
lines=['# Controlled Rank 5,000-Step Confirmation Report','','## Scope','','24 fresh WikiText-2 runs: LoRA and Symmetric Quadratic, ranks 1/2/4/8, seeds 42/123/456, each trained for 5,000 steps. This is independent of historical work and the 1,000-step screening.','','## Final validation and held-out test','','| Rank | LoRA final val | Symmetric final val | paired S−L val | LoRA test | Symmetric test | paired S−L test |']
for rank in (1,2,4,8):
 l=next(x for x in agg if x['rank']==rank and x['method']=='LoRA');q=next(x for x in agg if x['rank']==rank and x['method']=='Symmetric Quadratic');pv=np.array([x['symmetric_minus_lora_final_validation_loss'] for x in paired if x['rank']==rank]);pt=np.array([x['symmetric_minus_lora_final_test_loss'] for x in paired if x['rank']==rank]);lines.append(f"| {rank} | {l['final_validation_loss_mean']:.4f} ± {l['final_validation_loss_sample_sd']:.4f} | {q['final_validation_loss_mean']:.4f} ± {q['final_validation_loss_sample_sd']:.4f} | {pv.mean():+.4f} ± {pv.std(ddof=1):.4f} | {l['final_test_loss_mean']:.4f} ± {l['final_test_loss_sample_sd']:.4f} | {q['final_test_loss_mean']:.4f} ± {q['final_test_loss_sample_sd']:.4f} | {pt.mean():+.4f} ± {pt.std(ddof=1):.4f} |")
lines+=['','Final validation is measured at step 5,000. Test loss uses that final checkpoint. PPL is computed per seed and then averaged. Negative paired differences favour Symmetric. Three seeds show variability but do not establish broad significance. Rank also changes alpha/r (4, 2, 1, 0.5), so this does not isolate pure rank.','','## Remaining work','','The separate constant-scale ablation, spectral analysis and interaction analysis are not yet executed.']
(out/'CONTROLLED_RANK_5000_REPORT.md').write_text('\n'.join(lines)+'\n')
print(out)
