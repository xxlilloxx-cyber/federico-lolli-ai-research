"""Generate audit and publication figures from existing local result summaries only."""
from pathlib import Path
import csv,json,math,statistics
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]; DOC=ROOT/'docs'; FIG=DOC/'assets'/'publication'; FIG.mkdir(parents=True,exist_ok=True)
roots=['results','results_extended','results_depth','results_local_rank','results_long_convergence_5000','results_multilayer']
def val(d,*keys):
 for k in keys:
  if k in d:return d[k]
 return 'NOT AVAILABLE'
rows=[]
for name in roots:
 for p in (ROOT/name).rglob('summary.json') if (ROOT/name).exists() else []:
  try:d=json.loads(p.read_text())
  except:continue
  if not isinstance(d, dict): continue
  rows.append(dict(experiment_family=name,model=val(d,'model','model_name'),dataset=val(d,'dataset'),adapter=val(d,'kind','architecture'),formulation='NOT AVAILABLE',rank=val(d,'rank'),placement=','.join(d.get('projections',[])) or 'NOT AVAILABLE',layers=','.join(map(str,d.get('layers',[]))) or 'NOT AVAILABLE',adapted_layers=len(d.get('layers',[])) if d.get('layers') is not None else 'NOT AVAILABLE',trainable_parameters=val(d,'trainable_parameters'),steps=val(d,'steps'),seed=val(d,'seed'),best_validation_loss=val(d,'validation_loss'),final_validation_loss=val(d,'final_validation_loss'),test_loss='NOT AVAILABLE',parameter_budget=val(d,'target_budget','budget'),source_result_file=str(p.relative_to(ROOT))))
# final test data are public and map exact final 5k rows
for r in rows:
 if 'results_long_convergence_5000' in r['source_result_file']:
  r['formulation']={'lora':'(alpha/r)(xU)V','symmetric_quadratic':'(alpha/r)(xU)^2P'}.get(r['adapter'],'NOT AVAILABLE')
with (DOC/'data'/'experiment_audit.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
# concise markdown audit by every summary, CSV holds rows
groups={k:[x for x in rows if x['experiment_family']==k] for k in roots}
lines=['# Experiment audit','','This audit was generated from existing `summary.json` files without rerunning training. The complete per-run table is [`data/experiment_audit.csv`](data/experiment_audit.csv). `NOT AVAILABLE` means the original summary did not record that field; no value was inferred. Final 5,000-step held-out values remain in `data/long_final_per_seed.csv`.','','| Experiment family | Runs | Adapters observed | Step values observed |','|---|---:|---|---|']
for k,g in sorted(groups.items()): lines.append('| %s | %s | %s | %s |' % (k,len(g),', '.join(sorted({str(x["adapter"]) for x in g})),', '.join(sorted({str(x["steps"]) for x in g}))))
lines += ['', 'The source-result-file field is a local traceability reference; raw artifacts are deliberately not part of the public Pages payload.']
(DOC/'EXPERIMENT_AUDIT.md').write_text('\n'.join(lines)+'\n')
# architecture original
fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
def box(x,y,t,c):ax.add_patch(FancyBboxPatch((x,y),1.5,.55,boxstyle='round,pad=.04',fc=c,ec='#183153',lw=1.5));ax.text(x+.75,y+.275,t,ha='center',va='center',fontsize=11)
for x,t in [(0,'x'),(1.9,'A / U'),(3.8,'z'),(5.7,'z⊙z'),(7.6,'B / P'),(9.5,'Δy + frozen W₀x')]:box(x,.45,t,'#e9f3ff' if x in [1.9,3.8,5.7,7.6] else '#f4f6f8')
for x in [1.5,3.4,5.3,7.2,9.1]:ax.annotate('',xy=(x+.35,.725),xytext=(x,.725),arrowprops={'arrowstyle':'->'})
ax.text(2.65,1.45,'LoRA: x → A → B → Δy',ha='center',weight='bold');ax.text(6.5,1.45,'Symmetric: x → U → z → z⊙z → P → Δy',ha='center',weight='bold');ax.text(6,.05,'Frozen projection W₀ and bias remain frozen; blue stages are trainable adapter computation.',ha='center')
for ext in ['png','svg','pdf']:fig.savefig(FIG/f'adapter_architecture_comparison.{ext}',dpi=220,bbox_inches='tight')
plt.close(fig)
# Rank/placement from master historical summaries only
master=list(csv.DictReader((ROOT/'report'/'MASTER_RESULTS.csv').open()))
valid=[r for r in master if r['valid']=='True' and r['architecture'] in ('lora','symmetric_quadratic')]
g=defaultdict(list)
for r in valid:
 if r['local_rank'] and r['local_rank']!='':g[(r['architecture'],r['local_rank'])].append(float(r['final_val_loss']))
fig,ax=plt.subplots(figsize=(7,4))
for a,c in [('lora','#1769aa'),('symmetric_quadratic','#d96b27')]:
 pts=sorted((int(k[1]),statistics.mean(v),statistics.stdev(v) if len(v)>1 else 0) for k,v in g.items() if k[0]==a)
 if pts:ax.errorbar([p[0] for p in pts],[p[1] for p in pts],yerr=[p[2] for p in pts],fmt='o-',label=a.replace('_',' ').title(),color=c)
ax.set(xlabel='Local rank (mixed historical protocols)',ylabel='Recorded validation-loss summary',title='Observed rank summaries (not a controlled factorial comparison)');ax.legend();fig.tight_layout()
for ext in ['png','svg','pdf']:fig.savefig(FIG/f'performance_vs_rank.{ext}',dpi=220)
plt.close(fig)
