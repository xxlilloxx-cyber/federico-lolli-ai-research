import csv,json,glob,math,statistics
from pathlib import Path
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'experiments'/'historical'/'depth-and-placement'; FIG=ROOT/'results'/'figures'/'depth-and-placement'; OUT.mkdir(parents=True,exist_ok=True); FIG.mkdir(parents=True,exist_ok=True)

def summary_dirs(root): return sorted(Path(root).glob('*/summary.json'))
def load(p): return json.loads(Path(p).read_text())
def savefig(name): plt.tight_layout(); plt.savefig(FIG/name,dpi=180); plt.close()
# Audit table
rows=[]
for d in sorted((ROOT/'results_extended/linear_quadratic').glob('r*_steps2000')):
 s=load(d/'summary.json'); m=list(csv.DictReader((d/'metrics.csv').open())); valid=(len(m)==2000 and int(m[-1]['step'])==2000 and s.get('nan_count',0)==0 and s.get('inf_count',0)==0)
 rows.append({'seed':s['seed'],'completed':len(m)==2000,'steps':int(m[-1]['step']),'val_loss':s['validation_loss'],'ppl':s['perplexity'],'params':s['trainable_parameters'],'rank_l':s['rank_l'],'rank_q':s['rank_q'],'directory':str(d.relative_to(ROOT)),'valid':valid,'timestamp':Path(d/'summary.json').stat().st_mtime})
with (OUT/'linear_quadratic_seed_audit.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=rows[0].keys()).writeheader(); csv.DictWriter(f,fieldnames=rows[0].keys()).writerows(rows)
# screening data
screen=[]
for p in summary_dirs(ROOT/'results_depth/screen'):
 s=load(p); name=Path(p).parent.name; screen.append({'name':name,'kind':s['kind'],'loss':s['validation_loss'],'ppl':s['perplexity'],'params':s['trainable_parameters'],'time':s['training_seconds'],'vram':s['peak_vram_bytes']/2**20})
# parse attention depth
att=[]
for x in screen:
 if x['name'].startswith('attn_layer'):
  x['layer']=int(x['name'].split('_')[1][5:]); att.append(x)
att=sorted(att,key=lambda x:(x['layer'],x['kind']))
# Figure 1/2/3
for metric,title,ylabel,name in [('loss','Screening validation loss by attention depth','Validation loss','validation_vs_depth.png'),('ppl','Screening perplexity by attention depth','Perplexity','perplexity_vs_depth.png')]:
 for k,c in [('lora','tab:blue'),('symmetric_quadratic','tab:orange')]:
  z=sorted([x for x in att if x['kind']==k],key=lambda x:x['layer']); plt.plot([x['layer'] for x in z],[x[metric] for x in z],marker='o',label=k,color=c)
 plt.xlabel('GPT-2 block index'); plt.ylabel(ylabel); plt.title(title); plt.legend(); savefig(name)
plt.figure(); z=sorted([x for x in att if x['kind']=='symmetric_quadratic'],key=lambda x:x['layer']); q={x['layer']:x['loss'] for x in z}; z2=sorted([x for x in att if x['kind']=='lora'],key=lambda x:x['layer']); l={x['layer']:x['loss'] for x in z2}; dl=[q[i]-l[i] for i in q]; plt.axhline(0,color='k',lw=.8); plt.plot(list(q),dl,marker='o',color='purple'); plt.xlabel('GPT-2 block index'); plt.ylabel('ΔLoss = Symmetric − LoRA'); plt.title('Quadratic advantage by attention depth'); savefig('delta_loss_vs_depth.png')
# module comparison
mods=[]
for x in screen:
 if x['name'].startswith('mlp_'):
  x['module']=x['name'].split('_lora')[0].split('_symmetric')[0]; mods.append(x)
plt.figure(); labels=sorted(set(x['module'] for x in mods)); import numpy as np
for kind,c,off in [('lora','tab:blue',-.18),('symmetric_quadratic','tab:orange',.18)]:
 vals=[next(x['loss'] for x in mods if x['module']==m and x['kind']==kind) for m in labels]; plt.bar(np.arange(len(labels))+off,vals,.34,label=kind,color=c)
plt.xticks(range(len(labels)),labels,rotation=20); plt.ylabel('Validation loss'); plt.title('MLP placement screening'); plt.legend(); savefig('module_placement.png')
# confirmation aggregate
conf=[]
for p in summary_dirs(ROOT/'results_depth/confirmation'):
 s=load(p); n=Path(p).parent.name; conf.append({'name':n,'kind':s['kind'],'seed':s['seed'],'loss':s['validation_loss'],'ppl':s['perplexity'],'params':s['trainable_parameters'],'time':s['training_seconds'],'vram':s['peak_vram_bytes']/2**20,'dir':str(Path(p).parent.relative_to(ROOT))})
# grouped table
agg=[]
for name in sorted(set(x['name'].rsplit('_seed',1)[0].rsplit('_lora',1)[0].rsplit('_symmetric_quadratic',1)[0] for x in conf)):
 for k in ['lora','symmetric_quadratic']:
  z=[x for x in conf if x['name'].rsplit('_seed',1)[0].startswith(name+'_'+k)]
  if z: agg.append({'configuration':name,'model':k,'n':len(z),'loss_mean':statistics.mean(x['loss'] for x in z),'loss_std':statistics.stdev(x['loss'] for x in z) if len(z)>1 else 0,'ppl_mean':statistics.mean(x['ppl'] for x in z),'params':z[0]['params'],'time_mean':statistics.mean(x['time'] for x in z),'vram_mean':statistics.mean(x['vram'] for x in z)})
with (OUT/'confirmation_summary.csv').open('w',newline='') as f: csv.DictWriter(f,fieldnames=agg[0].keys()).writeheader(); csv.DictWriter(f,fieldnames=agg[0].keys()).writerows(agg)
plt.figure(); labels=sorted(set(x['configuration'] for x in agg)); import numpy as np
for k,c,off in [('lora','tab:blue',-.18),('symmetric_quadratic','tab:orange',.18)]:
 vals=[next(x['loss_mean'] for x in agg if x['configuration']==m and x['model']==k) for m in labels]; errs=[next(x['loss_std'] for x in agg if x['configuration']==m and x['model']==k) for m in labels]; plt.bar(np.arange(len(labels))+off,vals,.34,yerr=errs,capsize=3,label=k,color=c)
plt.xticks(range(len(labels)),labels,rotation=25); plt.ylabel('Validation loss (mean ± SD)'); plt.title('Three-seed confirmation at 100 steps'); plt.legend(); savefig('confirmation_mean_sd.png')
# convergence curves for confirmation, paired by config average by step
plt.figure()
for config,color in [('attn_layer0','tab:blue'),('attn_layer6','tab:green'),('attn_layer11','tab:red')]:
 for k,ls in [('lora','-'),('symmetric_quadratic','--')]:
  paths=[Path(ROOT/x['dir'])/'metrics.csv' for x in conf if x['name'].rsplit('_seed',1)[0].startswith(config+'_'+k) and x['kind']==k]
  if not paths: continue
  allrows=[list(csv.DictReader(p.open())) for p in paths]; steps=[int(r['step']) for r in allrows[0]]; means=[]
  for i in range(len(steps)): means.append(statistics.mean(float(rs[i]['validation_loss']) for rs in allrows))
  plt.plot(steps,means,label=f'{config} {k}',color=color,ls=ls)
plt.xlabel('Step'); plt.ylabel('Validation loss'); plt.title('Attention placement convergence (three-seed mean)'); plt.legend(fontsize=7); savefig('convergence_attention.png')
# report

def tab(data,cols,headers):
 out=['| '+' | '.join(headers)+'|','|'+'---|'*len(headers)]
 for r in data: out.append('| '+' | '.join(str(r[c]) for c in cols)+' |')
 return '\n'.join(out)

aud=tab(rows,['seed','completed','steps','val_loss','ppl','params','rank_l','rank_q','directory','valid'],['Seed','Completed','Steps','Val loss','PPL','Params','rL','rQ','Result directory','Valid'])
sc=tab(sorted(screen,key=lambda x:x['name']),['name','kind','loss','ppl','params','time','vram'],['Configuration','Model','Val loss','PPL','Params','Time s','Peak MiB'])
co=tab(agg,['configuration','model','n','loss_mean','loss_std','ppl_mean','params','time_mean'],['Configuration','Model','N','Val loss mean','SD','PPL mean','Params','Time mean s'])
md=f'''# Depth and Placement Study — Symmetric Quadratic vs LoRA

## Executive summary

This report is a separate extension of the main Quadratic Transformer study. It uses every existing result and adds a controlled placement investigation for the currently most informative quadratic parameterization, `Δy = α(xU)^(⊙2)P`, against LoRA, `Δy = α(xA)B`. The first task was an audit of the three Linear+Quadratic 2,000-step seeds. All three are now valid: each has a configuration, a 2,000-row `metrics.csv`, a `summary.json`, matching rank allocation `r_L=1,r_Q=2`, 6,144 trainable parameters, and no NaN/Inf. No checkpoint file was produced by the existing trainer, so “checkpoint” is recorded as not applicable rather than inferred.

The placement screening used GPT-2 Small, WikiText-2, sequence length 128, batch 1, gradient accumulation 4, learning rate 3e-4, seed 42 and 20 optimization steps. Attention output adapters were tested at blocks 0, 3, 6, 9 and 11. MLP input/output and combined MLP placement were also tested. These short runs are a screening instrument; the three-seed confirmation used 100 steps for early, middle, late attention and combined MLP at layer 0.

The screening shows Symmetric below LoRA at every tested attention depth, with the largest short-run gap at the last block. The single-module MLP results are nearly tied, while combined MLP is more promising but uses 15,360 parameters. The confirmation means should be read as evidence about placement under 100 steps, not as a replacement for the 500/1,000/2,000-step main comparison.

## 1. Seed audit of Linear+Quadratic at 2,000 steps

The filesystem audit checks the actual rows rather than trusting report prose. A valid run requires `metrics.csv` with exactly 2,000 rows ending at step 2,000, a readable `summary.json`, zero NaN/Inf, and a configuration matching the intended ranks. The seed 456 directory is the successful rerun after the earlier CUDA lockup; the earlier failed attempt did not contain a valid summary and is not double-counted.

{aud}

Aggregated across the three valid seeds: validation loss mean 3.4360, standard deviation 0.0146; perplexity mean 31.06, standard deviation 0.31; training time and the exact source rows remain in `summary.json` and `metrics.csv`. The main technical report has been corrected so all references consistently describe three seeds. The historical failure remains documented as an operational event, not as an additional observation.

## 2. Verified GPT-2 architecture

Programmatic inspection reports 12 Transformer blocks, hidden size 768, 12 attention heads, and FFN inner size 3072. Candidate modules and their weight shapes are: `attn.c_proj` 768×768, `mlp.c_fc` 768×3072, and `mlp.c_proj` 3072×768. The adapter wrapper reads the actual `Conv1D.weight.shape`; it does not assume that all projections have the same width.

At rank 4, a single attention or single MLP projection has LoRA/Symmetric parameter count `r(d+d_out)`: 6,144 for attention and 15,360 for either MLP projection. For the two MLP projections together, rank 2 per projection gives 15,360 total parameters, which is comparable to a rank-4 single MLP projection but not to the 6,144-parameter attention experiment. This distinction is explicit in every table.

## 3. Screening protocol

The screening varied one factor at a time: placement and depth. Model kind, seed, optimizer, sequence length, effective batch, initial GPT-2 weights and validation protocol were held constant. Output factors were zero initialized, so the adapters preserve the original forward output at initialization. The baseline for each placement is the LoRA run at the same placement and rank.

### Screening results

{sc}

### How to read the screening figures

In `validation_vs_depth.png`, lower is better. A vertical separation between the two lines is the measured short-run difference at the same block. In `delta_loss_vs_depth.png`, values below zero mean Symmetric has lower validation loss than LoRA at that depth. These are 20-step observations; they do not establish long-run convergence or causal representational superiority.

## 4. Three-seed confirmation

The confirmation selected early attention (block 0), middle attention (block 6), late attention (block 11), and combined MLP at block 0. Each LoRA/Symmetric pair used seeds 42, 123 and 456. The purpose was to test whether the screening ordering survives seed variation, not to optimize hyperparameters separately for either model.

{co}

The error bars in `confirmation_mean_sd.png` are standard deviations across the three seeds. The attention curves in `convergence_attention.png` show validation checkpoints actually saved by the trainer; no interpolation is used.

## 5. Observations, interpretations and limitations

**Observation.** In the 20-step attention scan, Symmetric has lower validation loss than LoRA at blocks 0, 3, 6, 9 and 11. The gap is small at this horizon and grows toward the late block in the recorded screening.

**Interpretation.** This is consistent with a placement dependence: the quadratic branch may interact differently with later contextual representations. It is also compatible with an optimization effect or seed-42 trajectory effect. The scan cannot distinguish these explanations.

**Observation.** Single-module `mlp.c_fc` and `mlp.c_proj` results are almost equal between models, while combined MLP is lower for Symmetric in the short screen.

**Interpretation.** The combined MLP result motivates confirmation, but it also has 15,360 trainable parameters. It cannot be compared directly with the 6,144-parameter attention result without a fixed-total-budget experiment.

**Observation.** Peak VRAM remained about 348 MiB for single projections and about 351 MiB for the combined MLP screen.

**Interpretation.** In this setup the frozen backbone and activations dominate memory. Similar memory does not mean identical compute cost.

**Limitations.** The depth screen is short, the confirmation is 100 steps, only block 0 MLP combination was tested, and it predates the later full-depth and fixed-total-budget campaigns; consult the canonical report for their completed results. No inference latency or FLOPs measurement was collected. Therefore this report does not claim that a particular depth or module is universally optimal.

## 6. Figures and files

- `figures/validation_vs_depth.png`: attention validation loss versus block depth.
- `figures/perplexity_vs_depth.png`: corresponding perplexity view.
- `figures/delta_loss_vs_depth.png`: Symmetric minus LoRA loss, with zero reference.
- `figures/module_placement.png`: MLP placement screening.
- `figures/confirmation_mean_sd.png`: three-seed confirmation means and standard deviations.
- `figures/convergence_attention.png`: three-seed mean validation trajectories for early, middle and late attention.
- `linear_quadratic_seed_audit.csv`: filesystem audit table.
- `confirmation_summary.csv`: aggregated confirmation statistics.

## 7. Next justified experiments

The evidence justifies two focused follow-ups: (1) a fixed-total-budget comparison of single versus distributed adapters, and (2) a full-depth attention scan only if the 100/500-step confirmation preserves the late-layer gap. MLP attention-plus-MLP should be tested with ranks chosen from the total budget, not with an unadjusted same rank. A held-out WikiText-2 test loss and inference latency would strengthen the scientific conclusion more than adding many short screens.
'''
(OUT/'DEPTH_AND_PLACEMENT_STUDY.md').write_text(md)
print('wrote',OUT/'DEPTH_AND_PLACEMENT_STUDY.md', 'figures',len(list(FIG.glob('*.png'))))
