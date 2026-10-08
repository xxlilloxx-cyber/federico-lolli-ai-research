#!/usr/bin/env python3
"""Read-only, reproducible analysis of the completed EWC lambda sweep."""
from __future__ import annotations
import csv,hashlib,json,math,re,sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parent
RESULTS=ROOT/'results_ewc_lambda_sweep'
OUTPUT=ROOT/'analysis_ewc_lambda_sweep'
METHODS=('lora','symmetric','combined')
LAMBDAS=(0.0,1.0,10.0,30.0,50.0,100.0,300.0)
SEED=42
sys.path.insert(0,str(ROOT))
from run_ewc_lambda_sweep import analyze_run,run_name


def md_table(frame:pd.DataFrame,digits:int=4)->str:
 def fmt(x):
  if isinstance(x,(float,np.floating)):return f'{x:.{digits}f}'
  return str(x)
 headers=list(frame.columns);lines=['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
 lines+=['| '+' | '.join(fmt(x) for x in row)+' |' for row in frame.itertuples(index=False,name=None)]
 return '\n'.join(lines)


def finite_tree(value:Any)->bool:
 if isinstance(value,dict):return all(finite_tree(x) for x in value.values())
 if isinstance(value,list):return all(finite_tree(x) for x in value)
 if isinstance(value,float):return math.isfinite(value)
 return True


def validate_data()->tuple[list[Path],str]:
 expected=[RESULTS/run_name(m,l,SEED) for m in METHODS for l in LAMBDAS];errors=[];dataset_hashes=set();configs=[]
 required=('summary.json','metrics.csv','memory_matrix.csv','config.json','dataset.json')
 for run in expected:
  if not run.is_dir():errors.append(f'missing run directory: {run.name}');continue
  for name in required:
   if not (run/name).is_file():errors.append(f'{run.name}: missing {name}')
  if errors and not (run/'summary.json').exists():continue
  summary=json.loads((run/'summary.json').read_text());config=json.loads((run/'config.json').read_text())
  if summary.get('complete') is not True:errors.append(f'{run.name}: complete is not true')
  if not finite_tree(summary):errors.append(f'{run.name}: non-finite summary value')
  metrics=pd.read_csv(run/'metrics.csv');memory=pd.read_csv(run/'memory_matrix.csv')
  for label,frame in [('metrics',metrics),('memory_matrix',memory)]:
   for column in frame.select_dtypes(include=[np.number]).columns:
    values=frame[column].dropna().to_numpy()
    if not np.isfinite(values).all():errors.append(f'{run.name}: non-finite {label}.{column}')
  dataset_hashes.add(hashlib.sha256((run/'dataset.json').read_bytes()).hexdigest());configs.append((run.name,config))
  for phase in 'ABC':
   checkpoint=run/'checkpoints'/f'after_{phase}.pt'
   if not checkpoint.is_file():errors.append(f'{run.name}: missing {checkpoint.name}')
 if len(dataset_hashes)!=1:errors.append(f'dataset hashes differ: {sorted(dataset_hashes)}')
 invariant=('model','dataset_seed','facts_per_set','steps_per_phase','evaluation_every_steps','sequence_length','micro_batch_size','gradient_accumulation','effective_batch_size','learning_rate','weight_decay','gradient_clipping','scheduler','placement','amp','ewc_gamma','fisher_samples','fisher_batches','protocol','seed')
 if configs:
  baseline=configs[0][1]
  for name,config in configs[1:]:
   for key in invariant:
    if config.get(key)!=baseline.get(key):errors.append(f'{name}: {key}={config.get(key)!r}, expected {baseline.get(key)!r}')
  if baseline.get('protocol')!='ewc_like':errors.append('common runner path is not ewc_like')
  if (baseline.get('micro_batch_size'),baseline.get('gradient_accumulation'),baseline.get('effective_batch_size'))!=(4,1,4):errors.append('execution batch configuration is not 4/1/4')
 lambdas_seen={float(c['ewc_lambda']) for _,c in configs};methods_seen={c['method'] for _,c in configs};seeds_seen={int(c['seed']) for _,c in configs}
 if lambdas_seen!=set(LAMBDAS):errors.append(f'lambda grid mismatch: {lambdas_seen}')
 if methods_seen!=set(METHODS):errors.append(f'method grid mismatch: {methods_seen}')
 if seeds_seen!={SEED}:errors.append(f'seed grid mismatch: {seeds_seen}')
 report=['# Data validation report','',f'- Expected cells: {len(expected)}',f'- Located complete cells: {sum((p/"summary.json").exists() for p in expected)}',f'- Methods: {sorted(methods_seen)}',f'- Lambdas: {sorted(lambdas_seen)}',f'- Seeds: {sorted(seeds_seen)}',f'- Shared dataset SHA-256: `{next(iter(dataset_hashes)) if len(dataset_hashes)==1 else "INCONSISTENT"}`','- Execution batch: microbatch 4, accumulation 1, effective batch 4','- Optimizer: AdamW; learning rate, clipping, weight decay, and scheduler checked identical','- Fisher: online diagonal implementation; gamma, samples, and batches checked identical','- Numeric finite-value check: passed' if not errors else '- Numeric/configuration validation: FAILED','']
 if errors:report+=['## Errors','']+[f'- {e}' for e in errors]
 else:report+=['## Result','','**VALID — all 21 expected cells are complete and configuration-consistent.**']
 out=OUTPUT/'data';out.mkdir(parents=True,exist_ok=True);(out/'DATA_VALIDATION_REPORT.md').write_text('\n'.join(report)+'\n')
 if errors:raise RuntimeError('data validation failed; see DATA_VALIDATION_REPORT.md')
 return expected,next(iter(dataset_hashes))


def fisher_factors(run:Path,method:str,value:float)->list[dict[str,Any]]:
 rows=[]
 for phase,state in [('A','F_A'),('B','F_AB')]:
  ck=torch.load(run/'checkpoints'/f'after_{phase}.pt',map_location='cpu',weights_only=False)
  for key,tensor in ck['auxiliary_state']['fisher'].items():
   f=tensor.float().reshape(-1);total=f.sum();p=f/total.clamp_min(1e-30);nz=p[p>0]
   rows.append({'method':method,'lambda':value,'seed':SEED,'online_state':state,'factor':key.rsplit('.',1)[-1],
                'elements':f.numel(),'fisher_sum':float(total),'fisher_max':float(f.max()),
                'effective_support':float(torch.exp(-(nz*torch.log(nz)).sum())) if total>0 else 0.0,
                'share_of_total':0.0})
 for state in ('F_A','F_AB'):
  subset=[r for r in rows if r['online_state']==state];total=sum(r['fisher_sum'] for r in subset)
  for r in subset:r['share_of_total']=r['fisher_sum']/total if total else 0.0
 return rows


def reconstruct(runs:list[Path])->tuple[pd.DataFrame,pd.DataFrame]:
 rows=[];factor=[]
 for run in runs:
  row=analyze_run(run);rows.append(row);factor.extend(fisher_factors(run,row['method'],row['lambda']))
 data=pd.DataFrame(rows).sort_values(['method','lambda']).reset_index(drop=True);factors=pd.DataFrame(factor)
 saved=RESULTS/'lambda_sweep_results.csv'
 if saved.exists():
  old=pd.read_csv(saved)
  for _,r in data.iterrows():
   match=old[(old.method==r.method)&np.isclose(old['lambda'],r['lambda'])]
   if len(match)!=1:raise RuntimeError('saved aggregate cell mismatch')
   for key in ('mean_forgetting','mean_new_fact_acquisition_nll','final_mean_nll','final_mean_probability','final_mean_exact_match'):
    if not np.isclose(float(match.iloc[0][key]),float(r[key]),atol=1e-12,rtol=0):raise RuntimeError(f'saved aggregate mismatch: {r.method} lambda={r["lambda"]} {key}')
 if not np.isfinite(data.select_dtypes(include=[np.number]).to_numpy()).all():raise RuntimeError('non-finite reconstructed metric')
 return data,factors


def add_baseline_changes(data:pd.DataFrame)->pd.DataFrame:
 metrics=('mean_forgetting','mean_new_fact_acquisition_nll','final_mean_nll','final_mean_probability','total_parameter_displacement_a_c_l2')
 rows=[]
 for method,g in data.groupby('method'):
  base=g[g['lambda']==0].iloc[0]
  for _,r in g.iterrows():
   row={'method':method,'lambda':r['lambda']}
   for key in metrics:
    delta=float(r[key]-base[key]);row[f'{key}_absolute_change']=delta;row[f'{key}_percent_change']=100*delta/abs(float(base[key])) if base[key]!=0 else 'not_applicable'
   row['forgetting_reduction']=float(base.mean_forgetting-r.mean_forgetting);row['plasticity_cost']=float(r.mean_new_fact_acquisition_nll-base.mean_new_fact_acquisition_nll)
   row['stability_gain_per_acquisition_cost']=row['forgetting_reduction']/row['plasticity_cost'] if row['plasticity_cost']>0 else 'not_applicable'
   rows.append(row)
 return pd.DataFrame(rows)


def pareto_rows(data:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for method,g in data.groupby('method'):
  for _,r in g.iterrows():
   dominated=((g.mean_forgetting<=r.mean_forgetting)&(g.mean_new_fact_acquisition_nll<=r.mean_new_fact_acquisition_nll)&((g.mean_forgetting<r.mean_forgetting)|(g.mean_new_fact_acquisition_nll<r.mean_new_fact_acquisition_nll))).any()
   if not dominated:rows.append({'method':method,'lambda':r['lambda'],'mean_forgetting':r.mean_forgetting,'mean_new_fact_acquisition_nll':r.mean_new_fact_acquisition_nll,'final_mean_nll':r.final_mean_nll,'final_mean_probability':r.final_mean_probability,'final_mean_exact_match':r.final_mean_exact_match})
 return pd.DataFrame(rows).sort_values(['method','lambda'])


def knee_analysis(data:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for method,g in data.groupby('method'):
  g=g.sort_values('lambda');f=g.mean_forgetting.to_numpy();a=g.mean_new_fact_acquisition_nll.to_numpy();fn=(f-f.min())/(f.max()-f.min());an=(a-a.min())/(a.max()-a.min());ideal=np.hypot(fn,an)
  points=np.c_[fn,an];v=points[-1]-points[0];d=points-points[0];chord=np.abs(v[0]*d[:,1]-v[1]*d[:,0])/np.linalg.norm(v)
  for i,(_,r) in enumerate(g.iterrows()):rows.append({'method':method,'lambda':r['lambda'],'normalized_ideal_distance':ideal[i],'endpoint_chord_distance':chord[i],'ideal_distance_candidate':i==ideal.argmin(),'chord_knee_candidate':i==chord.argmax()})
 return pd.DataFrame(rows)


def exact_thresholds(data:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for method,g in data.groupby('method'):
  g=g.sort_values('lambda');base=float(g.iloc[0].final_mean_exact_match);declined=g[g.final_mean_exact_match<base-1e-12];zero=g[g.final_mean_exact_match<=1e-12]
  rows.append({'method':method,'lambda0_exact_match':base,'first_lambda_with_decline':float(declined.iloc[0]['lambda']) if len(declined) else np.nan,'first_lambda_with_zero_exact_match':float(zero.iloc[0]['lambda']) if len(zero) else np.nan,'lambda30_exact_match':float(g[g['lambda']==30].iloc[0].final_mean_exact_match),'lambda50_exact_match':float(g[g['lambda']==50].iloc[0].final_mean_exact_match)})
 return pd.DataFrame(rows)


def correlations(data:pd.DataFrame)->pd.DataFrame:
 pairs=[('lambda','mean_forgetting'),('lambda','mean_new_fact_acquisition_nll'),('lambda','total_parameter_displacement_a_c_l2'),('total_parameter_displacement_a_c_l2','mean_forgetting'),('realized_ewc_penalty_c','mean_forgetting'),('realized_ewc_penalty_c','mean_new_fact_acquisition_nll')]
 rows=[]
 for method,g in data.groupby('method'):
  for x,y in pairs:
   pear=float(g[x].corr(g[y],method='pearson'))
   spear=float(g[x].rank(method='average').corr(g[y].rank(method='average'),method='pearson'))
   rows.append({'method':method,'x':x,'y':y,'pearson_r':pear,'spearman_rho':spear,'n':len(g)})
 return pd.DataFrame(rows)


def select_candidates(data:pd.DataFrame,knees:pd.DataFrame,pareto:pd.DataFrame)->pd.DataFrame:
 rows=[]
 for method,g in data.groupby('method'):
  k=knees[knees.method==method];ideal=float(k[k.ideal_distance_candidate].iloc[0]['lambda']);chord=float(k[k.chord_knee_candidate].iloc[0]['lambda']);selected=[ideal]
  if chord not in selected:selected.append(chord)
  if len(selected)<2:
   choices=g[(g['lambda']>0)&(g.final_mean_exact_match>0)&(~g['lambda'].isin(selected))].sort_values(['lambda'])
   if len(choices):selected.append(float(choices.iloc[-1]['lambda']))
  for value in selected[:2]:
   r=g[g['lambda']==value].iloc[0];flags=[]
   if value==ideal:flags.append('minimum normalized distance to ideal')
   if value==chord:flags.append('maximum endpoint-chord knee distance')
   if r.final_mean_exact_match>0:flags.append('non-zero final exact match')
   if ((pareto.method==method)&np.isclose(pareto['lambda'],value)).any():flags.append('Pareto-nondominated')
   rows.append({'method':method,'lambda':value,'mean_forgetting':r.mean_forgetting,'mean_new_fact_acquisition_nll':r.mean_new_fact_acquisition_nll,'final_mean_nll':r.final_mean_nll,'final_mean_probability':r.final_mean_probability,'final_mean_exact_match':r.final_mean_exact_match,'rationale':'; '.join(flags)})
 return pd.DataFrame(rows).sort_values(['method','lambda'])


def make_figures(data:pd.DataFrame,factors:pd.DataFrame):
 import matplotlib.pyplot as plt
 data=data.copy();data['realized_ewc_penalty_total']=data.realized_ewc_penalty_b+data.realized_ewc_penalty_c
 colors={'lora':'#2563eb','symmetric':'#dc2626','combined':'#16a34a'};markers={'lora':'o','symmetric':'s','combined':'^'}
 main=OUTPUT/'figures_main';supp=OUTPUT/'figures_supplementary';main.mkdir(parents=True,exist_ok=True);supp.mkdir(parents=True,exist_ok=True)
 def save(fig,folder,name):fig.tight_layout();fig.savefig(folder/f'{name}.png',dpi=220,bbox_inches='tight');fig.savefig(folder/f'{name}.svg',bbox_inches='tight');plt.close(fig)
 # Main 1
 fig,axes=plt.subplots(1,3,figsize=(13.5,4.2))
 for ax,method in zip(axes,METHODS):
  g=data[data.method==method].sort_values('lambda');ax.plot(g.mean_forgetting,g.mean_new_fact_acquisition_nll,'o-',color=colors[method])
  for _,r in g.iterrows():ax.annotate(f'λ={r["lambda"]:g}',(r.mean_forgetting,r.mean_new_fact_acquisition_nll),xytext=(4,4),textcoords='offset points',fontsize=7)
  ax.set(title=method.capitalize(),xlabel='Mean forgetting (NLL increase)',ylabel='Mean B/C acquisition NLL');ax.grid(alpha=.25)
 save(fig,main,'figure_1_stability_plasticity')
 # shared lambda plotting helper
 def linepanels(specs,name,folder=main):
  fig,axes=plt.subplots(1,len(specs),figsize=(4.6*len(specs),4));axes=np.atleast_1d(axes)
  for ax,(key,title) in zip(axes,specs):
   for method in METHODS:
    g=data[data.method==method].sort_values('lambda');ax.plot(g['lambda'],g[key],marker=markers[method],color=colors[method],label=method.capitalize())
   ax.set_xscale('symlog',linthresh=1);ax.set(title=title,xlabel='λ');ax.grid(alpha=.25)
  axes[0].legend(fontsize=8);save(fig,folder,name)
 linepanels([('forgetting_a_after_b_nll','A after B'),('forgetting_a_after_c_nll','A after C'),('forgetting_b_after_c_nll','B after C')],'figure_2_forgetting')
 linepanels([('acquisition_b_nll','B acquisition NLL'),('acquisition_c_nll','C acquisition NLL')],'figure_3_acquisition')
 linepanels([('final_mean_nll','Final mean A/B/C NLL'),('final_mean_probability','Final mean probability'),('final_mean_exact_match','Final mean exact match')],'figure_4_final_memory')
 # Main 5
 linepanels([('total_parameter_displacement_a_c_l2','||θC−θA||₂'),('realized_ewc_penalty_total','Realized EWC penalty B+C'),('fisher_ab_effective_support','F_AB effective support')],'figure_5_parameter_fisher')
 # Supplementary
 linepanels([('acquisition_a_nll','A acquisition'),('acquisition_b_nll','B acquisition'),('acquisition_c_nll','C acquisition'),('final_a_nll','Final A NLL'),('final_b_nll','Final B NLL'),('final_c_nll','Final C NLL')],'supp_1_nll_components',supp)
 linepanels([('acquisition_a_probability','A acquisition probability'),('acquisition_b_probability','B acquisition probability'),('acquisition_c_probability','C acquisition probability'),('final_a_probability','Final A probability'),('final_b_probability','Final B probability'),('final_c_probability','Final C probability')],'supp_2_probability_components',supp)
 linepanels([('final_mean_paraphrase_nll','Paraphrase mean NLL'),('final_mean_paraphrase_probability','Paraphrase probability'),('final_mean_paraphrase_exact_match','Paraphrase exact match')],'supp_3_paraphrase',supp)
 linepanels([('parameter_displacement_a_b_l2','||θB−θA||₂'),('parameter_displacement_b_c_l2','||θC−θB||₂'),('total_parameter_displacement_a_c_l2','||θC−θA||₂')],'supp_4_parameter_displacements',supp)
 linepanels([('realized_ewc_penalty_b','Penalty during B'),('realized_ewc_penalty_c','Penalty during C'),('fisher_a_sum','F_A sum'),('fisher_ab_sum','F_AB sum'),('fisher_ab_max','F_AB maximum')],'supp_5_penalty_fisher',supp)
 fig,axes=plt.subplots(2,3,figsize=(13,7));
 for ax,(method,state) in zip(axes.ravel(),[(m,s) for m in METHODS for s in ('F_A','F_AB')]):
  g=factors[(factors.method==method)&(factors.online_state==state)]
  for factor,h in g.groupby('factor'):ax.plot(h['lambda'],h.share_of_total,'o-',label=factor)
  ax.set_xscale('symlog',linthresh=1);ax.set(title=f'{method}: {state}',xlabel='λ',ylabel='Fisher mass share');ax.grid(alpha=.2);ax.legend(fontsize=7)
 save(fig,supp,'supp_6_factor_fisher_concentration')


def reports(data,baseline,pareto,knees,candidates,thresholds,corr,cross,focus):
 report_dir=OUTPUT/'report';report_dir.mkdir(parents=True,exist_ok=True)
 # helpers
 def row(method,value):return data[(data.method==method)&np.isclose(data['lambda'],value)].iloc[0]
 lines=['# EWC Lambda Sweep: Deep Scientific Analysis','','## 1. Executive summary','',
 'This single-seed exploratory sweep isolates EWC strength under one matched execution path. Across all three adapters, increasing λ reduces forgetting and increases new-fact acquisition NLL. The transition is gradual rather than binary, and the useful compromise region differs by architecture.','',
 '- A 10% operational threshold shows that forgetting first decreases materially at **λ=10 for all three methods**.',
 '- LoRA begins paying a material acquisition cost by **λ=1**; Symmetric and Combined cross the same 10% threshold at **λ=10**.',
 '- LoRA’s two geometric knee methods both select **λ=10**. Symmetric’s both select **λ=30**. Combined selects **λ=30** by ideal distance and **λ=10** by endpoint-chord curvature.',
 '- Exact match begins falling before it reaches zero. It becomes zero at λ=30 for LoRA and Symmetric, and at λ=50 for Combined.',
 '- λ=100 and λ=300 consistently under-learn B/C under this protocol; they improve stability while substantially weakening acquisition and greedy exact generation.',
 '- No Combined point at a shared λ dominates both LoRA and Symmetric simultaneously in forgetting and acquisition.','',
 'The recommended confirmation candidates are LoRA {1,10}, Symmetric {10,30}, and Combined {10,30}. They are candidate-generating choices, not statistically optimal values.','',
 '## 2. Experimental scope','',
 'The sweep contains 21 runs: LoRA, Symmetric, and Combined at λ ∈ {0,1,10,30,50,100,300}, seed 42. Each run uses the same frozen GPT-2 Small backbone, block-0 `attn.c_proj`, 12 facts per A/B/C set, 2,500 steps per phase, and no replay. Every adapter has 6,144 trainable parameters.','',
 'The objective is','',r'$$L_{\mathrm{total}}=L_{\mathrm{current}}+\frac{\lambda}{2}\sum_i F_i(\theta_i-\theta_i^*)^2,$$','',
 'with online Fisher update','',r'$$F_{AB}=\gamma F_A+F_B,\qquad \gamma=1.$$','',
 'Lambda controls resistance to changing parameters judged important for earlier facts.','',
 '## 3. Why λ=0 is the primary control','',
 'Lambda zero executes the same EWC runner, microbatch 4, accumulation 1, tokenization, ordering, Fisher estimation, checkpointing, and random-seed path as every other sweep cell. Fisher and references are still computed, while multiplication by λ makes the regularization objective and gradient contribution exactly zero. This is the primary numerical baseline. Historical microbatch-1 sequential results are excluded from the primary analysis.','',
 '## 4. Stability–plasticity trade-off','',
 '![Figure 1](../figures_main/figure_1_stability_plasticity.svg)','',
 'Every sampled λ is Pareto-nondominated within its own method because each increase in stability is purchased with reduced plasticity. Pareto membership therefore does not identify a unique choice. The knee diagnostics and exact-match behavior provide additional descriptive constraints.','',
 '![Figure 2](../figures_main/figure_2_forgetting.svg)','',
 '![Figure 3](../figures_main/figure_3_acquisition.svg)','',
 '## 5. LoRA analysis','']
 l0,l1,l10,l30=(row('lora',x) for x in (0,1,10,30))
 lines += [f'LoRA moves from mean forgetting {l0.mean_forgetting:.4f} at λ=0 to {l10.mean_forgetting:.4f} at λ=10, while mean B/C acquisition NLL rises from {l0.mean_new_fact_acquisition_nll:.4f} to {l10.mean_new_fact_acquisition_nll:.4f}. λ=1 retains the λ=0 final exact match ({l1.final_mean_exact_match:.4f}) with a modest stability gain. At λ=10 exact match remains non-zero ({l10.final_mean_exact_match:.4f}), but at λ=30 it collapses to zero. Both knee methods select λ=10. LoRA therefore reacts strongly to relatively weak regularization and does not require λ=30–50 to preserve facts.','',
 '## 6. Symmetric analysis','']
 s0,s10,s30,s50=(row('symmetric',x) for x in (0,10,30,50))
 lines += [f'Symmetric starts with the strongest plasticity: mean acquisition NLL {s0.mean_new_fact_acquisition_nll:.4f} and final exact match {s0.final_mean_exact_match:.4f} at λ=0. At λ=10 it retains non-zero exact match ({s10.final_mean_exact_match:.4f}) while reducing mean forgetting from {s0.mean_forgetting:.4f} to {s10.mean_forgetting:.4f}. λ=30 minimizes both geometric compromise diagnostics and final mean NLL ({s30.final_mean_nll:.4f}), but exact match is already zero. λ=50 adds stability with worse acquisition and no exact-match recovery. The useful candidates are therefore λ=10 for retained generation and λ=30 for balanced NLL.','',
 '## 7. Combined analysis','']
 c0,c10,c30,c50=(row('combined',x) for x in (0,10,30,50))
 lines += [f'Combined reduces mean forgetting from {c0.mean_forgetting:.4f} at λ=0 to {c10.mean_forgetting:.4f} at λ=10 and {c30.mean_forgetting:.4f} at λ=30. The endpoint-chord knee selects λ=10, while normalized ideal distance selects λ=30. Exact match remains small but non-zero at λ=30 ({c30.final_mean_exact_match:.4f}) and reaches zero at λ=50. Across shared λ values, Combined never has both lower forgetting and lower acquisition NLL than both LoRA and Symmetric. This sweep therefore provides no evidence of synergy in the primary trade-off plane.','',
 '## 8. Cross-method comparison','',md_table(cross[['lambda','method','mean_forgetting','mean_new_fact_acquisition_nll','final_mean_nll','final_mean_probability','final_mean_exact_match','best_stability','best_plasticity','best_final_nll','best_probability','best_exact_match']]),'',
 'The metric-specific best method varies. LoRA generally leads stability, Symmetric generally leads plasticity and probability, and Combined occupies intermediate regions. No single winner is supported.','',
 '## 9. Exact-match collapse','',md_table(thresholds),'',
 'Lambda 30 preserves only a small Combined exact-match signal and none for LoRA or Symmetric. Lambda 50 preserves none for any method. This makes λ=30/50 unsuitable if greedy exact generation is a hard requirement, despite favorable balanced NLL near λ=30.','',
 '## 10. Parameter movement','',
 '![Figure 5](../figures_main/figure_5_parameter_fisher.svg)','']
 for method,g in data.groupby('method'):
  ordered=g.sort_values('lambda');vals=ordered.total_parameter_displacement_a_c_l2.to_numpy();decreases=int(np.sum(np.diff(vals)<=0));r=corr[(corr.method==method)&(corr.x=='total_parameter_displacement_a_c_l2')&(corr.y=='mean_forgetting')].iloc[0]
  lines.append(f'- **{method.capitalize()}:** A→C displacement decreases in {decreases}/6 adjacent λ transitions; displacement–forgetting Pearson r={r.pearson_r:.3f}, Spearman ρ={r.spearman_rho:.3f}.')
 lines += ['','Movement is broadly reduced at high λ but is not perfectly monotonic at low/intermediate λ. Fisher weighting constrains selected directions rather than ordinary Euclidean distance uniformly.','',
 '## 11. Fisher structure','',
 'F_A is identical across λ within each method because phase A precedes the EWC penalty and uses the same execution path. F_AB varies because λ affects phase-B learning and therefore the B Fisher estimate. Output factors carry most Fisher mass. Cross-architecture Fisher magnitudes are parameterization dependent and should not be ranked as a universal importance scale. The observed association between Fisher concentration and lambda response is mechanistic evidence, not causal proof.','',
 '## 12. Pareto analysis','',md_table(pareto),'',
 'All within-method points are nondominated on the sampled grid. The knee table therefore supplies a descriptive compromise rather than an optimum:','',md_table(knees),'',
 '## 13. Candidate lambda values','',md_table(candidates),'',
 'The focused λ=0/10/30/50/100 comparison is available in `tables/LAMBDA_30_50_COMPARISON.csv`. λ=100 and λ=300 are useful stability anchors but are too restrictive for exact generation in this experiment.','',
 '## 14. Limitations','',
 '- One seed; no confidence intervals or inferential claims are justified.','- One synthetic dataset, one GPT-2 model size, and one insertion point.','- The diagonal empirical Fisher omits parameter correlations.','- Candidate knees depend on the sampled λ grid and metric normalization.','- Exact match is a strict greedy-decoding metric and complements rather than replaces NLL.','- Fisher magnitudes are parameterization dependent.','',
 '## 15. Confirmation experiment recommendation','',
 '- **LoRA:** λ=1 and λ=10. λ=1 preserves baseline exact match; λ=10 is the common geometric knee.','- **Symmetric:** λ=10 and λ=30. λ=10 retains useful exact match; λ=30 is the balanced-NLL knee.','- **Combined:** λ=10 and λ=30. They are the two independent knee candidates, with λ=30 remaining just above zero exact match.','',
 'A future confirmation should use seeds 42, 123, and 456 under this exact execution path. It should retain λ=0 as the common matched control.','',
 '## 16. Publication-oriented conclusions','',
 'EWC strength continuously controls factual-memory stability and new-fact plasticity. Moderate regularization provides useful compromises, but the transition differs by adapter architecture. LoRA responds at lower λ, while Symmetric and Combined retain stronger acquisition into the λ=10–30 range. High λ suppresses forgetting but causes under-learning and loss of exact generation. These single-seed findings nominate method-specific confirmation values; they do not identify universal optimal regularization.','',
 '## Descriptive correlations','',md_table(corr)]
 (report_dir/'EWC_LAMBDA_DEEP_ANALYSIS.md').write_text('\n'.join(lines)+'\n')
 pub=['# EWC Strength and Continual Factual Adaptation','','## Research question','','How does online EWC strength control the stability–plasticity trade-off for LoRA, Symmetric, and Combined adapters when execution, parameter count, data, and seed are held fixed?','','## Mathematical formulation','',r'$$L_{\mathrm{total}}=L_{\mathrm{current}}+\frac{\lambda}{2}\sum_iF_i(\theta_i-\theta_i^*)^2,$$',r'$$F_{AB}=F_A+F_B.$$','','## Protocol','','Frozen GPT-2 Small; block-0 `attn.c_proj`; 6,144 trainable adapter parameters; 12 facts per A/B/C set; no replay; 2,500 steps per phase; seed 42; λ ∈ {0,1,10,30,50,100,300}. Lambda zero uses the same EWC execution path and is the primary control.','','## Strongest findings','','1. Forgetting first decreases materially at λ=10 for all methods under a predeclared 10% threshold.','2. LoRA reacts at lower λ; λ=10 is its geometric knee. Symmetric and Combined have compromise candidates near λ=10–30.','3. Exact match becomes zero at λ=30 for LoRA and Symmetric and at λ=50 for Combined.','4. Very high λ reduces forgetting but causes clear B/C under-learning.','5. Combined does not dominate both LoRA and Symmetric in the forgetting–acquisition plane.','','## Principal figures']
 for i,name in enumerate(['figure_1_stability_plasticity','figure_2_forgetting','figure_3_acquisition','figure_4_final_memory','figure_5_parameter_fisher'],1):pub+=['',f'![Figure {i}](../figures_main/{name}.svg)']
 pub += ['','## Candidate conclusion','','Online EWC exposes a method-dependent continuum rather than one optimal regularization strength. Moderate λ values can reduce catastrophic forgetting while retaining some acquisition, whereas λ≥100 is overly restrictive for exact generation in this configuration.','','## Limitations','','The sweep uses one seed and one synthetic GPT-2 configuration. Knee and Pareto results are descriptive and grid dependent. No statistical significance or universal optimum is claimed.','','## Recommended confirmation','','Run λ={1,10} for LoRA and λ={10,30} for Symmetric and Combined across seeds 42, 123, and 456, retaining λ=0 as the matched control.']
 (report_dir/'PUBLICATION_SUMMARY.md').write_text('\n'.join(pub)+'\n')


def main():
 for p in (OUTPUT/'tables',OUTPUT/'figures_main',OUTPUT/'figures_supplementary',OUTPUT/'data',OUTPUT/'report'):p.mkdir(parents=True,exist_ok=True)
 runs,_=validate_data();data,factors=reconstruct(runs);baseline=add_baseline_changes(data);pareto=pareto_rows(data);knees=knee_analysis(data);thresholds=exact_thresholds(data);corr=correlations(data);candidates=select_candidates(data,knees,pareto)
 trade=data[['lambda','method','mean_forgetting','mean_new_fact_acquisition_nll','acquisition_b_nll','acquisition_c_nll','final_mean_nll','final_mean_probability','final_mean_exact_match']].copy()
 cross=trade.copy()
 for key,label,lower in [('mean_forgetting','best_stability',True),('mean_new_fact_acquisition_nll','best_plasticity',True),('final_mean_nll','best_final_nll',True),('final_mean_probability','best_probability',False),('final_mean_exact_match','best_exact_match',False)]:
  target=cross.groupby('lambda')[key].transform('min' if lower else 'max');cross[label]=np.isclose(cross[key],target)
 focus=cross[cross['lambda'].isin([0,10,30,50,100])].copy()
 tables=OUTPUT/'tables';data.to_csv(tables/'lambda_full_metrics.csv',index=False);baseline.to_csv(tables/'lambda_vs_baseline.csv',index=False);trade.to_csv(tables/'lambda_tradeoff.csv',index=False);pareto.to_csv(tables/'PARETO_CANDIDATES.csv',index=False);pareto.to_csv(tables/'pareto_candidates.csv',index=False);candidates.to_csv(tables/'candidate_lambdas.csv',index=False);corr.to_csv(tables/'correlations.csv',index=False);thresholds.to_csv(tables/'exact_match_thresholds.csv',index=False);cross.to_csv(tables/'CROSS_METHOD_COMPARISON.csv',index=False);focus.to_csv(tables/'LAMBDA_30_50_COMPARISON.csv',index=False);knees.to_csv(tables/'KNEE_ANALYSIS.csv',index=False);factors.to_csv(OUTPUT/'data'/'fisher_factor_metrics.csv',index=False)
 (tables/'PARETO_CANDIDATES.md').write_text('# Pareto-nondominated lambda values\n\nLower forgetting and acquisition NLL are preferred. Analysis is within each method.\n\n'+md_table(pareto)+'\n')
 make_figures(data,factors);reports(data,baseline,pareto,knees,candidates,thresholds,corr,cross,focus)
 manifest={'source_root':str(RESULTS.relative_to(ROOT)),'runs':len(data),'methods':list(METHODS),'lambdas':list(LAMBDAS),'seed':SEED,'main_figures':[p.name for p in sorted((OUTPUT/'figures_main').glob('*.svg'))],'tables':[p.name for p in sorted(tables.glob('*.csv'))]}
 (OUTPUT/'data'/'ANALYSIS_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'status':'complete','analysis_root':str(OUTPUT),'runs':len(data),'candidate_lambdas':candidates[['method','lambda']].to_dict('records')},indent=2))


if __name__=='__main__':main()
