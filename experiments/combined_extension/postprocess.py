#!/usr/bin/env python3
"""Aggregate extension summaries and build plots only from saved files."""
from __future__ import annotations
import argparse,csv,json,statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
NUMERIC=('best_validation_loss','final_validation_loss','final_test_loss','test_loss','test_accuracy','test_macro_f1','forward_ms_mean','backward_ms_mean','tokens_per_second_mean','peak_allocated_vram_mib')
def main(root,campaign):
 base=root/campaign;summaries=[]
 for path in base.glob('*/summary.json'):
  row=json.loads(path.read_text())
  if row.get('complete') is True:row['source']=str(path.relative_to(root));summaries.append(row)
 if not summaries:return
 aggregate=[];groups=defaultdict(list)
 for row in summaries:groups[(row.get('method'),row.get('rank'),row.get('rank_lora'),row.get('rank_symmetric'),str(row.get('layers')),str(row.get('projections')),row.get('scaling_mode'))].append(row)
 for key,rows in groups.items():
  out={'method':key[0],'rank':key[1],'rank_lora':key[2],'rank_symmetric':key[3],'layers':key[4],'projections':key[5],'scaling_mode':key[6],'n':len(rows)}
  for field in NUMERIC:
   values=[float(r[field]) for r in rows if r.get(field) is not None]
   if values:out[field+'_mean']=statistics.mean(values);out[field+'_sample_sd']=statistics.stdev(values) if len(values)>1 else None
  aggregate.append(out)
 outdir=root/'aggregate';outdir.mkdir(exist_ok=True)
 (outdir/f'{campaign}.json').write_text(json.dumps({'individual':summaries,'aggregate':aggregate},indent=2)+'\n')
 fields=sorted({k for row in aggregate for k in row});
 with (outdir/f'{campaign}.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(aggregate)
 if campaign=='rank_confirmation':
  figdir=root/'figures';figdir.mkdir(exist_ok=True)
  for metric,name,label in (('final_test_loss','rank_vs_test_loss','Held-out test loss'),('final_validation_loss','rank_vs_final_validation_loss','Final validation loss')):
   fig,ax=plt.subplots(figsize=(7,4.5))
   for method in ('lora','symmetric','combined'):
    rows=[r for r in aggregate if r['method']==method];rows.sort(key=lambda r:(r['rank'] or (r['rank_lora']+r['rank_symmetric'])))
    if rows:
     x=[r['rank'] or r['rank_lora']+r['rank_symmetric'] for r in rows];y=[r[metric+'_mean'] for r in rows];e=[r.get(metric+'_sample_sd') or 0 for r in rows];ax.errorbar(x,y,yerr=e,marker='o',label=method)
   ax.set(xlabel='Total nominal rank',ylabel=label,title='Controlled 5000-step rank confirmation');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=220);fig.savefig(figdir/f'{name}.svg');plt.close(fig)
  matched=[];same_local=[]
  for budget,single,pair in ((3072,2,1),(6144,4,2),(12288,8,4)):
   for row in summaries:
    if (row['method'] in ('lora','symmetric') and row.get('rank')==single) or (row['method']=='combined' and row.get('rank_lora')==pair):matched.append({'budget':budget,**row})
  for rank in (1,2,4):
   for row in summaries:
    if (row['method'] in ('lora','symmetric') and row.get('rank')==rank) or (row['method']=='combined' and row.get('rank_lora')==rank):same_local.append({'local_rank':rank,'parameter_matched':False,**row})
  for name,data in (('rank_matched_budget',matched),('rank_same_local_rank_NOT_parameter_matched',same_local)):
   fields=sorted({k for r in data for k,v in r.items() if isinstance(v,(str,int,float,bool)) or v is None})
   with (outdir/f'{name}.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(data)
  convergence=[]
  for summary in summaries:
   source=root/summary['source'];metrics=source.parent/'metrics.csv'
   for row in csv.DictReader(metrics.open()):
    if int(row['step']) in (1000,2000,3000,4000,5000):convergence.append({'method':summary['method'],'rank':summary.get('rank'),'rank_lora':summary.get('rank_lora'),'rank_symmetric':summary.get('rank_symmetric'),'seed':summary['seed'],**row})
  fields=sorted({k for r in convergence for k in r});
  with (outdir/'rank_convergence.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(convergence)
  fig,ax=plt.subplots(figsize=(7,4.5))
  for method in ('lora','symmetric','combined'):
   selected=[r for r in convergence if r['method']==method and ((method!='combined' and int(r['rank'])==4) or (method=='combined' and int(r['rank_lora'])==2))];by_step=defaultdict(list)
   for r in selected:by_step[int(r['step'])].append(float(r['validation_loss']))
   steps=sorted(by_step);ax.plot(steps,[statistics.mean(by_step[s]) for s in steps],marker='o',label=method)
  ax.set(xlabel='Optimizer step',ylabel='Validation loss',title='Matched 6,144-parameter convergence');ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(figdir/'convergence_matched_6144.png',dpi=220);fig.savefig(figdir/'convergence_matched_6144.svg');plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4.5));labels=[];means=[];errors=[]
  for budget in (3072,6144,12288):
   for method in ('lora','symmetric','combined'):
    values=[r['final_test_loss'] for r in matched if r['budget']==budget and r['method']==method]
    if values:labels.append(f'{method}\n{budget}');means.append(statistics.mean(values));errors.append(statistics.stdev(values) if len(values)>1 else 0)
  if labels:
   ax.bar(labels,means,yerr=errors,capsize=3);ax.set(ylabel='Held-out test loss',title='Matched-budget rank comparison');ax.tick_params(axis='x',labelsize=7);fig.tight_layout();fig.savefig(figdir/'matched_budget_comparison.png',dpi=220);fig.savefig(figdir/'matched_budget_comparison.svg')
  plt.close(fig)
 if campaign=='placement':
  fig,ax=plt.subplots(figsize=(8,4.5))
  for method in ('lora','symmetric','combined'):
   rows=sorted((r for r in summaries if r['method']==method and len(r['layers'])==1),key=lambda r:r['layers'][0]);ax.plot([r['layers'][0] for r in rows],[r['final_validation_loss'] for r in rows],marker='o',label=method)
  ax.set(xlabel='Zero-based GPT-2 block',ylabel='Final validation loss',title='Exploratory 12-block placement scan');ax.legend();ax.grid(alpha=.2);fig.tight_layout();figdir=root/'figures';figdir.mkdir(exist_ok=True);fig.savefig(figdir/'placement_vs_block.png',dpi=220);fig.savefig(figdir/'placement_vs_block.svg');plt.close(fig)
  differences=[]
  for block in range(12):
   found={r['method']:r['final_validation_loss'] for r in summaries if r['layers']==[block]}
   if len(found)==3:differences.append({'block':block,'symmetric_minus_lora':found['symmetric']-found['lora'],'combined_minus_lora':found['combined']-found['lora']})
  with (outdir/'placement_method_differences.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=differences[0]);w.writeheader();w.writerows(differences)
 if campaign=='multilayer':
  fig,ax=plt.subplots(figsize=(8,4.5));labels=[];values=[]
  for row in summaries:labels.append(f"{row['method']}\n{row['expected_trainable_parameters']}\n{len(row['layers'])}L");values.append(row['final_validation_loss'])
  ax.bar(labels,values);ax.set(ylabel='Final validation loss',title='Exploratory fixed-budget multilayer geometries');ax.tick_params(axis='x',labelsize=7);fig.tight_layout();figdir=root/'figures';figdir.mkdir(exist_ok=True);fig.savefig(figdir/'multilayer_geometry_comparison.png',dpi=220);fig.savefig(figdir/'multilayer_geometry_comparison.svg');plt.close(fig)
 if campaign=='scaling':
  fig,ax=plt.subplots(figsize=(8,4.5))
  for method in ('lora','symmetric','combined'):
   rows=[r for r in summaries if r['method']==method]
   for policy in sorted({r.get('scaling_policy') for r in rows}):
    chosen=sorted((r for r in rows if r.get('scaling_policy')==policy),key=lambda r:r.get('rank') or 2*r.get('rank_lora'));ax.plot([r.get('rank') or 2*r.get('rank_lora') for r in chosen],[r['final_validation_loss'] for r in chosen],marker='o',label=f'{method}: {policy}')
  ax.set(xlabel='Nominal total rank',ylabel='Final validation loss',title='Exploratory scaling ablation');ax.legend(fontsize=7);ax.grid(alpha=.2);fig.tight_layout();figdir=root/'figures';figdir.mkdir(exist_ok=True);fig.savefig(figdir/'scaling_ablation.png',dpi=220);fig.savefig(figdir/'scaling_ablation.svg');plt.close(fig)
 if campaign=='ag_news':
  figdir=root/'figures';figdir.mkdir(exist_ok=True)
  for metric in ('test_accuracy','test_macro_f1'):
   fig,ax=plt.subplots(figsize=(6.5,4));methods=('lora','symmetric','combined');vals=[[r[metric] for r in summaries if r['method']==m and r['expected_adapter_parameters']==6144] for m in methods];means=[statistics.mean(v) for v in vals];sds=[statistics.stdev(v) for v in vals];ax.bar(methods,means,yerr=sds,capsize=4);[ax.scatter([i]*len(v),v,color='black',s=15) for i,v in enumerate(vals)];ax.set(ylabel=metric.replace('_',' '),title='Controlled AG News matched-budget comparison');fig.tight_layout();fig.savefig(figdir/f'ag_news_{metric}.png',dpi=220);fig.savefig(figdir/f'ag_news_{metric}.svg');plt.close(fig)
 if campaign=='benchmark':
  figdir=root/'figures';figdir.mkdir(exist_ok=True)
  for metric,name in (('forward_ms_mean','forward_runtime_comparison'),('backward_ms_mean','backward_runtime_comparison'),('peak_allocated_vram_mib','vram_comparison'),('tokens_per_second_mean','throughput_comparison')):
   fig,ax=plt.subplots(figsize=(7,4));labels=[f"{r['method']}\n{r['expected_trainable_parameters']}" for r in summaries];ax.bar(labels,[r[metric] for r in summaries]);ax.set(ylabel=metric.replace('_',' '));fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=220);fig.savefig(figdir/f'{name}.svg');plt.close(fig)
 if campaign=='mechanism':
  metrics=base/'checkpoint_analysis'/'metrics.csv';rows=list(csv.DictReader(metrics.open()));figdir=root/'figures';figdir.mkdir(exist_ok=True)
  for field,name,title in (('jacobian_effective_rank_mean','effective_rank','Local Jacobian effective rank'),('branch_cosine_mean','combined_branch_cosine_similarity','Combined branch cosine similarity')):
   chosen=[r for r in rows if r.get(field) not in ('',None)];fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(chosen)),[float(r[field]) for r in chosen]);ax.set(ylabel=title,xlabel='Checkpoint');fig.tight_layout();fig.savefig(figdir/f'{name}.png',dpi=220);fig.savefig(figdir/f'{name}.svg');plt.close(fig)
  chosen=[r for r in rows if r['method']=='combined'];fig,ax=plt.subplots(figsize=(8,4));x=range(len(chosen));ax.plot(x,[float(r['lora_output_norm_mean']) for r in chosen],marker='o',label='LoRA branch');ax.plot(x,[float(r['symmetric_output_norm_mean']) for r in chosen],marker='o',label='Symmetric branch');ax.legend();ax.set(ylabel='Mean output norm',xlabel='Combined checkpoint');fig.tight_layout();fig.savefig(figdir/'combined_branch_norms.png',dpi=220);fig.savefig(figdir/'combined_branch_norms.svg');plt.close(fig)
  fig,ax=plt.subplots(figsize=(8,4));ax.bar(range(len(chosen)),[float(r['without_lora_test_degradation']) for r in chosen],label='disable LoRA');ax.bar(range(len(chosen)),[float(r['without_symmetric_test_degradation']) for r in chosen],bottom=[float(r['without_lora_test_degradation']) for r in chosen],label='disable Symmetric');ax.legend();ax.set(ylabel='Test-loss degradation',xlabel='Combined checkpoint');fig.tight_layout();fig.savefig(figdir/'branch_ablation_loss_changes.png',dpi=220);fig.savefig(figdir/'branch_ablation_loss_changes.svg');plt.close(fig)
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',required=True);a=p.parse_args();main(a.root,a.campaign)
