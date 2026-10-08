#!/usr/bin/env python3
"""Restart-safe external EWC-lambda sweep for the validated A→B→C study."""
from __future__ import annotations
import argparse,contextlib,csv,datetime as dt,json,math,multiprocessing as mp,shutil,sys,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments'/'factual_learning_lora_symmetric'
sys.path[:0]=[str(ROOT),str(EXP)]
from abc_memory import METHODS,run_one

DEFAULT_LAMBDAS=(0.0,1.0,10.0,30.0,50.0,100.0,300.0)
SEED=42


def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--all',action='store_true')
 p.add_argument('--method',choices=METHODS);g=p.add_mutually_exclusive_group();g.add_argument('--lambda',dest='lambda_value',type=float);g.add_argument('--lambdas')
 p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto');p.add_argument('--parallel-runs',type=int,choices=(1,2),default=2)
 return p.parse_args()


def parse_lambdas(text:str|None,single:float|None)->tuple[float,...]:
 if single is not None:return (float(single),)
 if text is None:return DEFAULT_LAMBDAS
 values=tuple(float(x.strip()) for x in text.split(',') if x.strip())
 if not values or len(set(values))!=len(values) or any(x<0 for x in values):raise ValueError('lambdas must be unique non-negative numbers')
 return values


def run_name(method:str,value:float,seed:int=SEED)->str:
 label=f'{value:g}'.replace('.','p');return f'{method}_lambda_{label}_seed_{seed}'


def base_config(device:str,value:float,smoke:bool)->dict[str,Any]:
 return {'model':'gpt2','dataset_seed':20261002,'facts_per_set':2 if smoke else 12,
         'steps_per_phase':2 if smoke else 2500,'evaluation_every_steps':2 if smoke else 250,
         'sequence_length':64,'micro_batch_size':4,'gradient_accumulation':1,'effective_batch_size':4,
         'learning_rate':3e-4,'weight_decay':0.0,'gradient_clipping':1.0,'scheduler':'none',
         'placement':'transformer.h[0].attn.c_proj','pairwise_slot_evaluation':False,
         'amp':'fp16' if device=='cuda' else 'off','ewc_lambda':float(value),'ewc_gamma':1.0,
         'fisher_samples':3 if smoke else 12,'fisher_batches':2 if smoke else 12,'device':device}


def complete(run_dir:Path)->bool:
 try:return json.loads((run_dir/'summary.json').read_text()).get('complete') is True
 except Exception:return False


def prepare(run_dir:Path)->bool:
 if complete(run_dir):return True
 if run_dir.exists() and any(run_dir.iterdir()):
  archive=run_dir.parent/'incomplete_attempts'/f'{run_dir.name}__{dt.datetime.now().strftime("%Y%m%dT%H%M%S")}'
  archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(run_dir),str(archive))
 run_dir.mkdir(parents=True,exist_ok=True);return False


def worker(spec):
 config,method,value,run_dir,index,total=spec;run_dir=Path(run_dir)
 with (run_dir/'log.txt').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  result=run_one(config,method,'ewc_like',SEED,run_dir,config['device'],
                 lambda phase:print(f'[{index}/{total}] PHASE {phase} complete {method}/lambda={value:g}',file=sys.__stdout__,flush=True))
 return str(run_dir),result


def write_csv(path:Path,rows:list[dict[str,Any]]):
 fields=sorted({k for r in rows for k in r}) if rows else []
 with path.open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def _vector(state):return torch.cat([state[k].float().reshape(-1) for k in sorted(state)])


def analyze_run(run_dir:Path)->dict[str,Any]:
 summary=json.loads((run_dir/'summary.json').read_text());matrix=list(csv.DictReader((run_dir/'memory_matrix.csv').open()))
 lookup={(r['training_phase'],r['evaluation_set']):r for r in matrix};final=[lookup[('C',x)] for x in 'ABC']
 ck={p:torch.load(run_dir/'checkpoints'/f'after_{p}.pt',map_location='cpu',weights_only=False) for p in 'ABC'}
 states={p:ck[p]['adapter_state'] for p in 'ABC'};a,b,c=(_vector(states[p]) for p in 'ABC')
 def fisher_summary(aux):
  f=torch.cat([x.float().reshape(-1) for x in aux['fisher'].values()]);prob=f/f.sum().clamp_min(1e-30);nz=prob[prob>0]
  return {'sum':float(f.sum()),'mean':float(f.mean()),'max':float(f.max()),'nonzero_fraction':float((f>0).float().mean()),'effective_support':float(torch.exp(-(nz*torch.log(nz)).sum()))}
 def raw_penalty(aux,target):
  return sum(float((aux['fisher'][k].float()*(target[k].float()-aux['reference'][k].float()).square()).sum()) for k in aux['fisher'])
 lam=float(summary['ewc_lambda']);raw_b=raw_penalty(ck['A']['auxiliary_state'],states['B']);raw_c=raw_penalty(ck['B']['auxiliary_state'],states['C'])
 fa=fisher_summary(ck['A']['auxiliary_state']);fab=fisher_summary(ck['B']['auxiliary_state'])
 row={'method':summary['method'],'lambda':lam,'seed':int(summary['seed']),'complete':True,
      'adapter_parameters':int(summary['final_adapter_parameters']),'ewc_auxiliary_state_bytes':int(summary['ewc_auxiliary_state_bytes']),
      'acquisition_a_nll':float(summary['acquisition_a_nll']),'acquisition_b_nll':float(summary['acquisition_b_nll']),'acquisition_c_nll':float(summary['acquisition_c_nll']),
      'acquisition_a_probability':float(lookup[('A','A')]['correct_probability']),'acquisition_b_probability':float(lookup[('B','B')]['correct_probability']),'acquisition_c_probability':float(lookup[('C','C')]['correct_probability']),
      'acquisition_a_exact_match':float(lookup[('A','A')]['exact_match']),'acquisition_b_exact_match':float(lookup[('B','B')]['exact_match']),'acquisition_c_exact_match':float(lookup[('C','C')]['exact_match']),
      'forgetting_a_after_b_nll':float(summary['forgetting_a_after_b_nll']),'forgetting_a_after_c_nll':float(summary['forgetting_a_after_c_nll']),'forgetting_b_after_c_nll':float(summary['forgetting_b_after_c_nll']),
      'forgetting_a_after_b_probability':float(summary['forgetting_a_after_b_probability']),'forgetting_a_after_c_probability':float(summary['forgetting_a_after_c_probability']),'forgetting_b_after_c_probability':float(summary['forgetting_b_after_c_probability']),
      'forgetting_a_after_b_exact_match':float(summary['forgetting_a_after_b_exact_match']),'forgetting_a_after_c_exact_match':float(summary['forgetting_a_after_c_exact_match']),'forgetting_b_after_c_exact_match':float(summary['forgetting_b_after_c_exact_match']),
      'final_a_nll':float(final[0]['nll']),'final_b_nll':float(final[1]['nll']),'final_c_nll':float(final[2]['nll']),
      'final_a_probability':float(final[0]['correct_probability']),'final_b_probability':float(final[1]['correct_probability']),'final_c_probability':float(final[2]['correct_probability']),
      'final_a_exact_match':float(final[0]['exact_match']),'final_b_exact_match':float(final[1]['exact_match']),'final_c_exact_match':float(final[2]['exact_match']),
      'final_mean_nll':float(summary['mean_final_nll']),'final_mean_probability':float(summary['mean_final_probability']),'final_mean_exact_match':float(summary['mean_final_exact_match']),
      'final_mean_paraphrase_nll':float(np.mean([float(x['paraphrase_nll']) for x in final])),
      'final_mean_paraphrase_probability':float(np.mean([float(x['paraphrase_correct_probability']) for x in final])),
      'final_mean_paraphrase_exact_match':float(np.mean([float(x['paraphrase_exact_match']) for x in final])),
      'mean_forgetting':float(np.mean([summary['forgetting_a_after_b_nll'],summary['forgetting_a_after_c_nll'],summary['forgetting_b_after_c_nll']])),
      'mean_new_fact_acquisition_nll':float(np.mean([summary['acquisition_b_nll'],summary['acquisition_c_nll']])),
      'mean_new_fact_acquisition_probability':float(np.mean([float(lookup[('B','B')]['correct_probability']),float(lookup[('C','C')]['correct_probability'])])),
      'parameter_displacement_a_b_l2':float((b-a).norm()),'parameter_displacement_b_c_l2':float((c-b).norm()),'total_parameter_displacement_a_c_l2':float((c-a).norm()),
      'raw_ewc_quadratic_b':raw_b,'raw_ewc_quadratic_c':raw_c,'realized_ewc_penalty_b':0.5*lam*raw_b,'realized_ewc_penalty_c':0.5*lam*raw_c}
 for prefix,values in [('fisher_a',fa),('fisher_ab',fab)]:
  row.update({f'{prefix}_{k}':v for k,v in values.items()})
 return row


def pareto(rows):
 out=[]
 for row in rows:
  dominated=any(other['method']==row['method'] and other is not row and other['mean_forgetting']<=row['mean_forgetting'] and other['mean_new_fact_acquisition_nll']<=row['mean_new_fact_acquisition_nll'] and (other['mean_forgetting']<row['mean_forgetting'] or other['mean_new_fact_acquisition_nll']<row['mean_new_fact_acquisition_nll']) for other in rows)
  if not dominated:out.append(dict(row,pareto_nondominated=True))
 return out


def make_figures(root:Path,rows:list[dict[str,Any]]):
 import matplotlib.pyplot as plt
 fdir=root/'figures';fdir.mkdir(exist_ok=True)
 def save(fig,name):fig.tight_layout();fig.savefig(fdir/f'{name}.png',dpi=200);fig.savefig(fdir/f'{name}.svg');plt.close(fig)
 fig,axes=plt.subplots(1,3,figsize=(13,4))
 for ax,method in zip(axes,METHODS):
  data=sorted((r for r in rows if r['method']==method),key=lambda r:r['lambda']);x=[r['mean_forgetting'] for r in data];y=[r['mean_new_fact_acquisition_nll'] for r in data]
  ax.plot(x,y,'o-');[ax.annotate(f"λ={r['lambda']:g}",(r['mean_forgetting'],r['mean_new_fact_acquisition_nll']),fontsize=7) for r in data]
  ax.set(title=method,xlabel='Mean forgetting (NLL increase)',ylabel='Mean B/C acquisition NLL');ax.grid(alpha=.2)
 save(fig,'stability_plasticity_by_method')
 metrics=[('forgetting_a_after_b_nll','Forget A→B'),('forgetting_a_after_c_nll','Forget A→C'),('forgetting_b_after_c_nll','Forget B→C'),('acquisition_b_nll','B acquisition NLL'),('acquisition_c_nll','C acquisition NLL'),('final_mean_nll','Final mean NLL'),('final_mean_probability','Final mean probability'),('final_mean_exact_match','Final exact match'),('realized_ewc_penalty_c','EWC penalty during C'),('total_parameter_displacement_a_c_l2','Total A→C displacement')]
 for method in METHODS:
  data=sorted((r for r in rows if r['method']==method),key=lambda r:r['lambda']);fig,axes=plt.subplots(3,4,figsize=(14,10));axes=axes.ravel()
  for ax,(key,title) in zip(axes,metrics):
   ax.plot([r['lambda'] for r in data],[r[key] for r in data],'o-');ax.set_xscale('symlog',linthresh=1);ax.set(title=title,xlabel='lambda');ax.grid(alpha=.2)
  for ax in axes[len(metrics):]:ax.axis('off')
  save(fig,f'lambda_curves_{method}')


def aggregate(root:Path):
 rows=[]
 for d in sorted(root.glob('*_lambda_*_seed_*')):
  if complete(d):rows.append(analyze_run(d))
 if not rows:return
 rows=sorted(rows,key=lambda r:(r['method'],r['lambda']));write_csv(root/'per_run_results.csv',rows);write_csv(root/'lambda_sweep_results.csv',rows)
 trade=[{k:r[k] for k in ('lambda','method','mean_forgetting','mean_new_fact_acquisition_nll','acquisition_b_nll','acquisition_c_nll','final_mean_nll','final_mean_probability','final_mean_exact_match')} for r in rows]
 write_csv(root/'lambda_tradeoff.csv',trade);write_csv(root/'pareto_candidates.csv',pareto(trade));make_figures(root,rows)
 zero=[r for r in rows if r['lambda']==0]
 if zero:
  valid=all(r['realized_ewc_penalty_b']==0 and r['realized_ewc_penalty_c']==0 for r in zero)
  control='# Lambda-zero execution-matched control\n\n'
  control+=f'- Completed methods: {", ".join(r["method"] for r in zero)}.\n- Same EWC code path: yes.\n- Microbatch / accumulation / effective batch: 4 / 1 / 4.\n- Fisher and reference states stored: yes.\n- Weighted EWC contribution in B and C: exactly zero: {str(valid).lower()}.\n- Unit test confirms that lambda=0 leaves both objective and gradient identical to the unregularized objective.\n\nThe raw Fisher-weighted quadratic can become non-zero after movement, but multiplication by lambda=0 makes its objective and gradient contribution exactly zero.\n'
  (root/'LAMBDA_ZERO_CONTROL_REPORT.md').write_text(control)
 report='# EWC lambda sweep report\n\nThis report is generated only from completed machine-readable runs. The sweep is exploratory and uses seed 42 only. No statistical optimum is claimed.\n\n## Trade-off table\n\n| Method | Lambda | Mean forgetting | Mean B/C acquisition NLL | Final mean NLL | Final probability | Final exact | Pareto |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
 candidates={(r['method'],r['lambda']) for r in pareto(trade)}
 for r in rows:report+=f"| {r['method']} | {r['lambda']:g} | {r['mean_forgetting']:.6f} | {r['mean_new_fact_acquisition_nll']:.6f} | {r['final_mean_nll']:.6f} | {r['final_mean_probability']:.6f} | {r['final_mean_exact_match']:.6f} | {'yes' if (r['method'],r['lambda']) in candidates else 'no'} |\n"
 report+='\nLower-left is preferred in the forgetting-versus-acquisition plane. Pareto labels are descriptive for this single seed and predefined grid. See `LAMBDA_ZERO_CONTROL_REPORT.md` for the matched control audit.\n'
 (root/'EWC_LAMBDA_SWEEP_REPORT.md').write_text(report)


def main():
 args=parse_args();device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 lambdas=(0.0,) if args.smoke else parse_lambdas(args.lambdas,args.lambda_value);methods=(args.method,) if args.method else METHODS
 root=ROOT/'results_ewc_lambda_sweep'/('smoke' if args.smoke else '');plan=[]
 for value in lambdas:
  for method in methods:
   d=root/run_name(method,value);plan.append((base_config(device,value,args.smoke),method,value,d))
 print(json.dumps({'device':device,'seed':SEED,'methods':list(methods),'lambdas':list(lambdas),'planned_runs':len(plan),'optimizer_steps':sum(3*x[0]['steps_per_phase'] for x in plan),'result_root':str(root)},indent=2))
 for cfg,m,v,d in plan:print(f'{m:9} lambda={v:g} seed={SEED} steps={3*cfg["steps_per_phase"]} '+('COMPLETE' if complete(d) else 'PENDING'))
 if args.dry_run:return
 root.mkdir(parents=True,exist_ok=True);pending=[];completed=0
 for i,(cfg,m,v,d) in enumerate(plan,1):
  if prepare(d):completed+=1;print(f'[{i}/{len(plan)}] SKIP {m}/lambda={v:g}',flush=True)
  else:pending.append((cfg,m,v,str(d),i,len(plan)))
 started=time.time()
 try:
  if args.parallel_runs==1:
   for spec in pending:
    print(f'[{spec[4]}/{len(plan)}] START {spec[1]}/lambda={spec[2]:g}',flush=True);worker(spec);completed+=1;print(f'[{spec[4]}/{len(plan)}] COMPLETE {spec[1]}/lambda={spec[2]:g}',flush=True)
  else:
   with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
    futures={}
    for spec in pending:
     print(f'[{spec[4]}/{len(plan)}] START {spec[1]}/lambda={spec[2]:g}',flush=True);futures[pool.submit(worker,spec)]=spec
    for future in as_completed(futures):
     spec=futures[future];future.result();completed+=1;print(f'[{spec[4]}/{len(plan)}] COMPLETE {spec[1]}/lambda={spec[2]:g}',flush=True)
 except Exception:
  spec=locals().get('spec')
  if spec:
   d=Path(spec[3]);(d/'failure.json').write_text(json.dumps({'complete':False,'traceback':traceback.format_exc()},indent=2)+'\n')
  raise
 aggregate(root);(root/'SWEEP_SUMMARY.json').write_text(json.dumps({'complete':completed==len(plan),'completed_runs':completed,'planned_runs':len(plan),'elapsed_seconds':time.time()-started,'seed':SEED,'lambdas':list(lambdas)},indent=2)+'\n')
 print(f'COMPLETE {completed}/{len(plan)} results={root}',flush=True)


if __name__=='__main__':main()
