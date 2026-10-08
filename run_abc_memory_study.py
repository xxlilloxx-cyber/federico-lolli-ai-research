#!/usr/bin/env python3
"""One-command, restart-safe launcher for the A→B→C memory study."""
from __future__ import annotations
import argparse,contextlib,datetime as dt,json,multiprocessing as mp,platform,shutil,statistics,subprocess,sys,threading,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path

ROOT=Path(__file__).resolve().parent
EXP=ROOT/'experiments'/'factual_learning_lora_symmetric'
sys.path[:0]=[str(ROOT),str(EXP)]
from abc_memory import METHODS,PROTOCOLS,aggregate,run_one


def parse():
 p=argparse.ArgumentParser()
 p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--all',action='store_true')
 p.add_argument('--campaign',choices=('principal','ewc_lambda_sweep'),default='principal')
 p.add_argument('--protocol',choices=PROTOCOLS);p.add_argument('--method',choices=METHODS)
 p.add_argument('--device',choices=('auto','cuda','cpu'),default='auto')
 p.add_argument('--micro-batch-size',type=int,default=4);p.add_argument('--gradient-accumulation',type=int,default=1)
 p.add_argument('--parallel-runs',type=int,choices=(1,2),default=1);p.add_argument('--eval-interval',type=int,default=250)
 p.add_argument('--amp',choices=('auto','fp16','bf16','off'),default='auto')
 p.add_argument('--ewc-lambda',type=float,default=100.0);p.add_argument('--ewc-gamma',type=float,default=1.0)
 p.add_argument('--fisher-samples',type=int,default=12);p.add_argument('--fisher-batches',type=int,default=12)
 p.add_argument('--performance-benchmark',action='store_true',help=argparse.SUPPRESS)
 return p.parse_args()


def _device_and_amp(args):
 import torch
 device='cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
 if device=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 if args.amp=='auto':amp='fp16' if device=='cuda' else 'off'
 else:amp=args.amp
 if device!='cuda' and amp!='off':raise ValueError('fp16/bf16 AMP requires CUDA')
 if amp=='bf16' and not torch.cuda.is_bf16_supported():raise ValueError('BF16 is not supported by this GPU')
 return device,amp


def _base_config(args,smoke=False):
 if args.micro_batch_size*args.gradient_accumulation!=4:
  raise ValueError('micro-batch-size * gradient-accumulation must equal the fixed effective batch size 4')
 return {'model':'gpt2','dataset_seed':20261002,'facts_per_set':2 if smoke else 12,
         'steps_per_phase':2 if smoke else 2500,'evaluation_every_steps':2 if smoke else args.eval_interval,
         'sequence_length':64,'micro_batch_size':args.micro_batch_size,
         'gradient_accumulation':args.gradient_accumulation,'effective_batch_size':4,
         'learning_rate':3e-4,'weight_decay':0.0,'gradient_clipping':1.0,'scheduler':'none',
         'placement':'transformer.h[0].attn.c_proj','pairwise_slot_evaluation':False,'amp':args._resolved_amp,
         'ewc_lambda':args.ewc_lambda,'ewc_gamma':args.ewc_gamma,
         'fisher_samples':min(args.fisher_samples,6 if smoke else 36),'fisher_batches':args.fisher_batches}


def _complete(path):
 try:return json.loads(path.read_text()).get('complete') is True
 except Exception:return False


def _prepare_run_dir(run_dir):
 if _complete(run_dir/'summary.json'):return True
 if run_dir.exists() and any(run_dir.iterdir()):
  archive=run_dir.parents[3]/'incomplete_attempts'/f'{run_dir.parents[1].name}__{run_dir.parent.name}__{run_dir.name}__{dt.datetime.now().strftime("%Y%m%dT%H%M%S")}'
  archive.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(run_dir),str(archive))
 run_dir.mkdir(parents=True,exist_ok=True);return False


def _worker(spec):
 config,method,protocol,seed,run_dir,index,total=spec;run_dir=Path(run_dir)
 with (run_dir/'log.txt').open('w',buffering=1) as log,contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
  result=run_one(config,method,protocol,seed,run_dir,config['device'],
                 lambda phase:print(f'[{index}/{total}] PHASE {phase} complete {protocol}/{method}/seed{seed}',file=sys.__stdout__,flush=True))
 return str(run_dir),result


class GPUSampler:
 def __init__(self):self.values=[];self.stop=threading.Event();self.thread=None
 def __enter__(self):
  def sample():
   while not self.stop.wait(.2):
    try:self.values.append(float(subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines()[0]))
    except Exception:pass
  self.thread=threading.Thread(target=sample,daemon=True);self.thread.start();return self
 def __exit__(self,*_):self.stop.set();self.thread.join(timeout=2)
 @property
 def mean(self):return statistics.mean(self.values) if self.values else None


def performance_benchmark(args,device):
 if device!='cuda':raise RuntimeError('the requested GPU performance benchmark requires CUDA')
 out=ROOT/'results_abc_memory'/'performance_benchmark';out.mkdir(parents=True,exist_ok=True)
 rows=[]
 for micro,accum in ((1,4),(2,2),(4,1)):
  bench=argparse.Namespace(**vars(args));bench.micro_batch_size=micro;bench.gradient_accumulation=accum
  cfg=_base_config(bench,True);cfg.update({'steps_per_phase':5,'evaluation_every_steps':5,'device':device})
  run_dir=out/f'micro{micro}_accum{accum}'
  if run_dir.exists():shutil.rmtree(run_dir)
  with GPUSampler() as sampler:
   started=time.perf_counter();summary=run_one(cfg,'lora','sequential_single',42,run_dir,device);wall=time.perf_counter()-started
  rows.append({'micro_batch_size':micro,'gradient_accumulation':accum,'effective_batch_size':4,
               'optimizer_steps_per_second':summary['optimizer_steps_per_second'],
               'tokens_per_second':summary['training_tokens_per_second'],'peak_gpu_memory_mb':summary['peak_gpu_memory_mb'],
               'mean_gpu_utilization_percent':sampler.mean,'wall_seconds':wall})
 # Equal total work: two six-step runs sequentially, then the same two cells concurrently.
 concurrency=[]
 cfg=_base_config(args,True);cfg.update({'steps_per_phase':2,'evaluation_every_steps':2,'device':device})
 for workers in (1,2):
  group=out/f'parallel_{workers}';
  if group.exists():shutil.rmtree(group)
  specs=[]
  for j,seed in enumerate((9101,9102),1):
   d=group/f'seed{seed}';d.mkdir(parents=True);specs.append((cfg,'lora','sequential_single',seed,str(d),j,2))
  with GPUSampler() as sampler:
   started=time.perf_counter()
   if workers==1:
    for spec in specs:_worker(spec)
   else:
    with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn')) as pool:
     list(pool.map(_worker,specs))
   wall=time.perf_counter()-started
  summaries=[json.loads((group/f'seed{s}'/'summary.json').read_text()) for s in (9101,9102)]
  concurrency.append({'parallel_runs':workers,'total_optimizer_steps':12,'wall_seconds':wall,
                      'total_completed_optimizer_steps_per_second':12/wall,
                      'sum_training_tokens_per_second':sum(s['training_tokens_per_second'] for s in summaries),
                      'max_peak_gpu_memory_mb':max(s['peak_gpu_memory_mb'] for s in summaries),
                      'mean_gpu_utilization_percent':sampler.mean})
 payload={'microbatch':rows,'parallel':concurrency,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total','--format=csv,noheader'],text=True).strip()}
 (out/'benchmark_results.json').write_text(json.dumps(payload,indent=2)+'\n')
 best=max(rows,key=lambda x:x['optimizer_steps_per_second']);p1,p2=concurrency;parallel_speedup=p2['total_completed_optimizer_steps_per_second']/p1['total_completed_optimizer_steps_per_second']
 report=f'''# Performance optimization report\n\n## Protocol\n\nAll microbatch candidates used effective batch size 4 and the same short LoRA workload. Measurements exclude full campaign training. GPU utilization was sampled with `nvidia-smi` and is approximate.\n\n| Microbatch | Accumulation | Effective batch | Optimizer steps/s | Tokens/s | Peak GPU MiB | Mean GPU util. | Wall s |\n|---:|---:|---:|---:|---:|---:|---:|---:|\n'''
 for x in rows:report+=f"| {x['micro_batch_size']} | {x['gradient_accumulation']} | 4 | {x['optimizer_steps_per_second']:.3f} | {x['tokens_per_second']:.1f} | {x['peak_gpu_memory_mb']:.1f} | {x['mean_gpu_utilization_percent'] if x['mean_gpu_utilization_percent'] is not None else 'N/A'} | {x['wall_seconds']:.2f} |\n"
 report+=f'''\nSelected candidate: **microbatch {best['micro_batch_size']} / accumulation {best['gradient_accumulation']}**, based on the highest finite optimizer throughput while fitting in memory. The historical configuration was microbatch 1 / accumulation 4.\n\n## Independent-run concurrency\n\n| Parallel runs | Total steps/s | Peak per-process GPU MiB | Mean GPU util. | Wall s |\n|---:|---:|---:|---:|---:|\n'''
 for x in concurrency:report+=f"| {x['parallel_runs']} | {x['total_completed_optimizer_steps_per_second']:.3f} | {x['max_peak_gpu_memory_mb']:.1f} | {x['mean_gpu_utilization_percent'] if x['mean_gpu_utilization_percent'] is not None else 'N/A'} | {x['wall_seconds']:.2f} |\n"
 report+=f"\nObserved N=2 whole-workload speedup: **{parallel_speedup:.2f}×**. Parallel execution remains opt-in; the default is 1.\n"
 (ROOT/'PERFORMANCE_OPTIMIZATION_REPORT.md').write_text(report)
 print(json.dumps(payload,indent=2));return best,parallel_speedup


def main():
 args=parse();device,amp=_device_and_amp(args);args._resolved_amp=amp
 if args.performance_benchmark:performance_benchmark(args,device);return
 config=_base_config(args,args.smoke);config['device']=device
 methods=(args.method,) if args.method else METHODS
 if args.campaign=='ewc_lambda_sweep':
  if args.protocol and args.protocol!='ewc_like':raise ValueError('lambda sweep supports only ewc_like')
  protocols=('ewc_like',);seeds=(42,);lambdas=(0.0,1.0,10.0,100.0,1000.0)
 else:
  protocols=(args.protocol,) if args.protocol else PROTOCOLS;seeds=(42,) if args.smoke else (42,123,456);lambdas=(args.ewc_lambda,)
 root=ROOT/'results_abc_memory'/('smoke_ewc' if args.smoke else '')
 plan=[]
 for lam in lambdas:
  for protocol in protocols:
   for method in methods:
    for seed in seeds:
     cfg=dict(config);cfg['ewc_lambda']=lam
     if args.campaign=='ewc_lambda_sweep':run_dir=root/'ewc_lambda_sweep'/f'lambda_{lam:g}'/method/f'seed{seed}'
     else:run_dir=root/'runs'/protocol/method/f'seed{seed}'
     plan.append((cfg,method,protocol,seed,run_dir))
 historical=[]
 for path in (ROOT/'results_factual_learning'/'principal_matched_budget').glob('*_seed*/summary.json'):
  try:
   row=json.loads(path.read_text());
   if row.get('condition')!='frozen':historical.append(float(row['seconds_per_optimizer_step']))
  except Exception:pass
 base=statistics.median(historical) if historical else None;steps=sum(3*c['steps_per_phase'] for c,*_ in plan)
 pending_count=sum(not _complete(d/'summary.json') for *_,d in plan)
 pending_steps=sum(3*c['steps_per_phase'] for c,m,p,s,d in plan if not _complete(d/'summary.json'))
 hours=pending_steps*base/3600/args.parallel_runs if base else None
 print(json.dumps({'device':device,'amp':amp,'selected_runs':len(plan),'pending_runs':pending_count,
                   'principal_matrix_runs':36,'optimizer_steps':steps,'pending_optimizer_steps':pending_steps,
                   'effective_batch_size':4,'parameters_per_slot':6144,
                   'final_parameters':{'sequential_single':6144,'ewc_like':6144,'grow_unfrozen':18432,'hard_consolidation':18432},
                   'estimated_hours_from_prior_rate':hours,'result_root':str(root)},indent=2))
 for cfg,m,p,s,d in plan:print(f'{p:20} {m:9} seed={s} steps={3*cfg["steps_per_phase"]} lambda={cfg["ewc_lambda"]:g} '+('COMPLETE' if _complete(d/'summary.json') else 'PENDING'))
 if args.dry_run:return
 pending=[];done=0
 for index,(cfg,method,protocol,seed,run_dir) in enumerate(plan,1):
  if _prepare_run_dir(run_dir):done+=1;print(f'[{index}/{len(plan)}] SKIP {protocol}/{method}/seed{seed}',flush=True);continue
  pending.append((cfg,method,protocol,seed,str(run_dir),index,len(plan)))
 started=time.time()
 try:
  if args.parallel_runs==1:
   for spec in pending:
    print(f'[{spec[5]}/{len(plan)}] START {spec[2]}/{spec[1]}/seed{spec[3]}',flush=True)
    _worker(spec);done+=1;print(f'[{spec[5]}/{len(plan)}] COMPLETE {spec[2]}/{spec[1]}/seed{spec[3]}',flush=True)
  else:
   with ProcessPoolExecutor(max_workers=args.parallel_runs,mp_context=mp.get_context('spawn')) as pool:
    futures={}
    for spec in pending:
     print(f'[{spec[5]}/{len(plan)}] START {spec[2]}/{spec[1]}/seed{spec[3]}',flush=True)
     futures[pool.submit(_worker,spec)]=spec
    for future in as_completed(futures):
     spec=futures[future]
     try:future.result();done+=1;print(f'[{spec[5]}/{len(plan)}] COMPLETE {spec[2]}/{spec[1]}/seed{spec[3]}',flush=True)
     except Exception as exc:print(f'ERROR {spec[2]}/{spec[1]}/seed{spec[3]}: {exc}',file=sys.stderr,flush=True);raise
 except Exception:
  spec=locals().get('spec');
  if spec:
   d=Path(spec[4]);(d/'failure.json').write_text(json.dumps({'complete':False,'traceback':traceback.format_exc(),'time_utc':dt.datetime.now(dt.timezone.utc).isoformat()},indent=2)+'\n')
  raise
 aggregate(root)
 principal_complete=sum(_complete(root/'runs'/p/m/f'seed{s}'/'summary.json') for p in PROTOCOLS for m in METHODS for s in (42,123,456))
 payload={'complete':principal_complete==36,'completed_principal_runs':principal_complete,'planned_principal_runs':36,
          'selected_runs_completed':done,'selected_runs':len(plan),'elapsed_seconds':time.time()-started,
          'python':platform.python_version(),'device':device,'amp':amp}
 name='CAMPAIGN_SUMMARY.json' if payload['complete'] else 'CAMPAIGN_SUMMARY_LATEST.json'
 (root/name).write_text(json.dumps(payload,indent=2)+'\n');print(f'COMPLETE {done}/{len(plan)} principal={principal_complete}/36 results={root}',flush=True)


if __name__=='__main__':main()
