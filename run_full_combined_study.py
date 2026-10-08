#!/usr/bin/env python3
"""One-command, resume-safe orchestrator for the LoRA/Symmetric/Combined study."""
from __future__ import annotations
import argparse,csv,datetime as dt,json,os,platform,shutil,statistics,subprocess,sys,time,traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parent
DEFAULT_CONFIG=ROOT/'config'/'full_combined_study.json'
EXP=ROOT/'experiments'/'combined_extension'

@dataclass
class Cell:
 campaign:str;name:str;runner:str;config:dict[str,Any];steps:int=0

def adapter_count(method,rank=None,rank_lora=None,rank_symmetric=None,layers=(0,),projections=('attn.c_proj',)):
 dims={'attn.c_proj':1536,'mlp.c_fc':3840,'mlp.c_proj':3840};per_rank=sum(dims[p] for _ in layers for p in projections)
 if method in ('lora','symmetric'):return per_rank*rank
 return per_rank*(rank_lora+rank_symmetric)

def base_cfg(master,device):
 c=master['controlled'];return {'model':master['model'],'alpha':master['alpha'],'device':device,'sequence_length':c['sequence_length'],'batch_size':c['batch_size'],'gradient_accumulation':c['gradient_accumulation'],'learning_rate':c['learning_rate'],'weight_decay':c['weight_decay'],'gradient_clipping':c['gradient_clipping'],'scheduler':c['scheduler'],'evaluation_every_steps':c['evaluation_every_steps'],'checkpoint_steps':c['checkpoint_steps'],'validation_blocks':8}

def method_cfg(method,total_rank):
 if method=='combined':return {'method':'combined','rank_lora':total_rank//2,'rank_symmetric':total_rank//2,'scaling_mode':'total_rank'}
 return {'method':method,'rank':total_rank}

def build_plan(master,device):
 cells=[];base=base_cfg(master,device);seeds=master['seeds'];steps=master['controlled']['steps']
 for method,ranks in (('lora',(1,2,4,8)),('symmetric',(1,2,4,8)),('combined',(2,4,8))):
  for total_rank in ranks:
   mc=method_cfg(method,total_rank)
   for seed in seeds:
    name=f"{method}_r{total_rank}_seed{seed}";cfg={**base,**mc,'seed':seed,'steps':steps,'layers':[0],'projections':['attn.c_proj'],'expected_trainable_parameters':1536*total_rank}
    cells.append(Cell('rank_confirmation',name,'run_wikitext_cell.py',cfg,steps))
 sc=master['scaling']
 for method in ('lora','symmetric'):
  for rank in (1,2,4,8):
   for policy,alpha in (('fixed_alpha',4.0),('constant_scale',float(rank))):
    for seed in sc['seeds']:
     cfg={**base,'method':method,'rank':rank,'alpha':alpha,'scaling_policy':policy,'seed':seed,'steps':sc['steps'],'evaluation_every_steps':sc['evaluation_every_steps'],'checkpoint_steps':[sc['steps']],'layers':[0],'projections':['attn.c_proj'],'expected_trainable_parameters':1536*rank}
     cells.append(Cell('scaling',f'{method}_r{rank}_{policy}_seed{seed}','run_wikitext_cell.py',cfg,sc['steps']))
 for pair in (1,2,4):
  for mode in ('total_rank','branch_wise'):
   for seed in sc['seeds']:
    cfg={**base,'method':'combined','rank_lora':pair,'rank_symmetric':pair,'alpha':4.0,'scaling_mode':mode,'scaling_policy':mode,'seed':seed,'steps':sc['steps'],'evaluation_every_steps':sc['evaluation_every_steps'],'checkpoint_steps':[sc['steps']],'layers':[0],'projections':['attn.c_proj'],'expected_trainable_parameters':3072*pair}
    cells.append(Cell('scaling',f'combined_{pair}p{pair}_{mode}_seed{seed}','run_wikitext_cell.py',cfg,sc['steps']))
 pl=master['placement']
 for block in range(12):
  for method in ('lora','symmetric','combined'):
   cfg={**base,**method_cfg(method,4),'seed':pl['seeds'][0],'steps':pl['steps'],'evaluation_every_steps':pl['steps'],'checkpoint_steps':[pl['steps']],'layers':[block],'projections':['attn.c_proj'],'expected_trainable_parameters':6144,'scientific_label':'EXPLORATORY'}
   cells.append(Cell('placement',f'block{block}_{method}', 'run_wikitext_cell.py',cfg,pl['steps']))
 if pl.get('long_confirmation'):
  for block in pl.get('selected_blocks',[]):
   for method in ('lora','symmetric','combined'):
    cfg={**base,**method_cfg(method,4),'seed':pl['seeds'][0],'steps':pl['long_steps'],'layers':[block],'projections':['attn.c_proj'],'expected_trainable_parameters':6144,'scientific_label':'CONTROLLED SELECTED-BLOCK CONFIRMATION'}
    cells.append(Cell('placement',f'long_block{block}_{method}','run_wikitext_cell.py',cfg,pl['long_steps']))
 am=master['attention_mlp'];placements=[('attn_c_proj',['attn.c_proj'],4,2),('mlp_c_fc',['mlp.c_fc'],4,2),('mlp_c_proj',['mlp.c_proj'],4,2),('mlp_both',['mlp.c_fc','mlp.c_proj'],2,1)]
 for label,projections,single_rank,pair_rank in placements:
  for method in ('lora','symmetric','combined'):
   mc={'method':method,'rank':single_rank} if method!='combined' else {'method':'combined','rank_lora':pair_rank,'rank_symmetric':pair_rank,'scaling_mode':'total_rank'}
   expected=adapter_count(method,single_rank,pair_rank,pair_rank,(0,),tuple(projections));cfg={**base,**mc,'seed':am['seeds'][0],'steps':am['steps'],'evaluation_every_steps':am['steps'],'checkpoint_steps':[am['steps']],'layers':[0],'projections':projections,'expected_trainable_parameters':expected,'scientific_label':'EXPLORATORY'}
   cells.append(Cell('attention_mlp',f'{label}_{method}','run_wikitext_cell.py',cfg,am['steps']))
 ml=master['multilayer'];geometries={6144:[('layer2',[2],4),('two',[0,11],2),('four',[0,3,7,11],1)],12288:[('layer2',[2],8),('two',[0,11],4),('four',[0,3,7,11],2)]}
 for budget,items in geometries.items():
  for label,layers,rank in items:
   for method in ('lora','symmetric'):
    cfg={**base,'method':method,'rank':rank,'seed':ml['seeds'][0],'steps':ml['steps'],'evaluation_every_steps':25,'checkpoint_steps':[ml['steps']],'layers':layers,'projections':['attn.c_proj'],'expected_trainable_parameters':budget,'budget':budget,'scientific_label':'EXPLORATORY'}
    cells.append(Cell('multilayer',f'budget{budget}_{label}_{method}','run_wikitext_cell.py',cfg,ml['steps']))
 for label,layers,pair in [('layer2',[2],4),('two',[0,11],2),('four',[0,3,7,11],1)]:
  cfg={**base,'method':'combined','rank_lora':pair,'rank_symmetric':pair,'scaling_mode':'total_rank','seed':ml['seeds'][0],'steps':ml['steps'],'evaluation_every_steps':25,'checkpoint_steps':[ml['steps']],'layers':layers,'projections':['attn.c_proj'],'expected_trainable_parameters':12288,'budget':12288,'scientific_label':'EXPLORATORY'}
  cells.append(Cell('multilayer',f'budget12288_{label}_combined','run_wikitext_cell.py',cfg,ml['steps']))
 ag=master['ag_news'];budgets=[(4,2,6144)]+([(8,4,12288)] if ag['optional_12288'] else [])
 for single,pair,budget in budgets:
  for method in ('lora','symmetric','combined'):
   for seed in seeds:
    mc={'method':method,'rank':single} if method!='combined' else {'method':'combined','rank_lora':pair,'rank_symmetric':pair,'scaling_mode':'total_rank'}
    cfg={**mc,'model':master['model'],'alpha':4.0,'device':device,'seed':seed,'steps':ag['steps'],'train_samples':ag['train_samples'],'validation_samples':ag['validation_samples'],'test_samples':ag['test_samples'],'max_length':ag['max_length'],'gradient_accumulation':4,'gradient_clipping':1.0,'learning_rate':3e-4,'evaluation_every_steps':ag['evaluation_every_steps'],'expected_adapter_parameters':budget,'scientific_label':'CONTROLLED'}
    cells.append(Cell('ag_news',f'budget{budget}_{method}_seed{seed}','run_agnews_cell.py',cfg,ag['steps']))
 bm=master['benchmark']
 for single,pair,budget in ((4,2,6144),(8,4,12288)):
  for method in ('lora','symmetric','combined'):
   mc={'method':method,'rank':single} if method!='combined' else {'method':'combined','rank_lora':pair,'rank_symmetric':pair}
   cfg={**mc,'model':master['model'],'alpha':4.0,'device':device,'expected_trainable_parameters':budget,**bm}
   cells.append(Cell('benchmark',f'budget{budget}_{method}','run_benchmark_cell.py',cfg,0))
 mechanism=master['mechanism'];cells.append(Cell('mechanism','checkpoint_analysis','run_mechanism.py',{'model':master['model'],'device':device,'rank_result_root':None,**mechanism},0))
 return cells

def environment(device):
 import torch,transformers,datasets
 selected='cuda' if device=='auto' and torch.cuda.is_available() else ('cpu' if device=='auto' else device)
 if selected=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 gpu=None;memory=None
 if torch.cuda.is_available():gpu=torch.cuda.get_device_name(0);memory=torch.cuda.get_device_properties(0).total_memory
 git=subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,text=True,capture_output=True,check=True).stdout.strip()
 return selected,{'python':platform.python_version(),'pytorch':torch.__version__,'transformers':transformers.__version__,'datasets':datasets.__version__,'cuda':torch.version.cuda,'gpu':gpu,'gpu_memory_bytes':memory,'git_commit':git}

def complete(run_dir):
 path=run_dir/'summary.json'
 try:return path.exists() and json.loads(path.read_text()).get('complete') is True
 except Exception:return False

def archive_incomplete(run_dir):
 if not run_dir.exists() or not any(run_dir.iterdir()):return
 archive=run_dir/'attempts'/dt.datetime.now().strftime('%Y%m%dT%H%M%S');archive.mkdir(parents=True)
 for path in list(run_dir.iterdir()):
  if path.name!='attempts':shutil.move(str(path),archive/path.name)

def run_visible(command,log,cwd):
 """Run one cell while mirroring its line-buffered log to the terminal."""
 process=subprocess.Popen(command,cwd=cwd,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
 try:
  assert process.stdout is not None
  for line in process.stdout:
   log.write(line);sys.stdout.write(line);sys.stdout.flush()
  return_code=process.wait()
  if return_code:
   raise subprocess.CalledProcessError(return_code,command)
 except KeyboardInterrupt:
  process.terminate()
  try:process.wait(timeout=10)
  except subprocess.TimeoutExpired:process.kill();process.wait()
  raise

def selected_campaigns(arg,master):
 mapping={'rank':['rank_confirmation'],'placement':['placement','attention_mlp'],'multilayer':['multilayer'],'agnews':['ag_news'],'benchmark':['benchmark'],'mechanism':['mechanism']}
 if arg=='all':
  aliases={'rank_confirmation':'rank_confirmation','scaling_ablation':'scaling','placement_scan':'placement','attention_mlp':'attention_mlp','multilayer':'multilayer','ag_news':'ag_news','mechanism':'mechanism','benchmark':'benchmark'}
  return [aliases[name] for name,flag in master['campaigns'].items() if flag and name in aliases]
 return mapping[arg]

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,default=DEFAULT_CONFIG);p.add_argument('--dry-run',action='store_true');p.add_argument('--smoke',action='store_true');p.add_argument('--campaign',choices=['rank','placement','multilayer','agnews','mechanism','benchmark','all'],default='all');p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto');p.add_argument('--continue-on-error',action='store_true');a=p.parse_args()
 master=json.loads(a.config.read_text());device,env=environment(a.device);root=ROOT/master['result_root'];campaigns=selected_campaigns(a.campaign,master);plan=[c for c in build_plan(master,device) if c.campaign in campaigns]
 if 'mechanism' in campaigns:print('Mechanism analysis is post-training and uses completed rank checkpoints; it is listed separately in the manifest.')
 if a.smoke:
  root=root/'smoke';plan=[c for c in build_plan(master,device) if c.campaign=='rank_confirmation' and c.config['seed']==42 and ((c.config['method']!='combined' and c.config.get('rank')==4) or (c.config['method']=='combined' and c.config.get('rank_lora')==2))]
  for c in plan:c.steps=2;c.config.update(steps=2,evaluation_every_steps=1,checkpoint_steps=[1,2],test_blocks=8,validation_blocks=2)
 historical=[]
 for path in (ROOT/'results_rank_factorial_controlled'/'confirmation_5000').glob('*/summary.json'):
  try:s=json.loads(path.read_text());historical.append(float(s.get('training_time_seconds',s.get('training_seconds')))/5000)
  except Exception:pass
 sec_per_step=statistics.median(historical) if historical else None;total_steps=sum(c.steps for c in plan);estimated_hours=total_steps*sec_per_step/3600 if sec_per_step else None
 print(json.dumps({'device':device,'environment':env,'campaign_counts':{name:sum(c.campaign==name for c in plan) for name in sorted(set(c.campaign for c in plan))},'total_runs':len(plan),'total_optimizer_steps':total_steps,'estimated_gpu_hours':estimated_hours,'estimated_disk_gib':round(len(plan)*0.012,2),'parameter_budgets':{'lora_r1':1536,'lora_r2':3072,'lora_r4':6144,'lora_r8':12288,'symmetric_same':True,'combined_1+1':3072,'combined_2+2':6144,'combined_4+4':12288}},indent=2))
 for c in plan:print(f"{c.campaign:18} {c.name:45} steps={c.steps:5} params={c.config.get('expected_trainable_parameters',c.config.get('expected_adapter_parameters'))}")
 if a.dry_run:return
 root.mkdir(parents=True,exist_ok=True);(root/'logs'/'launch_configs').mkdir(parents=True,exist_ok=True);started=dt.datetime.now(dt.timezone.utc)
 manifest={'started_utc':started.isoformat(),'environment':env,'master_config':master,'plan':[{'campaign':c.campaign,'name':c.name,'steps':c.steps,'config':c.config} for c in plan],'completed_cells':[],'failures':[]};(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 completed_campaigns=set();remaining=Counter(c.campaign for c in plan)
 for cell_index,cell in enumerate(plan,1):
  run_dir=root/cell.campaign/cell.name;cell.config['output_dir']=str(run_dir)
  if cell.campaign=='mechanism':cell.config['rank_result_root']=str(root/'rank_confirmation')
  if complete(run_dir):
   print('SKIP complete',cell.campaign,cell.name,flush=True);completed_campaigns.add(cell.campaign);remaining[cell.campaign]-=1
   if remaining[cell.campaign]==0:subprocess.run([sys.executable,str(EXP/'postprocess.py'),'--root',str(root),'--campaign',cell.campaign],cwd=ROOT,check=True)
   continue
  archive_incomplete(run_dir);run_dir.mkdir(parents=True,exist_ok=True);cell.config['orchestration_started_utc']=dt.datetime.now(dt.timezone.utc).isoformat();launch=root/'logs'/'launch_configs'/f'{cell.campaign}__{cell.name}.json';launch.write_text(json.dumps(cell.config,indent=2)+'\n')
  print(f"START [{cell_index}/{len(plan)}] {cell.campaign}/{cell.name}",flush=True)
  print(f"LOG   {run_dir/'log.txt'}",flush=True)
  with (run_dir/'log.txt').open('a',buffering=1) as log:
   try:
    cell_start=time.monotonic();run_visible([sys.executable,'-u',str(EXP/cell.runner),'--config',str(launch)],log,ROOT);summary_path=run_dir/'summary.json';summary=json.loads(summary_path.read_text());summary['orchestration_ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat();summary['orchestration_elapsed_seconds']=time.monotonic()-cell_start;summary_path.write_text(json.dumps(summary,indent=2)+'\n');completed_campaigns.add(cell.campaign);manifest['completed_cells'].append({'campaign':cell.campaign,'name':cell.name});(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f"DONE  [{cell_index}/{len(plan)}] {cell.campaign}/{cell.name}",flush=True)
   except KeyboardInterrupt:
    interruption={'campaign':cell.campaign,'name':cell.name,'time':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'interrupted_by_user'};manifest['failures'].append(interruption);(run_dir/'interrupted.json').write_text(json.dumps(interruption,indent=2)+'\n');(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n');print(f"INTERRUPTED {cell.campaign}/{cell.name}; completed runs remain reusable.",file=sys.stderr,flush=True);raise
   except Exception as exc:
    failure={'campaign':cell.campaign,'name':cell.name,'time':dt.datetime.now(dt.timezone.utc).isoformat(),'error':repr(exc)};manifest['failures'].append(failure);(run_dir/'failure.json').write_text(json.dumps(failure,indent=2)+'\n');(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if not (a.continue_on_error or master.get('failure_policy')=='continue'):raise
  remaining[cell.campaign]-=1
  if remaining[cell.campaign]==0:subprocess.run([sys.executable,str(EXP/'postprocess.py'),'--root',str(root),'--campaign',cell.campaign],cwd=ROOT,check=True)
 summaries=[]
 for path in root.glob('*/*/summary.json'):
  row=json.loads(path.read_text());row['source']=str(path.relative_to(root));summaries.append(row)
 aggregate=root/'aggregate';aggregate.mkdir(exist_ok=True);(aggregate/'MASTER_RESULTS.json').write_text(json.dumps(summaries,indent=2)+'\n')
 fields=sorted({k for row in summaries for k,v in row.items() if isinstance(v,(str,int,float,bool)) or v is None})
 with (aggregate/'MASTER_RESULTS.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(summaries)
 manifest['ended_utc']=dt.datetime.now(dt.timezone.utc).isoformat();manifest['complete']=not manifest['failures'];(root/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
 artifacts=[]
 for path in sorted((root/'aggregate').glob('*'))+sorted((root/'figures').glob('*')):
  artifacts.append({'artifact':str(path.relative_to(root)),'source':'per-run summary.json and metrics.csv files listed by campaign in MANIFEST.json'})
 (root/'PROVENANCE.json').write_text(json.dumps(artifacts,indent=2)+'\n')
if __name__=='__main__':main()
