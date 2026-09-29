import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]; out=root/'results_local_rank'/'budget12288_screen1000'; out.mkdir(parents=True,exist_ok=True)
sets=[('concentrated',[2],8),('two_distributed',[0,11],4),('four_distributed',[0,3,7,11],2)]
for name,layers,rank in sets:
 for kind in ['lora','symmetric_quadratic']:
  exp=f'{name}_{kind}_seed42'; d=out/exp
  if (d/'summary.json').exists() and (d/'metrics.csv').exists(): print('SKIP',exp,flush=True); continue
  d.mkdir(parents=True,exist_ok=True); cfg={'kind':kind,'model':'gpt2','seed':42,'steps':1000,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':['attn.c_proj'],'strategy':name,'budget_type':'fixed_total_12288','target_budget':12288,'output_dir':str(d)}; cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'; print('START',exp,flush=True); r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT); print('DONE',exp,'rc',r.returncode,flush=True)
  try:
   h=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10); print('HEALTH',h.returncode,h.stdout.strip(),flush=True)
  except Exception as e: print('HEALTH_ERROR',repr(e),flush=True)
  if r.returncode!=0: print('STOP',exp,flush=True); raise SystemExit(1)
