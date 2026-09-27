import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'results_local_rank'/'budget6144_2000';
runs=[('four_distributed','lora',456,[0,3,7,11],1),('four_distributed','symmetric_quadratic',42,[0,3,7,11],1),('four_distributed','symmetric_quadratic',123,[0,3,7,11],1),('four_distributed','symmetric_quadratic',456,[0,3,7,11],1)]
for name,kind,seed,layers,rank in runs:
 exp=f'{name}_{kind}_seed{seed}'; d=out/exp
 if (d/'summary.json').exists() and (d/'metrics.csv').exists(): print('SKIP',exp,flush=True); continue
 d.mkdir(parents=True,exist_ok=True)
 cfg={'kind':kind,'model':'gpt2','seed':seed,'steps':2000,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':['attn.c_proj'],'strategy':name,'budget_type':'fixed_total_6144','target_budget':6144,'output_dir':str(d)}
 cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'; print('START',exp,flush=True)
 r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT); print('DONE',exp,'rc',r.returncode,flush=True)
 try:
  h=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10); print('HEALTH',h.returncode,h.stdout.strip(),flush=True)
 except Exception as e: print('HEALTH_ERROR',repr(e),flush=True)
 if r.returncode!=0: print('STOP',exp,flush=True); break
