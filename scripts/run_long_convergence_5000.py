import json, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'results_long_convergence_5000'/'budget12288_5000'; out.mkdir(parents=True,exist_ok=True)
# Priority order: four-layer, two-layer, concentrated.
sets=[('four_distributed',[0,3,7,11],2),('two_distributed',[0,11],4),('concentrated',[2],8)]
for strategy,layers,rank in sets:
 for kind in ['lora','symmetric_quadratic']:
  for seed in [42,123,456]:
   exp=f'{strategy}_{kind}_seed{seed}'; d=out/exp
   if (d/'summary.json').exists() and (d/'checkpoints'/'adapter_final.pt').exists():
    print('SKIP',exp,flush=True); continue
   d.mkdir(parents=True,exist_ok=True)
   cfg={'kind':kind,'model':'gpt2','seed':seed,'steps':5000,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':['attn.c_proj'],'strategy':strategy,'budget_type':'fixed_total_12288_long','target_budget':12288,'diagnostics':True,'eval_every':25,'checkpoint_steps':[1000,2000,3000,4000,5000],'output_dir':str(d)}
   cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'
   print('START',exp,flush=True)
   r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT)
   print('DONE',exp,'rc',r.returncode,flush=True)
   try:
    h=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10)
    print('HEALTH',h.returncode,h.stdout.strip(),flush=True)
   except Exception as e: print('HEALTH_ERROR',repr(e),flush=True)
   if r.returncode!=0 or not (d/'summary.json').exists():
    print('STOP',exp,flush=True); raise SystemExit(1)
