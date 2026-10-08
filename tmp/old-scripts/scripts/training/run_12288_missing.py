import json,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]; d=root/'results_local_rank/budget12288_screen1000/four_distributed_symmetric_quadratic_seed42'; d.mkdir(parents=True,exist_ok=True)
cfg={'kind':'symmetric_quadratic','model':'gpt2','seed':42,'steps':1000,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':2,'alpha':4.0,'layers':[0,3,7,11],'projections':['attn.c_proj'],'strategy':'four_distributed','budget_type':'fixed_total_12288','target_budget':12288,'output_dir':str(d)}
cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); print('START',flush=True)
r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=(d/'run.log').open('w'),stderr=subprocess.STDOUT); print('DONE',r.returncode,flush=True)
try:
 h=subprocess.run(['nvidia-smi','--query-gpu=name,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,timeout=10); print('HEALTH',h.returncode,h.stdout.strip(),flush=True)
except Exception as e: print('HEALTH_ERROR',repr(e),flush=True)
