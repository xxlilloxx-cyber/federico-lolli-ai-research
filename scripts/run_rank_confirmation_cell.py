#!/usr/bin/env python3
"""One interruption-safe 5,000-step confirmation cell plus final held-out test."""
import argparse,json,math,subprocess,sys,yaml
from pathlib import Path
root=Path(__file__).resolve().parents[1]; ap=argparse.ArgumentParser();ap.add_argument('--method',choices=['lora','symmetric_quadratic'],required=True);ap.add_argument('--rank',type=int,required=True);ap.add_argument('--seed',type=int,required=True);a=ap.parse_args()
out=root/'results_rank_factorial_controlled'/'confirmation_5000'/f'{a.method}_r{a.rank}_seed{a.seed}_steps5000'; summary=out/'summary.json'; config=out/'config.json'; ck=out/'checkpoints'/'adapter_final.pt'
def valid():
 try:
  s=json.loads(summary.read_text());return config.exists() and ck.exists() and s.get('steps')==5000 and math.isfinite(s.get('validation_loss',float('nan')))
 except: return False
if not valid():
 out.mkdir(parents=True,exist_ok=True); cfg={'model':'gpt2','kind':a.method,'rank':a.rank,'alpha':4.0,'layers':[0],'projections':['attn.c_proj'],'seed':a.seed,'sequence_length':128,'gradient_accumulation':4,'learning_rate':3e-4,'steps':5000,'eval_every':25,'output_dir':str(out),'checkpoint_steps':[1000,2000,3000,4000,5000]}; y=out/'launch_config.yaml';y.write_text(yaml.safe_dump(cfg));subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(y)],cwd=root,check=True)
test=out/'final_test.json'
if not test.exists(): subprocess.run([str(root/'.venv/bin/python'),str(root/'scripts/evaluate_checkpoint.py'),'--config',str(config),'--checkpoint',str(ck),'--output',str(test)],cwd=root,check=True)
s=json.loads(summary.read_text());t=json.loads(test.read_text());s.update({'best_recorded_validation_loss':s['validation_loss'],'final_validation_loss':json.loads(open(out/'metrics.csv').read().splitlines()[-1].split(',')[2]) if False else None,'final_test_loss':t['test_loss'],'final_test_perplexity':t['test_perplexity'],'alpha_over_rank':4/a.rank});
# preserve exact final validation from final metric row without relying on column order
import csv
rows=list(csv.DictReader(open(out/'metrics.csv')));s['final_validation_loss']=float(rows[-1]['validation_loss']);s['final_validation_perplexity']=math.exp(s['final_validation_loss']);s['training_steps']=5000;s['peak_gpu_memory_mb']=s.get('peak_vram_bytes',0)/2**20
summary.write_text(json.dumps(s,indent=2))
print(out)
