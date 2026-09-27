import json, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'results_depth'/'screen'; out.mkdir(parents=True,exist_ok=True)

def run(name,kind,layers,projections,rank=4):
 d=out/name; d.mkdir(parents=True,exist_ok=True)
 cfg={'kind':kind,'model':'gpt2','seed':42,'steps':20,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':projections,'screening':'single_seed_20_steps','output_dir':str(d)}
 cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2))
 log=d/'run.log'
 print('START',name,flush=True)
 r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT)
 print('DONE',name,'rc',r.returncode,flush=True)

for layer in [0,3,6,9,11]:
 for kind in ['lora','symmetric_quadratic']:
  run(f'attn_layer{layer}_{kind}',kind,[layer],['attn.c_proj'])
for mod in ['mlp.c_fc','mlp.c_proj']:
 for kind in ['lora','symmetric_quadratic']:
  run(f'{mod.replace(".","_")}_{kind}',kind,[0],[mod])
for kind in ['lora','symmetric_quadratic']:
 run(f'mlp_both_layer0_{kind}',kind,[0],['mlp.c_fc','mlp.c_proj'],rank=2)
