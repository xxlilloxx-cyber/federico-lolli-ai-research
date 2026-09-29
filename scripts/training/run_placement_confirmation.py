import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]; out=root/'results_depth'/'confirmation'; out.mkdir(parents=True,exist_ok=True)
experiments=[('attn_layer0',['attn.c_proj'],[0],4),('attn_layer6',['attn.c_proj'],[6],4),('attn_layer11',['attn.c_proj'],[11],4),('mlp_both_layer0',['mlp.c_fc','mlp.c_proj'],[0],2)]
for base,mods,layers,rank in experiments:
 for kind in ['lora','symmetric_quadratic']:
  for seed in [42,123,456]:
   name=f'{base}_{kind}_seed{seed}'; d=out/name; d.mkdir(parents=True,exist_ok=True)
   cfg={'kind':kind,'model':'gpt2','seed':seed,'steps':100,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':mods,'protocol':'three_seed_confirmation_100_steps','output_dir':str(d)}
   cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'
   print('START',name,flush=True)
   r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT)
   print('DONE',name,'rc',r.returncode,flush=True)
