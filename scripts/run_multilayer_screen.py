import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]; out=root/'results_multilayer'/'screen'; out.mkdir(parents=True,exist_ok=True)
sets=[
 ('concentrated_layer2',[2],4,'concentrated'),
 ('distributed_two',[0,11],2,'distributed'),
 ('distributed_four',[0,3,7,11],1,'distributed'),
 ('data_driven_two',[2,4],2,'data_driven'),
 ('early_four',[0,1,2,3],1,'regional_early'),
 ('middle_four',[4,5,6,7],1,'regional_middle'),
 ('late_four',[8,9,10,11],1,'regional_late'),
]
for name,layers,rank,strategy in sets:
 for kind in ['lora','symmetric_quadratic']:
  exp=f'{name}_{kind}_seed42'; d=out/exp; d.mkdir(parents=True,exist_ok=True)
  cfg={'kind':kind,'model':'gpt2','seed':42,'steps':1000,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':rank,'alpha':4.0,'layers':layers,'projections':['attn.c_proj'],'strategy':strategy,'budget_type':'fixed_total_rank4','target_budget':6144,'output_dir':str(d)}
  cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'; print('START',exp,flush=True)
  r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT); print('DONE',exp,'rc',r.returncode,flush=True)
