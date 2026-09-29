import json, subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[2]; out=root/'results_depth'/'full_attention'; out.mkdir(parents=True,exist_ok=True)
for layer in range(12):
 for kind in ['lora','symmetric_quadratic']:
  name=f'layer{layer}_{kind}'; d=out/name; d.mkdir(parents=True,exist_ok=True)
  cfg={'kind':kind,'model':'gpt2','seed':42,'steps':20,'batch_size':1,'gradient_accumulation':4,'sequence_length':128,'learning_rate':0.0003,'rank':4,'alpha':4.0,'layers':[layer],'projections':['attn.c_proj'],'protocol':'full_attention_depth_scan_20_steps','output_dir':str(d)}
  cp=d/'config.json'; cp.write_text(json.dumps(cfg,indent=2)); log=d/'run.log'; print('START',name,flush=True)
  r=subprocess.run([str(root/'.venv/bin/python'),str(root/'src/train.py'),str(cp)],cwd=root,text=True,stdout=log.open('w'),stderr=subprocess.STDOUT); print('DONE',name,'rc',r.returncode,flush=True)
