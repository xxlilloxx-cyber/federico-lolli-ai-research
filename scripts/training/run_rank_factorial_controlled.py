#!/usr/bin/env python3
"""Launch one fresh, controlled WikiText-2 rank-factorial cell."""
import argparse, subprocess, tempfile, yaml
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--method',choices=['lora','symmetric_quadratic'],required=True);p.add_argument('--rank',type=int,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--steps',type=int,default=1000);p.add_argument('--alpha',type=float,default=4.0);p.add_argument('--output',default='results_rank_factorial_controlled/screening');a=p.parse_args()
out=f'{a.output}/{a.method}_r{a.rank}_a{a.alpha:g}_seed{a.seed}_steps{a.steps}'
cfg={'model':'gpt2','kind':a.method,'rank':a.rank,'alpha':a.alpha,'layers':[0],'projections':['attn.c_proj'],'seed':a.seed,'sequence_length':128,'gradient_accumulation':4,'learning_rate':3e-4,'steps':a.steps,'eval_every':25,'output_dir':out,'checkpoint_steps':[a.steps]}
Path(out).mkdir(parents=True,exist_ok=True); c=Path(out)/'launch_config.yaml';c.write_text(yaml.safe_dump(cfg));subprocess.run(['.venv/bin/python','src/train.py',str(c)],check=True)
