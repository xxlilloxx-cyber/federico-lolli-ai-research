#!/usr/bin/env python3
import csv,json,statistics,time,sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
from adapters import freeze_and_insert,load_adapter_checkpoint,parameter_counts
ROOT=Path(__file__).resolve().parents[2]; rows=[];device=torch.device('cuda')
for kind in ('lora','symmetric_quadratic'):
 d=ROOT/'results_rank_factorial_controlled'/'confirmation_5000'/f'{kind}_r4_seed42_steps5000';cfg=json.loads((d/'config.json').read_text());m=AutoModelForCausalLM.from_pretrained('gpt2',torch_dtype=torch.float16).to(device);m.config.use_cache=False;freeze_and_insert(m,kind,4,4.0,(0,),('attn.c_proj',));load_adapter_checkpoint(m,d/'checkpoints'/'adapter_final.pt',cfg)
 x=torch.randint(0,50257,(1,128),device=device);m.train();
 for _ in range(10):
  with torch.autocast('cuda',dtype=torch.float16):m(x,labels=x).loss.backward()
  m.zero_grad(set_to_none=True)
 torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats();fw=[];bw=[]
 for _ in range(30):
  torch.cuda.synchronize();t=time.perf_counter()
  with torch.autocast('cuda',dtype=torch.float16):loss=m(x,labels=x).loss
  torch.cuda.synchronize();mid=time.perf_counter();loss.backward();torch.cuda.synchronize();end=time.perf_counter();m.zero_grad(set_to_none=True);fw.append((mid-t)*1000);bw.append((end-mid)*1000)
 counts=parameter_counts(m);rows.append({'method':'LoRA' if kind=='lora' else 'Symmetric Quadratic','rank':4,'sequence_length':128,'batch_size':1,'warmup_iterations':10,'measured_iterations':30,'forward_ms_mean':statistics.mean(fw),'forward_ms_sample_sd':statistics.stdev(fw),'backward_ms_mean':statistics.mean(bw),'backward_ms_sample_sd':statistics.stdev(bw),'tokens_per_second_forward':128/(statistics.mean(fw)/1000),'peak_vram_mb':torch.cuda.max_memory_allocated()/2**20,'trainable_parameters':counts['trainable_parameters']})
 del m;torch.cuda.empty_cache()
out=ROOT/'docs'/'data'/'computational_cost.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
