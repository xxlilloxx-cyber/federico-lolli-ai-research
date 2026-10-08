#!/usr/bin/env python3
"""Execute one resume-safe WikiText-2 cell for the Combined extension."""
from __future__ import annotations
import argparse, csv, json, math, os, random, sys, time
from pathlib import Path
import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from experiments.combined_extension.adapters import insert_adapter, parameter_counts, trainable_state

def seed_all(seed):
 random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def main(path: Path):
 cfg=json.loads(path.read_text());seed_all(cfg['seed']);device=torch.device(cfg['device'])
 if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable')
 dtype=torch.float16 if device.type=='cuda' else torch.float32
 tok=AutoTokenizer.from_pretrained(cfg['model']);model=AutoModelForCausalLM.from_pretrained(cfg['model'],dtype=dtype).to(device);model.config.use_cache=False
 insert_adapter(model,cfg['method'],rank=cfg.get('rank'),rank_lora=cfg.get('rank_lora'),rank_symmetric=cfg.get('rank_symmetric'),alpha=cfg['alpha'],scaling_mode=cfg.get('scaling_mode','total_rank'),layers=cfg['layers'],projections=cfg['projections'])
 counts=parameter_counts(model); expected=cfg['expected_trainable_parameters']
 if counts.total_trainable!=expected:raise ValueError(f'parameter count {counts.total_trainable} != {expected}')
 raw=load_dataset('wikitext','wikitext-2-raw-v1');seq=cfg['sequence_length']
 def blocks(split):
  ids=tok('\n'.join(x for x in raw[split]['text'] if x.strip()),add_special_tokens=False)['input_ids']
  return [torch.tensor(ids[i:i+seq],dtype=torch.long) for i in range(0,len(ids)-seq,seq)]
 train,valid,test=blocks('train'),blocks('validation'),blocks('test')
 def one_loss(t):
  x=t.unsqueeze(0).to(device)
  context=torch.autocast('cuda',dtype=torch.float16) if device.type=='cuda' else torch.autocast('cpu',enabled=False)
  with context:return model(x,labels=x,use_cache=False).loss
 def evaluate(data,limit=None):
  model.eval(); chosen=data if limit is None else data[:limit]
  with torch.no_grad():value=float(torch.stack([one_loss(t).float() for t in chosen]).mean())
  return value
 out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 initial=evaluate(valid,cfg.get('validation_blocks',8));params=[p for p in model.parameters() if p.requires_grad]
 optimizer=torch.optim.AdamW(params,lr=cfg['learning_rate'],weight_decay=cfg.get('weight_decay',0.0));scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda')
 rows=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats() if device.type=='cuda' else None
 checkpoints=set(cfg['checkpoint_steps']);eval_every=cfg['evaluation_every_steps'];total_tokens=0
 for step in range(1,cfg['steps']+1):
  model.train();optimizer.zero_grad(set_to_none=True);losses=[]
  for offset in range(cfg['gradient_accumulation']):
   item=train[((step-1)*cfg['gradient_accumulation']+offset)%len(train)];loss=one_loss(item)
   if not torch.isfinite(loss):raise FloatingPointError(f'non-finite loss at step {step}')
   scaler.scale(loss/cfg['gradient_accumulation']).backward();losses.append(float(loss.detach()));total_tokens+=len(item)
  scaler.unscale_(optimizer);gradient_norm=float(torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clipping']));scaler.step(optimizer);scaler.update()
  if step==1 or step%eval_every==0 or step==cfg['steps']:
   val=evaluate(valid,cfg.get('validation_blocks',8));row={'step':step,'training_loss':float(np.mean(losses)),'validation_loss':val,'validation_perplexity':math.exp(val),'gradient_norm':gradient_norm,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20 if device.type=='cuda' else None};rows.append(row);print(json.dumps(row),flush=True)
  if step in checkpoints or step==cfg['steps']:
   ck=out/'checkpoints';ck.mkdir(exist_ok=True);torch.save({'trainable_state':trainable_state(model),'config':cfg,'step':step},ck/f'adapter_step_{step:05d}.pt')
 with (out/'metrics.csv').open('w',newline='') as handle:
  writer=csv.DictWriter(handle,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)
 best=min(rows,key=lambda row:row['validation_loss']);final=next(row for row in reversed(rows) if row['step']==cfg['steps']);test_loss=evaluate(test,cfg.get('test_blocks'))
 elapsed=time.monotonic()-start
 summary={**cfg,'complete':True,'trainable_parameters':counts.total_trainable,'lora_parameters':counts.lora,'symmetric_parameters':counts.symmetric,'total_model_parameters':counts.total_model,'initial_validation_loss':initial,'best_validation_loss':best['validation_loss'],'best_validation_step':best['step'],'final_validation_loss':final['validation_loss'],'final_test_loss':test_loss,'final_test_perplexity':math.exp(test_loss),'training_seconds':elapsed,'tokens_per_second':total_tokens/elapsed,'peak_gpu_memory_mb':max((r['peak_gpu_memory_mb'] or 0) for r in rows),'nan_count':0,'inf_count':0}
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);main(parser.parse_args().config)
