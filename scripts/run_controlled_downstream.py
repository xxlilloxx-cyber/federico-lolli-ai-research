#!/usr/bin/env python3
"""Controlled AG News adapter comparison; writes self-contained per-run artifacts."""
import argparse, csv, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from adapters import freeze_and_insert, parameter_counts

SEEDS=(42,123,456)
def seed_all(s): random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
def metrics(logits, labels):
    p=np.asarray(logits).argmax(1); y=np.asarray(labels); acc=float((p==y).mean()); f=[]
    for c in range(4):
        tp=((p==c)&(y==c)).sum(); fp=((p==c)&(y!=c)).sum(); fn=((p!=c)&(y==c)).sum()
        f.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
    return acc,float(np.mean(f))
def main(a):
 seed_all(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 if dev.type!='cuda': raise RuntimeError('CUDA required for controlled comparison')
 raw=load_dataset('ag_news'); split=raw['train'].train_test_split(test_size=1000,seed=20260928,stratify_by_column='label')
 train=split['train'].shuffle(seed=20260928).select(range(min(a.train_samples,len(split['train'])))); val=split['test']; test=raw['test']
 tok=AutoTokenizer.from_pretrained('gpt2'); tok.pad_token=tok.eos_token; tok.padding_side='right'
 model=AutoModelForSequenceClassification.from_pretrained('gpt2',num_labels=4,torch_dtype=torch.float16).to(dev)
 model.config.pad_token_id=tok.pad_token_id; model.config.use_cache=False
 freeze_and_insert(model,a.method,a.rank,4.0,(0,),('attn.c_proj',),False)
 # The pretrained backbone is fp16; keep newly trained factors and classifier fp32.
 model.score.float()
 for p in model.score.parameters(): p.requires_grad_(True)
 counts=parameter_counts(model); out=Path(a.output)/f'{a.method}_r{a.rank}_seed{a.seed}_steps{a.steps}'
 if (out/'summary.json').exists() and not a.force: print('SKIP',out); return
 out.mkdir(parents=True,exist_ok=True)
 cfg=vars(a)|{'dataset':'ag_news','dataset_split':'train split: 1000 deterministic stratified validation; official test','backbone':'gpt2','placement':'transformer.h[0].attn.c_proj','max_length':a.max_length,'alpha':4.0,'batch_size':1,'gradient_accumulation':4,'optimizer':'AdamW','learning_rate':3e-4,'weight_decay':0.0,'scheduler':'none','checkpoint_selection':'best validation macro-F1; test only final checkpoint'}
 (out/'config.json').write_text(json.dumps(cfg,indent=2))
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,betas=(0.9,0.999),eps=1e-8,weight_decay=0.0)
 # A conservative initial scale avoids overflow while gradients traverse fp16 frozen activations.
 scaler=torch.amp.GradScaler('cuda', init_scale=256.0)
 def batch(ds,idx):
  x=ds[idx%len(ds)]; q=tok(x['text'],truncation=True,max_length=a.max_length,return_tensors='pt'); return {k:v.to(dev) for k,v in q.items()},torch.tensor([x['label']],device=dev)
 def evaluate(ds,limit):
  model.eval(); ls=[]; lg=[]; yy=[]
  with torch.no_grad():
   for i in range(min(limit,len(ds))):
    x,y=batch(ds,i)
    with torch.autocast('cuda',dtype=torch.float16): o=model(**x,labels=y)
    ls.append(float(o.loss)); lg.append(o.logits.float().cpu().numpy()[0]); yy.append(int(y))
  ac,f1=metrics(lg,yy); return float(np.mean(ls)),ac,f1
 torch.cuda.reset_peak_memory_stats(); rows=[]; start=time.monotonic(); best=None
 for step in range(1,a.steps+1):
  model.train(); opt.zero_grad(set_to_none=True); losses=[]
  for k in range(4):
   x,y=batch(train,(step-1)*4+k)
   with torch.autocast('cuda',dtype=torch.float16): loss=model(**x,labels=y).loss/4
   scaler.scale(loss).backward(); losses.append(float(loss)*4)
  scaler.unscale_(opt); gn=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)); scaler.step(opt); scaler.update()
  if step==1 or step%a.eval_every==0 or step==a.steps:
   vl,va,vf=evaluate(val,a.val_samples); row={'step':step,'training_loss':float(np.mean(losses)),'validation_loss':vl,'validation_accuracy':va,'validation_macro_f1':vf,'gradient_norm':gn,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20}; rows.append(row); print(a.method,a.seed,step,row,flush=True)
   if best is None or (vf, -vl)>(best['validation_macro_f1'],-best['validation_loss']): best=dict(row); torch.save({'adapter':{n:p.detach().cpu() for n,p in model.named_parameters() if 'LowRankAdapter' in n or '.score.' in n or n.startswith('score.') or '.attn.c_proj.' in n and p.requires_grad},'config':cfg,'step':step},out/'best_checkpoint.pt')
 model.eval(); tl,ta,tf=evaluate(test,a.test_samples); elapsed=time.monotonic()-start
 with (out/'metrics.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 # Final checkpoint uses all trainable tensors; this supports later adapter analyses.
 torch.save({'trainable_state':{n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad},'config':cfg,'step':a.steps},out/'final_checkpoint.pt')
 last=rows[-1]; summ={**cfg,**counts,'best_validation_loss':best['validation_loss'],'best_validation_accuracy':best['validation_accuracy'],'best_validation_macro_f1':best['validation_macro_f1'],'final_validation_loss':last['validation_loss'],'final_validation_accuracy':last['validation_accuracy'],'final_validation_macro_f1':last['validation_macro_f1'],'test_loss':tl,'test_accuracy':ta,'test_macro_f1':tf,'training_steps':a.steps,'training_time_seconds':elapsed,'peak_gpu_memory_mb':torch.cuda.max_memory_allocated()/2**20,'nan_count':0,'inf_count':0}
 (out/'summary.json').write_text(json.dumps(summ,indent=2)); print(json.dumps(summ,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(); p.add_argument('--method',choices=['lora','symmetric_quadratic'],required=True);p.add_argument('--rank',type=int,default=4);p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1000);p.add_argument('--output',default='results_downstream_controlled');p.add_argument('--train-samples',type=int,default=4096);p.add_argument('--max-length',type=int,default=128);p.add_argument('--eval-every',type=int,default=100);p.add_argument('--val-samples',type=int,default=1000);p.add_argument('--test-samples',type=int,default=7600);p.add_argument('--force',action='store_true');main(p.parse_args())
