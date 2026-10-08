#!/usr/bin/env python3
"""Execute one controlled AG News cell including Combined."""
from __future__ import annotations
import argparse,csv,json,random,sys,time
from pathlib import Path
import numpy as np, torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.combined_extension.adapters import insert_adapter,parameter_counts,trainable_state

def seed_all(s):random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s) if torch.cuda.is_available() else None
def scores(logits,labels):
 p=np.asarray(logits).argmax(1);y=np.asarray(labels);fs=[]
 for c in range(4):
  tp=((p==c)&(y==c)).sum();fp=((p==c)&(y!=c)).sum();fn=((p!=c)&(y==c)).sum();fs.append(0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn))
 return float((p==y).mean()),float(np.mean(fs))
def main(path):
 cfg=json.loads(path.read_text());seed_all(cfg['seed']);dev=torch.device(cfg['device']);dtype=torch.float16 if dev.type=='cuda' else torch.float32
 raw=load_dataset('ag_news');split=raw['train'].train_test_split(test_size=1000,seed=20260928,stratify_by_column='label');train=split['train'].shuffle(seed=20260928).select(range(cfg['train_samples']));val=split['test'];test=raw['test']
 tok=AutoTokenizer.from_pretrained('gpt2');tok.pad_token=tok.eos_token;tok.padding_side='right'
 model=AutoModelForSequenceClassification.from_pretrained('gpt2',num_labels=4,dtype=dtype).to(dev);model.config.pad_token_id=tok.eos_token_id;model.config.use_cache=False
 insert_adapter(model,cfg['method'],rank=cfg.get('rank'),rank_lora=cfg.get('rank_lora'),rank_symmetric=cfg.get('rank_symmetric'),alpha=cfg['alpha'],scaling_mode=cfg.get('scaling_mode','total_rank'),layers=[0],projections=['attn.c_proj']);model.score.float()
 for p in model.score.parameters():p.requires_grad_(True)
 counts=parameter_counts(model);expected=cfg['expected_adapter_parameters']+3072
 if counts.total_trainable!=expected:raise ValueError(f'trainable count {counts.total_trainable} != {expected}')
 out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 def batch(ds,i):
  row=ds[i%len(ds)];x=tok(row['text'],truncation=True,max_length=cfg['max_length'],return_tensors='pt');return {k:v.to(dev) for k,v in x.items()},torch.tensor([row['label']],device=dev)
 def evaluate(ds,limit):
  model.eval();losses=[];logits=[];labels=[]
  with torch.no_grad():
   for i in range(min(limit,len(ds))):
    x,y=batch(ds,i)
    with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):o=model(**x,labels=y)
    losses.append(float(o.loss));logits.append(o.logits.float().cpu().numpy()[0]);labels.append(int(y))
  acc,f1=scores(logits,labels);return float(np.mean(losses)),acc,f1
 params=[p for p in model.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=cfg['learning_rate']);scaler=torch.amp.GradScaler('cuda',enabled=dev.type=='cuda',init_scale=256.0);rows=[];start=time.monotonic()
 for step in range(1,cfg['steps']+1):
  model.train();opt.zero_grad(set_to_none=True);losses=[]
  for offset in range(cfg['gradient_accumulation']):
   x,y=batch(train,(step-1)*cfg['gradient_accumulation']+offset)
   with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=dev.type=='cuda'):loss=model(**x,labels=y).loss
   if not torch.isfinite(loss):raise FloatingPointError(f'non-finite loss at {step}')
   scaler.scale(loss/cfg['gradient_accumulation']).backward();losses.append(float(loss))
  scaler.unscale_(opt);gn=float(torch.nn.utils.clip_grad_norm_(params,cfg['gradient_clipping']));scaler.step(opt);scaler.update()
  if step==1 or step%cfg['evaluation_every_steps']==0 or step==cfg['steps']:
   vl,va,vf=evaluate(val,cfg['validation_samples']);rows.append({'step':step,'training_loss':float(np.mean(losses)),'validation_loss':vl,'validation_accuracy':va,'validation_macro_f1':vf,'gradient_norm':gn});print(json.dumps(rows[-1]),flush=True)
 with (out/'metrics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 tl,ta,tf=evaluate(test,cfg['test_samples']);best=max(rows,key=lambda r:(r['validation_macro_f1'],-r['validation_loss']));final=rows[-1];torch.save({'trainable_state':trainable_state(model),'config':cfg,'step':cfg['steps']},out/'final_checkpoint.pt')
 summary={**cfg,'complete':True,'trainable_parameters':counts.total_trainable,'adapter_parameters':counts.lora+counts.symmetric,'best_validation_loss':best['validation_loss'],'best_validation_step':best['step'],'final_validation_loss':final['validation_loss'],'test_loss':tl,'test_accuracy':ta,'test_macro_f1':tf,'training_seconds':time.monotonic()-start,'nan_count':0,'inf_count':0};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
