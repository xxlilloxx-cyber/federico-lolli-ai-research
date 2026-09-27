import argparse,csv,json,math,sys
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from adapters import freeze_and_insert,load_adapter_checkpoint

def blocks(tok, texts, seq):
 ids=tok('\n'.join(x for x in texts if x.strip()),add_special_tokens=False)['input_ids']
 return [torch.tensor(ids[i:i+seq],dtype=torch.long) for i in range(0,len(ids)-seq,seq)]
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True); ap.add_argument('--checkpoint'); ap.add_argument('--output',required=True); ap.add_argument('--base',action='store_true'); a=ap.parse_args(); cfg=json.load(open(a.config)); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 tok=AutoTokenizer.from_pretrained(cfg['model']); model=AutoModelForCausalLM.from_pretrained(cfg['model'],torch_dtype=torch.float16).to(device); model.config.use_cache=False
 if not a.base:
  freeze_and_insert(model,cfg['kind'],cfg.get('rank',4),cfg.get('alpha',4.0),tuple(cfg['layers']),tuple(cfg['projections']),False,rank_l=cfg.get('rank_l'),rank_q=cfg.get('rank_q'),alpha_l=cfg.get('alpha_l'),alpha_q=cfg.get('alpha_q'))
  load_adapter_checkpoint(model,a.checkpoint,cfg,map_location='cpu')
 raw=load_dataset('wikitext','wikitext-2-raw-v1'); seq=cfg.get('sequence_length',128); test=blocks(tok,raw['test']['text'],seq); model.eval(); losses=[]
 with torch.no_grad():
  for i,t in enumerate(test):
   x=t.unsqueeze(0).to(device)
   with torch.autocast('cuda',dtype=torch.float16,enabled=device.type=='cuda'): losses.append(float(model(x,labels=x).loss.float().cpu()))
 loss=sum(losses)/len(losses); out={'model':cfg['model'],'kind':'baseline' if a.base else cfg['kind'],'seed':cfg.get('seed'), 'layers':cfg.get('layers'), 'rank':cfg.get('rank'),'checkpoint':None if a.base else str(a.checkpoint),'test_blocks':len(test),'test_loss':loss,'test_perplexity':math.exp(min(loss,20))}
 Path(a.output).write_text(json.dumps(out,indent=2)); print(json.dumps(out,indent=2))
if __name__=='__main__': main()
