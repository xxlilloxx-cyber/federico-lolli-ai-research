#!/usr/bin/env python3
"""Matched full-model cost benchmark for one adapter configuration."""
from __future__ import annotations
import argparse,csv,json,statistics,sys,time
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.combined_extension.adapters import insert_adapter,parameter_counts
def main(path):
 cfg=json.loads(path.read_text());dev=torch.device(cfg['device']);dtype=torch.float16 if dev.type=='cuda' else torch.float32
 model=AutoModelForCausalLM.from_pretrained(cfg['model'],dtype=dtype).to(dev);model.config.use_cache=False
 insert_adapter(model,cfg['method'],rank=cfg.get('rank'),rank_lora=cfg.get('rank_lora'),rank_symmetric=cfg.get('rank_symmetric'),alpha=cfg['alpha'],scaling_mode='total_rank',layers=[0],projections=['attn.c_proj']);counts=parameter_counts(model)
 if counts.total_trainable!=cfg['expected_trainable_parameters']:raise ValueError('parameter budget mismatch')
 ids=torch.randint(0,model.config.vocab_size,(cfg['batch_size'],cfg['sequence_length']),device=dev);sync=(lambda:torch.cuda.synchronize()) if dev.type=='cuda' else (lambda:None)
 def forward():return model(ids,use_cache=False).logits
 for _ in range(cfg['warmup_iterations']):forward();sync()
 rows=[];torch.cuda.reset_peak_memory_stats() if dev.type=='cuda' else None
 for i in range(cfg['measured_iterations']):
  model.eval();sync();start=time.perf_counter()
  with torch.no_grad():forward()
  sync();forward_ms=1000*(time.perf_counter()-start)
  model.train();model.zero_grad(set_to_none=True);loss=model(ids,labels=ids,use_cache=False).loss;sync();start=time.perf_counter();loss.backward();sync();backward_ms=1000*(time.perf_counter()-start)
  rows.append({'iteration':i,'forward_ms':forward_ms,'backward_ms':backward_ms,'tokens_per_second':cfg['batch_size']*cfg['sequence_length']/(forward_ms/1000)})
 out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
 with (out/'metrics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
 def pair(k):v=[r[k] for r in rows];return statistics.mean(v),statistics.stdev(v)
 fm,fs=pair('forward_ms');bm,bs=pair('backward_ms');tm,ts=pair('tokens_per_second')
 summary={**cfg,'complete':True,'trainable_parameters':counts.total_trainable,'forward_ms_mean':fm,'forward_ms_sample_sd':fs,'backward_ms_mean':bm,'backward_ms_sample_sd':bs,'tokens_per_second_mean':tm,'tokens_per_second_sample_sd':ts,'peak_allocated_vram_mib':torch.cuda.max_memory_allocated()/2**20 if dev.type=='cuda' else None};(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
