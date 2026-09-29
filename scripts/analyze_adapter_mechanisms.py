#!/usr/bin/env python3
"""Checkpoint-only spectra, local Jacobians, covariance, and adapter Hessians."""
import csv, json, math, sys
from pathlib import Path
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'results_rank_factorial_controlled'/'confirmation_5000'
OUT=ROOT/'analysis_mechanisms_2026-09-29'; OUT.mkdir(exist_ok=True)
DATA=ROOT/'docs'/'data'; DATA.mkdir(exist_ok=True)

def eff(s):
 s=s.double().clamp_min(0); total=s.sum()
 if total==0:return 0.0
 p=s/total;p=p[p>0];return float(torch.exp(-(p*p.log()).sum()))
def stats(s):
 s=s.double(); top=float(s.max()) if s.numel() else 0.0
 return {'numerical_rank':int((s>max(top*1e-7,1e-12)).sum()),'effective_rank':eff(s),'stable_rank':float((s.square().sum()/(top*top)).item()) if top else 0.0,'spectral_norm':top,'frobenius_norm':float(torch.linalg.vector_norm(s))}
def lowrank_svals(a,b):
 qa,ra=torch.linalg.qr(a.double(),mode='reduced');qb,rb=torch.linalg.qr(b.double().T,mode='reduced');return torch.linalg.svdvals(ra@rb.T)
def write(name,rows):
 with (DATA/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)

# Fixed held-out sample: WikiText-2 public test, first consecutive 128-token block,
# positions 0..31 at the input of block-0 attn.c_proj. This activation is upstream
# of the adapter and therefore identical for all compared checkpoints.
tok=AutoTokenizer.from_pretrained('gpt2'); raw=load_dataset('wikitext','wikitext-2-raw-v1')
ids=tok('\n'.join(x for x in raw['test']['text'] if x.strip()),add_special_tokens=False)['input_ids'][:128]
model=AutoModelForCausalLM.from_pretrained('gpt2'); captured=[]
hook=model.transformer.h[0].attn.c_proj.register_forward_pre_hook(lambda m,a: captured.append(a[0].detach().double()))
with torch.no_grad(): model(torch.tensor(ids).unsqueeze(0));hook.remove()
x=captured[0][0,:32,:]; base=model.transformer.h[0].attn.c_proj(x.float()).double(); c=base.mean(0);c=c/(c.norm()+1e-12)
(OUT/'heldout_sample.json').write_text(json.dumps({'dataset':'wikitext-2-raw-v1','split':'test','block_index':0,'sequence_length':128,'token_positions':list(range(32)),'activation':'input to transformer.h[0].attn.c_proj','scalar_reduction':'dot(adapter_output, normalized mean frozen c_proj output over fixed sample)'},indent=2))

lora=[];sym=[];jac=[];inter=[]
for ck in sorted(RUNS.glob('*/checkpoints/adapter_final.pt')):
 payload=torch.load(ck,map_location='cpu',weights_only=False);md=payload['metadata'];state=payload['adapter_state'];kind=md['kind'];r=int(md['rank']);seed=int(md['seed']);scale=float(md['alpha'])/r
 prefix='transformer.h.0.attn.c_proj.';U=state[prefix+'U'].double()
 if kind=='lora':
  V=state[prefix+'V'].double();sv=lowrank_svals(scale*U,V); row={'method':'LoRA','rank':r,'seed':seed,**stats(sv)};lora.append(row)
  js=sv
  hnorm=0.0;off=0.0;diag=0.0
 else:
  P=state[prefix+'P'].double();su=torch.linalg.svdvals(U);sp=torch.linalg.svdvals(P);outputs=scale*((x@U).square()@P);cov=torch.cov(outputs.T);scov=torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
  for name,svv in [('U',su),('P',sp),('output_covariance',scov)]:sym.append({'method':'Symmetric Quadratic','component':name,'rank':r,'seed':seed,**stats(svv)})
  jstats=[]
  for xx in x:
   z=xx@U;sv=lowrank_svals(2*scale*U* z.unsqueeze(0),P);jstats.append(stats(sv))
  js=None
  weights=P@c;H=2*scale*(U*weights.unsqueeze(0))@U.T;hnorm=float(H.norm());diag=float(H.diag().abs().mean());off=float((H.abs().sum()-H.diag().abs().sum())/(H.numel()-H.shape[0]));density=float((H.abs()>H.abs().max()*1e-3).double().mean()) if H.abs().max()>0 else 0
  inter.append({'method':'Symmetric Quadratic','rank':r,'seed':seed,'hessian_frobenius_norm':hnorm,'mean_abs_diagonal':diag,'mean_abs_off_diagonal':off,'offdiag_to_diag':off/(diag+1e-12),'interaction_density_at_1e-3_max':density})
 for pos in range(len(x)):
  if kind=='lora': st=stats(js)
  else: st=jstats[pos]
  jac.append({'method':'LoRA' if kind=='lora' else 'Symmetric Quadratic','rank':r,'seed':seed,'token_position':pos,**st})
 if kind=='lora':inter.append({'method':'LoRA','rank':r,'seed':seed,'hessian_frobenius_norm':0.0,'mean_abs_diagonal':0.0,'mean_abs_off_diagonal':0.0,'offdiag_to_diag':0.0,'interaction_density_at_1e-3_max':0.0})
write('lora_spectral_analysis.csv',lora);write('symmetric_spectral_analysis.csv',sym);write('jacobian_analysis.csv',jac);write('interaction_analysis.csv',inter)
relation=[]
for ck in sorted(RUNS.glob('*/summary.json')):
 s=json.loads(ck.read_text());method='LoRA' if s['kind']=='lora' else 'Symmetric Quadratic';rank=int(s['rank']);seed=int(s['seed'])
 jj=[q for q in jac if q['method']==method and q['rank']==rank and q['seed']==seed];hh=next(q for q in inter if q['method']==method and q['rank']==rank and q['seed']==seed)
 cov=next((q for q in sym if q['component']=='output_covariance' and q['rank']==rank and q['seed']==seed),None)
 relation.append({'method':method,'rank':rank,'seed':seed,'final_test_loss':s['final_test_loss'],'mean_local_jacobian_effective_rank':sum(q['effective_rank'] for q in jj)/len(jj),'mean_local_jacobian_spectral_norm':sum(q['spectral_norm'] for q in jj)/len(jj),'output_covariance_effective_rank':None if cov is None else cov['effective_rank'],'adapter_hessian_frobenius_norm':hh['hessian_frobenius_norm']})
write('structure_performance_relation.csv',relation)
print(json.dumps({'lora_checkpoints':len(lora),'symmetric_components':len(sym),'jacobian_rows':len(jac),'interaction_rows':len(inter)},indent=2))
