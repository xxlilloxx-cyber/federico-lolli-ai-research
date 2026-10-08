#!/usr/bin/env python3
"""Post-training spectra, local Jacobians, Hessians, and Combined ablations."""
from __future__ import annotations
import argparse,csv,json,math,sys
from pathlib import Path
import numpy as np,torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM,AutoTokenizer
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from experiments.combined_extension.adapters import CombinedAdapter,insert_adapter
from src.adapters import LowRankAdapter

def erank(s):
 s=s.double().clamp_min(0);total=s.sum()
 if total==0:return 0.0
 p=s/total;p=p[p>0];return float(torch.exp(-(p*p.log()).sum()))
def lowrank_singular(left,right):
 ql,rl=torch.linalg.qr(left.double(),mode='reduced');qr,rr=torch.linalg.qr(right.double().T,mode='reduced');return torch.linalg.svdvals(rl@rr.T)
def main(path):
 cfg=json.loads(path.read_text());dev=torch.device(cfg['device']);tok=AutoTokenizer.from_pretrained(cfg['model']);raw=load_dataset('wikitext','wikitext-2-raw-v1');seq=128
 def blocks(split):
  ids=tok('\n'.join(x for x in raw[split]['text'] if x.strip()),add_special_tokens=False)['input_ids'];return [torch.tensor(ids[i:i+seq]) for i in range(0,len(ids)-seq,seq)]
 validation,test=blocks('validation'),blocks('test');rows=[];rank_root=Path(cfg['rank_result_root'])
 for summary_path in sorted(rank_root.glob('*/summary.json')):
  summary=json.loads(summary_path.read_text());run_cfg=json.loads((summary_path.parent/'config.json').read_text());final=summary_path.parent/'checkpoints'/f"adapter_step_{run_cfg['steps']:05d}.pt"
  if not summary.get('complete') or not final.exists():continue
  dtype=torch.float16 if dev.type=='cuda' else torch.float32;model=AutoModelForCausalLM.from_pretrained(cfg['model'],dtype=dtype).to(dev);model.config.use_cache=False
  insert_adapter(model,run_cfg['method'],rank=run_cfg.get('rank'),rank_lora=run_cfg.get('rank_lora'),rank_symmetric=run_cfg.get('rank_symmetric'),alpha=run_cfg['alpha'],scaling_mode=run_cfg.get('scaling_mode','total_rank'),layers=run_cfg['layers'],projections=run_cfg['projections'])
  payload=torch.load(final,map_location=dev,weights_only=False);model.load_state_dict(payload['trainable_state'],strict=False);module=model.transformer.h[0].attn.c_proj;captured=[]
  hook=module.register_forward_pre_hook(lambda _m,args:captured.append(args[0].detach().float().cpu()));x=validation[0].unsqueeze(0).to(dev);model.eval()
  with torch.no_grad():model(x,use_cache=False)
  hook.remove();activations=captured[0][0,:cfg['token_positions']];method=run_cfg['method'];factor={};jac=[];jac_spectra=[];outputs=[];lora_outputs=[];sym_outputs=[]
  if isinstance(module,CombinedAdapter):
   A,B,U,P=[v.detach().float().cpu() for v in (module.A,module.B,module.U,module.P)];sl,ss=module.scales
   sa,sb,su,sp=(torch.linalg.svdvals(v) for v in (A,B,U,P));factor={'a_effective_rank':erank(sa),'b_effective_rank':erank(sb),'u_effective_rank':erank(su),'p_effective_rank':erank(sp),'a_singular_values':json.dumps(sa.tolist()),'b_singular_values':json.dumps(sb.tolist()),'u_singular_values':json.dumps(su.tolist()),'p_singular_values':json.dumps(sp.tolist())}
   for sample in activations:
    zl=sl*(sample@A)@B;zs=ss*((sample@U).square())@P;lora_outputs.append(zl);sym_outputs.append(zs);outputs.append(zl+zs)
    left=torch.cat((sl*A,2*ss*U*(sample@U).unsqueeze(0)),dim=1);right=torch.cat((B,P),dim=0);s=lowrank_singular(left,right);jac.append(erank(s));jac_spectra.append(s)
  elif module.kind=='lora':
   A,B=module.U.detach().float().cpu(),module.V.detach().float().cpu();scale=module.alpha/module.rank;s=lowrank_singular(scale*A,B);sa,sb=torch.linalg.svdvals(A),torch.linalg.svdvals(B);factor={'a_effective_rank':erank(sa),'b_effective_rank':erank(sb),'update_effective_rank':erank(s),'a_singular_values':json.dumps(sa.tolist()),'b_singular_values':json.dumps(sb.tolist()),'update_singular_values':json.dumps(s.tolist())};jac=[erank(s)]*len(activations);jac_spectra=[s]*len(activations);outputs=[scale*(sample@A)@B for sample in activations]
  else:
   U,P=module.U.detach().float().cpu(),module.P.detach().float().cpu();scale=module.alpha/module.rank;su,sp=torch.linalg.svdvals(U),torch.linalg.svdvals(P);factor={'u_effective_rank':erank(su),'p_effective_rank':erank(sp),'u_singular_values':json.dumps(su.tolist()),'p_singular_values':json.dumps(sp.tolist())}
   for sample in activations:
    outputs.append(scale*((sample@U).square())@P);s=lowrank_singular(2*scale*U*(sample@U).unsqueeze(0),P);jac.append(erank(s));jac_spectra.append(s)
  output=torch.stack(outputs);cov_s=torch.linalg.svdvals(output-output.mean(0));maxlen=max(len(s) for s in jac_spectra);mean_s=torch.stack([torch.nn.functional.pad(s,(0,maxlen-len(s))) for s in jac_spectra]).mean(0);row={'method':method,'seed':run_cfg['seed'],'rank':run_cfg.get('rank'),'rank_lora':run_cfg.get('rank_lora'),'rank_symmetric':run_cfg.get('rank_symmetric'),'jacobian_effective_rank_mean':float(np.mean(jac)),'jacobian_singular_values_mean':json.dumps(mean_s.tolist()),'output_covariance_effective_rank':erank(cov_s),'output_covariance_singular_values':json.dumps(cov_s.tolist()),**factor}
  if method in ('symmetric','combined'):
   U=module.U.detach().float().cpu();P=module.P.detach().float().cpu();scale=(module.alpha/module.rank if method=='symmetric' else module.scales[1]);c=output.mean(0);c=c/(c.norm()+1e-12);H=2*scale*U@torch.diag(P@c)@U.T;row['adapter_hessian_frobenius_norm']=float(H.norm())
  if method=='combined':
   lo,so=torch.stack(lora_outputs),torch.stack(sym_outputs);row.update(lora_output_norm_mean=float(lo.norm(dim=1).mean()),symmetric_output_norm_mean=float(so.norm(dim=1).mean()),combined_output_norm_mean=float((lo+so).norm(dim=1).mean()),branch_cosine_mean=float(torch.nn.functional.cosine_similarity(lo,so,dim=1).mean()))
   def eval_loss(data,limit):
    with torch.no_grad():return float(torch.stack([model(t.unsqueeze(0).to(dev),labels=t.unsqueeze(0).to(dev),use_cache=False).loss.float() for t in data[:limit]]).mean())
   module.lora_enabled=module.symmetric_enabled=True;fv,ft=eval_loss(validation,8),eval_loss(test,cfg['test_blocks']);module.lora_enabled=False;lv,lt=eval_loss(validation,8),eval_loss(test,cfg['test_blocks']);module.lora_enabled=True;module.symmetric_enabled=False;sv,st=eval_loss(validation,8),eval_loss(test,cfg['test_blocks']);module.symmetric_enabled=True
   row.update(full_validation_loss=fv,full_test_loss=ft,without_lora_validation_degradation=lv-fv,without_lora_test_degradation=lt-ft,without_symmetric_validation_degradation=sv-fv,without_symmetric_test_degradation=st-ft)
  rows.append(row);del model
 if not rows:raise RuntimeError('no complete rank checkpoints available for mechanism analysis')
 out=Path(cfg['output_dir']);out.mkdir(parents=True,exist_ok=True);(out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n');fields=sorted({k for r in rows for k in r})
 with (out/'metrics.csv').open('w',newline='') as h:w=csv.DictWriter(h,fieldnames=fields);w.writeheader();w.writerows(rows)
 (out/'summary.json').write_text(json.dumps({**cfg,'complete':True,'analyzed_checkpoints':len(rows),'heldout_definition':f"WikiText-2 validation block 0, first {cfg['token_positions']} tokens; branch ablation test uses first {cfg['test_blocks']} test blocks"},indent=2)+'\n')
if __name__=='__main__':p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);main(p.parse_args().config)
