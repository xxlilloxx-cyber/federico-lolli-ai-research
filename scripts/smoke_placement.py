import sys, torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,'src')
from adapters import freeze_and_insert, parameter_counts
mods=['attn.c_proj','mlp.c_fc','mlp.c_proj']
for mod in mods:
 for kind in ['lora','symmetric_quadratic']:
  m=AutoModelForCausalLM.from_pretrained('gpt2',torch_dtype=torch.float16)
  freeze_and_insert(m,kind,rank=4,alpha=4.0,layers=(0,),projections=(mod,))
  x=torch.randint(0,100,(1,16)); out=m(x,labels=x).loss
  out.backward()
  grads=[p.grad for p in m.parameters() if p.requires_grad]
  finite=all(g is not None and torch.isfinite(g).all() for g in grads)
  print(mod,kind,parameter_counts(m)['trainable_parameters'],float(out),finite)
