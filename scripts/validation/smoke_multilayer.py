import sys, torch
from transformers import AutoModelForCausalLM
sys.path.insert(0,'src')
from adapters import freeze_and_insert, parameter_counts
for kind in ['lora','symmetric_quadratic']:
 for layers,rank in [([2],4),([0,3],2),([0,3,7,11],1)]:
  m=AutoModelForCausalLM.from_pretrained('gpt2',torch_dtype=torch.float16)
  freeze_and_insert(m,kind,rank=rank,alpha=4.0,layers=tuple(layers),projections=('attn.c_proj',))
  c=parameter_counts(m); x=torch.randint(0,100,(1,16)); loss=m(x,labels=x).loss; loss.backward()
  finite=all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters() if p.requires_grad)
  frozen=all(p.grad is None for n,p in m.named_parameters() if not p.requires_grad)
  print(kind,layers,'rank',rank,'params',c['trainable_parameters'],'loss',float(loss),'finite',finite,'frozen',frozen)
