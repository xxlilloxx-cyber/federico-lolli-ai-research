from __future__ import annotations
import importlib.util,json,sys
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.combined_extension.adapters import CombinedAdapter,insert_adapter,parameter_counts

class ConvLike(nn.Module):
 def __init__(self,din=768,dout=768):super().__init__();self.weight=nn.Parameter(torch.randn(din,dout));self.bias=nn.Parameter(torch.randn(dout))
 def forward(self,x):return x@self.weight+self.bias
class FakeModel(nn.Module):
 def __init__(self):
  super().__init__();block=nn.Module();block.attn=nn.Module();block.attn.c_proj=ConvLike();self.transformer=nn.Module();self.transformer.h=nn.ModuleList([block])

def test_primary_total_rank_formula_and_branch_ablation():
 torch.manual_seed(3);base=ConvLike(8,8);module=CombinedAdapter(base,2,2,alpha=4,scaling_mode='total_rank')
 with torch.no_grad():module.B.normal_();module.P.normal_()
 x=torch.randn(3,8);linear,symmetric=module.branch_outputs(x)
 assert module.scales==(1.0,1.0);assert torch.allclose(module(x),base(x)+linear+symmetric)
 module.lora_enabled=False;assert torch.allclose(module(x),base(x)+symmetric)
 module.lora_enabled=True;module.symmetric_enabled=False;assert torch.allclose(module(x),base(x)+linear)

def test_branchwise_scaling_is_explicit_and_different():
 base=ConvLike(8,8);primary=CombinedAdapter(base,1,3,4,'total_rank');ablation=CombinedAdapter(base,1,3,4,'branch_wise')
 assert primary.scales==(1.0,1.0);assert ablation.scales==(4.0,4/3)

def test_programmatic_gpt2_projection_counts():
 expected={1:1536,2:3072,4:6144,8:12288}
 for rank,count in expected.items():
  for method in ('lora','symmetric'):
   model=FakeModel();insert_adapter(model,method,rank=rank);assert parameter_counts(model).total_trainable==count
 for pair,count in ((1,3072),(2,6144),(4,12288)):
  model=FakeModel();insert_adapter(model,'combined',rank_lora=pair,rank_symmetric=pair);assert parameter_counts(model).total_trainable==count

def test_combined_both_branches_receive_gradients():
 model=FakeModel();insert_adapter(model,'combined',rank_lora=2,rank_symmetric=2);module=model.transformer.h[0].attn.c_proj;opt=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1)
 for _ in range(2):opt.zero_grad();module(torch.randn(2,768)).square().mean().backward();opt.step()
 opt.zero_grad();module(torch.randn(2,768)).square().mean().backward()
 assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm()>0 for p in (module.A,module.B,module.U,module.P))

def test_master_plan_counts_and_multilayer_budget():
 spec=importlib.util.spec_from_file_location('master',ROOT/'run_full_combined_study.py');master=importlib.util.module_from_spec(spec);sys.modules['master']=master;spec.loader.exec_module(master)
 cfg=json.loads((ROOT/'config/full_combined_study.json').read_text());plan=master.build_plan(cfg,'cpu')
 counts={name:sum(c.campaign==name for c in plan) for name in {c.campaign for c in plan}}
 assert counts=={'rank_confirmation':33,'scaling':22,'placement':36,'attention_mlp':12,'multilayer':15,'ag_news':9,'benchmark':6,'mechanism':1}
 combined=[c for c in plan if c.campaign=='multilayer' and c.config['method']=='combined']
 assert len(combined)==3 and all(c.config['expected_trainable_parameters']==12288 for c in combined)
 assert all(c.config['scaling_mode']=='total_rank' for c in combined)
