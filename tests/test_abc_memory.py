from __future__ import annotations
import sys
from pathlib import Path
import torch
from torch import nn
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'experiments'/'factual_learning_lora_symmetric'
sys.path[:0]=[str(ROOT),str(EXP)]
from abc_memory import (EncodedExample,MemoryAdapterBank,_mean_per_example_causal_loss,
                        _save_checkpoint,capture_ewc_reference,compare_snapshot,
                        estimate_diagonal_fisher,ewc_quadratic_penalty,
                        load_bank_checkpoint,slot_snapshot)
from dataset import generate_facts


class ConvLike(nn.Module):
    def __init__(self,d=8):
        super().__init__();self.weight=nn.Parameter(torch.randn(d,d));self.bias=nn.Parameter(torch.randn(d))
    def forward(self,x):return x@self.weight+self.bias


class FakeModel(nn.Module):
    def __init__(self,d=8):
        super().__init__();block=nn.Module();block.attn=nn.Module();block.attn.c_proj=ConvLike(d)
        self.transformer=nn.Module();self.transformer.h=nn.ModuleList([block])


def test_abc_dataset_is_deterministic_disjoint_and_complete():
    first=generate_facts(20261002,12,("A","B","C"));second=generate_facts(20261002,12,("A","B","C"))
    assert first==second and len(first)==36
    assert {s:sum(f.fact_set==s for f in first) for s in "ABC"}=={"A":12,"B":12,"C":12}
    assert len({f.subject for f in first})==36 and len({f.answer for f in first})==36
    assert len(generate_facts(42,2))==4  # legacy A/B default remains unchanged


def test_slot_parameter_budgets():
    for method in ("lora","symmetric","combined"):
        bank=MemoryAdapterBank(ConvLike(768),method)
        bank.add_slot("A");assert bank.adapter_parameter_count()==6144
        bank.add_slot("B");bank.add_slot("C");assert bank.adapter_parameter_count()==18432


def test_frozen_slot_is_bit_identical_while_new_slot_learns():
    torch.manual_seed(4);bank=MemoryAdapterBank(ConvLike(),"combined");bank.add_slot("A")
    with torch.no_grad():
        bank.slots["A"].B.normal_();bank.slots["A"].P.normal_()
    reference=slot_snapshot(bank,"A");bank.add_slot("B");bank.set_trainable({"B"})
    optimizer=torch.optim.SGD(bank.slot_parameters("B"),lr=.1)
    for _ in range(3):
        optimizer.zero_grad();bank(torch.randn(4,8)).square().mean().backward();optimizer.step()
    check=compare_snapshot(bank,"A",reference)
    assert check=={"exact_equal":True,"max_abs_difference":0.0,"l2_difference":0.0}
    assert any(float(p.detach().abs().sum())>0 for name,p in bank.slots["B"].named_parameters() if name.endswith(("B","P")))


def test_slot_ablation_activation_changes_only_selected_corrections():
    torch.manual_seed(8);bank=MemoryAdapterBank(ConvLike(),"lora");bank.add_slot("A");bank.add_slot("B")
    with torch.no_grad():
        bank.slots["A"].V.normal_();bank.slots["B"].V.normal_()
    x=torch.randn(2,8);bank.activate({"A"});a=bank(x);bank.activate({"B"});b=bank(x);bank.activate();both=bank(x)
    assert torch.allclose(both,a+b-bank.base(x),atol=1e-6)


def test_checkpoint_reloads_all_named_slots(tmp_path):
    source=FakeModel();bank=MemoryAdapterBank(source.transformer.h[0].attn.c_proj,"symmetric")
    source.transformer.h[0].attn.c_proj=bank;bank.add_slot("A");bank.add_slot("B")
    with torch.no_grad():bank.slots["A"].P.normal_();bank.slots["B"].P.normal_()
    path=tmp_path/'after_B.pt';_save_checkpoint(bank,path,{"phase":"B"})
    target=FakeModel();loaded,metadata=load_bank_checkpoint(target,"symmetric",path)
    assert metadata["phase"]=="B" and list(loaded.slots)==["A","B"]
    for key,value in slot_snapshot(bank,"A").items():assert torch.equal(value,slot_snapshot(loaded,"A")[key])


def test_microbatch_loss_preserves_mean_per_example_objective():
    torch.manual_seed(11);logits=torch.randn(2,6,13,dtype=torch.double)
    labels=torch.tensor([[-100,-100,2,3,4,-100],[-100,5,6,-100,-100,-100]])
    batched=_mean_per_example_causal_loss(logits,labels)
    separate=torch.stack([_mean_per_example_causal_loss(logits[i:i+1],labels[i:i+1]) for i in range(2)]).mean()
    assert torch.allclose(batched,separate,atol=0,rtol=0)


def test_ewc_penalty_zero_then_positive_and_changes_gradient():
    torch.manual_seed(12);bank=MemoryAdapterBank(ConvLike(),"lora");bank.add_slot("shared")
    reference=capture_ewc_reference(bank);fisher={name:torch.ones_like(value) for name,value in reference.items()}
    assert ewc_quadratic_penalty(bank,reference,fisher).item()==0.0
    parameter=next(iter(bank.slot_parameters("shared")))
    with torch.no_grad():parameter.add_(0.01)
    penalty=ewc_quadratic_penalty(bank,reference,fisher);assert penalty.item()>0
    data_loss=bank(torch.randn(2,8)).square().mean()
    grad_plain=torch.autograd.grad(data_loss,parameter,retain_graph=True)[0]
    grad_zero=torch.autograd.grad(data_loss+0.5*0.0*penalty,parameter,retain_graph=True)[0]
    grad_ewc=torch.autograd.grad(data_loss+50.0*penalty,parameter)[0]
    assert torch.equal(grad_plain,grad_zero)  # lambda=0 is exactly the sequential objective
    assert not torch.equal(grad_zero,grad_ewc)
    assert all(not p.requires_grad for p in bank.base.parameters())
    assert sum(p.numel() for p in bank.parameters() if p.requires_grad)==64


class TinyLM(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.Embedding(17,8);self.bank=MemoryAdapterBank(ConvLike(),"combined");self.bank.add_slot("shared");self.head=nn.Linear(8,17,bias=False)
        for p in self.embedding.parameters():p.requires_grad_(False)
        for p in self.head.parameters():p.requires_grad_(False)
    def forward(self,input_ids,attention_mask=None,use_cache=False):
        return SimpleNamespace(logits=self.head(self.bank(self.embedding(input_ids))))


def test_fisher_is_adapter_only_finite_and_nonnegative():
    torch.manual_seed(13);model=TinyLM();examples=[]
    for token in (2,3,4):
        ids=torch.tensor([1,token,5]);labels=torch.tensor([-100,token,5]);prompt=torch.tensor([1])
        examples.append(EncodedExample(ids,labels,prompt,(token,5)))
    fisher=estimate_diagonal_fisher(model,model.bank,examples,samples=3,batches=2,pad_id=0,device=torch.device('cpu'))
    assert fisher.keys()==capture_ewc_reference(model.bank).keys()
    assert sum(value.numel() for value in fisher.values())==sum(p.numel() for p in model.bank.parameters() if p.requires_grad)
    assert all(torch.isfinite(value).all() and (value>=0).all() for value in fisher.values())
    assert any(float(value.sum())>0 for value in fisher.values())
