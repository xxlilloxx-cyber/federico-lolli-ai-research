"""A→B→C continual factual learning with reusable factual-data/adapters."""
from __future__ import annotations

import csv
import json
import math
import random
import statistics
import time
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import nn

from dataset import examples_for, generate_facts, serializable_dataset
from modeling import ParallelLoRASymmetricAdapter
from run_experiment import _encoded, evaluate, set_seed
from src.adapters import LowRankAdapter


METHODS = ("lora", "symmetric", "combined")
PROTOCOLS = ("sequential_single", "grow_unfrozen", "hard_consolidation", "ewc_like")
SETS = ("A", "B", "C")


@dataclass(frozen=True)
class EncodedExample:
    """Tokenized factual example retained in host memory."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    prompt_ids: torch.Tensor
    answer_ids: tuple[int, ...]


def _encode_examples(examples: list, tokenizer: Any, max_length: int,
                     pin_memory: bool = False) -> list[EncodedExample]:
    encoded=[]
    for example in examples:
        prompt_ids=tokenizer.encode(example.prompt,add_special_tokens=False)
        answer_ids=tokenizer.encode(example.answer,add_special_tokens=False)
        if not answer_ids:
            raise ValueError(f"answer tokenized to zero tokens: {example.fact_id}")
        if len(prompt_ids)+len(answer_ids)>max_length:
            keep=max_length-len(answer_ids)
            if keep<1:raise ValueError("sequence length is too short for the target answer")
            prompt_ids=prompt_ids[-keep:]
        ids=torch.tensor(prompt_ids+answer_ids,dtype=torch.long)
        labels=ids.clone();labels[:len(prompt_ids)]=-100
        prompt=torch.tensor(prompt_ids,dtype=torch.long)
        if pin_memory:
            ids=ids.pin_memory();labels=labels.pin_memory();prompt=prompt.pin_memory()
        encoded.append(EncodedExample(ids,labels,prompt,tuple(answer_ids)))
    return encoded


def _batch(examples: list[EncodedExample], indices: list[int], pad_id: int,
           device: torch.device) -> tuple[torch.Tensor,torch.Tensor,torch.Tensor]:
    chosen=[examples[i] for i in indices];length=max(x.input_ids.numel() for x in chosen)
    pinned=device.type=="cuda"
    ids=torch.full((len(chosen),length),pad_id,dtype=torch.long,pin_memory=pinned)
    labels=torch.full((len(chosen),length),-100,dtype=torch.long,pin_memory=pinned)
    attention=torch.zeros((len(chosen),length),dtype=torch.long,pin_memory=pinned)
    for row,item in enumerate(chosen):
        n=item.input_ids.numel();ids[row,:n]=item.input_ids;labels[row,:n]=item.labels;attention[row,:n]=1
    non_blocking=pinned
    return (ids.to(device,non_blocking=non_blocking),labels.to(device,non_blocking=non_blocking),
            attention.to(device,non_blocking=non_blocking))


def _mean_per_example_causal_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Match the historical mean of per-example target-token losses in a microbatch."""
    shifted_logits=logits[:,:-1,:].contiguous();shifted_labels=labels[:,1:].contiguous()
    losses=torch.nn.functional.cross_entropy(
        shifted_logits.float().view(-1,shifted_logits.shape[-1]),shifted_labels.view(-1),
        ignore_index=-100,reduction="none").view(shifted_labels.shape)
    mask=shifted_labels.ne(-100);counts=mask.sum(dim=1).clamp_min(1)
    return ((losses*mask).sum(dim=1)/counts).mean()


def adapter_named_parameters(bank: "MemoryAdapterBank") -> dict[str,nn.Parameter]:
    return {name:parameter for name,parameter in bank.named_parameters()
            if name.startswith("slots.")}


def capture_ewc_reference(bank: "MemoryAdapterBank") -> dict[str,torch.Tensor]:
    return {name:p.detach().clone() for name,p in adapter_named_parameters(bank).items()}


def ewc_quadratic_penalty(bank: "MemoryAdapterBank", reference: dict[str,torch.Tensor],
                          fisher: dict[str,torch.Tensor]) -> torch.Tensor:
    parameters=adapter_named_parameters(bank)
    if parameters.keys()!=reference.keys() or parameters.keys()!=fisher.keys():
        raise ValueError("EWC state does not match adapter parameters")
    terms=[(fisher[name]*(parameter-reference[name]).square()).sum()
           for name,parameter in parameters.items()]
    return sum(terms,torch.zeros((),device=next(iter(parameters.values())).device))


def estimate_diagonal_fisher(model: nn.Module, bank: "MemoryAdapterBank",
                             examples: list[EncodedExample], *, samples: int,
                             batches: int, pad_id: int, device: torch.device,
                             autocast_context: Callable[[],Any] | None = None,
                             selection_seed: int = 0) -> dict[str,torch.Tensor]:
    """Exact mean of squared per-example score gradients for adapter parameters."""
    if samples<1 or batches<1:raise ValueError("fisher samples and batches must be positive")
    count=min(samples,len(examples));indices=list(range(len(examples)))
    random.Random(selection_seed).shuffle(indices);indices=indices[:count]
    groups=np.array_split(indices,min(batches,count));parameters=adapter_named_parameters(bank)
    fisher={name:torch.zeros_like(parameter) for name,parameter in parameters.items()}
    model.eval()
    # Grouping controls deterministic work partitioning; each score is still differentiated
    # separately so the result is the requested mean of per-example squared gradients.
    for group in groups:
        for index in group.tolist():
            model.zero_grad(set_to_none=True)
            ids,labels,attention=_batch(examples,[index],pad_id,device)
            context=autocast_context() if autocast_context else nullcontext()
            with context:
                output=model(input_ids=ids,attention_mask=attention,use_cache=False)
                log_probability=-_mean_per_example_causal_loss(output.logits,labels)
            gradients=torch.autograd.grad(log_probability,tuple(parameters.values()),allow_unused=False)
            for (name,_),gradient in zip(parameters.items(),gradients):
                fisher[name].add_(gradient.detach().square())
    for value in fisher.values():value.div_(count)
    model.zero_grad(set_to_none=True)
    if not all(torch.isfinite(value).all() for value in fisher.values()):
        raise FloatingPointError("non-finite Fisher diagonal")
    return fisher


class ZeroProjection(nn.Module):
    """Shape-compatible zero base used to reuse validated adapter modules."""
    def __init__(self, d_in: int, d_out: int, device: torch.device):
        super().__init__()
        self.register_buffer("weight", torch.empty(d_in, d_out, device=device), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((*x.shape[:-1], self.weight.shape[1]))


class MemoryAdapterBank(nn.Module):
    """Frozen projection plus an additive bank of independently addressable slots."""
    def __init__(self, base: nn.Module, method: str, alpha: float = 4.0):
        super().__init__()
        if method not in METHODS:
            raise ValueError(method)
        self.base, self.method, self.alpha = base, method, float(alpha)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        self.slots = nn.ModuleDict()
        self.active_slots: set[str] = set()

    def add_slot(self, name: str) -> nn.Module:
        if name in self.slots:
            raise ValueError(f"duplicate memory slot: {name}")
        d_in, d_out = self.base.weight.shape
        zero = ZeroProjection(d_in, d_out, self.base.weight.device)
        if self.method == "lora":
            slot = LowRankAdapter(zero, 4, self.alpha, "lora")
        elif self.method == "symmetric":
            slot = LowRankAdapter(zero, 4, self.alpha, "symmetric_quadratic")
        else:
            slot = ParallelLoRASymmetricAdapter(zero, 2, 2, self.alpha, self.alpha)
        self.slots[name] = slot.to(device=self.base.weight.device)
        self.active_slots.add(name)
        return slot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.base(x)
        for name, slot in self.slots.items():
            if name in self.active_slots:
                output = output + slot(x)
        return output

    def activate(self, names: set[str] | None = None) -> None:
        self.active_slots = set(self.slots) if names is None else set(names)
        unknown = self.active_slots.difference(self.slots)
        if unknown:
            raise ValueError(f"unknown slots: {sorted(unknown)}")

    def set_trainable(self, names: set[str]) -> None:
        for slot_name, slot in self.slots.items():
            for parameter in slot.parameters():
                parameter.requires_grad_(slot_name in names)

    def slot_parameters(self, name: str) -> list[nn.Parameter]:
        return list(self.slots[name].parameters())

    def adapter_parameter_count(self) -> int:
        return sum(parameter.numel() for slot in self.slots.values() for parameter in slot.parameters())


def insert_bank(model: nn.Module, method: str) -> MemoryAdapterBank:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    parent = model.transformer.h[0].attn
    bank = MemoryAdapterBank(parent.c_proj, method)
    parent.c_proj = bank
    return bank


def slot_snapshot(bank: MemoryAdapterBank, name: str) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in bank.slots[name].state_dict().items()
            if key.split(".")[-1] in {"A", "B", "U", "V", "P"}}


def compare_snapshot(bank: MemoryAdapterBank, name: str,
                     reference: dict[str, torch.Tensor]) -> dict[str, float | bool]:
    current = slot_snapshot(bank, name)
    if current.keys() != reference.keys():
        raise ValueError(f"slot {name} tensor keys changed")
    differences = [(current[key] - reference[key]) for key in current]
    exact = all(torch.equal(current[key], reference[key]) for key in current)
    max_abs = max((float(d.abs().max()) for d in differences if d.numel()), default=0.0)
    l2 = math.sqrt(sum(float(d.float().square().sum()) for d in differences))
    return {"exact_equal": exact, "max_abs_difference": max_abs, "l2_difference": l2}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _save_checkpoint(bank: MemoryAdapterBank, path: Path, metadata: dict[str, Any],
                     auxiliary_state: dict[str,Any] | None = None) -> None:
    state = {name: value.detach().cpu() for name, value in bank.state_dict().items()
             if name.startswith("slots.") and name.split(".")[-1] in {"A", "B", "U", "V", "P"}}
    path.parent.mkdir(parents=True, exist_ok=True)
    payload={"metadata": metadata,"adapter_state":state,"slots":list(bank.slots)}
    if auxiliary_state is not None:
        payload["auxiliary_state"]={group:{name:value.detach().cpu() for name,value in tensors.items()}
                                    for group,tensors in auxiliary_state.items()}
    torch.save(payload,path)


def load_bank_checkpoint(model: nn.Module, method: str, path: Path,
                         map_location: str | torch.device = "cpu") -> tuple[MemoryAdapterBank, dict[str, Any]]:
    """Rebuild and load every named memory slot from an adapter-only checkpoint."""
    payload = torch.load(path, map_location=map_location, weights_only=False)
    bank = insert_bank(model, method)
    for name in payload["slots"]:
        bank.add_slot(name)
    incompatible = bank.load_state_dict(payload["adapter_state"], strict=False)
    if incompatible.unexpected_keys:
        raise ValueError(f"unexpected checkpoint tensors: {incompatible.unexpected_keys}")
    if any(not key.startswith("base.") for key in incompatible.missing_keys):
        raise ValueError(f"missing adapter tensors: {incompatible.missing_keys}")
    bank.activate()
    return bank, payload["metadata"]


def _summary_metrics(matrix: list[dict[str, Any]]) -> dict[str, float]:
    lookup = {(row["training_phase"], row["evaluation_set"]): row for row in matrix}
    a_a, b_b, c_c = lookup[("A", "A")], lookup[("B", "B")], lookup[("C", "C")]
    a_b, a_c, b_c = lookup[("B", "A")], lookup[("C", "A")], lookup[("C", "B")]
    final = [lookup[("C", fact_set)] for fact_set in SETS]
    result = {
        "acquisition_a_nll": a_a["nll"], "acquisition_b_nll": b_b["nll"],
        "acquisition_c_nll": c_c["nll"],
        "forgetting_a_after_b_nll": a_b["nll"] - a_a["nll"],
        "forgetting_a_after_c_nll": a_c["nll"] - a_a["nll"],
        "forgetting_b_after_c_nll": b_c["nll"] - b_b["nll"],
        "forgetting_a_after_b_probability": a_b["correct_probability"] - a_a["correct_probability"],
        "forgetting_a_after_c_probability": a_c["correct_probability"] - a_a["correct_probability"],
        "forgetting_b_after_c_probability": b_c["correct_probability"] - b_b["correct_probability"],
        "forgetting_a_after_b_exact_match": a_b["exact_match"] - a_a["exact_match"],
        "forgetting_a_after_c_exact_match": a_c["exact_match"] - a_a["exact_match"],
        "forgetting_b_after_c_exact_match": b_c["exact_match"] - b_b["exact_match"],
        "mean_final_nll": statistics.mean(row["nll"] for row in final),
        "mean_final_probability": statistics.mean(row["correct_probability"] for row in final),
        "mean_final_exact_match": statistics.mean(row["exact_match"] for row in final),
    }
    return result


@torch.inference_mode()
def _evaluate_encoded(model: nn.Module, examples: list[EncodedExample], pad_id: int,
                      device: torch.device, exact_match: bool = True) -> dict[str,float]:
    model.eval();nlls=[];exact=[]
    for index,item in enumerate(examples):
        ids,labels,attention=_batch(examples,[index],pad_id,device)
        output=model(input_ids=ids,attention_mask=attention,use_cache=False)
        nll=float(_mean_per_example_causal_loss(output.logits,labels).detach())
        nlls.append(nll)
        if exact_match:
            prompt=item.prompt_ids.unsqueeze(0).to(device,non_blocking=device.type=="cuda")
            generated=model.generate(prompt,attention_mask=torch.ones_like(prompt),
                                     max_new_tokens=len(item.answer_ids),do_sample=False,
                                     pad_token_id=pad_id)[0,prompt.shape[1]:].tolist()
            exact.append(float(generated==list(item.answer_ids)))
    values=np.asarray(nlls,dtype=np.float64)
    return {"target_answer_nll":float(values.mean()),
            "correct_answer_probability":float(np.exp(-values).mean()),
            "exact_match_accuracy":float(np.mean(exact)) if exact else float("nan")}


def run_one(config: dict[str, Any], method: str, protocol: str, seed: int,
            run_dir: Path, device_name: str,
            phase_callback: Callable[[str], None] | None = None) -> dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if protocol not in PROTOCOLS:raise ValueError(protocol)
    set_seed(seed);device=torch.device(device_name);amp_mode=str(config.get("amp","fp16" if device.type=="cuda" else "off"))
    if device.type!="cuda" and amp_mode!="off":raise ValueError("AMP modes require CUDA")
    if amp_mode=="bf16" and not torch.cuda.is_bf16_supported():raise ValueError("BF16 is not supported by this GPU")
    dtype={"fp16":torch.float16,"bf16":torch.bfloat16,"off":torch.float32}[amp_mode]
    autocast_factory=(lambda:torch.autocast(device_type="cuda",dtype=dtype)) if amp_mode!="off" else (lambda:nullcontext())
    # Adapter-only FP16 gradients were historically trained without loss scaling.
    # Start at one to preserve that validated numerical regime while retaining
    # GradScaler's overflow detection and adaptive behavior.
    scaler=torch.cuda.amp.GradScaler(init_scale=1.0,growth_interval=2000,
                                     enabled=device.type=="cuda" and amp_mode=="fp16")
    tokenizer=AutoTokenizer.from_pretrained(config["model"]);tokenizer.pad_token=tokenizer.eos_token
    model=AutoModelForCausalLM.from_pretrained(config["model"],dtype=dtype,attn_implementation="eager").to(device)
    model.config.use_cache=False;bank=insert_bank(model,method)
    facts=generate_facts(config["dataset_seed"],config["facts_per_set"],SETS)
    by_set={name:[fact for fact in facts if fact.fact_set==name] for name in SETS}
    pin=device.type=="cuda";max_length=int(config["sequence_length"])
    train={name:_encode_examples(examples_for(by_set[name],"train"),tokenizer,max_length,pin) for name in SETS}
    validation={name:_encode_examples(examples_for(by_set[name],"validation"),tokenizer,max_length,pin) for name in SETS}
    paraphrase={name:_encode_examples(examples_for(by_set[name],"paraphrase"),tokenizer,max_length,pin) for name in SETS}
    micro=int(config.get("micro_batch_size",1));accum=int(config["gradient_accumulation"])
    if micro<1 or accum<1:raise ValueError("micro batch and accumulation must be positive")
    if micro*accum!=int(config.get("effective_batch_size",4)):
        raise ValueError("micro_batch_size * gradient_accumulation must equal effective_batch_size")
    run_dir.mkdir(parents=True,exist_ok=True)
    resolved={**config,"method":method,"protocol":protocol,"seed":seed,"device":device_name,"dtype":str(dtype),
              "amp":amp_mode,"effective_batch_size":micro*accum,"output_dir":str(run_dir)}
    (run_dir/"config.json").write_text(json.dumps(resolved,indent=2)+"\n")
    (run_dir/"dataset.json").write_text(json.dumps(serializable_dataset(facts),indent=2)+"\n")
    metrics=[];matrix=[];freeze_checks={};optimizer=None;global_step=0;snapshots={};started=time.perf_counter()
    ewc_reference:dict[str,torch.Tensor]|None=None;ewc_fisher:dict[str,torch.Tensor]|None=None
    training_elapsed=0.0;training_tokens=0
    if device.type=="cuda":torch.cuda.reset_peak_memory_stats(device)

    def evaluate_set(fact_set:str,data:list[EncodedExample],exact:bool=True)->dict[str,float]:
        return _evaluate_encoded(model,data,tokenizer.eos_token_id,device,exact)

    def record_matrix(phase:str)->None:
        for fact_set in SETS:
            primary=evaluate_set(fact_set,validation[fact_set],True);para=evaluate_set(fact_set,paraphrase[fact_set],True)
            matrix.append({"method":method,"protocol":protocol,"seed":seed,"training_phase":phase,
                           "evaluation_set":fact_set,"nll":primary["target_answer_nll"],
                           "correct_probability":primary["correct_answer_probability"],
                           "exact_match":primary["exact_match_accuracy"],"paraphrase_nll":para["target_answer_nll"],
                           "paraphrase_correct_probability":para["correct_answer_probability"],
                           "paraphrase_exact_match":para["exact_match_accuracy"]})
        _write_csv(run_dir/"memory_matrix.csv",matrix)

    # Required step-zero baseline, kept in the learning-curve file rather than the phase memory matrix.
    for fact_set in SETS:
        score=evaluate_set(fact_set,validation[fact_set],False)
        metrics.append({"step":0,"phase":"initial","local_step":0,"training_loss":"",
                        "evaluation_set":fact_set,"validation_nll":score["target_answer_nll"],
                        "correct_probability":score["correct_answer_probability"],
                        "adapter_parameters":0,"trainable_adapter_parameters":0,"ewc_penalty":0.0})
    _write_csv(run_dir/"metrics.csv",metrics)

    for phase_index,phase in enumerate(SETS):
        if protocol in {"sequential_single","ewc_like"}:
            if phase_index==0:
                bank.add_slot("shared");bank.set_trainable({"shared"})
                optimizer=torch.optim.AdamW(bank.slot_parameters("shared"),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        else:
            bank.add_slot(phase)
            if protocol=="grow_unfrozen":
                bank.set_trainable(set(bank.slots))
                if optimizer is None:optimizer=torch.optim.AdamW(bank.slot_parameters(phase),lr=config["learning_rate"],weight_decay=config["weight_decay"])
                else:optimizer.add_param_group({"params":bank.slot_parameters(phase)})
            else:
                bank.set_trainable({phase});optimizer=torch.optim.AdamW(bank.slot_parameters(phase),lr=config["learning_rate"],weight_decay=config["weight_decay"])
        assert optimizer is not None
        rng=random.Random(seed+100000*phase_index);order=[];phase_losses=[]
        local_steps=int(config["steps_per_phase"]);eval_interval=int(config["evaluation_every_steps"])
        if device.type=="cuda":torch.cuda.synchronize(device)
        segment_started=time.perf_counter()
        for local_step in range(1,local_steps+1):
            model.train();optimizer.zero_grad(set_to_none=True);losses=[];penalty_value=0.0
            for _ in range(accum):
                indices=[]
                for _ in range(micro):
                    if not order:order=list(range(len(train[phase])));rng.shuffle(order)
                    indices.append(order.pop())
                ids,labels,attention=_batch(train[phase],indices,tokenizer.eos_token_id,device)
                training_tokens+=int(attention.sum())
                with autocast_factory():
                    output=model(input_ids=ids,attention_mask=attention,use_cache=False)
                    data_loss=_mean_per_example_causal_loss(output.logits,labels)
                    penalty=ewc_quadratic_penalty(bank,ewc_reference,ewc_fisher) if protocol=="ewc_like" and ewc_reference is not None else data_loss.new_zeros(())
                    objective=data_loss+0.5*float(config.get("ewc_lambda",100.0))*penalty
                if not torch.isfinite(objective):raise FloatingPointError(f"non-finite loss at {phase}:{local_step}")
                scaler.scale(objective/accum).backward();losses.append(float(data_loss.detach()));penalty_value=float(penalty.detach())
            if scaler.is_enabled():scaler.unscale_(optimizer)
            parameters=[p for p in model.parameters() if p.requires_grad]
            if not all(p.grad is None or torch.isfinite(p.grad).all() for p in parameters):
                raise FloatingPointError(f"non-finite gradient at {phase}:{local_step}")
            torch.nn.utils.clip_grad_norm_(parameters,config["gradient_clipping"]);scaler.step(optimizer);scaler.update()
            global_step+=1;phase_losses.extend(losses)
            if local_step%eval_interval==0 or local_step==local_steps:
                if device.type=="cuda":torch.cuda.synchronize(device)
                training_elapsed+=time.perf_counter()-segment_started
                scored=evaluate_set(phase,validation[phase],False)
                metrics.append({"step":global_step,"phase":phase,"local_step":local_step,
                                "training_loss":float(np.mean(losses)),"evaluation_set":phase,
                                "validation_nll":scored["target_answer_nll"],
                                "correct_probability":scored["correct_answer_probability"],
                                "adapter_parameters":bank.adapter_parameter_count(),
                                "trainable_adapter_parameters":sum(p.numel() for p in bank.parameters() if p.requires_grad),
                                "ewc_penalty":penalty_value})
                _write_csv(run_dir/"metrics.csv",metrics)
                if device.type=="cuda":torch.cuda.synchronize(device)
                segment_started=time.perf_counter()
        record_matrix(phase)
        if protocol=="ewc_like" and phase in {"A","B"}:
            new_fisher=estimate_diagonal_fisher(model,bank,train[phase],samples=int(config["fisher_samples"]),
                                                batches=int(config["fisher_batches"]),pad_id=tokenizer.eos_token_id,
                                                device=device,autocast_context=autocast_factory,
                                                selection_seed=int(config["dataset_seed"])+phase_index)
            if phase=="A":ewc_fisher=new_fisher
            else:
                assert ewc_fisher is not None
                gamma=float(config.get("ewc_gamma",1.0));ewc_fisher={name:gamma*ewc_fisher[name]+new_fisher[name] for name in new_fisher}
            ewc_reference=capture_ewc_reference(bank)
        auxiliary=None
        if protocol=="ewc_like" and ewc_reference is not None and ewc_fisher is not None:
            auxiliary={"reference":ewc_reference,"fisher":ewc_fisher}
        _save_checkpoint(bank,run_dir/"checkpoints"/f"after_{phase}.pt",
                         {**resolved,"phase":phase,"step":global_step,"slots":list(bank.slots)},auxiliary)
        if protocol=="hard_consolidation":
            snapshots[phase]=slot_snapshot(bank,phase)
            if phase=="B":
                check=compare_snapshot(bank,"A",snapshots["A"]);freeze_checks["A_after_B"]=check
                if not check["exact_equal"]:raise RuntimeError("Adapter A changed during phase B")
            if phase=="C":
                for old in ("A","B"):
                    check=compare_snapshot(bank,old,snapshots[old]);freeze_checks[f"{old}_after_C"]=check
                    if not check["exact_equal"]:raise RuntimeError(f"Adapter {old} changed during phase C")
        if phase_callback:phase_callback(phase)

    bank.activate();ablations=[];selections=[("all",set(bank.slots))]
    if protocol in {"sequential_single","ewc_like"}:selections.append(("shared_only",{"shared"}))
    else:
        selections.extend((f"{name}_only",{name}) for name in SETS)
        if config.get("pairwise_slot_evaluation"):selections.extend((("A+B",{"A","B"}),("A+C",{"A","C"}),("B+C",{"B","C"})))
    for label,active in selections:
        bank.activate(active)
        for fact_set in SETS:
            scored=evaluate_set(fact_set,validation[fact_set],True)
            ablations.append({"method":method,"protocol":protocol,"seed":seed,"active_slots":label,
                              "evaluation_set":fact_set,"nll":scored["target_answer_nll"],
                              "correct_probability":scored["correct_answer_probability"],"exact_match":scored["exact_match_accuracy"]})
    bank.activate();_write_csv(run_dir/"memory_slot_ablation.csv",ablations);derived=_summary_metrics(matrix)
    auxiliary_bytes=0
    if ewc_reference is not None and ewc_fisher is not None:
        auxiliary_bytes=sum(x.numel()*x.element_size() for state in (ewc_reference,ewc_fisher) for x in state.values())
    summary={**resolved,"complete":True,"final_step":global_step,"slots":list(bank.slots),"parameters_per_slot":6144,
             "final_adapter_parameters":bank.adapter_parameter_count(),
             "final_trainable_adapter_parameters":sum(p.numel() for p in bank.parameters() if p.requires_grad),
             "ewc_auxiliary_state_bytes":auxiliary_bytes,"freeze_integrity":freeze_checks,
             "training_time_seconds":time.perf_counter()-started,"optimizer_training_seconds":training_elapsed,
             "optimizer_steps_per_second":global_step/training_elapsed,"training_tokens_per_second":training_tokens/training_elapsed,
             "peak_gpu_memory_mb":float(torch.cuda.max_memory_allocated(device)/1024**2) if device.type=="cuda" else 0.0,**derived}
    for value in summary.values():
        if isinstance(value,float) and not math.isfinite(value):raise FloatingPointError("non-finite summary")
    (run_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n");return summary

def aggregate(root: Path) -> None:
    summaries=[];memory=[];ablations=[]
    for path in sorted(root.glob("runs/*/*/*/summary.json")):
        row=json.loads(path.read_text())
        if row.get("complete") is not True:continue
        row["source"]=str(path.relative_to(root));summaries.append(row)
        memory.extend(csv.DictReader((path.parent/"memory_matrix.csv").open()))
        ablations.extend(csv.DictReader((path.parent/"memory_slot_ablation.csv").open()))
    if not summaries:return
    scalar=["method","protocol","seed","final_step","parameters_per_slot","final_adapter_parameters",
            "acquisition_a_nll","acquisition_b_nll","acquisition_c_nll","forgetting_a_after_b_nll",
            "forgetting_a_after_c_nll","forgetting_b_after_c_nll","forgetting_a_after_b_probability",
            "forgetting_a_after_c_probability","forgetting_b_after_c_probability","mean_final_nll",
            "mean_final_probability","mean_final_exact_match","training_time_seconds","source"]
    _write_csv(root/"per_run_results.csv",[{k:r.get(k) for k in scalar} for r in summaries])
    metrics=scalar[6:-2];agg=[]
    for (method,protocol),group in sorted(_groups(summaries,lambda r:(r["method"],r["protocol"])).items()):
        row={"method":method,"protocol":protocol,"n_seeds":len(group)}
        for key in metrics:
            values=[float(r[key]) for r in group];row[key+"_mean"]=statistics.mean(values);row[key+"_sample_sd"]=statistics.stdev(values) if len(values)>1 else ""
        agg.append(row)
    _write_csv(root/"aggregate_results.csv",agg)
    memagg=[]
    for key,group in sorted(_groups(memory,lambda r:(r["method"],r["protocol"],r["training_phase"],r["evaluation_set"])).items()):
        row=dict(zip(("method","protocol","training_phase","evaluation_set"),key));row["n_seeds"]=len(group)
        for field in ("nll","correct_probability","exact_match","paraphrase_nll","paraphrase_correct_probability","paraphrase_exact_match"):
            values=[float(r[field]) for r in group];row[field+"_mean"]=statistics.mean(values);row[field+"_sample_sd"]=statistics.stdev(values) if len(values)>1 else ""
        memagg.append(row)
    _write_csv(root/"memory_matrix_aggregate.csv",memagg)
    forgetting=[]
    for r in summaries:
        forgetting.append({k:r[k] for k in ("method","protocol","seed","forgetting_a_after_b_nll","forgetting_a_after_c_nll","forgetting_b_after_c_nll","forgetting_a_after_b_probability","forgetting_a_after_c_probability","forgetting_b_after_c_probability","forgetting_a_after_b_exact_match","forgetting_a_after_c_exact_match","forgetting_b_after_c_exact_match")})
    _write_csv(root/"forgetting_metrics.csv",forgetting);_write_csv(root/"memory_slot_ablation.csv",ablations)
    _write_csv(root/"protocol_parameter_budgets.csv",[
        {"protocol":"sequential_single","final_adapter_parameters":6144,"auxiliary_state":"none",
         "comparison_group":"6,144 reused trainable parameters"},
        {"protocol":"ewc_like","final_adapter_parameters":6144,"auxiliary_state":"Fisher diagonal + reference snapshot",
         "comparison_group":"6,144 reused trainable parameters; auxiliary EWC state is not trainable"},
        {"protocol":"grow_unfrozen","final_adapter_parameters":18432,"auxiliary_state":"none",
         "comparison_group":"18,432 growing parameters; not parameter matched to sequential/EWC"},
        {"protocol":"hard_consolidation","final_adapter_parameters":18432,"auxiliary_state":"none",
         "comparison_group":"18,432 stored parameters; 6,144 trainable per phase; not parameter matched to sequential/EWC"},
    ])
    make_figures(root,summaries,memory,ablations)


def _groups(rows, key):
    out=defaultdict(list)
    for row in rows:out[key(row)].append(row)
    return out


def make_figures(root: Path, summaries: list[dict], memory: list[dict], ablations: list[dict]) -> None:
    import matplotlib.pyplot as plt
    figures=root/"figures";figures.mkdir(exist_ok=True)
    def save(fig,name):fig.tight_layout();fig.savefig(figures/f"{name}.png",dpi=200);fig.savefig(figures/f"{name}.svg");plt.close(fig)
    colors={"A":"#3b82f6","B":"#f59e0b","C":"#10b981"}
    for field,name,ylabel in (("correct_probability","abc_probability_by_phase","Correct-answer probability"),("nll","abc_nll_by_phase","Target NLL")):
        fig,axes=plt.subplots(1,len(PROTOCOLS),figsize=(4.3*len(PROTOCOLS),4),sharey=True)
        for ax,protocol in zip(axes,PROTOCOLS):
            for method,style in zip(METHODS,("-","--",":")):
                for fact_set in SETS:
                    points=[]
                    for phase in SETS:
                        v=[float(r[field]) for r in memory if r["protocol"]==protocol and r["method"]==method and r["training_phase"]==phase and r["evaluation_set"]==fact_set]
                        if v:points.append(statistics.mean(v))
                    if points:ax.plot(SETS,points,style,color=colors[fact_set],alpha=.8,label=f"{method}:{fact_set}")
            ax.set_title(protocol);ax.set_xlabel("Training phase");ax.grid(alpha=.2)
        axes[0].set_ylabel(ylabel);axes[-1].legend(fontsize=6,ncol=2);save(fig,name)
    for key,name,title in (("forgetting_a_after_b_nll","forgetting_a_after_b","A after B"),("forgetting_a_after_c_nll","forgetting_a_after_c","A after C"),("forgetting_b_after_c_nll","forgetting_b_after_c","B after C")):
        fig,ax=plt.subplots(figsize=(9,4));labels=[];values=[]
        for protocol in PROTOCOLS:
            for method in METHODS:
                v=[float(r[key]) for r in summaries if r["protocol"]==protocol and r["method"]==method]
                if v:labels.append(f"{protocol}\n{method}");values.append(statistics.mean(v))
        ax.bar(labels,values);ax.set_ylabel("NLL increase");ax.set_title(f"Forgetting {title}");ax.tick_params(axis='x',labelsize=7);save(fig,name)
    final=[r for r in memory if r["training_phase"]=="C"]
    fig,ax=plt.subplots(figsize=(10,4));labels=[];values=[]
    for protocol in PROTOCOLS:
        for method in METHODS:
            for fact_set in SETS:
                v=[float(r["nll"]) for r in final if r["protocol"]==protocol and r["method"]==method and r["evaluation_set"]==fact_set]
                if v:labels.append(f"{protocol}\n{method}:{fact_set}");values.append(statistics.mean(v))
    ax.bar(labels,values);ax.set_ylabel("Final target NLL");ax.tick_params(axis='x',labelsize=5,rotation=45);save(fig,"final_abc_performance")
    fig,ax=plt.subplots(figsize=(9,4));labels=[];values=[]
    for protocol in PROTOCOLS:
        for method in METHODS:
            v=[r["mean_final_nll"] for r in summaries if r["protocol"]==protocol and r["method"]==method]
            if v:labels.append(f"{protocol}\n{method}");values.append(statistics.mean(v))
    ax.bar(labels,values);ax.set_ylabel("Mean final A/B/C NLL");ax.tick_params(axis='x',labelsize=7);save(fig,"final_mean_memory_performance")
    hard=[r for r in ablations if r["protocol"]=="hard_consolidation"]
    fig,ax=plt.subplots(figsize=(10,4));groups=[];values=[]
    for method in METHODS:
        for active in ("all","A_only","B_only","C_only"):
            v=[float(r["nll"]) for r in hard if r["method"]==method and r["active_slots"]==active and r["evaluation_set"]==active[:1]] if active!="all" else [float(r["nll"]) for r in hard if r["method"]==method and r["active_slots"]==active]
            if v:groups.append(f"{method}\n{active}");values.append(statistics.mean(v))
    ax.bar(groups,values);ax.set_ylabel("Target NLL");ax.set_title("Hard-consolidation slot ablation");save(fig,"hard_consolidation_slot_ablation")
    fig,ax=plt.subplots(figsize=(7,5))
    for protocol,marker in zip(PROTOCOLS,("o","s","^")):
        for method in METHODS:
            group=[r for r in summaries if r["protocol"]==protocol and r["method"]==method]
            if not group:continue
            x=statistics.mean((r["forgetting_a_after_c_nll"]+r["forgetting_b_after_c_nll"])/2 for r in group);y=statistics.mean(-r["acquisition_c_nll"] for r in group)
            ax.scatter(x,y,marker=marker,label=f"{protocol}:{method}")
    ax.set(xlabel="Mean previous-fact NLL increase",ylabel="Current-fact acquisition (−NLL)",title="Stability–plasticity diagnostic")
    handles,labels=ax.get_legend_handles_labels()
    if handles:ax.legend(fontsize=6)
    ax.grid(alpha=.2);save(fig,"stability_plasticity")
