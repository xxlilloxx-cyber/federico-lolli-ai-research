#!/usr/bin/env python3
"""Run one isolated factual-learning condition and seed."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for path in (ROOT, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dataset import FactualExample, examples_for, generate_facts, serializable_dataset
from modeling import (branch_displacements, branch_gradient_norms, branch_norms,
                      branch_parameter_snapshot, insert_adaptation, parameter_budget)


METRIC_FIELDS = [
    "step", "phase", "evaluation", "fact_set", "prompt_split", "training_loss",
    "validation_loss", "target_answer_nll", "correct_answer_probability",
    "answer_sequence_probability", "exact_match_accuracy", "lora_a_norm", "lora_b_norm",
    "lora_effective_update_norm", "lora_parameter_norm", "symmetric_u_norm",
    "symmetric_p_norm", "symmetric_parameter_norm", "total_adaptation_norm",
    "lora_parameter_displacement_l2", "lora_parameter_displacement_max_abs",
    "symmetric_parameter_displacement_l2", "symmetric_parameter_displacement_max_abs",
    "total_parameter_displacement_l2", "total_parameter_displacement_max_abs", "lora_gradient_norm",
    "symmetric_gradient_norm", "trainable_parameters", "lora_parameters",
    "symmetric_parameters",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def write_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in METRIC_FIELDS})


def _encoded(example: FactualExample, tokenizer: Any, max_length: int, device: torch.device):
    prompt_ids = tokenizer.encode(example.prompt, add_special_tokens=False)
    answer_ids = tokenizer.encode(example.answer, add_special_tokens=False)
    if not answer_ids:
        raise ValueError(f"answer tokenized to zero tokens: {example.fact_id}")
    if len(prompt_ids) + len(answer_ids) > max_length:
        keep = max_length - len(answer_ids)
        if keep < 1:
            raise ValueError("sequence length is too short for the target answer")
        prompt_ids = prompt_ids[-keep:]
    ids = torch.tensor([prompt_ids + answer_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, :len(prompt_ids)] = -100
    return ids, labels, prompt_ids, answer_ids


@torch.no_grad()
def evaluate(model: torch.nn.Module, tokenizer: Any, examples: list[FactualExample],
             max_length: int, device: torch.device, exact_match: bool = True) -> dict[str, float]:
    model.eval()
    per_token_nll, sequence_probability, exact = [], [], []
    for example in examples:
        ids, labels, prompt_ids, answer_ids = _encoded(example, tokenizer, max_length, device)
        output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
        nll = float(output.loss.detach().float())
        per_token_nll.append(nll)
        sequence_probability.append(math.exp(-nll * len(answer_ids)))
        if exact_match:
            prompt = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            generated = model.generate(
                prompt, attention_mask=torch.ones_like(prompt), max_new_tokens=len(answer_ids), do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )[0, len(prompt_ids):].tolist()
            exact.append(float(generated == answer_ids))
    mean_nll = float(np.mean(per_token_nll))
    return {
        "validation_loss": mean_nll,
        "target_answer_nll": mean_nll,
        "correct_answer_probability": float(np.mean(np.exp(-np.asarray(per_token_nll)))),
        "answer_sequence_probability": float(np.mean(sequence_probability)),
        "exact_match_accuracy": float(np.mean(exact)) if exact else float("nan"),
    }


def _row(model: torch.nn.Module, budget: Any, step: int, phase: str, evaluation: str,
         fact_set: str, prompt_split: str, values: dict[str, float],
         initial_snapshot: dict[str, dict[str, torch.Tensor]],
         training_loss: float | None = None, gradients: dict[str, float] | None = None) -> dict[str, Any]:
    norms = branch_norms(model)
    displacements = branch_displacements(model, initial_snapshot)
    gradients = gradients or {"lora_gradient_norm": 0.0, "symmetric_gradient_norm": 0.0}
    return {
        "step": step, "phase": phase, "evaluation": evaluation, "fact_set": fact_set,
        "prompt_split": prompt_split, "training_loss": training_loss, **values, **norms, **displacements,
        **gradients, "trainable_parameters": budget.trainable_parameters,
        "lora_parameters": budget.lora_parameters,
        "symmetric_parameters": budget.symmetric_parameters,
    }


def _save_adapter(model: torch.nn.Module, path: Path, metadata: dict[str, Any]) -> None:
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()
             if name.endswith((".A", ".B", ".U", ".V", ".P"))}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"metadata": metadata, "adapter_state": state}, path)


def run(config: dict[str, Any], condition: str, seed: int, output_root: Path,
        force: bool = False) -> Path:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_dir = output_root / f"{condition}_seed{seed}"
    summary_path = run_dir / "summary.json"
    if summary_path.exists() and not force:
        summary = json.loads(summary_path.read_text())
        if summary.get("status") == "complete":
            print(f"SKIP complete run: {run_dir}")
            return run_dir
        raise RuntimeError(f"incomplete run exists; pass --force to replace only this run: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        handlers=[logging.FileHandler(run_dir / "logs" / "run.log"), logging.StreamHandler()],
                        force=True)
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(config["model"])
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["model"], dtype=dtype, attn_implementation="eager"
    ).to(device)
    model.config.use_cache = False
    if condition == "combined":
        lora_rank, symmetric_rank = config["combined_lora_rank"], config["combined_symmetric_rank"]
    else:
        lora_rank, symmetric_rank = config["lora_rank"], config["symmetric_rank"]
    insert_adaptation(model, condition, lora_rank=lora_rank, symmetric_rank=symmetric_rank,
                      alpha_lora=config["alpha_lora"], alpha_symmetric=config["alpha_symmetric"],
                      layers=config["layers"], projections=config["projections"])
    budget = parameter_budget(model)
    initial_snapshot = branch_parameter_snapshot(model)

    facts = generate_facts(config["dataset_seed"], config["facts_per_set"])
    facts_a = [fact for fact in facts if fact.fact_set == "A"]
    facts_b = [fact for fact in facts if fact.fact_set == "B"]
    train = {"A": examples_for(facts_a, "train"), "B": examples_for(facts_b, "train")}
    validation = {"A": examples_for(facts_a, "validation"), "B": examples_for(facts_b, "validation")}
    paraphrase = {"A": examples_for(facts_a, "paraphrase"), "B": examples_for(facts_b, "paraphrase")}

    resolved = dict(config)
    resolved.update({"condition": condition, "seed": seed, "device": str(device),
                     "dtype": str(dtype), "effective_lora_rank": lora_rank,
                     "effective_symmetric_rank": symmetric_rank,
                     "trainable_parameters": budget.trainable_parameters,
                     "lora_parameters": budget.lora_parameters,
                     "symmetric_parameters": budget.symmetric_parameters,
                     "frozen_backbone_parameters": budget.frozen_backbone_parameters,
                     "trainable_percentage_of_backbone": budget.trainable_percentage})
    (run_dir / "config.json").write_text(json.dumps(resolved, indent=2) + "\n")
    (run_dir / "dataset.json").write_text(json.dumps(serializable_dataset(facts), indent=2) + "\n")

    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"]) if parameters else None
    rows: list[dict[str, Any]] = []
    frozen_evaluation_cache: dict[tuple[str, str], dict[str, float]] = {}
    evaluation_every = int(config["evaluation_every_steps"])
    max_length = int(config["sequence_length"])
    started = time.perf_counter()

    def score(fact_set: str, split: str, data: list[FactualExample], exact: bool = True) -> dict[str, float]:
        key = (fact_set, split)
        if condition == "frozen" and key in frozen_evaluation_cache:
            return dict(frozen_evaluation_cache[key])
        values = evaluate(model, tokenizer, data, max_length, device, exact_match=exact)
        if condition == "frozen":
            frozen_evaluation_cache[key] = dict(values)
        return values

    def record(step: int, phase: str, label: str, sets: tuple[str, ...], exact: bool = True) -> None:
        for fact_set in sets:
            for split, data in (("validation", validation[fact_set]), ("paraphrase", paraphrase[fact_set])):
                values = score(fact_set, split, data, exact)
                rows.append(_row(model, budget, step, phase, label, fact_set, split, values,
                                 initial_snapshot))
        write_metrics(run_dir / "metrics.csv", rows)

    record(0, "initial", "initial", ("A", "B"))
    global_step = 0
    last_training_loss = None
    for phase, fact_set, steps in (("phase1", "A", int(config["phase1_steps"])),
                                   ("phase2", "B", int(config["phase2_steps"]))):
        rng = random.Random(seed + (0 if phase == "phase1" else 100000))
        order = list(range(len(train[fact_set])))
        if optimizer is not None:
            model.train()
        for local_step in range(1, steps + 1):
            gradients = None
            losses = []
            if optimizer is not None:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                for _ in range(int(config["gradient_accumulation"])):
                    if not order:
                        order = list(range(len(train[fact_set])))
                    if len(order) == len(train[fact_set]):
                        rng.shuffle(order)
                    example = train[fact_set][order.pop()]
                    ids, labels, _, _ = _encoded(example, tokenizer, max_length, device)
                    output = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, use_cache=False)
                    loss = output.loss / int(config["gradient_accumulation"])
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss at {phase} step {local_step}")
                    loss.backward()
                    losses.append(float(output.loss.detach().float()))
                gradients = branch_gradient_norms(model)
                torch.nn.utils.clip_grad_norm_(parameters, float(config["gradient_clipping"]))
                optimizer.step()
                last_training_loss = float(np.mean(losses))
            global_step += 1
            if local_step % evaluation_every == 0 or local_step == steps:
                values = score(fact_set, "validation", validation[fact_set], exact=True)
                rows.append(_row(model, budget, global_step, phase, "checkpoint", fact_set,
                                 "validation", values, initial_snapshot, last_training_loss, gradients))
                write_metrics(run_dir / "metrics.csv", rows)
                logging.info("%s step=%d fact_set=%s val_nll=%.6f", condition, global_step,
                             fact_set, values["target_answer_nll"])
        if phase == "phase1":
            record(global_step, phase, "pre_interference", ("A",))
            if optimizer is not None:
                _save_adapter(model, run_dir / "checkpoints" / "phase1_adapter.pt", resolved)

    record(global_step, "phase2", "post_interference", ("A", "B"))
    if optimizer is not None:
        _save_adapter(model, run_dir / "checkpoints" / "final_adapter.pt", resolved)
    for parameter in parameters:
        parameter.requires_grad_(False)
    record(global_step, "retention", "frozen_retention", ("A", "B"))

    validation_rows = [r for r in rows if r["prompt_split"] == "validation"
                       and r["evaluation"] in {"checkpoint", "post_interference"}]
    best = min(validation_rows, key=lambda r: r["validation_loss"])
    phase1_rows = [r for r in validation_rows if r["phase"] == "phase1" and r["fact_set"] == "A"]
    phase2_rows = [r for r in validation_rows if r["phase"] == "phase2" and r["fact_set"] == "B"]
    best_phase1 = min(phase1_rows, key=lambda r: r["validation_loss"])
    best_phase2 = min(phase2_rows, key=lambda r: r["validation_loss"])
    final_candidates = [r for r in rows if r["evaluation"] == "post_interference"
                        and r["fact_set"] == "B" and r["prompt_split"] == "validation"]
    final = final_candidates[-1]
    pre_a = next(r for r in rows if r["evaluation"] == "pre_interference"
                 and r["fact_set"] == "A" and r["prompt_split"] == "validation")
    post_a = next(r for r in rows if r["evaluation"] == "post_interference"
                  and r["fact_set"] == "A" and r["prompt_split"] == "validation")
    retention_a = next(r for r in rows if r["evaluation"] == "frozen_retention"
                       and r["fact_set"] == "A" and r["prompt_split"] == "validation")
    retention_b = next(r for r in rows if r["evaluation"] == "frozen_retention"
                       and r["fact_set"] == "B" and r["prompt_split"] == "validation")
    elapsed = time.perf_counter() - started
    summary = {
        "status": "complete", "condition": condition, "seed": seed,
        "best_validation_loss": best["validation_loss"], "best_validation_step": best["step"],
        "best_phase1_a_validation_loss": best_phase1["validation_loss"],
        "best_phase1_a_validation_step": best_phase1["step"],
        "best_phase2_b_validation_loss": best_phase2["validation_loss"],
        "best_phase2_b_validation_step": best_phase2["step"],
        "final_validation_loss": final["validation_loss"], "final_step": global_step,
        "final_fact_set": "B", "phase1_a_nll": pre_a["target_answer_nll"],
        "post_interference_a_nll": post_a["target_answer_nll"],
        "interference_a_nll_change": post_a["target_answer_nll"] - pre_a["target_answer_nll"],
        "interference_a_probability_change": post_a["correct_answer_probability"] - pre_a["correct_answer_probability"],
        "frozen_retention_a_nll": retention_a["target_answer_nll"],
        "frozen_retention_b_nll": retention_b["target_answer_nll"],
        "frozen_retention_a_change": retention_a["target_answer_nll"] - post_a["target_answer_nll"],
        "final_b_correct_answer_probability": final["correct_answer_probability"],
        "final_b_exact_match_accuracy": final["exact_match_accuracy"],
        "training_time_seconds": elapsed, "seconds_per_optimizer_step": elapsed / max(global_step, 1),
        **asdict(budget), "trainable_percentage_of_backbone": budget.trainable_percentage,
        **branch_norms(model), **branch_displacements(model, initial_snapshot),
    }
    for value in summary.values():
        if isinstance(value, float) and not math.isfinite(value):
            raise FloatingPointError("summary contains a non-finite metric")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    logging.info("complete: %s", run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--condition", choices=("frozen", "lora", "symmetric", "combined"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--phase1-steps", type=int)
    parser.add_argument("--phase2-steps", type=int)
    parser.add_argument("--facts-per-set", type=int)
    parser.add_argument("--evaluation-every-steps", type=int)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    for argument, key in ((args.phase1_steps, "phase1_steps"), (args.phase2_steps, "phase2_steps"),
                          (args.facts_per_set, "facts_per_set"),
                          (args.evaluation_every_steps, "evaluation_every_steps")):
        if argument is not None:
            config[key] = argument
    output_root = args.output_root or ROOT / config["output_root"]
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    run(config, args.condition, args.seed, output_root, args.force)


if __name__ == "__main__":
    main()
