#!/usr/bin/env python3
"""Restart-safe, publication-oriented multi-seed EWC confirmation campaign."""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import json
import math
import shutil
import statistics
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments" / "factual_learning_lora_symmetric"
SOURCE_ROOT = ROOT / "results_ewc_lambda_sweep"
RESULT_ROOT = ROOT / "results_ewc_confirmation"
EXPECTED_DATASET_SHA256 = "cc505ec681369c5f9aa4c7123cfff11da878bce17c0669890d416b5eb07f1923"
GRID = {"lora": (0.0, 1.0, 10.0), "symmetric": (0.0, 10.0, 30.0), "combined": (0.0, 10.0, 30.0)}
SEEDS = (42, 123, 456)
METHODS = tuple(GRID)
sys.path[:0] = [str(ROOT), str(EXP)]
from abc_memory import run_one  # noqa: E402
from run_ewc_lambda_sweep import analyze_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--seed", type=int, choices=SEEDS)
    parser.add_argument("--lambda", dest="lambda_value", type=float)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> str:
    device = "cuda" if requested == "auto" and torch.cuda.is_available() else ("cpu" if requested == "auto" else requested)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return device


def run_name(method: str, value: float, seed: int) -> str:
    return f"{method}_lambda_{value:g}_seed_{seed}".replace(".", "p")


def canonical_config(device: str, value: float, smoke: bool = False) -> dict[str, Any]:
    return {
        "model": "gpt2", "dataset_seed": 20261002, "facts_per_set": 2 if smoke else 12,
        "steps_per_phase": 2 if smoke else 2500, "evaluation_every_steps": 2 if smoke else 250,
        "sequence_length": 64, "micro_batch_size": 4, "gradient_accumulation": 1,
        "effective_batch_size": 4, "learning_rate": 3e-4, "weight_decay": 0.0,
        "gradient_clipping": 1.0, "scheduler": "none",
        "placement": "transformer.h[0].attn.c_proj", "pairwise_slot_evaluation": False,
        "amp": "fp16" if device == "cuda" else "off", "ewc_lambda": float(value),
        "ewc_gamma": 1.0, "fisher_samples": 2 if smoke else 12,
        "fisher_batches": 2 if smoke else 12, "device": device,
    }


REUSE_KEYS = (
    "model", "dataset_seed", "facts_per_set", "steps_per_phase", "evaluation_every_steps",
    "sequence_length", "micro_batch_size", "gradient_accumulation", "effective_batch_size",
    "learning_rate", "weight_decay", "gradient_clipping", "scheduler", "placement",
    "pairwise_slot_evaluation", "amp", "ewc_lambda", "ewc_gamma", "fisher_samples",
    "fisher_batches", "device", "dtype", "method", "protocol", "seed",
)
EXPECTED_SHAPES = {
    "lora": {"slots.shared.U": (768, 4), "slots.shared.V": (4, 768)},
    "symmetric": {"slots.shared.U": (768, 4), "slots.shared.P": (4, 768)},
    "combined": {"slots.shared.A": (768, 2), "slots.shared.B": (2, 768),
                 "slots.shared.U": (768, 2), "slots.shared.P": (2, 768)},
}


def _finite_tree(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_tree(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_tree(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def complete(run_dir: Path) -> bool:
    try:
        return json.loads((run_dir / "summary.json").read_text()).get("complete") is True
    except Exception:
        return False


def validate_run(run_dir: Path, method: str, value: float, seed: int,
                 expected: dict[str, Any], *, require_dataset_hash: bool = True) -> list[str]:
    errors: list[str] = []
    required = ("summary.json", "config.json", "dataset.json", "metrics.csv", "memory_matrix.csv",
                "memory_slot_ablation.csv", "checkpoints/after_A.pt", "checkpoints/after_B.pt",
                "checkpoints/after_C.pt")
    for relative in required:
        if not (run_dir / relative).is_file():
            errors.append(f"missing {relative}")
    if errors:
        return errors
    try:
        config = json.loads((run_dir / "config.json").read_text())
        summary = json.loads((run_dir / "summary.json").read_text())
    except Exception as exc:
        return [f"invalid JSON: {exc}"]
    if summary.get("complete") is not True:
        errors.append("summary complete is not true")
    if not _finite_tree(summary):
        errors.append("summary contains NaN/Inf")
    dtype = {"fp16": "torch.float16", "bf16": "torch.bfloat16", "off": "torch.float32"}[expected["amp"]]
    wanted = {**expected, "dtype": dtype, "method": method, "protocol": "ewc_like", "seed": seed}
    for key in REUSE_KEYS:
        if config.get(key) != wanted.get(key):
            errors.append(f"{key}: {config.get(key)!r} != {wanted.get(key)!r}")
    if int(summary.get("final_step", -1)) != 3 * int(expected["steps_per_phase"]):
        errors.append("final step mismatch")
    if int(summary.get("final_adapter_parameters", -1)) != 6144:
        errors.append("adapter parameter count is not 6144")
    if int(summary.get("final_trainable_adapter_parameters", -1)) != 6144:
        errors.append("trainable adapter parameter count is not 6144")
    digest = hashlib.sha256((run_dir / "dataset.json").read_bytes()).hexdigest()
    if require_dataset_hash and digest != EXPECTED_DATASET_SHA256:
        errors.append(f"dataset SHA-256 {digest} != {EXPECTED_DATASET_SHA256}")
    try:
        checkpoint = torch.load(run_dir / "checkpoints" / "after_A.pt", map_location="cpu", weights_only=False)
        shapes = {key: tuple(value.shape) for key, value in checkpoint["adapter_state"].items()}
        if shapes != EXPECTED_SHAPES[method]:
            errors.append(f"adapter tensor shapes {shapes} != {EXPECTED_SHAPES[method]}")
        auxiliary = checkpoint.get("auxiliary_state", {})
        if set(auxiliary) != {"reference", "fisher"}:
            errors.append("after_A checkpoint lacks online-EWC reference/Fisher state")
    except Exception as exc:
        errors.append(f"checkpoint validation failed: {exc}")
    for name in ("metrics.csv", "memory_matrix.csv"):
        try:
            with (run_dir / name).open() as handle:
                for row in csv.DictReader(handle):
                    for item in row.values():
                        if item not in (None, ""):
                            try:
                                number = float(item)
                            except ValueError:
                                continue
                            if not math.isfinite(number):
                                errors.append(f"non-finite value in {name}")
                                raise StopIteration
        except StopIteration:
            pass
    return errors


def discover_seed42(device: str = "cuda") -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reused, invalid = [], []
    for method, values in GRID.items():
        for value in values:
            source = SOURCE_ROOT / run_name(method, value, 42)
            expected = canonical_config(device, value)
            errors = validate_run(source, method, value, 42, expected)
            record = {"method": method, "lambda": value, "seed": 42,
                      "source": str(source.relative_to(ROOT)), "errors": errors}
            (invalid if errors else reused).append(record)
    return reused, invalid


def selected_cells(args: argparse.Namespace) -> list[tuple[str, float, int]]:
    methods = (args.method,) if args.method else METHODS
    seeds = (args.seed,) if args.seed is not None else SEEDS
    cells = []
    for method in methods:
        values = GRID[method]
        if args.lambda_value is not None:
            if args.lambda_value not in values:
                raise ValueError(f"lambda {args.lambda_value:g} is not in the confirmation grid for {method}")
            values = (float(args.lambda_value),)
        for value in values:
            cells.extend((method, value, seed) for seed in seeds)
    return cells


def prepare_directory(run_dir: Path) -> bool:
    if complete(run_dir):
        return True
    if run_dir.exists() and any(run_dir.iterdir()):
        archive = RESULT_ROOT / "incomplete_attempts" / f"{run_dir.name}__{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}"
        archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(run_dir), str(archive))
    run_dir.mkdir(parents=True, exist_ok=True)
    return False


def run_cell(config: dict[str, Any], method: str, value: float, seed: int,
             run_dir: Path, index: int, total: int) -> None:
    with (run_dir / "log.txt").open("w", buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        run_one(config, method, "ewc_like", seed, run_dir, config["device"],
                lambda phase: print(f"[{index}/{total}] PHASE {phase} complete {method}/lambda={value:g}/seed={seed}", file=sys.__stdout__, flush=True))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row)) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def resolve_all_runs() -> tuple[list[tuple[Path, str]], list[str]]:
    references = {}
    index = RESULT_ROOT / "REUSED_RUNS.json"
    if index.exists():
        for row in json.loads(index.read_text()).get("reused", []):
            references[(row["method"], float(row["lambda"]), int(row["seed"]))] = ROOT / row["source"]
    runs, errors = [], []
    for method, values in GRID.items():
        for value in values:
            for seed in SEEDS:
                key = (method, value, seed)
                path = references.get(key, RESULT_ROOT / run_name(*key))
                expected = canonical_config("cuda", value)
                found = validate_run(path, method, value, seed, expected)
                if found:
                    errors.append(f"{run_name(*key)}: {'; '.join(found)}")
                else:
                    runs.append((path, "reused" if key in references else "new"))
    return runs, errors


PRIMARY_METRICS = (
    "mean_forgetting", "mean_new_fact_acquisition_nll", "final_mean_nll",
    "final_mean_probability", "final_mean_exact_match",
)


def _pareto(rows: list[dict[str, Any]], within: bool) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        pool = [other for other in rows if not within or other["method"] == row["method"]]
        dominated = any(other is not row and other["mean_forgetting_mean"] <= row["mean_forgetting_mean"] and
                        other["mean_new_fact_acquisition_nll_mean"] <= row["mean_new_fact_acquisition_nll_mean"] and
                        (other["mean_forgetting_mean"] < row["mean_forgetting_mean"] or
                         other["mean_new_fact_acquisition_nll_mean"] < row["mean_new_fact_acquisition_nll_mean"])
                        for other in pool)
        result.append({**row, "pareto_nondominated": not dominated})
    return result


def aggregate_confirmation() -> None:
    runs, errors = resolve_all_runs()
    if errors:
        raise RuntimeError("confirmation aggregation requires all 27 valid cells:\n" + "\n".join(errors))
    rows = []
    for path, origin in runs:
        row = analyze_run(path)
        row["origin"] = origin; row["source_run"] = str(path.relative_to(ROOT))
        rows.append(row)
    rows.sort(key=lambda row: (row["method"], row["lambda"], row["seed"]))
    _write_csv(RESULT_ROOT / "per_run_results.csv", rows)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and key not in {"lambda", "seed"}]
    aggregate = []
    for method, values in GRID.items():
        for value in values:
            group = [row for row in rows if row["method"] == method and row["lambda"] == value]
            item: dict[str, Any] = {"method": method, "lambda": value, "n_seeds": len(group), "seeds": "42,123,456"}
            for key in numeric:
                samples = [float(row[key]) for row in group]
                item[key + "_mean"] = statistics.mean(samples)
                item[key + "_sample_sd"] = statistics.stdev(samples)
            aggregate.append(item)
    _write_csv(RESULT_ROOT / "aggregate_results.csv", aggregate)
    comparisons = {
        "lora": ((1.0, 0.0), (10.0, 0.0), (10.0, 1.0)),
        "symmetric": ((10.0, 0.0), (30.0, 0.0), (30.0, 10.0)),
        "combined": ((10.0, 0.0), (30.0, 0.0), (30.0, 10.0)),
    }
    paired = []
    for method, pairs in comparisons.items():
        for candidate, baseline in pairs:
            for seed in SEEDS:
                high = next(row for row in rows if (row["method"], row["lambda"], row["seed"]) == (method, candidate, seed))
                low = next(row for row in rows if (row["method"], row["lambda"], row["seed"]) == (method, baseline, seed))
                item = {"method": method, "candidate_lambda": candidate, "baseline_lambda": baseline, "seed": seed}
                item.update({key + "_difference": high[key] - low[key] for key in PRIMARY_METRICS})
                paired.append(item)
    _write_csv(RESULT_ROOT / "paired_differences.csv", paired)
    within = _pareto(aggregate, True); global_rows = _pareto(aggregate, False)
    _write_csv(RESULT_ROOT / "within_method_pareto.csv", within)
    _write_csv(RESULT_ROOT / "global_pareto.csv", global_rows)
    make_figures(aggregate, rows, global_rows)
    make_reports(aggregate, paired, global_rows)


def make_figures(aggregate: list[dict[str, Any]], rows: list[dict[str, Any]], global_rows: list[dict[str, Any]]) -> None:
    import matplotlib.pyplot as plt
    main = RESULT_ROOT / "figures_main"; supplementary = RESULT_ROOT / "figures_supplementary"
    main.mkdir(parents=True, exist_ok=True); supplementary.mkdir(parents=True, exist_ok=True)
    colors = {"lora": "#2563eb", "symmetric": "#dc2626", "combined": "#16a34a"}
    markers = {"lora": "o", "symmetric": "s", "combined": "^"}
    def save(fig, directory, name):
        fig.tight_layout(); fig.savefig(directory / f"{name}.png", dpi=220, bbox_inches="tight"); fig.savefig(directory / f"{name}.svg", bbox_inches="tight"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, method in zip(axes, METHODS):
        group = [row for row in aggregate if row["method"] == method]
        ax.errorbar([r["mean_forgetting_mean"] for r in group], [r["mean_new_fact_acquisition_nll_mean"] for r in group],
                    xerr=[r["mean_forgetting_sample_sd"] for r in group], yerr=[r["mean_new_fact_acquisition_nll_sample_sd"] for r in group], fmt="o-", color=colors[method], capsize=3)
        for row in group: ax.annotate(f'λ={row["lambda"]:g}', (row["mean_forgetting_mean"], row["mean_new_fact_acquisition_nll_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set(title=method.capitalize(), xlabel="Mean forgetting", ylabel="Mean B/C acquisition NLL"); ax.grid(alpha=.25)
    save(fig, main, "figure_1_stability_plasticity_confirmation")
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for method in METHODS:
        group = [row for row in global_rows if row["method"] == method]
        for row in group:
            filled = row["pareto_nondominated"]
            ax.scatter(row["mean_forgetting_mean"], row["mean_new_fact_acquisition_nll_mean"], marker=markers[method], s=100 if filled else 55, color=colors[method], facecolors=colors[method] if filled else "none")
            ax.annotate(f'{method[0].upper()} λ={row["lambda"]:g}', (row["mean_forgetting_mean"], row["mean_new_fact_acquisition_nll_mean"]), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.set(xlabel="Mean forgetting", ylabel="Mean B/C acquisition NLL", title="Global Pareto comparison (filled = nondominated)"); ax.grid(alpha=.25)
    save(fig, main, "figure_2_global_pareto")
    labels = [f'{r["method"]}\nλ={r["lambda"]:g}' for r in aggregate]; x = np.arange(len(aggregate))
    fig, ax = plt.subplots(figsize=(10, 4.5)); ax.bar(x, [r["final_mean_nll_mean"] for r in aggregate], yerr=[r["final_mean_nll_sample_sd"] for r in aggregate], color=[colors[r["method"]] for r in aggregate], capsize=3)
    ax.set(xticks=x, xticklabels=labels, ylabel="Final mean A/B/C NLL", title="Final memory NLL (mean ± sample SD)"); save(fig, main, "figure_3_final_memory_nll")
    components = (("forgetting_a_after_b_nll", "A after B"), ("forgetting_a_after_c_nll", "A after C"), ("forgetting_b_after_c_nll", "B after C"))
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    for ax, (key, title) in zip(axes, components):
        ax.errorbar(x, [r[key + "_mean"] for r in aggregate], yerr=[r[key + "_sample_sd"] for r in aggregate], fmt="o", capsize=3)
        ax.set(xticks=x, xticklabels=labels, title=title, ylabel="NLL increase"); ax.tick_params(axis="x", labelrotation=55); ax.grid(alpha=.25)
    save(fig, main, "figure_4_forgetting_components")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    width=.38
    axes[0].bar(x-width/2, [r["acquisition_b_nll_mean"] for r in aggregate], width, yerr=[r["acquisition_b_nll_sample_sd"] for r in aggregate], label="B", capsize=2)
    axes[0].bar(x+width/2, [r["acquisition_c_nll_mean"] for r in aggregate], width, yerr=[r["acquisition_c_nll_sample_sd"] for r in aggregate], label="C", capsize=2)
    axes[0].set(ylabel="Acquisition NLL", title="New-fact acquisition"); axes[0].legend()
    axes[1].bar(x, [r["final_mean_exact_match_mean"] for r in aggregate], yerr=[r["final_mean_exact_match_sample_sd"] for r in aggregate], color=[colors[r["method"]] for r in aggregate], capsize=2); axes[1].set(ylabel="Final exact match", title="Final exact-match memory")
    for ax in axes: ax.set(xticks=x, xticklabels=labels); ax.tick_params(axis="x", labelrotation=55); ax.grid(axis="y", alpha=.25)
    save(fig, main, "figure_5_acquisition_exact_match")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, key, title in zip(axes, ("total_parameter_displacement_a_c_l2", "fisher_ab_effective_support", "realized_ewc_penalty_c"), ("A→C displacement", "Fisher effective support", "Realized C penalty")):
        for method in METHODS:
            group = [r for r in aggregate if r["method"] == method]; ax.errorbar([r["lambda"] for r in group], [r[key + "_mean"] for r in group], yerr=[r[key + "_sample_sd"] for r in group], marker=markers[method], color=colors[method], label=method)
        ax.set(title=title, xlabel="λ"); ax.grid(alpha=.2)
    axes[0].legend(); save(fig, supplementary, "supplementary_parameter_fisher")


def _markdown_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for _, label in columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values=[]
        for key, _ in columns:
            value=row[key]; values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def make_reports(aggregate: list[dict[str, Any]], paired: list[dict[str, Any]], global_rows: list[dict[str, Any]]) -> None:
    compact = [{"method": r["method"], "lambda": r["lambda"], "forgetting": r["mean_forgetting_mean"], "forgetting_sd": r["mean_forgetting_sample_sd"], "acquisition": r["mean_new_fact_acquisition_nll_mean"], "acquisition_sd": r["mean_new_fact_acquisition_nll_sample_sd"], "final_nll": r["final_mean_nll_mean"], "final_probability": r["final_mean_probability_mean"], "exact_match": r["final_mean_exact_match_mean"]} for r in aggregate]
    table = _markdown_table(compact, [("method", "Method"), ("lambda", "λ"), ("forgetting", "Forgetting"), ("forgetting_sd", "SD"), ("acquisition", "Acquisition NLL"), ("acquisition_sd", "SD"), ("final_nll", "Final NLL"), ("final_probability", "Probability"), ("exact_match", "Exact match")])
    consistency=[]
    for item in paired:
        if item["baseline_lambda"] != 0: continue
        key=(item["method"],item["candidate_lambda"])
        existing=next((x for x in consistency if x[:2]==key),None)
        if existing is None: consistency.append([*key,0,0,0]) ; existing=consistency[-1]
        existing[2]+=item["mean_forgetting_difference"]<0; existing[3]+=item["mean_new_fact_acquisition_nll_difference"]>0; existing[4]+=item["final_mean_nll_difference"]<0
    consistency_text="\n".join(f"- {m} λ={v:g} versus λ=0: forgetting reduced in {stable}/3 seeds; acquisition NLL increased in {cost}/3; final memory NLL improved in {balanced}/3." for m,v,stable,cost,balanced in consistency)
    pareto_text=", ".join(f'{r["method"]} λ={r["lambda"]:g}' for r in global_rows if r["pareto_nondominated"])
    best_stability=min(aggregate,key=lambda r:r["mean_forgetting_mean"])
    best_plasticity=min(aggregate,key=lambda r:r["mean_new_fact_acquisition_nll_mean"])
    best_final=min(aggregate,key=lambda r:r["final_mean_nll_mean"])
    best_probability=max(aggregate,key=lambda r:r["final_mean_probability_mean"])
    best_exact=max(aggregate,key=lambda r:r["final_mean_exact_match_mean"])
    best_text=(f'- Best stability: {best_stability["method"]} λ={best_stability["lambda"]:g}.\n'
               f'- Best plasticity: {best_plasticity["method"]} λ={best_plasticity["lambda"]:g}.\n'
               f'- Best balanced final NLL: {best_final["method"]} λ={best_final["lambda"]:g}.\n'
               f'- Best final probability: {best_probability["method"]} λ={best_probability["lambda"]:g}.\n'
               f'- Best final exact match: {best_exact["method"]} λ={best_exact["lambda"]:g}.')
    reused=json.loads((RESULT_ROOT/"REUSED_RUNS.json").read_text()).get("reused",[])
    figures="\n".join(f"- [Figure {i}](figures_main/{name}.svg)" for i,name in enumerate(("figure_1_stability_plasticity_confirmation","figure_2_global_pareto","figure_3_final_memory_nll","figure_4_forgetting_components","figure_5_acquisition_exact_match"),1))
    report=f'''# EWC Multi-Seed Confirmation Report

## 1. Executive summary

This report confirms candidate online-EWC strengths across seeds 42, 123, and 456. Results are descriptive means ± sample standard deviations; three seeds do not establish universal optimality or statistical proof.

## 2. Confirmation question

Do the candidate λ regions identified in the seed-42 sweep reproduce across three matched seeds for LoRA, Symmetric, and Combined adapters?

## 3. Experimental protocol

Frozen GPT-2 Small, block-0 `attn.c_proj`, 6,144 adapter parameters, 12 disjoint facts per A/B/C phase, 2,500 optimizer steps per phase, no replay, microbatch 4, accumulation 1, online diagonal Fisher, γ=1.

## 4. Matched lambda=0 control

Lambda zero remains on the EWC runner path: Fisher/reference state is computed and the regularizer and its gradient are multiplied by zero. {len(reused)} validated seed-42 cells were referenced without checkpoint duplication.

## 5. LoRA results

The LoRA confirmation compares λ=0, 1, and 10 under the same execution path.

## 6. Symmetric results

The Symmetric confirmation compares λ=0, 10, and 30.

## 7. Combined results

The Combined confirmation compares λ=0, 10, and 30; no synergy is assumed.

## 8. Stability-plasticity comparison

{table}

{best_text}

## 9. Global Pareto analysis

Aggregated nondominated configurations: **{pareto_text}**.

## 10. Exact-match behavior

Exact-match values are reported separately in the table and Figure 5 rather than folded into one winner score.

## 11. Parameter-displacement evidence

Per-run A→B, B→C, and A→C displacement values are retained in `per_run_results.csv`; their aggregate means and sample SDs are in `aggregate_results.csv`.

## 12. Fisher observations

Fisher sums, maxima, and effective support are reconstructed from adapter-only checkpoint state and aggregated over seeds.

## 13. Paired-seed consistency

{consistency_text}

## 14. Limitations

The study uses three seeds, one synthetic factual dataset construction, GPT-2 Small, one placement, and a predefined λ grid. Paired differences are descriptive and no significance claim is made.

## 15. Confirmed conclusions

H1 is assessed from LoRA λ=1 and 10 relative to its matched λ=0 control. H2 is assessed from Symmetric λ=10 and 30, and H3 from Combined λ=10 and 30. Direction counts in Section 13 determine whether the seed-42 patterns reproduce. H4 is not directly confirmed because λ=50, 100, and 300 are outside this confirmation grid; those high-λ observations remain exploratory context only.

The evidence preserves separate stability, acquisition, final-memory, probability, and exact-match outcomes rather than forcing one universal winner.

## 16. Publication recommendation

The result is suitable for publication-oriented reporting as a controlled three-seed confirmation, with the above scope and limitations stated explicitly.
'''
    (RESULT_ROOT/"EWC_CONFIRMATION_REPORT.md").write_text(report)
    recap=f'''# EWC Confirmation Recap

- Grid: LoRA λ={{0,1,10}}; Symmetric λ={{0,10,30}}; Combined λ={{0,10,30}}.
- Seeds: 42, 123, 456.
- Reused seed-42 references: {len(reused)}; newly trained cells: 18.
- Global Pareto configurations: {pareto_text}.
- Strongest metric-specific outcomes:\n{best_text}
- Three-seed consistency:\n{consistency_text}
- Limitation: three seeds and one controlled factual-learning setting.
- Full report: [EWC_CONFIRMATION_REPORT.md](EWC_CONFIRMATION_REPORT.md)

## Main figures

{figures}
'''
    (RESULT_ROOT/"EWC_CONFIRMATION_RECAP.md").write_text(recap)
    publication=f'''# Publication Confirmation Summary

## Research question

How does online-EWC strength control stability and plasticity across LoRA, Symmetric, and Combined adapters over three matched seeds?

## Mathematical formulation

$$L_{{total}}=L_{{current}}+\\frac{{\\lambda}}{{2}}\\sum_iF_i(\\theta_i-\\theta_i^*)^2,\\qquad F_{{AB}}=F_A+F_B.$$

## Protocol

Frozen GPT-2 Small; block-0 `attn.c_proj`; 6,144 trainable adapter parameters; A→B→C without replay; 2,500 steps per phase; seeds 42, 123, 456.

## Confirmed three-seed findings and main numerical table

{table}

{best_text}

## Global Pareto result

Nondominated aggregated configurations: **{pareto_text}**.

## Principal figures

{figures}

## Limitations

Three seeds improve robustness but remain a limited sample. Results concern one model, dataset generator, placement, and λ grid.

## Candidate publication claim

Online EWC produces method-dependent, reproducible stability-plasticity changes under a matched continual factual-learning protocol; the evidence supports candidate operating regions rather than a universal optimal λ.
'''
    (RESULT_ROOT/"PUBLICATION_CONFIRMATION_SUMMARY.md").write_text(publication)


def write_reuse_index(reused: list[dict[str, Any]], invalid: list[dict[str, Any]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {"dataset_sha256": EXPECTED_DATASET_SHA256, "validation_time_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
               "reused": [{k:v for k,v in row.items() if k != "errors"} for row in reused], "invalid": invalid}
    (RESULT_ROOT / "REUSED_RUNS.json").write_text(json.dumps(payload, indent=2) + "\n")


def estimate_hours(cells: list[tuple[str, float, int]], reusable: set[tuple[str,float,int]]) -> float | None:
    durations=[]
    for method, values in GRID.items():
        for value in values:
            path=SOURCE_ROOT/run_name(method,value,42)/"summary.json"
            if path.exists():
                try: durations.append(float(json.loads(path.read_text())["training_time_seconds"]))
                except Exception: pass
    pending=sum((m,v,s) not in reusable and not complete(RESULT_ROOT/run_name(m,v,s)) for m,v,s in cells)
    return pending*statistics.median(durations)/3600 if durations else None


def main() -> None:
    args=parse_args(); device=resolve_device(args.device)
    if args.smoke:
        method=args.method or "lora"; value=args.lambda_value if args.lambda_value is not None else GRID[method][0]; seed=args.seed or 42
        config=canonical_config(device,value,True); run_dir=RESULT_ROOT/"smoke"/run_name(method,value,seed)
        if run_dir.exists(): shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True); print(f"SMOKE START {method} lambda={value:g} seed={seed}",flush=True)
        run_cell(config,method,value,seed,run_dir,1,1)
        errors=validate_run(run_dir,method,value,seed,config,require_dataset_hash=False)
        if errors: raise RuntimeError("smoke validation failed: "+"; ".join(errors))
        print(f"SMOKE COMPLETE {run_dir}",flush=True); return
    # Reuse is scientifically valid only for the canonical CUDA/fp16 execution.
    reused, invalid=discover_seed42("cuda")
    write_reuse_index(reused,invalid)
    reusable={(r["method"],float(r["lambda"]),42) for r in reused}
    cells=selected_cells(args); new=[]; completed=[]
    for cell in cells:
        if cell in reusable: continue
        path=RESULT_ROOT/run_name(*cell)
        if complete(path):
            errors=validate_run(path,*cell,canonical_config(device,cell[1]))
            if errors: raise RuntimeError(f"completed confirmation run is invalid: {path}: {'; '.join(errors)}")
            completed.append(cell)
        else:new.append(cell)
    hours=estimate_hours(cells,reusable)
    payload={"device":device,"amp":"fp16" if device=="cuda" else "off","total_scientific_cells":27,
             "selected_cells":len(cells),"reused_cells":sum(c in reusable for c in cells),
             "completed_new_cells":len(completed),"new_cells_required":len(new),
             "invalid_or_mismatched_seed42_cells":len(invalid),"estimated_sequential_hours":hours,
             "result_root":str(RESULT_ROOT)}
    print(json.dumps(payload,indent=2))
    for cell in cells:
        status="REUSED" if cell in reusable else ("COMPLETE" if cell in completed else "PENDING")
        print(f"{cell[0]:9} lambda={cell[1]:g} seed={cell[2]} {status}")
    if invalid:
        for row in invalid: print(f"INVALID {row['method']} lambda={row['lambda']:g} seed=42: {'; '.join(row['errors'])}")
    if args.dry_run:return
    if device!="cuda":raise RuntimeError("the publication confirmation must use CUDA/fp16 to match the reused seed-42 execution; CPU is supported for smoke tests")
    started=time.time()
    for index,(method,value,seed) in enumerate(new,1):
        run_dir=RESULT_ROOT/run_name(method,value,seed)
        if prepare_directory(run_dir):continue
        print(f"[{index}/{len(new)}] START {method}/lambda={value:g}/seed={seed}",flush=True)
        try:
            run_cell(canonical_config(device,value),method,value,seed,run_dir,index,len(new))
            errors=validate_run(run_dir,method,value,seed,canonical_config(device,value))
            if errors:raise RuntimeError("post-run validation failed: "+"; ".join(errors))
        except Exception:
            (run_dir/"failure.json").write_text(json.dumps({"complete":False,"time_utc":dt.datetime.now(dt.timezone.utc).isoformat(),"traceback":traceback.format_exc()},indent=2)+"\n")
            raise
        print(f"[{index}/{len(new)}] COMPLETE {method}/lambda={value:g}/seed={seed}",flush=True)
    all_runs,errors=resolve_all_runs()
    if not errors and len(all_runs)==27:aggregate_confirmation()
    (RESULT_ROOT/"CAMPAIGN_STATUS.json").write_text(json.dumps({"complete":not errors and len(all_runs)==27,"scientific_cells":len(all_runs),"elapsed_seconds":time.time()-started,"errors":errors},indent=2)+"\n")
    print(f"COMPLETE valid_cells={len(all_runs)}/27 results={RESULT_ROOT}",flush=True)


if __name__ == "__main__":
    main()
