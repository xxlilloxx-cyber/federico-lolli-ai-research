#!/usr/bin/env python3
"""Validate and aggregate factual-learning summaries."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

FIELDS = ["best_validation_loss", "best_validation_step", "best_phase1_a_validation_loss",
          "best_phase1_a_validation_step", "best_phase2_b_validation_loss",
          "best_phase2_b_validation_step", "final_validation_loss", "phase1_a_nll",
          "post_interference_a_nll", "interference_a_nll_change",
          "interference_a_probability_change", "frozen_retention_a_nll",
          "frozen_retention_b_nll", "frozen_retention_a_change",
          "final_b_correct_answer_probability", "final_b_exact_match_accuracy",
          "lora_a_norm", "lora_b_norm", "lora_effective_update_norm", "lora_parameter_norm",
          "symmetric_u_norm", "symmetric_p_norm", "symmetric_parameter_norm",
          "total_adaptation_norm", "lora_parameter_displacement_l2",
          "lora_parameter_displacement_max_abs", "symmetric_parameter_displacement_l2",
          "symmetric_parameter_displacement_max_abs", "total_parameter_displacement_l2",
          "total_parameter_displacement_max_abs"]


def fmt(mean: float, sd: float | str) -> str:
    return f"{mean:.6f}" if sd == "" else f"{mean:.6f} ± {float(sd):.6f}"


def write_report(root: Path, summaries: list[dict], aggregate: list[dict]) -> None:
    config = json.loads(next(root.glob("*_seed*/config.json")).read_text())
    by_method = {row["condition"]: row for row in aggregate}
    lines = ["# Principal matched-budget factual-learning experiment", "",
             "## Configuration", "",
             f"- Backbone: `{config['model']}` (frozen)",
             f"- Placement: zero-based block {config['layers'][0]}, `{config['projections'][0]}`",
             f"- Facts per sequential set: {config['facts_per_set']}",
             f"- Phase 1 / Phase 2 steps: {config['phase1_steps']} / {config['phase2_steps']}",
             f"- Total optimizer steps: {config['phase1_steps'] + config['phase2_steps']}",
             f"- Seeds: {', '.join(str(v) for v in config['seeds'])}",
             f"- Optimizer: AdamW; learning rate `{config['learning_rate']}`; scheduler `{config['scheduler']}`",
             f"- Gradient accumulation: {config['gradient_accumulation']}; evaluation every {config['evaluation_every_steps']} steps",
             "", "## Parameter budgets", "",
             "| Condition | LoRA parameters | Symmetric parameters | Total trainable |", "|---|---:|---:|---:|"]
    for method in ("frozen", "lora", "symmetric", "combined"):
        row = by_method[method]
        lines.append(f"| {method} | {int(row['lora_parameters']):,} | {int(row['symmetric_parameters']):,} | {int(row['trainable_parameters']):,} |")
    lines += ["", "## Final results across three seeds", "",
              "| Condition | Final set-B validation NLL | Correct-answer probability | Exact match |",
              "|---|---:|---:|---:|"]
    for method in ("frozen", "lora", "symmetric", "combined"):
        row = by_method[method]
        lines.append(f"| {method} | {fmt(row['final_validation_loss_mean'], row['final_validation_loss_sample_sd'])} | {fmt(row['final_b_correct_answer_probability_mean'], row['final_b_correct_answer_probability_sample_sd'])} | {fmt(row['final_b_exact_match_accuracy_mean'], row['final_b_exact_match_accuracy_sample_sd'])} |")
    lines += ["", "Individual seed values are retained below; the three-seed summaries are descriptive and are not treated as evidence of statistical superiority.", "",
              "| Condition | Seed | Best validation NLL (step) | Final validation NLL (step 5000) |",
              "|---|---:|---:|---:|"]
    for row in sorted(summaries, key=lambda item: (item["condition"], item["seed"])):
        lines.append(f"| {row['condition']} | {row['seed']} | {row['best_validation_loss']:.6f} ({row['best_validation_step']}) | {row['final_validation_loss']:.6f} |")
    lines += ["", "## Retention and sequential interference", "",
              "| Condition | Set-A NLL before B | Set-A NLL after B | Interference ΔNLL | Frozen-retention ΔNLL |",
              "|---|---:|---:|---:|---:|"]
    for method in ("frozen", "lora", "symmetric", "combined"):
        row = by_method[method]
        lines.append(f"| {method} | {fmt(row['phase1_a_nll_mean'], row['phase1_a_nll_sample_sd'])} | {fmt(row['post_interference_a_nll_mean'], row['post_interference_a_nll_sample_sd'])} | {fmt(row['interference_a_nll_change_mean'], row['interference_a_nll_change_sample_sd'])} | {fmt(row['frozen_retention_a_change_mean'], row['frozen_retention_a_change_sample_sd'])} |")
    lines += ["", "## Parameter displacement", "",
              "Final factor and branch norms were recomputed from the trained tensors rather than carried forward from initialization.", "",
              "| Condition | Final LoRA A / B norms | Final `||ΔW_LoRA||` | Final Symmetric U / P norms | Total adapter norm |",
              "|---|---:|---:|---:|---:|"]
    for method in ("frozen", "lora", "symmetric", "combined"):
        row = by_method[method]
        lines.append(f"| {method} | {fmt(row['lora_a_norm_mean'], row['lora_a_norm_sample_sd'])} / {fmt(row['lora_b_norm_mean'], row['lora_b_norm_sample_sd'])} | {fmt(row['lora_effective_update_norm_mean'], row['lora_effective_update_norm_sample_sd'])} | {fmt(row['symmetric_u_norm_mean'], row['symmetric_u_norm_sample_sd'])} / {fmt(row['symmetric_p_norm_mean'], row['symmetric_p_norm_sample_sd'])} | {fmt(row['total_adaptation_norm_mean'], row['total_adaptation_norm_sample_sd'])} |")
    lines += ["", "| Condition | LoRA branch L2 displacement | Symmetric branch L2 displacement | Total max-absolute displacement |",
              "|---|---:|---:|---:|"]
    for method in ("frozen", "lora", "symmetric", "combined"):
        row = by_method[method]
        lines.append(f"| {method} | {fmt(row['lora_parameter_displacement_l2_mean'], row['lora_parameter_displacement_l2_sample_sd'])} | {fmt(row['symmetric_parameter_displacement_l2_mean'], row['symmetric_parameter_displacement_l2_sample_sd'])} | {fmt(row['total_parameter_displacement_max_abs_mean'], row['total_parameter_displacement_max_abs_sample_sd'])} |")
    lines += ["", "All planned cells completed with finite metrics and the expected parameter budgets. No failed or replaced runs were omitted.", ""]
    (root / "PRINCIPAL_EXPERIMENT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-seeds", nargs="*", type=int)
    args = parser.parse_args()
    summaries = []
    for path in sorted(args.input.glob("*_seed*/summary.json")):
        row = json.loads(path.read_text())
        if row.get("status") != "complete":
            raise ValueError(f"incomplete result: {path}")
        for field in FIELDS:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"non-finite {field}: {path}")
        row["source"] = str(path.relative_to(args.input))
        summaries.append(row)
    if not summaries:
        raise ValueError("no completed summaries found")
    if args.expected_seeds:
        expected = {(method, seed) for method in ("frozen", "lora", "symmetric", "combined")
                    for seed in args.expected_seeds}
        actual = {(row["condition"], int(row["seed"])) for row in summaries}
        if actual != expected:
            raise ValueError(f"run cells differ: missing={sorted(expected-actual)} extra={sorted(actual-expected)}")
    per_fields = ["condition", "seed", *FIELDS, "trainable_parameters", "lora_parameters",
                  "symmetric_parameters", "trainable_percentage_of_backbone", "training_time_seconds", "source"]
    with (args.input / "per_run_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=per_fields)
        writer.writeheader(); writer.writerows({k: row[k] for k in per_fields} for row in summaries)
    aggregate = []
    for condition in sorted({row["condition"] for row in summaries}):
        group = [row for row in summaries if row["condition"] == condition]
        out = {"condition": condition, "n_seeds": len(group),
               "trainable_parameters": group[0]["trainable_parameters"],
               "lora_parameters": group[0]["lora_parameters"],
               "symmetric_parameters": group[0]["symmetric_parameters"]}
        for field in FIELDS:
            values = [float(row[field]) for row in group]
            out[field + "_mean"] = statistics.mean(values)
            out[field + "_sample_sd"] = statistics.stdev(values) if len(values) > 1 else ""
        aggregate.append(out)
    aggregate_fields = list(aggregate[0])
    with (args.input / "aggregate_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=aggregate_fields)
        writer.writeheader(); writer.writerows(aggregate)
    if args.expected_seeds and len(args.expected_seeds) == 3:
        write_report(args.input, summaries, aggregate)
    print(f"aggregated {len(summaries)} runs into {args.input}")


if __name__ == "__main__":
    main()
