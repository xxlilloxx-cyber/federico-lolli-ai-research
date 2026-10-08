#!/usr/bin/env python3
"""Generate figures only from saved factual-learning metrics."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

COLORS = {"frozen": "#666666", "lora": "#2468B4", "symmetric": "#D45500", "combined": "#328A52"}


def save(fig, out: Path, name: str) -> None:
    fig.tight_layout(); fig.savefig(out / f"{name}.png", dpi=220); fig.savefig(out / f"{name}.svg"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--input", type=Path, required=True); args = parser.parse_args()
    out = args.input / "figures"; out.mkdir(exist_ok=True)
    records, summaries = [], []
    for path in sorted(args.input.glob("*_seed*/metrics.csv")):
        condition = path.parent.name.rsplit("_seed", 1)[0]
        seed = int(path.parent.name.rsplit("_seed", 1)[1])
        with path.open() as handle:
            for row in csv.DictReader(handle):
                row.update(condition=condition, seed=seed); records.append(row)
        summaries.append(json.loads((path.parent / "summary.json").read_text()))
    if not records: raise ValueError("no metrics found")

    def curves(field: str, predicate, title: str, ylabel: str, name: str):
        fig, ax = plt.subplots(figsize=(7.2, 4.4)); grouped = defaultdict(list); per_seed = defaultdict(dict)
        for row in records:
            if predicate(row) and row[field] not in ("", "None"):
                key = (row["condition"], int(row["step"])); value = float(row[field])
                grouped[key].append(value); per_seed[(row["condition"], int(row["seed"]))][int(row["step"])] = value
        for condition in sorted({key[0] for key in grouped}):
            for (method, seed), trajectory in per_seed.items():
                if method == condition:
                    points = sorted(trajectory.items())
                    ax.plot([p[0] for p in points], [p[1] for p in points], color=COLORS[condition], alpha=.22, linewidth=1)
            steps = sorted(step for method, step in grouped if method == condition)
            means = [np.mean(grouped[(condition, step)]) for step in steps]
            sds = [np.std(grouped[(condition, step)], ddof=1) if len(grouped[(condition, step)]) > 1 else 0 for step in steps]
            ax.plot(steps, means, marker="o", label=condition, color=COLORS[condition], linewidth=2)
            ax.fill_between(steps, np.asarray(means)-sds, np.asarray(means)+sds, color=COLORS[condition], alpha=.12)
        ax.set(title=title, xlabel="Optimizer step", ylabel=ylabel); ax.grid(alpha=.25); ax.legend(); save(fig, out, name)

    curves("training_loss", lambda r: r["evaluation"] == "checkpoint", "Training objective at evaluation checkpoints", "Training loss", "training_loss_vs_step")
    curves("validation_loss", lambda r: r["evaluation"] == "checkpoint", "Validation target NLL", "NLL (lower is better)", "validation_loss_vs_step")
    curves("correct_answer_probability", lambda r: r["prompt_split"] == "validation" and r["evaluation"] == "checkpoint", "Correct-answer probability", "Geometric mean token probability", "correct_answer_probability_vs_step")
    curves("total_adaptation_norm", lambda r: r["evaluation"] == "checkpoint", "Current adaptation parameter norm", "L2/Frobenius norm", "parameter_norm_vs_step")
    curves("total_parameter_displacement_l2", lambda r: r["evaluation"] == "checkpoint", "Adapter displacement from initialization", "L2 displacement", "parameter_displacement_vs_step")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for field, label, axis in (("lora_b_norm", "LoRA B", axes[0]),
                               ("lora_effective_update_norm", "LoRA effective update", axes[0]),
                               ("symmetric_p_norm", "Symmetric P", axes[1])):
        for condition in ("lora", "symmetric", "combined"):
            points = defaultdict(list)
            for row in records:
                if row["condition"] == condition and row["evaluation"] == "checkpoint":
                    points[int(row["step"])].append(float(row[field]))
            if points and any(any(value != 0 for value in values) for values in points.values()):
                steps = sorted(points); means = [np.mean(points[step]) for step in steps]
                axis.plot(steps, means, label=f"{condition}: {label}", color=COLORS[condition],
                          linestyle="--" if field == "lora_effective_update_norm" else "-")
    axes[0].set(title="LoRA learned output and effective update", xlabel="Optimizer step", ylabel="Frobenius norm")
    axes[1].set(title="Symmetric learned output factor", xlabel="Optimizer step", ylabel="Frobenius norm")
    for axis in axes: axis.grid(alpha=.25); axis.legend(fontsize=8)
    save(fig, out, "factor_norms_vs_step")

    methods = [m for m in ("frozen", "lora", "symmetric", "combined") if any(s["condition"] == m for s in summaries)]
    fig, ax = plt.subplots(figsize=(7.2, 4.4)); vals = [[s["final_b_correct_answer_probability"] for s in summaries if s["condition"] == m] for m in methods]
    means=[np.mean(v) for v in vals]; sds=[np.std(v,ddof=1) if len(v)>1 else 0 for v in vals]
    ax.bar(methods, means, yerr=sds, capsize=4, color=[COLORS[m] for m in methods])
    for index, values in enumerate(vals): ax.scatter(np.full(len(values),index), values, color="black", s=18, zorder=3)
    ax.set(ylabel="Final set-B answer probability", title="Final factual-learning comparison"); save(fig, out, "method_comparison")

    fig, ax = plt.subplots(figsize=(7.2, 4.4)); x = np.arange(len(methods)); pre = [[s["phase1_a_nll"] for s in summaries if s["condition"] == m] for m in methods]; post = [[s["post_interference_a_nll"] for s in summaries if s["condition"] == m] for m in methods]
    ax.bar(x-.2, [np.mean(v) for v in pre], .4, label="Before learning B"); ax.bar(x+.2, [np.mean(v) for v in post], .4, label="After learning B"); ax.set_xticks(x, methods); ax.set(ylabel="Set-A target NLL", title="Sequential interference"); ax.legend(); save(fig, out, "interference_set_a")

    fig, ax = plt.subplots(figsize=(7.2, 4.4)); before = [[s["post_interference_a_nll"] for s in summaries if s["condition"] == m] for m in methods]; after = [[s["frozen_retention_a_nll"] for s in summaries if s["condition"] == m] for m in methods]
    ax.bar(x-.2, [np.mean(v) for v in before], .4, label="Before freezing"); ax.bar(x+.2, [np.mean(v) for v in after], .4, label="Frozen adapter"); ax.set_xticks(x, methods); ax.set(ylabel="Set-A target NLL", title="Frozen-adapter retention"); ax.legend(); save(fig, out, "frozen_adapter_retention")
    print(f"wrote 9 figure pairs to {out}")


if __name__ == "__main__": main()
