"""Regenerate Project 01 public figures from sanitized CSV exports only."""
import csv
import statistics
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "data"
ASSETS = ROOT / "docs" / "assets"
COLORS = {"lora": "#1769aa", "symmetric_quadratic": "#d96b27"}
LABELS = {"lora": "LoRA", "symmetric_quadratic": "Symmetric Quadratic"}
GEOMS = ["concentrated", "two_distributed", "four_distributed"]


def load(name):
    return list(csv.DictReader((DATA / name).open()))


def grouped(rows, keys):
    out = defaultdict(list)
    for row in rows:
        out[tuple(row[k] for k in keys)].append(row)
    return out


def mean_sd(xs):
    return statistics.mean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0


def curve(rows, geometry, name):
    points = grouped([r for r in rows if r["geometry"] == geometry and r["architecture"] == name], ["step"])
    points = {int(key[0]): value for key, value in points.items()}
    steps = sorted(points)
    means = [mean_sd([float(r["validation_loss"]) for r in points[s]]) for s in steps]
    return steps, [x[0] for x in means], [x[1] for x in means]


def main():
    ASSETS.mkdir(exist_ok=True)
    trajectory = load("long_validation_trajectories.csv")
    final = load("long_final_per_seed.csv")
    for geometry, filename, title in [
        ("concentrated", "validation-concentrated-5000.png", "Validation trajectory — concentrated placement"),
        ("two_distributed", "validation-two-layer-5000.png", "Validation trajectory — two-layer placement"),
        ("four_distributed", "validation-four-layer-5000.png", "Validation trajectory — four-layer placement"),
    ]:
        plt.figure(figsize=(8.5, 5.2))
        for architecture in LABELS:
            s, m, sd = curve(trajectory, geometry, architecture)
            plt.plot(s, m, label=LABELS[architecture], color=COLORS[architecture])
            plt.fill_between(s, [a-b for a,b in zip(m,sd)], [a+b for a,b in zip(m,sd)], color=COLORS[architecture], alpha=.15)
        plt.title(title); plt.xlabel("Optimizer step"); plt.ylabel("Validation loss (mean ± sample SD)")
        plt.grid(alpha=.22); plt.legend(); plt.tight_layout(); plt.savefig(ASSETS / filename, dpi=220); plt.close()
    plt.figure(figsize=(8.5,5.2))
    for geometry in GEOMS:
        sl, ml, _ = curve(trajectory, geometry, "lora")
        ss, ms, _ = curve(trajectory, geometry, "symmetric_quadratic")
        assert sl == ss
        plt.plot(sl, [b-a for a,b in zip(ml,ms)], label=geometry.replace("_", " "))
    plt.axhline(0,color="black",lw=.8); plt.title("Validation difference through training")
    plt.xlabel("Optimizer step"); plt.ylabel("Symmetric Quadratic − LoRA validation loss")
    plt.grid(alpha=.22); plt.legend(); plt.tight_layout(); plt.savefig(ASSETS / "delta-loss-5000.png", dpi=220); plt.close()
    plt.figure(figsize=(7.8,5.2))
    for i, geometry in enumerate(GEOMS):
        for offset, architecture in [(-.18,"lora"),(.18,"symmetric_quadratic")]:
            values=[float(r["test_loss_final_checkpoint"]) for r in final if r["geometry"]==geometry and r["architecture"]==architecture]
            m,sd=mean_sd(values); plt.bar(i+offset,m,.34,yerr=sd,capsize=4,color=COLORS[architecture],label=LABELS[architecture] if i==0 else None)
            plt.scatter([i+offset]*len(values),values,color="#15233c",s=18,zorder=3)
    plt.xticks(range(3),["Concentrated","Two-layer","Four-layer"]);plt.ylabel("Test loss (mean ± sample SD)");plt.title("Held-out WikiText-2 test loss")
    plt.legend();plt.tight_layout();plt.savefig(ASSETS / "test-loss-5000.png",dpi=220);plt.close()
    plt.figure(figsize=(7.8,5.2))
    for i, geometry in enumerate(GEOMS):
        for offset, architecture in [(-.18,"lora"),(.18,"symmetric_quadratic")]:
            values=[float(r["test_loss_final_checkpoint"])-float(r["final_validation_loss"]) for r in final if r["geometry"]==geometry and r["architecture"]==architecture]
            m,sd=mean_sd(values); plt.bar(i+offset,m,.34,yerr=sd,capsize=4,color=COLORS[architecture],label=LABELS[architecture] if i==0 else None)
    plt.axhline(0,color="black",lw=.8);plt.xticks(range(3),["Concentrated","Two-layer","Four-layer"]);plt.ylabel("Test loss − final-step validation loss");plt.title("Test–validation loss difference")
    plt.legend();plt.tight_layout();plt.savefig(ASSETS / "test-validation-difference-5000.png",dpi=220);plt.close()
    plt.figure(figsize=(7.8,5.2))
    for i, geometry in enumerate(GEOMS):
        for offset, architecture in [(-.18,"lora"),(.18,"symmetric_quadratic")]:
            values=[float(r["final_validation_loss"]) for r in final if r["geometry"]==geometry and r["architecture"]==architecture]
            plt.scatter([i+offset]*len(values),values,color=COLORS[architecture],s=52,label=LABELS[architecture] if i==0 else None)
    plt.xticks(range(3),["Concentrated","Two-layer","Four-layer"]);plt.ylabel("Final-step validation loss");plt.title("Per-seed final validation endpoints")
    plt.legend();plt.grid(axis="y",alpha=.22);plt.tight_layout();plt.savefig(ASSETS / "per-seed-final-validation-5000.png",dpi=220);plt.close()


if __name__ == "__main__":
    main()
