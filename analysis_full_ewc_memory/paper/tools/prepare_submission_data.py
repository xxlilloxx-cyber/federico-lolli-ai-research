#!/usr/bin/env python3
"""Prepare read-only, submission-specific retention and sensitivity outputs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cairosvg

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "analysis_full_ewc_memory"
PAPER = ANALYSIS / "paper"
OUT = PAPER / "submission_data"
FIG = PAPER / "submission_figures"
METHOD_ORDER = ["lora", "symmetric", "combined"]
METHOD_LABEL = {"lora": "LoRA", "symmetric": "Symmetric", "combined": "Combined"}
CONFIRMATION_GRID = {"lora": [0, 1, 10], "symmetric": [0, 10, 30], "combined": [0, 10, 30]}
SEEDS = [42, 123, 456]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def canonical_confirmation_runs() -> pd.DataFrame:
    runs = pd.read_csv(ANALYSIS / "data" / "CANONICAL_RUNS.csv")
    selected = runs[runs.scientific_role.str.contains("confirmation", na=False)].copy()
    assert len(selected) == 27
    assert selected.canonical_run_id.nunique() == 27
    assert not selected.duplicated(["method", "lambda", "seed"]).any()
    assert set(selected.seed) == set(SEEDS)
    for method, lambdas in CONFIRMATION_GRID.items():
        got = sorted(selected[selected.method.eq(method)]["lambda"].astype(int).unique())
        assert got == lambdas, (method, got)
    return selected


def load_retention() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    identity_rows = []
    runs = canonical_confirmation_runs()
    full = pd.read_csv(ANALYSIS / "tables" / "confirmation_full_metrics.csv")
    for _, run in runs.iterrows():
        run_dir = REPO / run.source_directory
        matrix_path = run_dir / "memory_matrix.csv"
        summary_path = run_dir / "summary.json"
        assert matrix_path.exists() and summary_path.exists(), run_dir
        summary = read_json(summary_path)
        assert summary.get("complete") is True
        matrix = pd.read_csv(matrix_path)
        assert len(matrix) == 9
        key = dict(method=run.method, lambda_value=int(run["lambda"]), seed=int(run.seed))

        def nll(phase: str, fact_set: str) -> float:
            hit = matrix[(matrix.training_phase == phase) & (matrix.evaluation_set == fact_set)]
            assert len(hit) == 1
            return float(hit.iloc[0].nll)

        values = {
            "a_theta_a": nll("A", "A"),
            "a_theta_b": nll("B", "A"),
            "a_theta_c": nll("C", "A"),
            "b_theta_b": nll("B", "B"),
            "b_theta_c": nll("C", "B"),
        }
        rows.append({**key, "canonical_run_id": run.canonical_run_id,
                     "source_directory": run.source_directory, **values})
        expected = full[(full.method == run.method) &
                        (full["lambda"].astype(int) == int(run["lambda"])) &
                        (full.seed == int(run.seed))]
        assert len(expected) == 1
        expected = expected.iloc[0]
        identities = {
            "f_a_b": values["a_theta_b"] - values["a_theta_a"],
            "f_a_c": values["a_theta_c"] - values["a_theta_a"],
            "f_b_c": values["b_theta_c"] - values["b_theta_b"],
        }
        targets = {
            "f_a_b": float(expected.forgetting_a_after_b_nll),
            "f_a_c": float(expected.forgetting_a_after_c_nll),
            "f_b_c": float(expected.forgetting_b_after_c_nll),
        }
        errors = {name: identities[name] - targets[name] for name in identities}
        assert max(abs(v) for v in errors.values()) < 1e-6, (key, errors)
        identity_rows.append({**key, **identities,
                              **{f"reported_{k}": v for k, v in targets.items()},
                              **{f"error_{k}": v for k, v in errors.items()}})

    per_run = pd.DataFrame(rows)
    identities = pd.DataFrame(identity_rows)
    return per_run, identities


def aggregate_retention(per_run: pd.DataFrame) -> pd.DataFrame:
    metrics = ["a_theta_a", "a_theta_b", "a_theta_c", "b_theta_b", "b_theta_c"]
    rows = []
    for (method, lam), group in per_run.groupby(["method", "lambda_value"]):
        row = {"method": method, "lambda": lam, "seeds": "42,123,456", "n": len(group)}
        for metric in metrics:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sample_sd"] = group[metric].std(ddof=1)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["_order"] = out.method.map({m: i for i, m in enumerate(METHOD_ORDER)})
    return out.sort_values(["_order", "lambda"]).drop(columns="_order").reset_index(drop=True)


def plot_retention(agg: pd.DataFrame) -> None:
    plt.rcParams.update({
        "font.size": 10.5, "axes.labelsize": 11, "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5, "ytick.labelsize": 9.5, "legend.fontsize": 8.5,
        "figure.dpi": 160, "savefig.bbox": "tight",
    })
    markers = ["o", "s", "^"]
    styles = ["-", "--", "-."]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.25), sharey=True)
    x = np.arange(3)
    for ax, method in zip(axes, METHOD_ORDER):
        group = agg[agg.method.eq(method)].sort_values("lambda")
        for idx, (_, row) in enumerate(group.iterrows()):
            means = [row.a_theta_a_mean, row.a_theta_b_mean, row.a_theta_c_mean]
            sds = [row.a_theta_a_sample_sd, row.a_theta_b_sample_sd, row.a_theta_c_sample_sd]
            ax.errorbar(x, means, yerr=sds, marker=markers[idx], linestyle=styles[idx],
                        linewidth=1.5, markersize=5, capsize=2.5, color=str(0.12 + 0.28 * idx),
                        label=rf"$\lambda={int(row['lambda'])}$")
        ax.set_title(METHOD_LABEL[method])
        ax.set_xticks(x, [r"$\theta_A$", r"$\theta_B$", r"$\theta_C$"])
        ax.set_xlabel("Adapter state")
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, loc="upper left")
    axes[0].set_ylabel("NLL on factual set A (lower is better)")
    fig.tight_layout(w_pad=1.0)
    FIG.mkdir(parents=True, exist_ok=True)
    for suffix in ["pdf", "svg", "png"]:
        fig.savefig(FIG / f"figure_absolute_retention_nll.{suffix}")
    plt.close(fig)


def convert_canonical_figures() -> None:
    """Convert existing canonical SVGs to vector PDFs without redrawing data."""
    source = ANALYSIS / "figures_main"
    FIG.mkdir(parents=True, exist_ok=True)
    for svg in sorted(source.glob("figure_*.svg")):
        cairosvg.svg2pdf(url=str(svg), write_to=str(FIG / f"{svg.stem}.pdf"))


def lambda_log_correlations() -> pd.DataFrame:
    data = pd.read_csv(ANALYSIS / "tables" / "exploratory_lambda_sweep.csv")
    outcomes = ["mean_forgetting", "mean_new_fact_acquisition_nll",
                "parameter_displacement_a_c_l2"]
    rows = []
    for method in METHOD_ORDER:
        group = data[data.method.eq(method)].sort_values("lambda").copy()
        group["log1p_lambda"] = np.log1p(group["lambda"])
        for outcome in outcomes:
            rows.append({
                "method": method,
                "outcome": outcome,
                "pearson_r_raw_lambda": group["lambda"].corr(group[outcome], method="pearson"),
                "pearson_r_log1p_lambda": group["log1p_lambda"].corr(group[outcome], method="pearson"),
                "spearman_rho": group["lambda"].rank().corr(group[outcome].rank(), method="pearson"),
                "n": len(group),
            })
    return pd.DataFrame(rows)


def run_breakdown() -> pd.DataFrame:
    runs = pd.read_csv(ANALYSIS / "data" / "CANONICAL_RUNS.csv")
    categories = [
        ("Sequential single-adapter baseline", runs.protocol.eq("sequential_single")),
        ("Grow-unfrozen mechanistic control", runs.protocol.eq("grow_unfrozen")),
        ("Hard-consolidation mechanistic control", runs.protocol.eq("hard_consolidation")),
        ("EWC lambda=100 mechanistic control", runs.protocol.eq("ewc_like") & runs["lambda"].eq(100)),
        ("Exploratory EWC sweep excluding lambda=100", runs.protocol.eq("ewc_like") & runs.scientific_role.str.contains("exploratory_lambda", na=False) & ~runs["lambda"].eq(100)),
        ("Additional EWC confirmation executions", runs.protocol.eq("ewc_like") & runs.scientific_role.eq("confirmation")),
    ]
    rows = [{"category": name, "canonical_runs": int(mask.sum())} for name, mask in categories]
    out = pd.DataFrame(rows)
    assert out.canonical_runs.sum() == 72, out
    return out


def dataset_summary() -> dict:
    dataset_paths = sorted(REPO.glob("results_ewc_lambda_sweep/*/dataset.json"))
    assert dataset_paths
    dataset = read_json(dataset_paths[0])
    digest = hashlib.sha256(dataset_paths[0].read_bytes()).hexdigest()
    return {"path": str(dataset_paths[0].relative_to(REPO)), "file_sha256": digest,
            "top_level_type": type(dataset).__name__}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_run, identities = load_retention()
    aggregate = aggregate_retention(per_run)
    per_run.to_csv(OUT / "absolute_retention_per_run.csv", index=False)
    aggregate.to_csv(OUT / "absolute_retention_aggregate.csv", index=False)
    identities.to_csv(OUT / "forgetting_identity_verification.csv", index=False)
    lambda_log_correlations().to_csv(OUT / "lambda_correlation_sensitivity.csv", index=False)
    run_breakdown().to_csv(OUT / "canonical_run_breakdown.csv", index=False)
    (OUT / "dataset_source.json").write_text(json.dumps(dataset_summary(), indent=2) + "\n")
    plot_retention(aggregate)
    convert_canonical_figures()
    print(f"Prepared {len(per_run)} confirmation runs; max forgetting identity error "
          f"{identities.filter(like='error_').abs().to_numpy().max():.3e}")


if __name__ == "__main__":
    main()
