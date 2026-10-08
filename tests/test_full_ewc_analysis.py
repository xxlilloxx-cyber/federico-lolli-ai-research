from __future__ import annotations

import numpy as np
import pandas as pd

from analyze_full_ewc_memory_study import discover, extract_metrics, pareto, validate


def test_discovery_deduplicates_reused_scientific_observations():
    runs, stats, errors = discover()
    assert errors == []
    assert stats == {
        "physical_runs": 75,
        "canonical_runs": 72,
        "physical_duplicates": 3,
        "confirmation_references": 9,
        "duplicate_or_reused_links": 12,
    }
    assert sum("confirmation" in run.roles for run in runs) == 27
    assert sum("exploratory_lambda" in run.roles for run in runs) == 21


def test_complete_study_validation_passes():
    runs, stats, errors = discover()
    assert validate(runs, errors, stats) == []


def test_one_reused_confirmation_run_reconstructs_finite_metrics():
    runs, _, _ = discover()
    run = next(run for run in runs if run.method == "lora" and run.value == 10 and run.seed == 42 and "confirmation" in run.roles)
    row, factors = extract_metrics(run)
    assert row["adapter_parameters"] == 6144
    assert np.isfinite([row["mean_forgetting"], row["mean_new_fact_acquisition_nll"], row["mean_final_memory_nll"]]).all()
    assert factors


def test_global_pareto_marks_dominated_points():
    frame = pd.DataFrame([
        {"method": "lora", "lambda": 0, "mean_forgetting": 4.0, "mean_new_fact_acquisition_nll": 0.2},
        {"method": "symmetric", "lambda": 10, "mean_forgetting": 2.0, "mean_new_fact_acquisition_nll": 0.5},
        {"method": "combined", "lambda": 30, "mean_forgetting": 3.0, "mean_new_fact_acquisition_nll": 0.8},
    ])
    found = pareto(frame)
    assert found.pareto_nondominated.tolist() == [True, True, False]
