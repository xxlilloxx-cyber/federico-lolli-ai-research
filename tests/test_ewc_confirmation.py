from __future__ import annotations

from types import SimpleNamespace

from run_ewc_confirmation import (
    EXPECTED_DATASET_SHA256,
    GRID,
    SEEDS,
    canonical_config,
    discover_seed42,
    selected_cells,
    validate_run,
)
from run_ewc_lambda_sweep import analyze_run


def test_confirmation_grid_has_exactly_27_cells():
    assert GRID == {
        "lora": (0.0, 1.0, 10.0),
        "symmetric": (0.0, 10.0, 30.0),
        "combined": (0.0, 10.0, 30.0),
    }
    args = SimpleNamespace(method=None, seed=None, lambda_value=None)
    cells = selected_cells(args)
    assert len(cells) == 27
    assert {seed for _, _, seed in cells} == set(SEEDS)


def test_cli_filters_preserve_predeclared_grid():
    args = SimpleNamespace(method="lora", seed=123, lambda_value=10.0)
    assert selected_cells(args) == [("lora", 10.0, 123)]


def test_all_nine_seed42_sources_pass_strict_reuse_validation():
    reused, invalid = discover_seed42("cuda")
    assert len(reused) == 9
    assert invalid == []
    assert all(row["seed"] == 42 and not row["errors"] for row in reused)


def test_reuse_validator_rejects_execution_shape_mismatch():
    reused, _ = discover_seed42("cuda")
    row = reused[0]
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    source = root / row["source"]
    expected = canonical_config("cuda", row["lambda"])
    expected["micro_batch_size"] = 2
    errors = validate_run(source, row["method"], row["lambda"], 42, expected)
    assert any("micro_batch_size" in error for error in errors)


def test_canonical_confirmation_configuration():
    config = canonical_config("cuda", 10.0)
    assert config["micro_batch_size"] == 4
    assert config["gradient_accumulation"] == 1
    assert config["effective_batch_size"] == 4
    assert config["fisher_samples"] == config["fisher_batches"] == 12
    assert config["steps_per_phase"] == 2500
    assert config["amp"] == "fp16"
    assert len(EXPECTED_DATASET_SHA256) == 64


def test_reused_run_uses_the_same_machine_readable_aggregation_path():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    row = analyze_run(root / "results_ewc_lambda_sweep" / "combined_lambda_10_seed_42")
    assert row["method"] == "combined" and row["lambda"] == 10.0 and row["seed"] == 42
    assert row["adapter_parameters"] == 6144
    assert all(map(lambda value: value == value and abs(value) != float("inf"),
                   (row["mean_forgetting"], row["mean_new_fact_acquisition_nll"], row["final_mean_nll"])))
