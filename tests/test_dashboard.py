"""Tests for the Plotly Dash reporting dashboard."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from reporting.dashboard import (
    build_latency_percentile_chart,
    build_latency_vs_batch_chart,
    build_throughput_chart,
    build_ttft_chart,
    compute_latency_percentiles,
    compute_mean_throughput,
    compute_mean_ttft,
    create_app,
    load_results,
)


@pytest.fixture()
def sample_results_dir(tmp_path: Path) -> str:
    df = pd.DataFrame(
        [
            {
                "request_id": 0,
                "backend": "nim",
                "batch_size": 1,
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "ttft_ms": 50.0,
                "tpot_ms": 5.0,
                "total_latency_ms": 150.0,
                "tokens_per_sec": 133.0,
            },
            {
                "request_id": 1,
                "backend": "vllm",
                "batch_size": 4,
                "prompt_tokens": 12,
                "completion_tokens": 25,
                "ttft_ms": 40.0,
                "tpot_ms": 4.0,
                "total_latency_ms": 140.0,
                "tokens_per_sec": 178.0,
            },
        ]
    )
    path = tmp_path / "bench.csv"
    df.to_csv(path, index=False)
    return str(tmp_path)


def test_load_results_returns_dataframe(sample_results_dir: str) -> None:
    df = load_results(sample_results_dir)
    assert df is not None
    assert len(df) == 2
    assert set(df["backend"]) == {"nim", "vllm"}


def test_load_results_empty_dir(tmp_path: Path) -> None:
    assert load_results(str(tmp_path)) is None


def test_compute_helpers(sample_results_dir: str) -> None:
    df = load_results(sample_results_dir)
    assert df is not None
    pct = compute_latency_percentiles(df)
    assert {"backend", "p50", "p95", "p99"}.issubset(pct.columns)
    ttft = compute_mean_ttft(df)
    assert "mean_ttft_ms" in ttft.columns
    tps = compute_mean_throughput(df)
    assert "mean_tps" in tps.columns


def test_chart_builders(sample_results_dir: str) -> None:
    df = load_results(sample_results_dir)
    assert df is not None
    for builder in (
        build_latency_percentile_chart,
        build_ttft_chart,
        build_throughput_chart,
        build_latency_vs_batch_chart,
    ):
        fig = builder(df)
        assert fig.data


def test_create_app_with_data(sample_results_dir: str) -> None:
    app = create_app(sample_results_dir)
    assert app.layout is not None


def test_create_app_empty_dir(tmp_path: Path) -> None:
    app = create_app(str(tmp_path))
    assert app.layout is not None


def test_load_results_injects_backend_from_filename(tmp_path: Path) -> None:
    path = tmp_path / "custom_backend.csv"
    pd.DataFrame({"total_latency_ms": [100.0], "ttft_ms": [10.0], "tokens_per_sec": [50.0]}).to_csv(
        path, index=False
    )
    df = load_results(str(tmp_path))
    assert df is not None
    assert df.iloc[0]["backend"] == "custom_backend"
