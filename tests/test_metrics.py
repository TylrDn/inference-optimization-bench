"""Tests for benchmark metrics module."""
from bench.metrics import BenchmarkReport, RequestResult


def make_result(i: int, latency_ms: float, tokens: int = 50) -> RequestResult:
    return RequestResult(
        request_id=i,
        prompt_tokens=20,
        completion_tokens=tokens,
        ttft_ms=latency_ms,
        tpot_ms=latency_ms / tokens,
        total_latency_ms=latency_ms,
        tokens_per_sec=tokens / (latency_ms / 1000),
    )


def test_benchmark_report_summary():
    report = BenchmarkReport(
        model="test", backend="vllm", quantization="none",
        batch_size=1, sequence_length=512,
    )
    for i in range(100):
        report.results.append(make_result(i, float(i + 1)))
    s = report.summary()
    assert s["n_requests"] == 100
    assert s["latency_p50_ms"] == 50.0
    assert s["latency_p95_ms"] == 95.0
    assert s["throughput_tokens_per_sec"] > 0


def test_empty_results_handled():
    report = BenchmarkReport(
        model="test", backend="vllm", quantization="none",
        batch_size=1, sequence_length=512,
    )
    report.results.append(make_result(0, 100.0))
    s = report.summary()
    assert s["n_requests"] == 1
