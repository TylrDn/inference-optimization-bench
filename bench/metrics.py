"""Inference benchmark metrics: TTFT, TPOT, throughput, VRAM."""
from __future__ import annotations
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


@dataclass
class RequestResult:
    request_id: int
    prompt_tokens: int
    completion_tokens: int
    ttft_ms: float        # Time to first token
    tpot_ms: float        # Time per output token
    total_latency_ms: float
    tokens_per_sec: float
    vram_used_mib: Optional[float] = None


@dataclass
class BenchmarkReport:
    model: str
    backend: str
    quantization: str
    batch_size: int
    sequence_length: int
    results: List[RequestResult] = field(default_factory=list)

    def summary(self) -> dict:
        lats = [r.total_latency_ms for r in self.results]
        ttfts = [r.ttft_ms for r in self.results]
        tps = [r.tokens_per_sec for r in self.results]
        n = len(lats)
        s = sorted(lats)
        return {
            "model": self.model,
            "backend": self.backend,
            "quantization": self.quantization,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "n_requests": n,
            "latency_p50_ms": s[int(0.50 * n)],
            "latency_p95_ms": s[int(0.95 * n)],
            "latency_p99_ms": s[int(0.99 * n)],
            "ttft_mean_ms": statistics.mean(ttfts),
            "ttft_p95_ms": sorted(ttfts)[int(0.95 * n)],
            "throughput_tokens_per_sec": statistics.mean(tps),
            "vram_peak_mib": max((r.vram_used_mib or 0) for r in self.results),
        }


def get_gpu_vram_used_mib(device_index: int = 0) -> Optional[float]:
    if not NVML_AVAILABLE:
        return None
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.used / (1024 ** 2)
    except Exception:
        return None
