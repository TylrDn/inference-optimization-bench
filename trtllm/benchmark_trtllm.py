"""Benchmark TRT-LLM engine: TTFT, throughput, VRAM."""
from __future__ import annotations
import argparse
import csv
import time
from pathlib import Path
from bench.metrics import RequestResult, BenchmarkReport, get_gpu_vram_used_mib

PROMPTS = [
    "Explain PagedAttention and how it reduces GPU memory fragmentation.",
    "What is TensorRT-LLM and how does it differ from vLLM?",
    "Describe the NVIDIA Triton Python backend execution model.",
]


def benchmark(engine_dir: str, num_requests: int, max_tokens: int, output_dir: str):
    from trtllm.run_engine import run_trtllm

    report = BenchmarkReport(
        model="trtllm-engine",
        backend="trtllm",
        quantization="none",
        batch_size=1,
        sequence_length=max_tokens,
    )

    print("Warmup...")
    for _ in range(3):
        run_trtllm(engine_dir, PROMPTS[0], max_tokens=64)

    print(f"Benchmarking {num_requests} requests...")
    for i in range(num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        vram = get_gpu_vram_used_mib()
        t_start = time.perf_counter()
        text = run_trtllm(engine_dir, prompt, max_tokens=max_tokens)
        t_end = time.perf_counter()
        total_ms = (t_end - t_start) * 1000
        completion_tokens = len(text.split())
        report.results.append(RequestResult(
            request_id=i,
            prompt_tokens=len(prompt.split()),
            completion_tokens=completion_tokens,
            ttft_ms=total_ms,
            tpot_ms=total_ms / max(completion_tokens, 1),
            total_latency_ms=total_ms,
            tokens_per_sec=completion_tokens / (total_ms / 1000),
            vram_used_mib=vram,
        ))

    summary = report.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    from dataclasses import asdict
    out_path = Path(output_dir) / f"trtllm_{int(time.time())}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(report.results[0]).keys())
        writer.writeheader()
        writer.writerows([asdict(r) for r in report.results])
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-dir", required=True)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()
    benchmark(args.engine_dir, args.num_requests, args.max_tokens, args.output_dir)
