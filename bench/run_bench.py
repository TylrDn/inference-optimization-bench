"""Single benchmark run CLI."""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

from bench.backends.llamacpp_backend import LlamaCppBackend
from bench.backends.triton_backend import TritonBackend
from bench.backends.vllm_backend import VLLMBackend
from bench.metrics import BenchmarkReport
from bench.warmup import warmup_backend

PROMPTS = [
    "Explain the difference between GPTQ and AWQ quantization in three sentences.",
    "What is DCGM and why does it matter for GPU cluster monitoring?",
    "Describe how continuous batching in vLLM improves GPU utilization.",
    "What are the tradeoffs between tensor parallelism and pipeline parallelism?",
    "Explain how KV-cache prefix caching reduces time-to-first-token.",
]

BACKEND_MAP = {
    "vllm": VLLMBackend,
    "triton": TritonBackend,
    "llamacpp": LlamaCppBackend,
}


def main():
    parser = argparse.ArgumentParser(description="Run single inference benchmark")
    parser.add_argument("--model", required=True, help="Model name (e.g. llama3-8b)")
    parser.add_argument("--backend", required=True, choices=list(BACKEND_MAP.keys()))
    parser.add_argument("--quant", default="none", help="Quantization method")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--num-requests", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output-dir", default="./results")
    args = parser.parse_args()

    backend_cls = BACKEND_MAP[args.backend]
    backend = backend_cls()

    if not backend.health_check():
        print(f"ERROR: {args.backend} backend not reachable. Is the server running?")
        return

    print(f"Warming up {args.backend} backend...")
    warmup_backend(lambda p: backend.infer(0, p, max_tokens=64), n_warmup=args.warmup)

    report = BenchmarkReport(
        model=args.model,
        backend=args.backend,
        quantization=args.quant,
        batch_size=args.batch_size,
        sequence_length=args.seq_len,
    )

    print(f"Running {args.num_requests} requests...")
    for i in range(args.num_requests):
        prompt = PROMPTS[i % len(PROMPTS)]
        result = backend.infer(i, prompt, max_tokens=args.seq_len)
        report.results.append(result)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{args.num_requests} completed")

    summary = report.summary()
    print("\n--- Benchmark Summary ---")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    run_id = f"{args.model}_{args.backend}_{args.quant}_{int(time.time())}"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(args.output_dir) / f"{run_id}.csv"
    with open(out_path, "w", newline="") as f:
        from dataclasses import asdict
        writer = csv.DictWriter(f, fieldnames=asdict(report.results[0]).keys())
        writer.writeheader()
        writer.writerows([asdict(r) for r in report.results])
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
