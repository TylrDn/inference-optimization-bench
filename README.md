# inference-optimization-bench

Systematic benchmarking suite for LLM inference optimization — quantization (GPTQ, AWQ, GGUF), KV-cache strategies, TRT-LLM compilation, and continuous batching analysis. Produces reproducible latency/throughput/VRAM reports across model sizes and hardware.

**Target Role:** [Solutions Architect, Agentic AI — NVIDIA JR2014517](https://nvidia.wd5.myworkdayjobs.com/en-US/NVIDIAExternalCareerSite/job/US-CA-Santa-Clara/Solutions-Architect--Agentic-AI_JR2014517) | LLM Model Builder JR2014441580

## Benchmark Dimensions

| Dimension | Options |
|---|---|
| Quantization | None (fp16), GPTQ 4-bit, AWQ 4-bit, GGUF Q4_K_M, GGUF Q8_0 |
| Serving Backend | vLLM, Triton+TRT-LLM, llama.cpp |
| KV-Cache | Baseline, prefix caching, paged attention |
| Batch Size | 1, 4, 8, 16, 32 |
| Sequence Length | 512, 1024, 2048, 4096 tokens |
| Hardware | A100 80GB, H100 80GB, RTX 4090, T4 (multi-target) |

## Quick Start

```bash
pip install -r requirements.txt
cp .env.template .env
# Run single benchmark
python bench/run_bench.py --model llama3-8b --backend vllm --quant none
# Run full sweep
python bench/sweep.py --config configs/full_sweep.yaml
# Generate report
python reporting/generate_report.py --results results/
```

## Architecture

```
inference-optimization-bench/
├── bench/                    # Core benchmarking runners
│   ├── run_bench.py           # Single benchmark run
│   ├── sweep.py               # Full sweep across configs
│   ├── metrics.py             # TTFT, TPOT, throughput, VRAM
│   ├── warmup.py              # GPU warmup + consistency checks
│   └── backends/
│       ├── vllm_backend.py
│       ├── triton_backend.py
│       └── llamacpp_backend.py
├── quantization/              # Quantization pipelines
│   ├── gptq_quantize.py
│   ├── awq_quantize.py
│   └── gguf_convert.sh
├── trtllm/                    # TRT-LLM compilation + engine runner
│   ├── build_engine.sh
│   ├── run_engine.py
│   └── benchmark_trtllm.py
├── configs/                   # Sweep + model config YAML
├── reporting/                 # Markdown + chart report generation
├── results/                   # Benchmark output CSVs
├── notebooks/                 # Analysis notebooks
└── README.md
```

## Outputs

- `results/{run_id}.csv` — per-request latency, TTFT, TPOT, tokens/sec, VRAM
- `reporting/report_{run_id}.md` — auto-generated markdown summary with Plotly charts
- P50/P95/P99 latency breakdown across batch sizes and quant methods

## Cross-Repo Integration

- Quantized models consumed by [`model-serving-stack`](https://github.com/TylrDn/model-serving-stack) for production serving
- Fine-tuned base models from [`llm-finetuning-lab`](https://github.com/TylrDn/llm-finetuning-lab)
- NIM agents benchmarked via [`nvidia-nim-agent-toolkit`](https://github.com/TylrDn/nvidia-nim-agent-toolkit)

## Topics

`llm-inference` `quantization` `gptq` `awq` `gguf` `trt-llm` `triton-inference-server` `vllm` `benchmarking` `nvidia` `python` `gpu`
