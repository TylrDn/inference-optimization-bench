"""
bench/workloads/tool_calling.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Tool-calling workload for LLM inference benchmarking.

Each generated sample contains a system message that defines a set of
available tools in OpenAI function-calling schema format, followed by a user
message that requires the model to call one or more of those tools.

Three tool schemas are defined:

1. ``get_gpu_metrics`` — query live GPU utilisation, VRAM, and temperature.
2. ``run_benchmark`` — trigger a benchmark sweep and return latency numbers.
3. ``query_model_registry`` — search available models by keyword.

The samples exercise the model's ability to output structured JSON tool-call
responses, which stresses instruction-following and structured-output paths
that may have different latency profiles from free-text generation.
"""

from __future__ import annotations

import json
import random
from typing import Final

from bench.workloads.base_workload import WorkloadBase, WorkloadSample

# ---------------------------------------------------------------------------
# Tool schemas (OpenAI function-calling format)
# ---------------------------------------------------------------------------

_GPU_METRICS_SCHEMA: Final[dict] = {
    "type": "function",
    "function": {
        "name": "get_gpu_metrics",
        "description": (
            "Retrieve real-time performance metrics for a specific GPU device. "
            "Returns GPU utilisation percentage, used VRAM in MiB, total VRAM "
            "in MiB, and temperature in Celsius."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "device_id": {
                    "type": "integer",
                    "description": (
                        "Zero-based index of the GPU device to query "
                        "(e.g. 0 for the first GPU)."
                    ),
                }
            },
            "required": ["device_id"],
        },
    },
}

_RUN_BENCHMARK_SCHEMA: Final[dict] = {
    "type": "function",
    "function": {
        "name": "run_benchmark",
        "description": (
            "Execute an inference benchmark sweep for a given backend and model "
            "configuration. Returns p50, p95, and p99 latency in milliseconds, "
            "mean throughput in tokens/s, and total elapsed time."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "backend": {
                    "type": "string",
                    "enum": ["vllm", "nim", "triton", "llamacpp"],
                    "description": "Inference backend to benchmark.",
                },
                "model": {
                    "type": "string",
                    "description": (
                        "Model identifier (e.g. 'meta/llama-3.1-70b-instruct')."
                    ),
                },
                "batch_size": {
                    "type": "integer",
                    "description": "Number of concurrent requests to issue.",
                },
                "max_tokens": {
                    "type": "integer",
                    "description": "Maximum tokens to generate per request.",
                    "default": 512,
                },
                "num_requests": {
                    "type": "integer",
                    "description": "Total number of requests in the sweep.",
                    "default": 100,
                },
            },
            "required": ["backend", "model", "batch_size"],
        },
    },
}

_QUERY_MODEL_REGISTRY_SCHEMA: Final[dict] = {
    "type": "function",
    "function": {
        "name": "query_model_registry",
        "description": (
            "Search the internal model registry for models matching a text "
            "query. Returns a list of matching model records including name, "
            "parameter count, supported precision formats, and availability "
            "on the current cluster."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text search query (e.g. 'llama 70b instruct', "
                        "'mixtral moe', 'code generation model')."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 5,
                },
                "available_only": {
                    "type": "boolean",
                    "description": (
                        "If true, only return models currently deployed on "
                        "the cluster."
                    ),
                    "default": True,
                },
            },
            "required": ["query"],
        },
    },
}

_ALL_TOOLS: Final[list[dict]] = [
    _GPU_METRICS_SCHEMA,
    _RUN_BENCHMARK_SCHEMA,
    _QUERY_MODEL_REGISTRY_SCHEMA,
]

_TOOLS_JSON: Final[str] = json.dumps(_ALL_TOOLS, indent=2)

# ---------------------------------------------------------------------------
# System message template
# ---------------------------------------------------------------------------

_SYSTEM_MESSAGE: Final[str] = (
    "You are an AI assistant embedded in an ML infrastructure platform. "
    "You have access to the following tools:\n\n"
    f"{_TOOLS_JSON}\n\n"
    "When the user asks you to perform an action that requires a tool, "
    "respond with a JSON object containing `tool_call` with `name` and "
    "`arguments` fields. Do not perform any action that cannot be expressed "
    "as a tool call. If multiple tool calls are needed, return them as a JSON "
    "array under `tool_calls`."
)

# ---------------------------------------------------------------------------
# User request templates
# ---------------------------------------------------------------------------

_USER_REQUESTS: Final[list[dict]] = [
    # --- get_gpu_metrics requests ---
    {
        "prompt": (
            "What's the current GPU utilisation and VRAM usage on GPU 0? "
            "I need to check before launching a large batch job."
        ),
        "expected_tools": ["get_gpu_metrics"],
        "category": "gpu_metrics",
    },
    {
        "prompt": (
            "Check the temperature on GPUs 0 through 3 — I want to make sure "
            "none of them are thermal throttling before I start the overnight "
            "benchmark sweep."
        ),
        "expected_tools": ["get_gpu_metrics"],
        "category": "gpu_metrics",
    },
    {
        "prompt": (
            "How much free VRAM is available on GPU 1 right now? I need at "
            "least 40 GB to load the 70B model in BF16."
        ),
        "expected_tools": ["get_gpu_metrics"],
        "category": "gpu_metrics",
    },
    # --- run_benchmark requests ---
    {
        "prompt": (
            "Run a benchmark sweep for the vLLM backend with "
            "meta/llama-3.1-8b-instruct at batch sizes 1, 8, and 32. "
            "Use max_tokens=256 and 200 total requests per sweep."
        ),
        "expected_tools": ["run_benchmark"],
        "category": "run_benchmark",
    },
    {
        "prompt": (
            "I need to compare NIM and vLLM for the 70B model at batch_size=16. "
            "Can you kick off benchmark runs for both backends?"
        ),
        "expected_tools": ["run_benchmark"],
        "category": "run_benchmark",
    },
    {
        "prompt": (
            "Run a quick 50-request benchmark for the triton backend using "
            "meta/llama-3.1-70b-instruct with batch_size=4 and max_tokens=128."
        ),
        "expected_tools": ["run_benchmark"],
        "category": "run_benchmark",
    },
    # --- query_model_registry requests ---
    {
        "prompt": (
            "What instruct-tuned Llama models are currently deployed on the "
            "cluster? Show me all available ones."
        ),
        "expected_tools": ["query_model_registry"],
        "category": "model_registry",
    },
    {
        "prompt": (
            "Search the model registry for any code generation models that "
            "support FP8 precision and are available right now."
        ),
        "expected_tools": ["query_model_registry"],
        "category": "model_registry",
    },
    {
        "prompt": (
            "List all Mixtral MoE variants in the model registry, including "
            "ones not currently deployed."
        ),
        "expected_tools": ["query_model_registry"],
        "category": "model_registry",
    },
    # --- multi-tool requests ---
    {
        "prompt": (
            "Before I run a benchmark, I want to check that GPU 0 has enough "
            "headroom (under 80% utilisation and at least 20 GB free VRAM). "
            "Also, can you look up what 8B models are available in the "
            "registry? I'll then kick off a benchmark once you have that info."
        ),
        "expected_tools": ["get_gpu_metrics", "query_model_registry"],
        "category": "multi_tool",
    },
    {
        "prompt": (
            "I want a full health check: (1) query GPU metrics for device 0, "
            "(2) find all available 70B instruct models, and (3) run a quick "
            "10-request benchmark with vllm at batch_size=1 using the first "
            "result from the registry."
        ),
        "expected_tools": [
            "get_gpu_metrics",
            "query_model_registry",
            "run_benchmark",
        ],
        "category": "multi_tool",
    },
    {
        "prompt": (
            "Check whether GPU 2 is free (utilisation < 10%), and if so, "
            "run a benchmark for nim backend with meta/llama-3.1-70b-instruct "
            "at batch_size=8."
        ),
        "expected_tools": ["get_gpu_metrics", "run_benchmark"],
        "category": "multi_tool",
    },
]


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------

class ToolCallingWorkload(WorkloadBase):
    """Tool-calling workload for LLM inference benchmarking.

    Each generated sample contains a system message with the tool schema
    definitions followed by a user request that requires one or more tool
    calls.  This tests structured-output generation latency and the model's
    ability to select the correct tool(s).

    Example::

        workload = ToolCallingWorkload()
        samples = workload.generate(n=60, seed=99)
    """

    @property
    def workload_type(self) -> str:
        """Return the workload type identifier ``"tool_calling"``."""
        return "tool_calling"

    def description(self) -> str:
        """Return a human-readable description of this workload."""
        return (
            "Tool-calling workload: 12 user request templates across three "
            "tool schemas (get_gpu_metrics, run_benchmark, query_model_registry) "
            "and multi-tool scenarios. System message includes full OpenAI "
            "function-calling schema definitions."
        )

    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]:
        """Generate *n* tool-calling benchmark samples.

        Each sample's ``messages`` list starts with the system tool-definition
        message followed by a single user request.

        Args:
            n: Number of samples to produce.
            seed: RNG seed for deterministic shuffling.

        Returns:
            A list of *n* :class:`~bench.workloads.base_workload.WorkloadSample`
            instances.
        """
        rng = random.Random(seed)
        pool = list(_USER_REQUESTS)
        rng.shuffle(pool)

        samples: list[WorkloadSample] = []
        for i in range(n):
            request = pool[i % len(pool)]
            prompt = request["prompt"]
            messages: list[dict[str, str]] = [
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ]
            samples.append(
                WorkloadSample(
                    prompt=prompt,
                    messages=messages,
                    expected_min_tokens=50,
                    metadata={
                        "category": request["category"],
                        "expected_tools": request["expected_tools"],
                        "template_index": i % len(pool),
                    },
                )
            )
        return samples
