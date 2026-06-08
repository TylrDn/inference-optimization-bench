"""
bench/workloads/single_turn.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Single-turn (one user message → one assistant response) workload generator.

Prompts are drawn from five categories that exercise different model
capabilities relevant to inference-infrastructure benchmarking:

1. **Code explanation** — technical deep-dives requiring verbose output.
2. **Technical Q&A** — factual questions about ML systems.
3. **Math reasoning** — multi-step arithmetic / dimensional analysis.
4. **Summarization** — condensing a supplied paragraph into bullet points.
5. **Instruction following** — structured, format-specific output tasks.
"""

from __future__ import annotations

import random
from typing import Final

from bench.workloads.base_workload import WorkloadBase, WorkloadSample

# ---------------------------------------------------------------------------
# Template catalogue — 5 prompts per category = 25 total
# ---------------------------------------------------------------------------

_CODE_EXPLANATION: Final[list[str]] = [
    (
        "Explain in detail how CUDA tensor cores work and how they differ "
        "from regular CUDA CUDA cores. Include information about supported "
        "data types, throughput differences, and which deep-learning "
        "operations benefit most from tensor cores."
    ),
    (
        "Walk me through the implementation of Flash Attention v2. Explain "
        "the tiling strategy, how it avoids materialising the full N×N "
        "attention matrix, and why it achieves near-memory-bandwidth-bound "
        "performance on modern GPUs."
    ),
    (
        "Describe the CUDA programming model in detail: grids, blocks, "
        "warps, and threads. Explain how shared memory, L1 cache, and L2 "
        "cache interact, and what 'occupancy' means in the context of kernel "
        "optimisation."
    ),
    (
        "Explain how continuous batching works in vLLM. What problem does it "
        "solve compared to static batching? Walk through the scheduler loop "
        "and describe how PagedAttention enables it."
    ),
    (
        "Describe how INT8 quantisation is applied to transformer weight "
        "matrices. Cover both weight-only and weight-activation quantisation, "
        "the role of calibration data, and the performance/accuracy tradeoffs "
        "involved."
    ),
]

_TECHNICAL_QA: Final[list[str]] = [
    (
        "What is KV-cache and how does it reduce TTFT and improve throughput "
        "in autoregressive LLM serving? Include an explanation of how memory "
        "is allocated per sequence and how eviction policies work."
    ),
    (
        "Compare tensor parallelism and pipeline parallelism for large-model "
        "inference. When would you choose each strategy, and how do they "
        "interact with batch size and sequence length?"
    ),
    (
        "What is speculative decoding? Explain the draft-then-verify loop, "
        "how acceptance probability is computed, and under what conditions "
        "it yields a meaningful speedup."
    ),
    (
        "Explain the difference between prefill and decode phases in "
        "autoregressive text generation. Why are they computationally "
        "different, and how does disaggregated prefill-decode scheduling "
        "improve GPU utilisation?"
    ),
    (
        "What is AWQ (Activation-Aware Weight Quantisation) and how does it "
        "differ from GPTQ? Describe the salient weight identification step "
        "and explain why preserving certain weights matters."
    ),
]

_MATH_REASONING: Final[list[str]] = [
    (
        "A model has 70 billion parameters stored in bfloat16. How much GPU "
        "VRAM is required to hold the weights alone? Show the calculation "
        "step by step and then estimate the additional memory needed for "
        "KV-cache when serving 32 concurrent sequences of 4096 tokens each, "
        "assuming 80 transformer layers with hidden size 8192 and 64 attention heads."
    ),
    (
        "An H100 SXM5 GPU has 3.35 TB/s of HBM3 bandwidth and can perform "
        "989 TFLOPS of BF16 tensor-core operations. A transformer layer "
        "performs a matrix multiplication of shape [batch=1, seq=1, "
        "hidden=8192] × [8192, 32768]. Is this operation compute-bound or "
        "memory-bound at batch size 1? Show your arithmetic-intensity "
        "calculation."
    ),
    (
        "You have a cluster of 8 × H100 80 GB GPUs. You want to serve a "
        "405B parameter model in FP8. Calculate: (a) total weight memory, "
        "(b) whether the weights fit on 8 GPUs, (c) how much headroom "
        "remains for KV-cache if each GPU also holds an equal shard of "
        "the model."
    ),
    (
        "A benchmark reports p50 TTFT of 120 ms and p99 TTFT of 800 ms at "
        "32 concurrent requests. The target SLA is p99 TTFT < 500 ms. "
        "If latency scales linearly with concurrency, what is the maximum "
        "concurrency level that satisfies the SLA? Show your derivation."
    ),
    (
        "A model generates tokens at 45 tokens/second per request on one "
        "GPU. You need to serve 1,000 simultaneous users each expecting "
        "responses of ~200 tokens within 10 seconds. How many GPUs do you "
        "need at minimum? State any assumptions you make."
    ),
]

_SUMMARIZATION: Final[list[str]] = [
    (
        "Summarise the following passage into exactly 3 concise bullet points, "
        "each starting with a bold keyword:\n\n"
        "PagedAttention is a memory management technique introduced by the vLLM "
        "project that treats the KV-cache like virtual memory in an OS. Instead "
        "of pre-allocating a contiguous block of GPU memory for each sequence's "
        "KV-cache, PagedAttention divides the cache into fixed-size pages that "
        "can be allocated on demand and shared across requests that have common "
        "prefixes. This dramatically reduces memory fragmentation and enables "
        "much higher effective batch sizes than traditional approaches. The "
        "technique was inspired by the paging mechanism in operating systems and "
        "allows the LLM serving engine to pack more concurrent sequences into "
        "the same GPU memory budget, increasing throughput without sacrificing "
        "per-request latency."
    ),
    (
        "Summarise the following passage into exactly 3 concise bullet points, "
        "each starting with a bold keyword:\n\n"
        "Speculative decoding is an inference acceleration technique that uses a "
        "small, fast draft model to generate candidate token sequences which are "
        "then verified in a single forward pass by the larger target model. "
        "Because the draft model is orders of magnitude cheaper to run, the "
        "wall-clock time per token is reduced when most draft tokens are accepted. "
        "The acceptance rate depends on how well the draft distribution matches "
        "the target distribution. When acceptance is high—typically for "
        "predictable text like code or structured output—speedups of 2–3× are "
        "achievable. The technique preserves the exact output distribution of the "
        "target model, making it lossless in the statistical sense."
    ),
    (
        "Summarise the following passage into exactly 3 concise bullet points, "
        "each starting with a bold keyword:\n\n"
        "Grouped-Query Attention (GQA) is a technique that reduces the memory "
        "bandwidth and KV-cache size of transformer models by sharing key and "
        "value projections across groups of query heads. Whereas standard "
        "Multi-Head Attention (MHA) maintains one K and V head per Q head, and "
        "Multi-Query Attention (MQA) uses a single K/V head for all queries, "
        "GQA interpolates between the two extremes by assigning each group of "
        "query heads to one K/V head. Models like Llama-3 and Mistral adopt GQA "
        "to reduce inference memory footprint while retaining most of the "
        "representational capacity of full MHA."
    ),
    (
        "Summarise the following passage into exactly 3 concise bullet points, "
        "each starting with a bold keyword:\n\n"
        "Tensor parallelism splits individual weight matrices across multiple "
        "GPUs so that each device holds a column or row shard of each weight "
        "tensor. During the forward pass, partial matrix products are computed "
        "on each GPU and the results are combined with an all-reduce collective "
        "operation. This allows models too large for a single GPU to be served "
        "without pipeline bubbles, but requires fast NVLink interconnects to "
        "keep the all-reduce overhead manageable. The Megatron-LM library "
        "pioneered this approach and it is now supported natively by TensorRT-LLM "
        "and vLLM."
    ),
    (
        "Summarise the following passage into exactly 3 concise bullet points, "
        "each starting with a bold keyword:\n\n"
        "Quantisation-Aware Training (QAT) inserts fake-quantisation nodes into "
        "the model graph during fine-tuning so that the model learns to be robust "
        "to the precision loss it will experience at inference time. This contrasts "
        "with Post-Training Quantisation (PTQ), which quantises a pre-trained "
        "model without retraining. QAT typically recovers most of the accuracy "
        "lost by aggressive quantisation schemes (e.g., INT4 weights) but requires "
        "access to training data and additional GPU hours. For large language "
        "models, QAT is expensive, so recent work focuses on improved PTQ methods "
        "that approach QAT accuracy without the training cost."
    ),
]

_INSTRUCTION_FOLLOWING: Final[list[str]] = [
    (
        "List the 5 most important GPU metrics to monitor during LLM inference. "
        "For each metric, provide: (1) its name, (2) the unit it is measured in, "
        "(3) the target threshold for a healthy deployment, and (4) what to do "
        "if it exceeds the threshold. Format your response as a numbered list "
        "with the four sub-items indented under each entry."
    ),
    (
        "Write a Python function called `compute_model_flops` that accepts "
        "`num_layers: int`, `hidden_size: int`, `seq_len: int`, and "
        "`batch_size: int` as arguments and returns the approximate number of "
        "floating-point operations for a single transformer forward pass. "
        "Include a docstring explaining the formula and add inline comments. "
        "Return only the function definition with no surrounding prose."
    ),
    (
        "Create a comparison table with exactly 5 rows and 4 columns. "
        "The columns should be: Technique | Memory Reduction | Accuracy Impact "
        "| Typical Speedup. The rows should cover: INT8 PTQ, INT4 AWQ, "
        "FP8 QAT, Pruning (50% sparsity), Speculative Decoding. "
        "Use markdown table syntax."
    ),
    (
        "Explain the steps required to deploy a Llama-3-70B model with vLLM "
        "on two H100 GPUs using tensor parallelism. Provide the exact shell "
        "commands and environment variables needed, formatted as a numbered "
        "step-by-step guide. Do not include any prose outside the numbered steps."
    ),
    (
        "Write a YAML configuration file for a benchmark sweep that tests "
        "the following combinations: backends = [vllm, nim], batch_sizes = "
        "[1, 4, 16, 32], max_tokens = [128, 512], temperature = 0.0. "
        "Include a top-level `name` field and a `description` field. "
        "Return only the YAML content with no surrounding explanation."
    ),
]

# Flat list of all 25 templates with category label in metadata
_ALL_TEMPLATES: Final[list[dict]] = (
    [{"prompt": p, "category": "code_explanation"} for p in _CODE_EXPLANATION]
    + [{"prompt": p, "category": "technical_qa"} for p in _TECHNICAL_QA]
    + [{"prompt": p, "category": "math_reasoning"} for p in _MATH_REASONING]
    + [{"prompt": p, "category": "summarization"} for p in _SUMMARIZATION]
    + [{"prompt": p, "category": "instruction_following"} for p in _INSTRUCTION_FOLLOWING]
)


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------

class SingleTurnWorkload(WorkloadBase):
    """Single-turn prompt workload for LLM inference benchmarking.

    Each generated sample contains exactly one user message.  Templates are
    drawn from five technical categories and shuffled deterministically using
    the supplied seed.  When ``n`` exceeds the number of available templates
    (25), the list is cycled through as many times as necessary.

    Example::

        workload = SingleTurnWorkload()
        samples = workload.generate(n=100, seed=0)
    """

    @property
    def workload_type(self) -> str:
        """Return the workload type identifier ``"single_turn"``."""
        return "single_turn"

    def description(self) -> str:
        """Return a human-readable description of this workload."""
        return (
            "Single-turn workload: 25 template prompts across five categories "
            "(code explanation, technical Q&A, math reasoning, summarisation, "
            "instruction following). Prompts are shuffled and cycled to fill "
            "the requested sample count."
        )

    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]:
        """Generate *n* single-turn benchmark samples.

        Args:
            n: Number of samples to produce.
            seed: RNG seed for deterministic shuffling.

        Returns:
            A list of *n* :class:`~bench.workloads.base_workload.WorkloadSample`
            instances, each containing a single user message.
        """
        rng = random.Random(seed)
        pool = list(_ALL_TEMPLATES)
        rng.shuffle(pool)

        samples: list[WorkloadSample] = []
        for i in range(n):
            template = pool[i % len(pool)]
            prompt = template["prompt"]
            samples.append(
                WorkloadSample(
                    prompt=prompt,
                    messages=[{"role": "user", "content": prompt}],
                    expected_min_tokens=50,
                    metadata={
                        "category": template["category"],
                        "template_index": i % len(pool),
                    },
                )
            )
        return samples
