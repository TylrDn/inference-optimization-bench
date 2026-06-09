"""
bench/workloads/multi_turn.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Multi-turn conversation workload for LLM inference benchmarking.

Each sample includes a 2-to-4 turn conversation history (alternating
user/assistant) followed by a final user message.  Templates cover four
real-world LLM-infrastructure dialogue scenarios:

* Technical debugging
* Architecture discussion
* Code review
* LLM optimization discussion

Generating multi-turn samples stresses the KV-cache pre-fill path because
the model must process the entire conversation history before generating the
next assistant turn.
"""

from __future__ import annotations

import random
from typing import Final

from bench.workloads.base_workload import WorkloadBase, WorkloadSample

# ---------------------------------------------------------------------------
# Conversation templates
# Each entry is a list of alternating user/assistant dicts followed by a
# final user turn.  The assistant turns represent "ideal" prior responses and
# are included verbatim so the benchmark faithfully exercises the KV-cache
# pre-fill path.
# ---------------------------------------------------------------------------

_TEMPLATES: Final[list[dict]] = [
    # ------------------------------------------------------------------
    # 1. CUDA OOM debugging conversation (4 turns)
    # ------------------------------------------------------------------
    {
        "category": "technical_debugging",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I'm running a vLLM server with Llama-3-70B on 4×A100 80 GB "
                    "GPUs with tensor_parallel_size=4. After about 20 minutes of "
                    "traffic the process dies with `torch.cuda.OutOfMemoryError`. "
                    "Any idea what's happening?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "That's a classic KV-cache exhaustion issue. A few questions "
                    "to narrow it down: (1) What's the `gpu_memory_utilization` "
                    "setting in your vLLM config? Default is 0.90. (2) Are you "
                    "seeing any very long sequences (>8k tokens) in your traffic? "
                    "(3) Are you using chunked prefill? If not, a single long "
                    "prefill can spike memory and trigger the OOM before the "
                    "scheduler has a chance to evict stale blocks."
                ),
            },
            {
                "role": "user",
                "content": (
                    "gpu_memory_utilization is 0.90. Most requests are short "
                    "(under 1k tokens) but we do occasionally get 16k-token "
                    "context requests. We're not using chunked prefill."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The 16k-token requests are almost certainly the culprit. "
                    "A single 16k-token prefill for Llama-3-70B with GQA "
                    "(8 KV heads, 80 layers, bfloat16) consumes roughly "
                    "2 × 8 × 80 × 16384 × 128 × 2 bytes ≈ 5.4 GB of KV-cache "
                    "memory *per request*. If several of these arrive "
                    "concurrently, you exhaust your reserved pool. "
                    "Enable chunked prefill (`enable_chunked_prefill=True`) "
                    "and cap `max_num_batched_tokens` to something like 4096."
                ),
            },
            {
                "role": "user",
                "content": (
                    "That makes sense. After enabling chunked prefill the OOMs "
                    "stopped, but now p99 TTFT for those long requests jumped "
                    "from 1.2s to 4.8s. Is there a way to get chunked prefill "
                    "without that TTFT regression?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 2. Architecture discussion: tensor vs. pipeline parallelism (3 turns)
    # ------------------------------------------------------------------
    {
        "category": "architecture_discussion",
        "messages": [
            {
                "role": "user",
                "content": (
                    "We're planning to serve a 405B parameter model on 16 H100s. "
                    "Should we use tensor parallelism, pipeline parallelism, or "
                    "a hybrid? What are the tradeoffs?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "For 405B on 16 H100s the practical choice is usually a "
                    "hybrid: tensor parallelism (TP) within each 8-GPU node "
                    "(where NVLink bandwidth is ~900 GB/s), and pipeline "
                    "parallelism (PP) across the two nodes (where you're limited "
                    "to 400 Gb/s InfiniBand). Pure TP across 16 GPUs over "
                    "InfiniBand would drown in all-reduce latency. Pure PP "
                    "introduces pipeline bubbles that hurt utilisation at low "
                    "batch sizes. TP=8, PP=2 is the standard starting point."
                ),
            },
            {
                "role": "user",
                "content": (
                    "What's the maximum batch size I can fit with TP=8, PP=2 "
                    "and FP8 weights, if I want to keep at least 20 GB per GPU "
                    "for KV-cache?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "FP8 weights for 405B = 405B × 1 byte = ~405 GB total, or "
                    "~25.3 GB per GPU across 16. Each H100 SXM5 has 80 GB HBM3, "
                    "so you have 80 − 25.3 ≈ 54.7 GB free per GPU. With your "
                    "20 GB KV-cache reservation you have ~34.7 GB for activations "
                    "and framework overhead, which is comfortable for batch sizes "
                    "up to ~64–128 depending on sequence length. "
                    "Use `vllm serve --max-model-len 4096 --gpu-memory-utilization 0.88` "
                    "as a starting point and watch the VRAM headroom metric."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Thanks. One more thing: does TensorRT-LLM handle TP=8, PP=2 "
                    "automatically, or do I need to build separate engine shards?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 3. Code review: async inference server (3 turns)
    # ------------------------------------------------------------------
    {
        "category": "code_review",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Can you review this Python function that batches inference "
                    "requests?\n\n"
                    "```python\n"
                    "async def batch_infer(prompts: list[str], backend) -> list[str]:\n"
                    "    results = []\n"
                    "    for p in prompts:\n"
                    "        r = await backend.infer(p)\n"
                    "        results.append(r.text)\n"
                    "    return results\n"
                    "```\n"
                    "Is there anything wrong with it?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Yes — the loop `await`s each request sequentially, so "
                    "you lose all concurrency benefit. Requests are sent one "
                    "at a time rather than in parallel. You should gather all "
                    "coroutines concurrently with `asyncio.gather`:\n\n"
                    "```python\n"
                    "import asyncio\n\n"
                    "async def batch_infer(prompts: list[str], backend) -> list[str]:\n"
                    "    coros = [backend.infer(p) for p in prompts]\n"
                    "    results = await asyncio.gather(*coros)\n"
                    "    return [r.text for r in results]\n"
                    "```\n"
                    "This sends all requests concurrently and collects results "
                    "in the original order. You may also want `return_exceptions=True` "
                    "if you need resilience to individual failures."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Good catch. What if I also want to limit concurrency to "
                    "at most 32 simultaneous in-flight requests to avoid "
                    "overwhelming the backend?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Use `asyncio.Semaphore`:\n\n"
                    "```python\n"
                    "import asyncio\n\n"
                    "async def batch_infer(\n"
                    "    prompts: list[str], backend, max_concurrency: int = 32\n"
                    ") -> list[str]:\n"
                    "    sem = asyncio.Semaphore(max_concurrency)\n\n"
                    "    async def _infer_one(p: str):\n"
                    "        async with sem:\n"
                    "            return await backend.infer(p)\n\n"
                    "    results = await asyncio.gather(*[_infer_one(p) for p in prompts])\n"
                    "    return [r.text for r in results]\n"
                    "```\n"
                    "The semaphore ensures at most `max_concurrency` coroutines "
                    "are inside `backend.infer()` at any one time, while still "
                    "queuing the rest for immediate execution when a slot opens."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Perfect. Should I propagate cancellations from the caller "
                    "into the in-flight requests, and how would I do that?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 4. LLM optimization: speculative decoding tuning (2 turns)
    # ------------------------------------------------------------------
    {
        "category": "llm_optimization",
        "messages": [
            {
                "role": "user",
                "content": (
                    "We enabled speculative decoding in vLLM using a 1B draft "
                    "model and `num_speculative_tokens=5`, but our throughput "
                    "actually went *down* by 15%. What could be going wrong?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Several things can cause spec-dec to regress throughput:\n\n"
                    "1. **Low acceptance rate**: If your request distribution is "
                    "diverse (creative writing, open-ended Q&A), the draft model "
                    "acceptance rate may be below ~0.7, meaning you're doing "
                    "extra work for little gain. Check `spec_decode_acceptance_rate` "
                    "in your Prometheus metrics.\n\n"
                    "2. **Draft model bottleneck**: If your GPU is near capacity, "
                    "running the draft model adds latency even when it saves "
                    "target-model calls.\n\n"
                    "3. **Batch size interaction**: Spec-dec helps most at "
                    "batch_size=1. At high concurrency the target model's "
                    "batched execution is already efficient and spec-dec "
                    "overhead dominates.\n\n"
                    "What is your typical concurrent request count?"
                ),
            },
            {
                "role": "user",
                "content": (
                    "Acceptance rate metrics show ~0.62. Typical concurrency "
                    "is 48 requests. Based on this, should we disable spec-dec "
                    "or try to improve the acceptance rate? If the latter, "
                    "how?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 5. Debugging a tokenizer mismatch (3 turns)
    # ------------------------------------------------------------------
    {
        "category": "technical_debugging",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Our benchmark reports that Llama-3-8B generates only "
                    "18 tokens for a request where we set max_new_tokens=200. "
                    "The prompt is: 'List the planets in our solar system.' "
                    "Why would it stop so early?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "The most likely cause is an EOS token being generated "
                    "early. Llama-3 uses a special `<|eot_id|>` end-of-turn "
                    "token that the serving framework may interpret as EOS. "
                    "Check two things:\n\n"
                    "1. Is `stop_token_ids` set in your generation config and "
                    "does it include token ID 128009 (`<|eot_id|>`)?\n\n"
                    "2. Is the prompt wrapped with the instruct chat template? "
                    "If so, the model expects to end its turn with `<|eot_id|>` "
                    "and the serving layer stops there."
                ),
            },
            {
                "role": "user",
                "content": (
                    "You're right — the chat template was being applied "
                    "automatically and the EOS list included `<|eot_id|>`. "
                    "We removed the chat template and now completions are "
                    "longer, but the output looks odd — no punctuation at the "
                    "end of sentences, and reasoning steps are incomplete. "
                    "What's happening now?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Stripping the chat template entirely swings to the other "
                    "extreme: the instruct-tuned model now sees a raw completion "
                    "prompt instead of the structured turn format it was trained "
                    "on, so its output degrades. The correct fix is to keep the "
                    "chat template but *not* add `<|eot_id|>` to the stop-token "
                    "list, or to pass `skip_special_tokens=False` and handle EOS "
                    "detection manually. Alternatively set `include_stop_str_in_output=True` "
                    "so you can inspect what triggered the stop."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Got it. Is there a way to benchmark both instruct-template "
                    "and raw-completion modes side by side without running two "
                    "separate vLLM instances?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 6. Architecture discussion: disaggregated prefill-decode (3 turns)
    # ------------------------------------------------------------------
    {
        "category": "architecture_discussion",
        "messages": [
            {
                "role": "user",
                "content": (
                    "I've been reading about disaggregated prefill-decode "
                    "architectures. Can you explain the core idea and when "
                    "it makes sense to adopt it?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "In a standard LLM serving setup, the same GPU handles both "
                    "prefill (processing the input tokens) and decode (generating "
                    "output tokens). These two phases have very different compute "
                    "profiles: prefill is compute-bound (large matmuls across the "
                    "full sequence), while decode is memory-bandwidth-bound "
                    "(reading model weights for a single new token). This "
                    "mismatch means that a single GPU must be optimised for two "
                    "conflicting workload types.\n\n"
                    "Disaggregated prefill-decode splits these phases across "
                    "separate GPU pools: a 'prefill fleet' processes incoming "
                    "context in parallel, then transfers the resulting KV-cache "
                    "state to a 'decode fleet' that autoregressively generates "
                    "the response. This lets each fleet be independently scaled "
                    "and tuned."
                ),
            },
            {
                "role": "user",
                "content": (
                    "What are the main engineering challenges? I'm particularly "
                    "interested in the KV-cache transfer overhead."
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "KV-cache transfer is the dominant engineering challenge. "
                    "For a 70B model with 80 layers and 8 GQA heads, a 4k-token "
                    "context takes roughly 2 × 8 × 80 × 4096 × 128 × 2 bytes "
                    "≈ 1.35 GB per request. At 100 req/s that's 135 GB/s of "
                    "cross-fleet bandwidth — easily saturating a 100 GbE link. "
                    "Production systems (e.g. Mooncake, Llumnix) use RDMA over "
                    "InfiniBand or NVLink-C2C to keep transfer latency under "
                    "10 ms. You also need a scheduler that co-ordinates KV "
                    "block IDs between fleets and handles failures gracefully "
                    "when a decode node goes down mid-generation."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Does vLLM currently support disaggregated prefill-decode, "
                    "or would we need to build it ourselves?"
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 7. Code review: metrics collection (2 turns)
    # ------------------------------------------------------------------
    {
        "category": "code_review",
        "messages": [
            {
                "role": "user",
                "content": (
                    "Here's my TTFT measurement code:\n\n"
                    "```python\n"
                    "t0 = time.time()\n"
                    "response = requests.post(url, json=payload)\n"
                    "t1 = time.time()\n"
                    "ttft_ms = (t1 - t0) * 1000\n"
                    "```\n"
                    "The numbers look too high — around 800ms even for short "
                    "prompts on a local GPU. What am I measuring wrong?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "You're measuring total round-trip latency, not TTFT. "
                    "`requests.post` with a non-streaming call blocks until "
                    "the *entire* response body is received, so `t1 − t0` "
                    "is `total_latency`, not time-to-first-token.\n\n"
                    "To measure true TTFT you need to:\n"
                    "1. Set `stream=True` in your JSON payload.\n"
                    "2. Use `requests.get(url, json=payload, stream=True)` "
                    "or `httpx.stream()`.\n"
                    "3. Record `t_first_token = time.perf_counter()` inside "
                    "the loop at the first non-empty SSE chunk.\n\n"
                    "Also prefer `time.perf_counter()` over `time.time()` for "
                    "sub-second interval measurements — it has higher resolution "
                    "and is monotonic."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Thanks. I switched to streaming with httpx. Now my TTFT "
                    "readings are 40–60 ms for short prompts locally, which "
                    "seems more realistic. But how do I accurately count the "
                    "completion tokens when streaming? The final usage chunk "
                    "isn't always present."
                ),
            },
        ],
    },
    # ------------------------------------------------------------------
    # 8. LLM optimization: quantisation accuracy regression (4 turns)
    # ------------------------------------------------------------------
    {
        "category": "llm_optimization",
        "messages": [
            {
                "role": "user",
                "content": (
                    "We quantised Mixtral-8x7B to INT4 using AWQ and noticed "
                    "a significant accuracy drop on our internal coding benchmark. "
                    "BF16 baseline scores 72.4, AWQ INT4 scores 61.8. That's a "
                    "14% regression. Is that expected for AWQ INT4?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "A 14% accuracy drop is larger than typical AWQ INT4 "
                    "regression — most published results show 2–5% degradation "
                    "on coding tasks. A few things to check:\n\n"
                    "1. **Group size**: What group size did you use for AWQ "
                    "calibration? Larger group sizes (e.g. 128) are more "
                    "aggressive than 64 or 32.\n\n"
                    "2. **Calibration data**: AWQ is sensitive to calibration "
                    "data quality. If you used generic text rather than "
                    "code-domain data, the salient weight search may have "
                    "protected the wrong weights.\n\n"
                    "3. **MoE routing layers**: Mixtral's routing layers are "
                    "often excluded from quantisation in reference implementations "
                    "— make sure those are in FP16."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Group size is 128 and we used The Pile as calibration data. "
                    "How should we re-run with better calibration data? Should "
                    "we use code-specific corpora?"
                ),
            },
            {
                "role": "assistant",
                "content": (
                    "Yes — calibration data should match your target distribution "
                    "as closely as possible. For a coding-heavy use case:\n\n"
                    "1. Use a mix of The Stack v2 or CodeSearchNet samples "
                    "(Python, JavaScript, Rust) as calibration.\n\n"
                    "2. Keep group_size=128 for now; if accuracy is still poor "
                    "try group_size=64 (doubles KV-cache but recovers accuracy).\n\n"
                    "3. Ensure the MoE router weights (`model.layers.*.block_sparse_moe.gate.weight`) "  # noqa: E501
                    "are excluded from quantisation with AWQ's `modules_to_not_convert` list.\n\n"
                    "Re-run with 512 calibration samples; more than that rarely "
                    "helps for AWQ."
                ),
            },
            {
                "role": "user",
                "content": (
                    "After re-calibrating on code data with group_size=64 and "
                    "excluding the router, the score improved to 68.9 — still "
                    "3.5 points below BF16. Is there anything else we can try "
                    "to close that gap further without going to FP8?"
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Workload class
# ---------------------------------------------------------------------------

class MultiTurnWorkload(WorkloadBase):
    """Multi-turn conversation workload for LLM inference benchmarking.

    Each generated sample provides a full conversation history (2–4 prior
    turns) followed by a final user question.  This exercises the KV-cache
    pre-fill path for longer contexts.

    Example::

        workload = MultiTurnWorkload()
        samples = workload.generate(n=50, seed=7)
    """

    @property
    def workload_type(self) -> str:
        """Return the workload type identifier ``"multi_turn"``."""
        return "multi_turn"

    def description(self) -> str:
        """Return a human-readable description of this workload."""
        return (
            "Multi-turn workload: 8 conversation templates covering technical "
            "debugging, architecture discussions, code review, and LLM "
            "optimisation. Each sample includes 2–4 prior turns of history "
            "plus a final user question."
        )

    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]:
        """Generate *n* multi-turn benchmark samples.

        Templates are shuffled deterministically and cycled to fill *n*.

        Args:
            n: Number of samples to produce.
            seed: RNG seed for deterministic shuffling.

        Returns:
            A list of *n* :class:`~bench.workloads.base_workload.WorkloadSample`
            instances.  Each sample's ``messages`` list contains the full
            conversation history.  The last message always has
            ``role="user"``.
        """
        rng = random.Random(seed)
        pool = list(_TEMPLATES)
        rng.shuffle(pool)

        samples: list[WorkloadSample] = []
        for i in range(n):
            template = pool[i % len(pool)]
            messages = template["messages"]

            # Sanity-check: last message must be from the user
            assert messages[-1]["role"] == "user", (
                f"Template {i % len(pool)} last message is not from 'user'"
            )

            # Use the final user turn as the plain-text prompt field
            prompt = messages[-1]["content"]

            samples.append(
                WorkloadSample(
                    prompt=prompt,
                    messages=messages,
                    expected_min_tokens=100,
                    metadata={
                        "category": template["category"],
                        "turn_count": len(messages),
                        "template_index": i % len(pool),
                    },
                )
            )
        return samples
