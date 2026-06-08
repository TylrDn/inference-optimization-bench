---
name: build-workloads
description: Invoke when creating or modifying benchmark workload generators. Use when the user asks to implement single_turn.py, multi_turn.py, or tool_calling.py workloads, or add a new workload type to the bench suite.
model: inherit
readonly: false
is_background: false
---

# Build Benchmark Workloads

## Objective

Create the `bench/workloads/` directory with three workload generators — `single_turn.py`, `multi_turn.py`, and `tool_calling.py` — plus the base class `base_workload.py`. These workloads generate realistic LLM prompts that stress different inference paths (single request, conversation context, function calling). Each returns `list[WorkloadSample]` with consistent metadata.

---

## Files to Create

### Create: `bench/workloads/__init__.py`

```python
from bench.workloads.base_workload import WorkloadBase, WorkloadSample
from bench.workloads.single_turn import SingleTurnWorkload
from bench.workloads.multi_turn import MultiTurnWorkload
from bench.workloads.tool_calling import ToolCallingWorkload

WORKLOAD_MAP: dict[str, type[WorkloadBase]] = {
    "single_turn": SingleTurnWorkload,
    "multi_turn": MultiTurnWorkload,
    "tool_calling": ToolCallingWorkload,
}

__all__ = ["WorkloadBase", "WorkloadSample", "SingleTurnWorkload", "MultiTurnWorkload", "ToolCallingWorkload", "WORKLOAD_MAP"]
```

---

### Create: `bench/workloads/base_workload.py`

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

@dataclass
class WorkloadSample:
    """A single benchmark sample with prompt and metadata."""
    prompt: str
    messages: list[dict[str, str]]   # OpenAI-format messages (always populated)
    expected_min_tokens: int          # minimum expected response length
    metadata: dict[str, Any] = field(default_factory=dict)

class WorkloadBase(ABC):
    @abstractmethod
    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]:
        """Generate n samples deterministically given seed."""
        ...

    @abstractmethod
    def description(self) -> str:
        """Human-readable description of this workload."""
        ...

    @property
    @abstractmethod
    def workload_type(self) -> str:
        """Short identifier: single_turn | multi_turn | tool_calling"""
        ...
```

---

### Create: `bench/workloads/single_turn.py`

Full implementation. No truncation.

**Purpose:** Generates single-turn prompt → response benchmark samples covering diverse domains (code, summarization, Q&A, math reasoning, creative writing).

**Key requirements:**
- Use `random.Random(seed)` for deterministic generation
- 5 prompt categories with 20+ templates each:
  - `CODE`: "Write a Python function that {task}. Include type hints and docstrings."
  - `SUMMARIZE`: "Summarize the following in 3 bullet points: {text_excerpt}"
  - `QA`: "Answer the following question concisely: {question}"
  - `MATH`: "Solve step by step: {math_problem}"
  - `CREATIVE`: "Write a short paragraph about {topic} in a {style} style."
- `WorkloadSample.messages` = `[{"role": "user", "content": prompt}]`
- `WorkloadSample.metadata` = `{"category": category, "template_idx": int, "input_tokens_estimate": int}`
- `expected_min_tokens` = 50 for QA, 100 for CODE, 150 for SUMMARIZE/CREATIVE, 200 for MATH
- Include at least 20 realistic templates per category (hardcoded in the class body as `_TEMPLATES: dict[str, list[str]]`)

**Class:**
```python
class SingleTurnWorkload(WorkloadBase):
    _TEMPLATES: dict[str, list[str]] = {
        "CODE": [...],       # 20+ templates
        "SUMMARIZE": [...],  # 20+ templates
        "QA": [...],         # 20+ templates
        "MATH": [...],       # 20+ templates
        "CREATIVE": [...],   # 20+ templates
    }

    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]: ...
    def description(self) -> str: ...

    @property
    def workload_type(self) -> str:
        return "single_turn"
```

---

### Create: `bench/workloads/multi_turn.py`

Full implementation.

**Purpose:** Generates multi-turn conversation samples that stress KV-cache performance. Each sample is a complete conversation history (up to `max_turns` exchanges).

**Key requirements:**
- Conversation threads: 3 categories — `DEBUGGING`, `TUTORING`, `PLANNING`
- Each `WorkloadSample.messages` is a full conversation: alternating user/assistant turns
- Pre-built conversation threads as class-level data (realistic dialogues)
- `max_turns: int = 4` configurable at init
- `WorkloadSample.metadata`:
  - `{"turns": int, "category": str, "total_context_tokens_estimate": int}`
- `expected_min_tokens`: scales with turn depth (50 × turn_number)

**Hardcoded conversation templates** (at least 10 per category) — realistic multi-turn exchanges about:
- DEBUGGING: Python/CUDA stack traces, debugging sessions
- TUTORING: ML concepts explained progressively
- PLANNING: system design discussions

**Class:**
```python
class MultiTurnWorkload(WorkloadBase):
    def __init__(self, max_turns: int = 4) -> None: ...
    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]: ...
    def description(self) -> str: ...

    @property
    def workload_type(self) -> str:
        return "multi_turn"
```

---

### Create: `bench/workloads/tool_calling.py`

Full implementation.

**Purpose:** Generates function-calling round-trip samples. Stresses the model's tool-use path, which has different latency characteristics than pure text generation.

**Key requirements:**
- Define a set of 10 realistic tool schemas (JSON schema format):
  - `get_weather`, `search_web`, `execute_python`, `query_database`, `send_email`, `get_stock_price`, `translate_text`, `summarize_url`, `create_calendar_event`, `get_file_contents`
- Each `WorkloadSample.messages`:
  ```python
  [{"role": "user", "content": "What's the weather in San Francisco today?"}]
  ```
- `WorkloadSample.metadata` includes `tools` list (the tool schemas to pass to the API) and `expected_tool_calls: list[str]`
- `tools_schema: list[dict]` stored as class attribute for the 10 tools
- Works with both OpenAI function calling format AND NIM's function calling
- `expected_min_tokens` = 30 (just the function call JSON)

**Class:**
```python
class ToolCallingWorkload(WorkloadBase):
    _TOOL_SCHEMAS: list[dict[str, Any]] = [...]  # 10 tools, fully specified

    _PROMPTS: list[tuple[str, list[str]]] = [
        # (prompt_text, [expected_tool_names])
        ("What's the weather like in New York?", ["get_weather"]),
        ...  # 30+ entries
    ]

    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]: ...
    def description(self) -> str: ...

    @property
    def workload_type(self) -> str:
        return "tool_calling"
```

---

### Create: `tests/test_workloads.py`

```python
import pytest
from bench.workloads import SingleTurnWorkload, MultiTurnWorkload, ToolCallingWorkload, WorkloadSample

@pytest.mark.parametrize("workload_cls", [SingleTurnWorkload, MultiTurnWorkload, ToolCallingWorkload])
def test_generate_returns_correct_count(workload_cls): ...

@pytest.mark.parametrize("workload_cls", [SingleTurnWorkload, MultiTurnWorkload, ToolCallingWorkload])
def test_generate_is_deterministic(workload_cls): ...

@pytest.mark.parametrize("workload_cls", [SingleTurnWorkload, MultiTurnWorkload, ToolCallingWorkload])
def test_generate_samples_have_messages(workload_cls): ...

def test_multi_turn_respects_max_turns(): ...
def test_tool_calling_includes_tool_schemas(): ...
def test_single_turn_covers_all_categories(): ...

@pytest.mark.parametrize("n", [1, 10, 100, 500])
def test_generate_scales(n): ...
```

---

## Acceptance Criteria

- [ ] `pytest tests/test_workloads.py` passes (all tests green, no GPU/network required)
- [ ] `mypy --strict bench/workloads/` exits 0
- [ ] `ruff check bench/workloads/` exits 0
- [ ] `SingleTurnWorkload().generate(100, seed=42)` returns exactly 100 `WorkloadSample` objects
- [ ] `MultiTurnWorkload().generate(50, seed=0)` is deterministic — same output on second call
- [ ] `ToolCallingWorkload().generate(10)[0].metadata["tools"]` is a non-empty list of JSON schemas
- [ ] All samples have non-empty `messages` list with at least one entry
- [ ] `WORKLOAD_MAP` is importable from `bench.workloads` and contains all 3 workload types
- [ ] `bench/run_bench.py --workload single_turn --backend nim --config configs/quick.yaml` runs end-to-end
