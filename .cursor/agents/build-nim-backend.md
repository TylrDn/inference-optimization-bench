---
name: build-nim-backend
description: Invoke when building or modifying the NVIDIA NIM inference backend. Use when the user asks to implement nim_backend.py, add NIM support, wire NIM into the benchmark suite, or fix NIM-related health/infer issues.
model: inherit
readonly: false
is_background: false
---

# Build NIM Inference Backend

## Objective

Create `bench/backends/nim_backend.py` — the NVIDIA NIM inference backend that integrates with the OpenAI-compatible NIM API at `https://integrate.api.nvidia.com/v1`. This is the **highest priority gap** in the repo and the primary NVIDIA differentiator. It must match the `AbstractBackend` interface exactly so it plugs into the existing `BACKEND_MAP` in `bench/backends/__init__.py`.

---

## Files to Create / Modify

### Create: `bench/backends/nim_backend.py`

Full implementation. No placeholders. Every method must be production-complete.

**Exact imports:**
```python
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import aiohttp
from openai import AsyncOpenAI, APIConnectionError, APIStatusError, APITimeoutError

from bench.backends.base_backend import AbstractBackend
from bench.config import BackendConfig
from bench.exceptions import BackendUnavailableError
from bench.metrics import InferenceResult
```

**Class signature:**
```python
class NIMBackend(AbstractBackend):
    """NVIDIA NIM inference backend using OpenAI-compatible REST API.

    Connects to https://integrate.api.nvidia.com/v1 (or a custom NIM_BASE_URL).
    Supports streaming for accurate TTFT measurement.
    """

    def __init__(self, config: BackendConfig) -> None: ...
    async def health_check(self) -> bool: ...
    async def infer(self, prompt: str, params: dict[str, Any]) -> InferenceResult: ...
    async def infer_stream(self, prompt: str, params: dict[str, Any]) -> InferenceResult: ...
    async def teardown(self) -> None: ...
```

**`__init__` must:**
- Read `NIM_API_KEY` and `NIM_BASE_URL` from environment (with `NIM_BASE_URL` defaulting to `https://integrate.api.nvidia.com/v1`)
- Instantiate `AsyncOpenAI(api_key=self._api_key, base_url=self._base_url)`
- Store `config.model_name` (e.g. `"meta/llama3-70b-instruct"`)
- Store `config.timeout_s` (default 120)
- Set up `self._logger = logging.getLogger(__name__)`

**`health_check` must:**
- Call `await self._client.models.list()` (async)
- Return `True` if successful, `False` on `APIConnectionError` or `APIStatusError`
- Log health status at INFO level
- Raise `BackendUnavailableError` if called and NIM_API_KEY is not set

**`infer` must:**
- Delegate to `infer_stream` (NIM always uses streaming for TTFT accuracy)
- Accept `params: dict[str, Any]` with keys: `max_tokens`, `temperature`, `top_p`, `seed`

**`infer_stream` must:**
- Use `time.perf_counter()` for timing (NOT `time.time()`)
- Record `t_start = time.perf_counter()` before API call
- Call `await self._client.chat.completions.create(model=..., messages=[{"role": "user", "content": prompt}], stream=True, **params)`
- Record `t_first_token` on receiving first chunk (`chunk.choices[0].delta.content is not None`)
- Accumulate full response text across all chunks
- Record `t_end` after stream closes
- Return `InferenceResult` with:
  - `ttft_ms = (t_first_token - t_start) * 1000`
  - `total_latency_ms = (t_end - t_start) * 1000`
  - `tokens_generated` from `chunk.usage.completion_tokens` if available, else `len(full_text.split())`
  - `prompt_tokens` from usage
  - `raw_response = {"text": full_text, "model": ..., "finish_reason": ...}`
- On `APITimeoutError`: raise `BackendUnavailableError(f"NIM request timed out after {self._timeout_s}s")`
- On `APIStatusError` with 429: implement exponential backoff (3 retries, 2/4/8s delays), log WARNING
- On `APIStatusError` with 5xx: raise `BackendUnavailableError`

**`teardown` must:**
- Call `await self._client.close()`
- Log INFO "NIM backend torn down"

---

### Modify: `bench/backends/__init__.py`

Add NIMBackend to `BACKEND_MAP`:

```python
from bench.backends.nim_backend import NIMBackend

BACKEND_MAP: dict[str, type[AbstractBackend]] = {
    "vllm": VLLMBackend,
    "triton": TritonBackend,
    "llamacpp": LlamaCppBackend,
    "nim": NIMBackend,  # ADD THIS
}
```

---

### Create: `tests/test_nim_backend.py`

Full unit test suite. Mock all network calls.

**Required tests:**
```python
async def test_health_check_success(mocker): ...
async def test_health_check_failure_connection_error(mocker): ...
async def test_health_check_raises_when_no_api_key(monkeypatch): ...
async def test_infer_stream_measures_ttft(mocker): ...
async def test_infer_stream_retries_on_429(mocker): ...
async def test_infer_stream_raises_on_timeout(mocker): ...
async def test_infer_stream_raises_on_5xx(mocker): ...
async def test_teardown_closes_client(mocker): ...

@pytest.mark.parametrize("model", [
    "meta/llama3-70b-instruct",
    "meta/llama3-8b-instruct",
    "mistralai/mixtral-8x7b-instruct-v0.1",
])
async def test_infer_with_different_models(model, mocker): ...
```

**Fixtures to add in `tests/conftest.py`:**
```python
@pytest.fixture
def nim_backend_config() -> BackendConfig:
    return BackendConfig(
        backend_type="nim",
        model_name="meta/llama3-8b-instruct",
        base_url="https://integrate.api.nvidia.com/v1",
        timeout_s=30,
    )
```

---

### Modify: `configs/full_sweep.yaml`

Add NIM backend section:

```yaml
backends:
  nim:
    model_name: meta/llama3-70b-instruct
    timeout_s: 120
    concurrent_requests: [1, 4, 8, 16, 32]
    quantization: null  # NIM handles this server-side
```

---

### Modify: `.env.template`

Add:
```
NIM_API_KEY=nvapi-your-key-here
NIM_BASE_URL=https://integrate.api.nvidia.com/v1
```

---

## Acceptance Criteria

- [ ] `pytest tests/test_nim_backend.py` passes (all 10+ tests green, no network calls)
- [ ] `mypy --strict bench/backends/nim_backend.py` exits 0
- [ ] `ruff check bench/backends/nim_backend.py` exits 0
- [ ] `NIMBackend` is importable from `bench.backends` via `BACKEND_MAP["nim"]`
- [ ] Running `python bench/run_bench.py --backend nim --config configs/quick.yaml` with valid `NIM_API_KEY` completes without errors and writes a JSON result file to `results/`
- [ ] TTFT is measured using `time.perf_counter()` streaming (verified in test)
- [ ] `health_check()` returns `False` (not raises) on connection error
- [ ] 429 retries are tested and verified to have exponential backoff delays
