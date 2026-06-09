"""
bench/backends/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface of the backends sub-package.

All four backend classes are imported here so that ``run_bench.py`` can
resolve a backend by string name via ``BACKEND_MAP`` without importing
individual modules.

Backend classes all share the same interface as :class:`VLLMBackend`:

* ``__init__(self, base_url: str, model: str, ...)``
* ``health_check(self) -> bool``
* ``infer(self, request_id: int, prompt: str,
  max_tokens: int, temperature: float) -> RequestResult``
"""

from bench.backends.llamacpp_backend import LlamaCppBackend
from bench.backends.nim_backend import NIMBackend
from bench.backends.triton_backend import TritonBackend
from bench.backends.vllm_backend import VLLMBackend

BACKEND_MAP: dict[str, type] = {
    "vllm": VLLMBackend,
    "triton": TritonBackend,
    "llamacpp": LlamaCppBackend,
    "nim": NIMBackend,
}

__all__ = [
    "VLLMBackend",
    "TritonBackend",
    "LlamaCppBackend",
    "NIMBackend",
    "BACKEND_MAP",
]
