"""Triton benchmark backend — calls Triton HTTP endpoint."""
from __future__ import annotations

import time

from bench.metrics import RequestResult, get_gpu_vram_used_mib

try:
    import tritonclient.http as httpclient
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class TritonBackend:
    def __init__(self, url: str = "localhost:8000", model_name: str = "llama3"):
        self.url = url
        self.model_name = model_name
        if TRITON_AVAILABLE:
            self.client = httpclient.InferenceServerClient(url=url)

    def health_check(self) -> bool:
        if not TRITON_AVAILABLE:
            return False
        try:
            return self.client.is_server_ready()
        except Exception:
            return False

    def infer(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RequestResult:
        import numpy as np
        text_input = httpclient.InferInput("text_input", [1], "BYTES")
        text_input.set_data_from_numpy(np.array([prompt.encode()], dtype=object))
        output = httpclient.InferRequestedOutput("text_output")
        t_start = time.perf_counter()
        result = self.client.infer(self.model_name, [text_input], outputs=[output])
        t_end = time.perf_counter()
        text = result.as_numpy("text_output")[0].decode("utf-8")
        completion_tokens = len(text.split())
        total_ms = (t_end - t_start) * 1000
        return RequestResult(
            request_id=request_id,
            prompt_tokens=len(prompt.split()),
            completion_tokens=completion_tokens,
            ttft_ms=total_ms,
            tpot_ms=total_ms / max(completion_tokens, 1),
            total_latency_ms=total_ms,
            tokens_per_sec=completion_tokens / (total_ms / 1000),
            vram_used_mib=get_gpu_vram_used_mib(),
        )
