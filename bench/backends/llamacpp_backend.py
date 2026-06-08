"""llama.cpp benchmark backend — calls llama-server HTTP endpoint."""
from __future__ import annotations
import time
import httpx
from bench.metrics import RequestResult, get_gpu_vram_used_mib


class LlamaCppBackend:
    def __init__(self, base_url: str = "http://localhost:8080", model: str = "llama3-gguf"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=120.0)

    def health_check(self) -> bool:
        try:
            r = self.client.get(f"{self.base_url}/health")
            return r.status_code == 200
        except Exception:
            return False

    def infer(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RequestResult:
        payload = {"prompt": prompt, "n_predict": max_tokens, "temperature": temperature}
        t_start = time.perf_counter()
        response = self.client.post(f"{self.base_url}/completion", json=payload)
        t_end = time.perf_counter()
        response.raise_for_status()
        data = response.json()
        content = data.get("content", "")
        completion_tokens = data.get("tokens_evaluated", len(content.split()))
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
