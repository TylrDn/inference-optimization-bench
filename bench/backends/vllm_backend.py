"""vLLM benchmark backend — calls local vLLM OpenAI-compatible server."""
from __future__ import annotations
import time
import httpx
from typing import Optional
from bench.metrics import RequestResult, get_gpu_vram_used_mib


class VLLMBackend:
    def __init__(self, base_url: str = "http://localhost:8010", model: str = "llama3-8b"):
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
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        vram_before = get_gpu_vram_used_mib()
        t_start = time.perf_counter()
        response = self.client.post(f"{self.base_url}/v1/chat/completions", json=payload)
        t_end = time.perf_counter()
        response.raise_for_status()
        data = response.json()
        content = data["choices"][0]["message"]["content"]
        completion_tokens = len(content.split())
        total_ms = (t_end - t_start) * 1000
        tpot_ms = total_ms / max(completion_tokens, 1)
        return RequestResult(
            request_id=request_id,
            prompt_tokens=len(prompt.split()),
            completion_tokens=completion_tokens,
            ttft_ms=total_ms,  # Non-streaming: total = TTFT proxy
            tpot_ms=tpot_ms,
            total_latency_ms=total_ms,
            tokens_per_sec=completion_tokens / (total_ms / 1000),
            vram_used_mib=get_gpu_vram_used_mib() or vram_before,
        )
