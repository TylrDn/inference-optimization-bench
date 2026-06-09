"""
bench/backends/nim_backend.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
NIM (NVIDIA Inference Microservice) backend for the inference-optimization bench.

This module provides :class:`NIMBackend`, which communicates with a NIM-compatible
OpenAI-style HTTP API using synchronous ``httpx`` streaming requests.  Timing
measurements are taken at the byte level so that TTFT reflects the actual
wall-clock delay until the first token arrives over the wire rather than a
client-side parse artefact.
"""

from __future__ import annotations

import json
import os
import time
from typing import Iterator

import httpx

from bench.metrics import RequestResult


# ---------------------------------------------------------------------------
# Helper: iterate over SSE lines from a streaming httpx response
# ---------------------------------------------------------------------------

def _iter_sse_chunks(response: httpx.Response) -> Iterator[dict]:
    """Yield parsed JSON data objects from a Server-Sent Events stream.

    Each non-empty, non-``[DONE]`` ``data:`` line is JSON-decoded and yielded.

    Args:
        response: An open ``httpx.Response`` object in streaming mode.

    Yields:
        Parsed JSON dictionaries representing individual SSE events.
    """
    for raw_line in response.iter_lines():
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            break
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


# ---------------------------------------------------------------------------
# NIMBackend
# ---------------------------------------------------------------------------

class NIMBackend:
    """Synchronous HTTP backend for NVIDIA Inference Microservices (NIM).

    The backend uses ``httpx`` with streaming enabled so that the
    time-to-first-token (TTFT) can be measured accurately:

    * ``t_start`` is captured immediately before the HTTP request is sent.
    * ``t_first_token`` is captured when the **first non-empty SSE chunk**
      (containing at least one token in ``choices[0].delta.content``) arrives.
    * ``t_end`` is captured after the stream is fully consumed and the
      connection is closed.

    Retry logic:
    * HTTP 429 (rate-limited): up to 3 retries with exponential back-off
      (2 s → 4 s → 8 s).
    * HTTP 5xx or network timeout: raises :class:`RuntimeError` immediately.

    Args:
        base_url: Root URL of the NIM-compatible OpenAI API endpoint.
        model: Model identifier string passed in every request.
        api_key: Bearer token for the NIM API.  If *None*, the value of the
            ``NIM_API_KEY`` environment variable is used.  If neither is set
            the client is constructed without an ``Authorization`` header,
            which is suitable for on-prem NIM deployments that do not require
            authentication.

    Example::

        backend = NIMBackend(
            base_url="https://integrate.api.nvidia.com/v1",
            model="meta/llama-3.1-70b-instruct",
        )
        if backend.health_check():
            result = backend.infer(request_id=1, prompt="Hello, world!")
    """

    _RETRY_STATUSES = {429}
    _MAX_RETRIES = 3
    _BACKOFF_BASE = 2.0  # seconds

    def __init__(
        self,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = "meta/llama-3.1-70b-instruct",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

        resolved_key = api_key or os.environ.get("NIM_API_KEY")
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"

        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=120.0,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health_check(self) -> bool:
        """Check whether the NIM endpoint is reachable and returns models.

        Sends a ``GET /v1/models`` request and considers any 2xx response a
        sign of health.  All exceptions (network errors, timeouts, non-2xx
        responses) are caught and result in ``False`` being returned so that
        callers can safely poll without try/except boilerplate.

        Returns:
            ``True`` if the endpoint responds with HTTP 2xx, ``False``
            otherwise.
        """
        try:
            response = self._client.get("/models")
            return response.is_success
        except Exception:
            return False

    def infer(
        self,
        request_id: int,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> RequestResult:
        """Run a single inference request against the NIM endpoint.

        The prompt is wrapped into a single user message and sent to
        ``POST /v1/chat/completions`` with ``stream=True``.  Token counts
        are extracted from the final ``usage`` chunk when available; if the
        endpoint does not return usage the method falls back to a whitespace-
        based token approximation.

        Args:
            request_id: Caller-supplied integer identifier for this request,
                stored verbatim in the returned :class:`~bench.metrics.RequestResult`.
            prompt: The user-facing text prompt.
            max_tokens: Maximum number of completion tokens the model may
                generate.
            temperature: Sampling temperature (0.0 = greedy / deterministic).

        Returns:
            A fully-populated :class:`~bench.metrics.RequestResult` instance.

        Raises:
            RuntimeError: On HTTP 5xx responses or request timeouts.
        """
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        attempt = 0
        last_exc: Exception | None = None

        while attempt <= self._MAX_RETRIES:
            try:
                result = self._do_streaming_request(request_id, prompt, payload)
                return result
            except _RateLimitError as exc:
                last_exc = exc
                attempt += 1
                if attempt > self._MAX_RETRIES:
                    break
                backoff = self._BACKOFF_BASE ** attempt  # 2, 4, 8
                time.sleep(backoff)
            except RuntimeError:
                raise

        raise RuntimeError(
            f"NIMBackend: request_id={request_id} exceeded retry limit "
            f"({self._MAX_RETRIES} retries). Last error: {last_exc}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _do_streaming_request(
        self,
        request_id: int,
        prompt: str,
        payload: dict,
    ) -> RequestResult:
        """Execute one streaming POST and return a :class:`RequestResult`.

        Args:
            request_id: Identifier forwarded to the result object.
            prompt: Original prompt text (used for fallback token counting).
            payload: Full JSON body to POST.

        Returns:
            Populated :class:`~bench.metrics.RequestResult`.

        Raises:
            _RateLimitError: When the server returns HTTP 429.
            RuntimeError: When the server returns HTTP 5xx or a timeout occurs.
        """
        t_start = time.perf_counter()
        t_first_token: float | None = None
        collected_content: list[str] = []
        usage: dict | None = None

        try:
            with self._client.stream(
                "POST",
                "/chat/completions",
                json=payload,
            ) as response:
                if response.status_code == 429:
                    raise _RateLimitError("HTTP 429 from NIM endpoint")
                if response.status_code >= 500:
                    raise RuntimeError(
                        f"NIMBackend: server error HTTP {response.status_code}"
                    )
                response.raise_for_status()

                for chunk in _iter_sse_chunks(response):
                    # Final usage chunk (some NIM builds append this last)
                    if chunk.get("usage"):
                        usage = chunk["usage"]

                    choices = chunk.get("choices") or []
                    if not choices:
                        continue

                    delta = choices[0].get("delta") or {}
                    content_piece = delta.get("content") or ""

                    if content_piece:
                        if t_first_token is None:
                            t_first_token = time.perf_counter()
                        collected_content.append(content_piece)

        except httpx.TimeoutException as exc:
            raise RuntimeError(
                f"NIMBackend: request_id={request_id} timed out: {exc}"
            ) from exc

        t_end = time.perf_counter()

        # ---- timing ------------------------------------------------
        if t_first_token is None:
            # No content token arrived — use t_end as fallback so that
            # ttft_ms is always ≤ total_latency_ms.
            t_first_token = t_end

        ttft_ms = (t_first_token - t_start) * 1000.0
        total_latency_ms = (t_end - t_start) * 1000.0

        # ---- token counts ------------------------------------------
        if usage:
            prompt_tokens: int = usage.get("prompt_tokens", 0)
            completion_tokens: int = usage.get("completion_tokens", len(collected_content))
        else:
            # Rough whitespace-based approximation as fallback
            prompt_tokens = max(1, len(prompt.split()))
            completion_tokens = max(1, len("".join(collected_content).split()))

        # ---- derived metrics ---------------------------------------
        # tpot = (total_latency - ttft) / completion_tokens
        # Guard against division by zero.
        if completion_tokens > 0 and (total_latency_ms - ttft_ms) > 0:
            tpot_ms = (total_latency_ms - ttft_ms) / completion_tokens
        else:
            tpot_ms = 0.0

        tokens_per_sec = (
            (completion_tokens / (total_latency_ms / 1000.0))
            if total_latency_ms > 0
            else 0.0
        )

        return RequestResult(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=total_latency_ms,
            tokens_per_sec=tokens_per_sec,
            vram_used_mib=None,  # NIM does not expose per-request VRAM via this API
        )

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"NIMBackend(base_url={self.base_url!r}, model={self.model!r})"
        )


# ---------------------------------------------------------------------------
# Internal sentinel exception (not part of public API)
# ---------------------------------------------------------------------------

class _RateLimitError(Exception):
    """Raised internally when NIM returns HTTP 429."""
