"""
tests/test_nim_backend.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
Unit tests for :class:`bench.backends.nim_backend.NIMBackend`.

All HTTP interactions are mocked with ``unittest.mock.patch`` so no live
network calls are made.  The tests validate:

* ``health_check`` returning ``True`` on a 200 /v1/models response.
* ``health_check`` returning ``False`` on a connection error.
* ``infer`` accurately measuring TTFT (> 0, < total_latency_ms).
* ``infer`` retrying exactly once on a 429 before succeeding.
* ``infer`` raising ``RuntimeError`` on a 5xx response.
* ``infer`` populating all :class:`~bench.metrics.RequestResult` fields.

Run with::

    pytest tests/test_nim_backend.py -v
"""

from __future__ import annotations

import json
import time
from types import TracebackType
from typing import Any, Iterator
from unittest.mock import MagicMock, patch

import pytest

from bench.backends.nim_backend import NIMBackend
from bench.metrics import RequestResult


# ---------------------------------------------------------------------------
# SSE stream helpers
# ---------------------------------------------------------------------------

def _make_sse_chunk(content: str, finish_reason: str | None = None) -> str:
    """Return a single SSE data line for a chat-completions streaming chunk."""
    delta: dict[str, Any] = {"role": "assistant", "content": content}
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "meta/llama-3.1-8b-instruct",
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n"


def _make_usage_chunk(prompt_tokens: int, completion_tokens: int) -> str:
    """Return a usage SSE chunk (stream_options include_usage=True)."""
    payload = {
        "id": "chatcmpl-test",
        "object": "chat.completion.chunk",
        "model": "meta/llama-3.1-8b-instruct",
        "choices": [],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
    return f"data: {json.dumps(payload)}\n"


def _sse_lines(*content_pieces: str, prompt_tokens: int = 5, add_usage: bool = True) -> list[str]:
    """Build a complete SSE response line list for the given content pieces."""
    lines: list[str] = []
    for i, piece in enumerate(content_pieces):
        finish = "stop" if i == len(content_pieces) - 1 else None
        lines.append(_make_sse_chunk(piece, finish_reason=finish))
    if add_usage:
        lines.append(_make_usage_chunk(prompt_tokens, len(content_pieces)))
    lines.append("data: [DONE]\n")
    return lines


# ---------------------------------------------------------------------------
# Mock response context manager
# ---------------------------------------------------------------------------

class _MockStreamResponse:
    """Minimal mock for an ``httpx.Response`` used as a context manager.

    Simulates the streaming interface: ``iter_lines()`` returns the supplied
    lines, and ``is_success`` / ``status_code`` are configurable.
    """

    def __init__(
        self,
        lines: list[str],
        status_code: int = 200,
        delay_between_lines: float = 0.0,
    ) -> None:
        self.status_code = status_code
        self._lines = lines
        self._delay = delay_between_lines

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        if not self.is_success:
            import httpx
            raise httpx.HTTPStatusError(
                message=f"HTTP {self.status_code}",
                request=MagicMock(),
                response=MagicMock(status_code=self.status_code),
            )

    def iter_lines(self) -> Iterator[str]:
        for line in self._lines:
            if self._delay > 0:
                time.sleep(self._delay)
            yield line

    def __enter__(self) -> "_MockStreamResponse":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        pass


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def backend() -> NIMBackend:
    """Return a :class:`NIMBackend` pointing at a dummy URL with no real key."""
    return NIMBackend(
        base_url="http://localhost:9999/v1",
        model="meta/llama-3.1-8b-instruct",
        api_key="test-key",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestHealthCheck:
    def test_health_check_success(self, backend: NIMBackend) -> None:
        """health_check returns True when GET /models returns 200."""
        mock_response = MagicMock()
        mock_response.is_success = True

        with patch.object(backend._client, "get", return_value=mock_response) as mock_get:
            result = backend.health_check()

        assert result is True
        mock_get.assert_called_once_with("/models")

    def test_health_check_failure_connection_error(self, backend: NIMBackend) -> None:
        """health_check returns False on a connection error (no exception propagated)."""
        import httpx

        with patch.object(
            backend._client, "get", side_effect=httpx.ConnectError("connection refused")
        ):
            result = backend.health_check()

        assert result is False

    def test_health_check_failure_non_2xx(self, backend: NIMBackend) -> None:
        """health_check returns False when the endpoint responds with 503."""
        mock_response = MagicMock()
        mock_response.is_success = False

        with patch.object(backend._client, "get", return_value=mock_response):
            result = backend.health_check()

        assert result is False


class TestInfer:
    def test_infer_measures_ttft(self, backend: NIMBackend) -> None:
        """ttft_ms is positive and strictly less than total_latency_ms."""
        # Use a small delay so that the second line arrives measurably later
        lines = _sse_lines("Hello", " world", "!", prompt_tokens=4)
        mock_stream = _MockStreamResponse(lines, delay_between_lines=0.005)

        with patch.object(backend._client, "stream", return_value=mock_stream):
            result = backend.infer(request_id=1, prompt="Hi")

        assert result.ttft_ms > 0, "TTFT must be positive"
        assert result.ttft_ms <= result.total_latency_ms, (
            "TTFT must not exceed total latency"
        )

    def test_infer_retries_on_429(self, backend: NIMBackend) -> None:
        """infer retries up to once after a 429 and succeeds on the second call."""
        lines = _sse_lines("Retry success", prompt_tokens=3)
        success_stream = _MockStreamResponse(lines, status_code=200)
        rate_limit_stream = _MockStreamResponse([], status_code=429)

        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return rate_limit_stream
            return success_stream

        with patch.object(backend._client, "stream", side_effect=side_effect):
            with patch("time.sleep"):  # avoid real backoff delay in tests
                result = backend.infer(request_id=2, prompt="Test retry")

        assert call_count == 2, "Expected exactly 2 stream calls (1 failure + 1 success)"
        assert isinstance(result, RequestResult)

    def test_infer_raises_on_5xx(self, backend: NIMBackend) -> None:
        """infer raises RuntimeError immediately on a 5xx response."""
        error_stream = _MockStreamResponse([], status_code=500)

        with patch.object(backend._client, "stream", return_value=error_stream):
            with pytest.raises(RuntimeError, match="server error HTTP 500"):
                backend.infer(request_id=3, prompt="Should fail")

    def test_infer_result_fields(self, backend: NIMBackend) -> None:
        """All RequestResult fields are populated with sane values."""
        prompt = "Explain tensor parallelism."
        prompt_tokens = 5
        completion_pieces = ["Tensor", " parallelism", " splits", " weights", "."]
        lines = _sse_lines(*completion_pieces, prompt_tokens=prompt_tokens, add_usage=True)
        mock_stream = _MockStreamResponse(lines)

        with patch.object(backend._client, "stream", return_value=mock_stream):
            result = backend.infer(request_id=42, prompt=prompt, max_tokens=100, temperature=0.0)

        # request_id is passed through
        assert result.request_id == 42

        # token counts come from the usage chunk
        assert result.prompt_tokens == prompt_tokens
        assert result.completion_tokens == len(completion_pieces)

        # timing values are non-negative floats
        assert result.ttft_ms >= 0.0
        assert result.tpot_ms >= 0.0
        assert result.total_latency_ms >= 0.0

        # throughput is positive
        assert result.tokens_per_sec > 0.0

        # vram is None for NIM (not reported per-request)
        assert result.vram_used_mib is None

    def test_infer_ttft_less_than_total_latency_streaming(self, backend: NIMBackend) -> None:
        """With multiple chunks, TTFT (first chunk) < total_latency (all chunks)."""
        # Add realistic delays to exaggerate the gap
        lines = _sse_lines(
            "First", " second", " third", " fourth", " fifth",
            prompt_tokens=8,
        )
        # 5 ms between each line to ensure measurable gap
        mock_stream = _MockStreamResponse(lines, delay_between_lines=0.005)

        with patch.object(backend._client, "stream", return_value=mock_stream):
            result = backend.infer(request_id=10, prompt="Generate five words")

        assert result.ttft_ms < result.total_latency_ms, (
            f"Expected ttft_ms ({result.ttft_ms:.2f}) < total_latency_ms "
            f"({result.total_latency_ms:.2f})"
        )

    def test_infer_raises_after_max_retries(self, backend: NIMBackend) -> None:
        """infer raises RuntimeError after exhausting all retries on persistent 429."""
        rate_limit_stream = _MockStreamResponse([], status_code=429)

        with patch.object(backend._client, "stream", return_value=rate_limit_stream):
            with patch("time.sleep"):
                with pytest.raises(RuntimeError, match="exceeded retry limit"):
                    backend.infer(request_id=99, prompt="Never succeeds")
