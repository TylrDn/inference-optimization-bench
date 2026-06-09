"""Unit tests for non-NIM inference backends."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import numpy as np

from bench.backends.llamacpp_backend import LlamaCppBackend
from bench.backends.triton_backend import TritonBackend
from bench.backends.vllm_backend import VLLMBackend


class TestVLLMBackend:
    def test_health_check_success(self) -> None:
        backend = VLLMBackend(base_url="http://localhost:8010")
        mock_response = MagicMock(status_code=200)
        with patch.object(backend.client, "get", return_value=mock_response):
            assert backend.health_check() is True

    def test_health_check_failure(self) -> None:
        backend = VLLMBackend()
        with patch.object(backend.client, "get", side_effect=httpx.ConnectError("down")):
            assert backend.health_check() is False

    def test_infer_parses_response(self) -> None:
        backend = VLLMBackend(model="test-model")
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "hello world"}}],
        }
        mock_response.raise_for_status = MagicMock()
        with patch.object(backend.client, "post", return_value=mock_response):
            with patch("bench.backends.vllm_backend.get_gpu_vram_used_mib", return_value=1024.0):
                result = backend.infer(1, "test prompt", max_tokens=32)
        assert result.request_id == 1
        assert result.completion_tokens == 2
        assert result.total_latency_ms >= 0


class TestLlamaCppBackend:
    def test_health_check_success(self) -> None:
        backend = LlamaCppBackend()
        mock_response = MagicMock(status_code=200)
        with patch.object(backend.client, "get", return_value=mock_response):
            assert backend.health_check() is True

    def test_infer_parses_response(self) -> None:
        backend = LlamaCppBackend()
        mock_response = MagicMock()
        mock_response.json.return_value = {"content": "quantized output", "tokens_evaluated": 5}
        mock_response.raise_for_status = MagicMock()
        with patch.object(backend.client, "post", return_value=mock_response):
            with patch(
                "bench.backends.llamacpp_backend.get_gpu_vram_used_mib", return_value=None
            ):
                result = backend.infer(2, "benchmark prompt")
        assert result.completion_tokens == 5
        assert result.tokens_per_sec > 0


class TestTritonBackend:
    def test_health_check_false_when_triton_unavailable(self) -> None:
        backend = TritonBackend()
        with patch("bench.backends.triton_backend.TRITON_AVAILABLE", False):
            assert backend.health_check() is False

    def test_infer_with_mock_client(self) -> None:
        backend = TritonBackend(url="localhost:8000", model_name="llama3")
        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.as_numpy.return_value = np.array([b"triton response text"])
        mock_client.infer.return_value = mock_result
        backend.client = mock_client

        mock_http = MagicMock()
        mock_infer_input = MagicMock()
        mock_output = MagicMock()
        mock_http.InferInput.return_value = mock_infer_input
        mock_http.InferRequestedOutput.return_value = mock_output

        with patch("bench.backends.triton_backend.TRITON_AVAILABLE", True):
            with patch("bench.backends.triton_backend.httpclient", mock_http, create=True):
                with patch(
                    "bench.backends.triton_backend.get_gpu_vram_used_mib", return_value=2048.0
                ):
                    result = backend.infer(3, "triton prompt")
        assert result.completion_tokens == 3
        mock_client.infer.assert_called_once()
