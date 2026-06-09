"""Tests for GPU metrics helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from bench.metrics import get_gpu_vram_used_mib


def test_get_gpu_vram_used_mib_when_nvml_unavailable() -> None:
    with patch("bench.metrics.NVML_AVAILABLE", False):
        assert get_gpu_vram_used_mib() is None


def test_get_gpu_vram_used_mib_success() -> None:
    mock_info = MagicMock()
    mock_info.used = 1024 * 1024 * 512
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlDeviceGetMemoryInfo.return_value = mock_info
    with patch("bench.metrics.NVML_AVAILABLE", True):
        with patch("bench.metrics.pynvml", mock_pynvml, create=True):
            result = get_gpu_vram_used_mib(device_index=0)
    assert result == 512.0


def test_get_gpu_vram_used_mib_handles_errors() -> None:
    mock_pynvml = MagicMock()
    mock_pynvml.nvmlInit.side_effect = RuntimeError("no gpu")
    with patch("bench.metrics.NVML_AVAILABLE", True):
        with patch("bench.metrics.pynvml", mock_pynvml, create=True):
            assert get_gpu_vram_used_mib() is None
