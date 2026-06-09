"""Tests for GPU warmup helper."""

from __future__ import annotations

from bench.warmup import warmup_backend


def test_warmup_backend_calls_infer_fn() -> None:
    calls: list[str] = []

    def infer_fn(prompt: str) -> str:
        calls.append(prompt)
        return "ok"

    warmup_backend(infer_fn, prompt="warm", n_warmup=3, verbose=False)
    assert calls == ["warm", "warm", "warm"]


def test_warmup_backend_verbose_output(capsys) -> None:
    warmup_backend(lambda p: "ok", n_warmup=1, verbose=True)
    captured = capsys.readouterr()
    assert "Warmup complete." in captured.out
