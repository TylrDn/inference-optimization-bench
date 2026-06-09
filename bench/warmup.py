"""GPU warmup and consistency checks before benchmarking."""
from __future__ import annotations

import time
from typing import Callable


def warmup_backend(
    infer_fn: Callable[[str], str],
    prompt: str = "Say hello.",
    n_warmup: int = 5,
    verbose: bool = True,
) -> None:
    """Run n_warmup inference calls to warm up GPU caches and JIT."""
    for i in range(n_warmup):
        start = time.perf_counter()
        _ = infer_fn(prompt)
        elapsed = (time.perf_counter() - start) * 1000
        if verbose:
            print(f"  Warmup {i+1}/{n_warmup}: {elapsed:.1f}ms")
    if verbose:
        print("Warmup complete.")
