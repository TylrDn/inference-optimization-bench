"""Tests for benchmark workload generators."""

from __future__ import annotations

import pytest

from bench.workloads import WORKLOAD_MAP
from bench.workloads.multi_turn import MultiTurnWorkload
from bench.workloads.single_turn import SingleTurnWorkload
from bench.workloads.tool_calling import ToolCallingWorkload


@pytest.mark.parametrize(
    "name,cls",
    [
        ("single_turn", SingleTurnWorkload),
        ("multi_turn", MultiTurnWorkload),
        ("tool_calling", ToolCallingWorkload),
    ],
)
def test_workload_map_exports(name: str, cls: type) -> None:
    assert WORKLOAD_MAP[name] is cls


@pytest.mark.parametrize(
    "workload",
    [
        SingleTurnWorkload(),
        MultiTurnWorkload(),
        ToolCallingWorkload(),
    ],
)
def test_generate_returns_n_samples(workload) -> None:
    samples = workload.generate(n=5, seed=42)
    assert len(samples) == 5
    for sample in samples:
        assert sample.prompt
        assert sample.messages
        assert sample.expected_min_tokens > 0
        assert isinstance(sample.metadata, dict)


@pytest.mark.parametrize(
    "workload",
    [
        SingleTurnWorkload(),
        MultiTurnWorkload(),
        ToolCallingWorkload(),
    ],
)
def test_generate_is_deterministic_for_same_seed(workload) -> None:
    first = workload.generate(n=8, seed=99)
    second = workload.generate(n=8, seed=99)
    assert [s.prompt for s in first] == [s.prompt for s in second]
    assert [s.messages for s in first] == [s.messages for s in second]


def test_single_turn_categories_present() -> None:
    workload = SingleTurnWorkload()
    samples = workload.generate(n=20, seed=7)
    categories = {s.metadata.get("category") for s in samples}
    assert len(categories) >= 3
    assert workload.workload_type == "single_turn"
    assert "single-turn" in workload.description().lower()


def test_multi_turn_has_conversation_history() -> None:
    workload = MultiTurnWorkload()
    sample = workload.generate(n=1, seed=1)[0]
    assert len(sample.messages) >= 3
    assert workload.workload_type == "multi_turn"


def test_tool_calling_includes_tool_metadata() -> None:
    workload = ToolCallingWorkload()
    sample = workload.generate(n=1, seed=3)[0]
    assert "expected_tools" in sample.metadata
    assert workload.workload_type == "tool_calling"
