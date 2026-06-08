"""
bench/workloads/base_workload.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Abstract base classes for benchmark workloads.

All concrete workload implementations must subclass :class:`WorkloadBase` and
implement the three abstract members.  :class:`WorkloadSample` is the unit of
work passed to a backend's ``infer`` method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkloadSample:
    """A single unit of work for the benchmark runner.

    Attributes:
        prompt: Plain text representation of the user request.  For
            single-turn workloads this is identical to
            ``messages[-1]["content"]``.  Backends that do not support the
            chat-messages format may use this field directly.
        messages: Full conversation in OpenAI chat-completions format
            (list of ``{"role": ..., "content": ...}`` dicts).  This is
            the authoritative representation used by all HTTP backends.
        expected_min_tokens: Lower bound on the number of completion tokens
            a correct response is expected to require.  Used by the reporter
            to flag responses that appear truncated.
        metadata: Arbitrary key-value pairs attached by the workload
            generator (e.g. ``category``, ``turn_count``, ``tool_names``).
            Not interpreted by the runner; preserved in the CSV output.
    """

    prompt: str
    messages: list[dict[str, str]]
    expected_min_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)


class WorkloadBase(ABC):
    """Abstract base class for all benchmark workload generators.

    Subclasses must implement :meth:`generate`, :meth:`description`, and
    the :attr:`workload_type` property.
    """

    @abstractmethod
    def generate(self, n: int, seed: int = 42) -> list[WorkloadSample]:
        """Generate *n* benchmark samples.

        Args:
            n: Number of samples to generate.
            seed: Integer seed for the random number generator so that
                repeated calls with the same arguments produce identical
                output.

        Returns:
            A list of exactly *n* :class:`WorkloadSample` instances.
        """
        ...

    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of this workload.

        Returns:
            A short sentence describing the workload type and its
            characteristics (e.g. prompt length, conversation depth).
        """
        ...

    @property
    @abstractmethod
    def workload_type(self) -> str:
        """Machine-readable workload type identifier.

        Returns:
            A snake_case string matching the key in ``WORKLOAD_MAP``
            (e.g. ``"single_turn"``, ``"multi_turn"``, ``"tool_calling"``).
        """
        ...
