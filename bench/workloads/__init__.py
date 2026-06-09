"""
bench/workloads/__init__.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Public surface of the workloads sub-package.

Import all concrete workload classes and expose ``WORKLOAD_MAP`` so that
``bench/run_bench.py`` can instantiate a workload by name without importing
individual modules.
"""

from bench.workloads.base_workload import WorkloadBase, WorkloadSample
from bench.workloads.multi_turn import MultiTurnWorkload
from bench.workloads.single_turn import SingleTurnWorkload
from bench.workloads.tool_calling import ToolCallingWorkload

WORKLOAD_MAP: dict[str, type[WorkloadBase]] = {
    "single_turn": SingleTurnWorkload,
    "multi_turn": MultiTurnWorkload,
    "tool_calling": ToolCallingWorkload,
}

__all__ = [
    "WorkloadBase",
    "WorkloadSample",
    "SingleTurnWorkload",
    "MultiTurnWorkload",
    "ToolCallingWorkload",
    "WORKLOAD_MAP",
]
