"""
aqflow: Unified namespace for Atlas–QE workflow.

This package re-exports core and software modules while the codebase
transitions from historical 'src.*' package naming.
"""

# Re-export core and software APIs from existing modules
from src.core import (  # type: ignore
    eos_controller,
    process_orchestrator,
    resources_simple,
    task_creation,
    task_processing,
    configuration,
)
from src.software import atlas, qe  # type: ignore

__all__ = [
    "eos_controller",
    "process_orchestrator",
    "resources_simple",
    "task_creation",
    "task_processing",
    "configuration",
    "atlas",
    "qe",
]

