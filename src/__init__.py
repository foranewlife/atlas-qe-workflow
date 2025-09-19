"""
ATLAS-QE Materials Workflow

A comprehensive automated workflow system for materials calculations using ATLAS (OFDFT)
and Quantum ESPRESSO (DFT) with intelligent parameter set management and caching.
"""

__version__ = "0.1.0"
__author__ = "TB-OFDFT Project Team"

from .core.parameter_set import ParameterSet, ParameterSetManager
from .core.cache_system import ParameterSetCache
from .core.workflow_engine import WorkflowEngine

__all__ = [
    "ParameterSet",
    "ParameterSetManager",
    "ParameterSetCache",
    "WorkflowEngine",
]