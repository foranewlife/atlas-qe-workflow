"""
Calculator interfaces for ATLAS and Quantum ESPRESSO.

This module provides unified interfaces for different materials calculation
software packages, enabling consistent workflow management across different
computational backends.
"""

from .base_calculator import BaseCalculator, CalculationResult, CalculationStatus
from .atlas_interface import ATLASCalculator
from .qe_interface import QECalculator

__all__ = [
    "BaseCalculator",
    "CalculationResult",
    "CalculationStatus",
    "ATLASCalculator",
    "QECalculator",
]