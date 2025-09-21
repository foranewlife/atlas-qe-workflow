"""
Software-specific adapters (Atlas, QE).

Provides factories to obtain input generators for each software.
Execution is handled by aqflow.core.executor.
"""

from .base import SoftwareInputGenerator
from .atlas import AtlasInputGenerator
from .qe import QEInputGenerator


def get_input_generator(software: str) -> SoftwareInputGenerator:
    s = software.lower()
    if s == "atlas":
        return AtlasInputGenerator()
    if s == "qe":
        return QEInputGenerator()
    raise ValueError(f"Unsupported software: {software}")


__all__ = [
    "SoftwareInputGenerator",
    "AtlasInputGenerator",
    "QEInputGenerator",
    "get_input_generator",
]
