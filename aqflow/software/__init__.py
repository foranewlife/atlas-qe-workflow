"""
Software-specific adapters (Atlas, QE).

Provides factories to obtain input generators and runners for each software.
"""

from .base import SoftwareInputGenerator, SoftwareRunner
from .atlas import AtlasInputGenerator, AtlasRunner
from .qe import QEInputGenerator, QERunner


def get_input_generator(software: str) -> SoftwareInputGenerator:
    s = software.lower()
    if s == "atlas":
        return AtlasInputGenerator()
    if s == "qe":
        return QEInputGenerator()
    raise ValueError(f"Unsupported software: {software}")


def get_runner(software: str) -> SoftwareRunner:
    s = software.lower()
    if s == "atlas":
        return AtlasRunner()
    if s == "qe":
        return QERunner()
    raise ValueError(f"Unsupported software: {software}")

