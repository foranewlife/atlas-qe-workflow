"""Core module namespace (kept minimal)."""

from .configuration import (
    WorkflowConfiguration,
    ConfigurationLoader,
    ParameterSpaceEnumerator,
    StructureConfig,
    ParameterCombination,
)
from .template_engine import TemplateProcessor, StructureProcessor

__all__ = [
    "WorkflowConfiguration",
    "ConfigurationLoader",
    "ParameterSpaceEnumerator",
    "StructureConfig",
    "ParameterCombination",
    "TemplateProcessor",
    "StructureProcessor",
]
