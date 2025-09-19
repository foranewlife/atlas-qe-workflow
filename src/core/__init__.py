"""
Core modules for parameter set management, caching, and workflow orchestration.
"""

# New modular components (working)
from .configuration import (
    WorkflowConfiguration, ConfigurationLoader, ParameterSpaceEnumerator,
    StructureConfig, ParameterCombination
)
from .template_engine import TemplateProcessor, StructureProcessor, InputFileGenerator
from .eos_workflow import EOSWorkflowRunner, CalculationTask, CalculationResult, WorkflowResults

# Legacy components (keep for compatibility but avoid complex imports)
try:
    from .parameter_set import ParameterSet, ParameterSetManager
    from .cache_system import ParameterSetCache
    from .pseudopotential import PseudopotentialManager
    _legacy_available = True
except ImportError:
    _legacy_available = False

__all__ = [
    "WorkflowConfiguration",
    "ConfigurationLoader",
    "ParameterSpaceEnumerator",
    "StructureConfig",
    "ParameterCombination",
    "TemplateProcessor",
    "StructureProcessor",
    "InputFileGenerator",
    "EOSWorkflowRunner",
    "CalculationTask",
    "CalculationResult",
    "WorkflowResults",
]

# Add legacy components if available
if _legacy_available:
    __all__.extend([
        "ParameterSet",
        "ParameterSetManager",
        "ParameterSetCache",
        "PseudopotentialManager",
    ])