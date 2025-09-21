"""
Base interfaces for software-specific adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict

from aqflow.core.configuration import StructureConfig, ParameterCombination, WorkflowConfiguration
from aqflow.core.template_engine import TemplateProcessor, StructureProcessor


class SoftwareInputGenerator(ABC):
    @abstractmethod
    def create_inputs(
        self,
        work_dir: Path,
        config: WorkflowConfiguration,
        config_dir: Path,
        structure: StructureConfig,
        combination: ParameterCombination,
        volume_scale: float,
        template_proc: TemplateProcessor,
        struct_proc: StructureProcessor,
    ) -> Path:
        """Create inputs in work_dir and return main input file path."""
        raise NotImplementedError


# Execution is handled by aqflow.core.executor; no runner abstraction needed here.
