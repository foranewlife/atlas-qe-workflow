"""
Base interfaces for software-specific adapters.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Tuple

from src.core.configuration import StructureConfig, ParameterCombination, WorkflowConfiguration
from src.core.template_engine import TemplateProcessor, StructureProcessor


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


class SoftwareRunner(ABC):
    @abstractmethod
    def build_command(self, binary_path: str, input_filename: str) -> str:
        """Return shell command to execute the calculation."""
        raise NotImplementedError

    def default_environment(self) -> Dict[str, str]:
        """Default environment variables for this software (override as needed)."""
        return {"OMP_NUM_THREADS": os.getenv("OMP_NUM_THREADS", "1")}

