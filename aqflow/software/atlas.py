"""
ATLAS software adapter: input generation.
"""

from __future__ import annotations

from pathlib import Path

from .base import SoftwareInputGenerator
from aqflow.core.configuration import StructureConfig, ParameterCombination, WorkflowConfiguration
from aqflow.core.template_engine import TemplateProcessor, StructureProcessor


class AtlasInputGenerator(SoftwareInputGenerator):
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
        work_dir.mkdir(parents=True, exist_ok=True)

        # Generate POSCAR
        poscar_content = struct_proc.generate_structure_content(structure=structure, volume_scale=volume_scale)
        (work_dir / "POSCAR").write_text(poscar_content)

        # Process template and write atlas.in
        text = template_proc.process_template(
            template_file=combination.template,
            substitutions=combination.template_substitutions,
            structure=structure,
            combination=combination,
        )
        input_path = work_dir / "atlas.in"
        input_path.write_text(text)
        return input_path


# Execution is handled by aqflow.core.executor
