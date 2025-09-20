"""
Simplified task creation module.

Responsibilities:
- Generate ATLAS/QE input files for a given (structure, combination, volume)
- Maintain per-task working directories under the results base dir
- Return a lightweight task definition that downstream processors can execute

Constraints:
- All file-existence validation should be done up-front by the controller
- This module assumes required files/templates exist
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from .configuration import StructureConfig, ParameterCombination, WorkflowConfiguration
from .template_engine import TemplateProcessor, StructureProcessor
from src.software import get_input_generator


@dataclass
class TaskDef:
    """Lightweight task definition for execution modules.

    Contains only what is needed to run the job in an existing folder.
    """

    task_id: str
    software: str  # "atlas" | "qe"
    work_dir: Path
    input_file: Path
    expected_outputs: List[str]
    meta: Dict[str, str]


class TaskCreator:
    """Create per-volume tasks and inputs for EOS runs.

    This class uses the existing TemplateProcessor and StructureProcessor
    to materialize inputs into the task working directory.
    """

    def __init__(self, config: WorkflowConfiguration, config_dir: Path, results_base: Path):
        self.config = config
        self.config_dir = Path(config_dir)
        self.results_base = Path(results_base)
        self.template_proc = TemplateProcessor(config, self.config_dir)
        self.struct_proc = StructureProcessor(self.config_dir)

    def create_task(
        self,
        structure: StructureConfig,
        combination: ParameterCombination,
        volume_scale: float,
        volume_index: int,
    ) -> TaskDef:
        """Create a single task folder and input files.

        Assumes validation of referenced files has already happened.
        """
        # Layout: results/{structure}/{combination}/{volume:.5f}/
        work_dir = self.results_base / structure.name / combination.name / f"{volume_scale:.5f}"
        work_dir.mkdir(parents=True, exist_ok=True)

        # Delegate software-specific input generation
        generator = get_input_generator(combination.software)
        input_path = generator.create_inputs(
            work_dir=work_dir,
            config=self.config,
            config_dir=self.config_dir,
            structure=structure,
            combination=combination,
            volume_scale=volume_scale,
            template_proc=self.template_proc,
            struct_proc=self.struct_proc,
        )

        # Copy pseudopotentials next to inputs if present in example data
        # We assume controller ensured files exist in configured directories; here we are permissive.
        pp_set = self.config.pseudopotential_sets[combination.pseudopotential_set]
        for _, ppfile in pp_set.items():
            # Prefer example-relative path if exists
            pp_src = self.config_dir / self.config.data_paths.get("pseudopotentials_directory", "") / ppfile
            if pp_src.exists():
                dest = work_dir / ppfile
                if not dest.exists():
                    dest.write_bytes(pp_src.read_bytes())

        task_id = f"{structure.name}_{combination.name}_{volume_index:02d}"
        expected = ["job.out", "eos_data.json"]

        return TaskDef(
            task_id=task_id,
            software=combination.software,
            work_dir=work_dir,
            input_file=input_path,
            expected_outputs=expected,
            meta={
                "structure": structure.name,
                "combination": combination.name,
                "volume_scale": f"{volume_scale:.5f}",
            },
        )
