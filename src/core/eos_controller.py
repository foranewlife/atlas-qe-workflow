"""
EOS Controller: orchestrates config validation, task creation, and execution.

Design goals:
- Validate inputs early (templates, PPs, structures) so downstream assumes existence
- Generate per-volume tasks and inputs
- Delegate execution to TaskProcessor
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

from .configuration import (
    ConfigurationLoader,
    ParameterSpaceEnumerator,
    WorkflowConfiguration,
    ResourceConfigurationLoader,
)
import yaml
from .task_creation import TaskCreator, TaskDef
from .process_orchestrator import ProcessOrchestrator
from .task_processing import RunResult

logger = logging.getLogger(__name__)


class EosController:
    def __init__(self, workflow_config_file: Path, resources_file: Path):
        self.workflow_cfg_path = Path(workflow_config_file)
        self.resources_file = Path(resources_file)
        self.config_dir = self.workflow_cfg_path.parent

        loader = ConfigurationLoader(self.workflow_cfg_path)
        self.config: WorkflowConfiguration = loader.load_configuration()

        # Load distributed/resource settings for scheduling and timeouts
        self.simple_resources = None
        self.distributed_config = None
        # Try simplified config first
        try:
            data = yaml.safe_load(Path(self.resources_file).read_text())
            if isinstance(data, dict) and (('resources' in data) or ('software_paths' in data) or ('local_resources' in data) or ('remote_resources' in data)):
                self.simple_resources = data
            else:
                # Fall back to legacy loader
                r_loader = ResourceConfigurationLoader(self.resources_file)
                _, self.distributed_config = r_loader.load_resource_configuration()
        except Exception:
            # Fall back to legacy loader on any error
            try:
                r_loader = ResourceConfigurationLoader(self.resources_file)
                _, self.distributed_config = r_loader.load_resource_configuration()
            except Exception:
                self.distributed_config = None

        # Resolve results directory
        results_base = Path(self.config.data_paths.get("base_directory", "results"))
        self.results_base = results_base

        # Prepare helpers
        self.creator = TaskCreator(self.config, self.config_dir, self.results_base)

    # ---------- Validation ----------
    def validate_inputs(self) -> None:
        """Validate that templates, structure files, and PPs referenced in config exist."""
        missing: List[str] = []

        # Templates
        for combo in self.config.parameter_combinations:
            t = Path(combo.template_file)
            if not t.is_absolute():
                t = self.config_dir / t
            if not t.exists():
                missing.append(f"template: {t}")

            # Pseudopotentials referenced by combo are checked per structure elements
            pp_dir = self.config.data_paths.get("pseudopotentials_directory", "")
            for struct_name in combo.applies_to_structures:
                struct = next(s for s in self.config.structures if s.name == struct_name)
                pp_set = self.config.pseudopotential_sets[combo.pseudopotential_set]
                for elem in struct.elements:
                    pp_path = self.config_dir / pp_dir / pp_set[elem]
                    if not pp_path.exists():
                        missing.append(f"pseudopotential: {pp_path}")

        # Structure files, if specified
        for s in self.config.structures:
            if s.file:
                p = Path(s.file)
                if not p.is_absolute():
                    p = self.config_dir / p
                if not p.exists():
                    missing.append(f"structure: {p}")

        if missing:
            raise FileNotFoundError("Missing required files:\n" + "\n".join(sorted(set(missing))))

        logger.info("Input validation passed (templates, PPs, structures)")

    # ---------- Task generation ----------
    def generate_tasks(self) -> List[TaskDef]:
        enum = ParameterSpaceEnumerator(self.config)
        pairs: List[Tuple] = enum.discover_parameter_space()

        tasks: List[TaskDef] = []
        for structure, combination in pairs:
            volumes = self.config.generate_volume_series(structure)
            for idx, v in enumerate(volumes):
                task = self.creator.create_task(structure, combination, v, idx)
                tasks.append(task)

        logger.info(f"Prepared {len(tasks)} tasks under {self.results_base}")
        return tasks

    # ---------- Execution ----------
    def execute(self, tasks: List[TaskDef]) -> List[RunResult]:
        """Execute tasks using the single state-based scheduler (ResourcePool + Orchestrator)."""
        orchestrator = ProcessOrchestrator(self.resources_file, self.results_base)
        orchestrator.load_tasks(tasks)
        orchestrator.run()
        results: List[RunResult] = []
        for t in tasks:
            job_out = t.work_dir / "job.out"
            rc = 0 if job_out.exists() else 1
            results.append(RunResult(task_id=t.task_id, returncode=rc, stdout_path=job_out, stderr_path=None))
        return results
