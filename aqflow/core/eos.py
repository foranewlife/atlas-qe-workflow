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
import time
import json
from .models import EosModel, EosMeta, EosTaskEntry

from .configuration import (
    ConfigurationLoader,
    ParameterSpaceEnumerator,
    WorkflowConfiguration,
)
from dataclasses import dataclass
from .tasks import TaskCreator, TaskDef
from .executor import Executor, BOARD_PATH

logger = logging.getLogger(__name__)


class EosController:
    def __init__(self, workflow_config_file: Path, resources_file: Path):
        self.workflow_cfg_path = Path(workflow_config_file)
        self.resources_file = Path(resources_file)
        self.config_dir = self.workflow_cfg_path.parent

        loader = ConfigurationLoader(self.workflow_cfg_path)
        self.config: WorkflowConfiguration = loader.load_configuration()

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
            t = Path(combo.template)
            if not t.is_absolute():
                t = self.config_dir / t
            if not t.exists():
                missing.append(f"template: {t}")

            # Pseudopotentials referenced by combo are checked per structure elements
            pp_dir = self.config.data_paths.get("pseudopotentials") or self.config.data_paths.get("pseudopotentials_directory", "")
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
        """Append tasks to board.json and run state machine until finish."""
        # Prepare board and append tasks as queued
        ex = Executor(self.resources_file, board_path=BOARD_PATH, run_meta={
            "tool": "eos",
            "args": ["aqflow", "eos", str(self.workflow_cfg_path)],
            "resources_file": str(self.resources_file),
            "root": str(Path.cwd()),
        })
        # Convert TaskDef -> dict entries
        entries = []
        for t in tasks:
            entries.append({
                "id": t.task_id,
                "name": f"{t.software} {t.meta.get('structure','')} {t.meta.get('volume_scale','')}".strip(),
                "type": t.software,
                "workdir": str(t.work_dir),
                "status": "queued",
            })
        ex.add_tasks(entries)
        ex.save()
        # Initialize eos.json using Pydantic model
        eos_tasks = [
            EosTaskEntry(
                id=t.task_id,
                structure=t.meta.get("structure", ""),
                combination=t.meta.get("combination", ""),
                volume_scale=float(t.meta.get("volume_scale", 0.0) or 0.0),
                workdir=(
                    str(t.work_dir.relative_to(Path.cwd()))
                    if str(t.work_dir).startswith(str(Path.cwd()))
                    else str(t.work_dir)
                ),
                status="queued",
            )
            for t in tasks
        ]
        # Snapshot structures and combinations for downstream post/analysis
        structures_info = {}
        for s in self.config.structures:
            vol_series = self.config.generate_volume_series(s)
            struct_file = None
            if s.file:
                p = Path(s.file)
                if not p.is_absolute():
                    p = self.config_dir / p
                struct_file = str(p)
            structures_info[s.name] = {
                "name": s.name,
                "elements": list(s.elements),
                "description": s.description,
                "volume_range": list(s.volume_range),
                "volume_points": int(s.volume_points),
                "volume_series": [float(x) for x in vol_series],
                "file": struct_file,
            }

        combinations_info = {}
        for c in self.config.parameter_combinations:
            tpl = Path(c.template)
            if not tpl.is_absolute():
                tpl = self.config_dir / tpl
            # Resolve pseudopotential files for convenience
            pp_set_name = c.pseudopotential_set
            pp_map = self.config.pseudopotential_sets.get(pp_set_name, {})
            pp_resolved = {}
            for elem, fn in (pp_map or {}).items():
                p = Path(fn)
                if not p.is_absolute():
                    p = self.config_dir / p
                pp_resolved[elem] = str(p)
            combinations_info[c.name] = {
                "name": c.name,
                "software": c.software,
                "template": str(tpl),
                "applies_to_structures": list(c.applies_to_structures),
                "pseudopotential_set": pp_set_name,
                "pseudopotentials": pp_resolved,
                "template_substitutions": dict(c.template_substitutions or {}),
            }

        # Curves index: group tasks by (structure, combination)
        curves_map = {}
        for t in tasks:
            key = f"{t.meta.get('structure','')}|{t.meta.get('combination','')}"
            cur = curves_map.setdefault(
                key,
                {
                    "key": key,
                    "structure": t.meta.get("structure", ""),
                    "combination": t.meta.get("combination", ""),
                    "task_ids": [],
                    "workdirs": [],
                    "volume_scales": [],
                },
            )
            cur["task_ids"].append(t.task_id)
            cur["workdirs"].append(
                str(t.work_dir.relative_to(Path.cwd()))
                if str(t.work_dir).startswith(str(Path.cwd()))
                else str(t.work_dir)
            )
            cur["volume_scales"].append(float(t.meta.get("volume_scale", 0.0) or 0.0))

        eos_model = EosModel(
            meta=EosMeta(
                system=self.config.system,
                description=self.config.description,
                config_path=str(self.workflow_cfg_path.resolve()),
                created_at=time.time(),
                last_update=time.time(),
            ),
            schema_version=1,
            units={"energy": "eV", "volume": "A^3"},
            run={
                "root": str(Path.cwd()),
                "resources_file": str(self.resources_file.resolve()),
                "results_base": str(self.results_base),
            },
            tasks=eos_tasks,
            structures_info=structures_info,
            combinations_info=combinations_info,
            curves_index=list(curves_map.values()),
        )
        self._write_eos(eos_model)

        ex.run()

        # Build results based on job.out presence
        results: List[RunResult] = []
        # Update eos.json with statuses
        for idx, t in enumerate(tasks):
            job_out = t.work_dir / "job.out"
            rc = 0 if job_out.exists() else 1
            results.append(RunResult(task_id=t.task_id, returncode=rc))
            e = eos_model.tasks[idx]
            e.status = "succeeded" if rc == 0 else "failed"
            e.exit_code = rc
            e.job_out = (
                str(job_out.relative_to(Path.cwd()))
                if str(job_out).startswith(str(Path.cwd()))
                else str(job_out)
            )
        eos_model.meta.last_update = time.time()
        self._write_eos(eos_model)
        return results


    def _write_eos(self, model: EosModel) -> None:
        p = Path.cwd() / "aqflow_data" / "eos.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(model.model_dump_json(indent=2))
        tmp.replace(p)


@dataclass
class RunResult:
    task_id: str
    returncode: int
