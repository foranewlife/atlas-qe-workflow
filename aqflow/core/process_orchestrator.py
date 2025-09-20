"""
Orchestrates execution using ResourcePool: assign tasks to resources and poll until done.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Dict, List

from .task_creation import TaskDef
from .resources_simple import ResourcePool


@dataclass
class TaskState:
    task: TaskDef
    status: str = "pending"  # pending|running|completed|failed
    returncode: int = 0
    started_at: float | None = None
    finished_at: float | None = None


class ProcessOrchestrator:
    def __init__(self, resources_cfg: Path, results_dir: Path):
        self.pool = ResourcePool(resources_cfg)
        self.results_dir = Path(results_dir)
        self.state_file = self.results_dir / "tasks_simple.json"
        self.states: Dict[str, TaskState] = {}
        self.logger = logging.getLogger("atlas-qe-workflow")

    def load_tasks(self, tasks: List[TaskDef]):
        for t in tasks:
            if t.task_id not in self.states:
                self.states[t.task_id] = TaskState(task=t)

    def _save(self):
        payload = {
            "tasks": [
                {
                    "task": {
                        "task_id": s.task.task_id,
                        "software": s.task.software,
                        "work_dir": str(s.task.work_dir),
                        "input_file": str(s.task.input_file),
                        "expected_outputs": s.task.expected_outputs,
                        "meta": s.task.meta,
                    },
                    "status": s.status,
                    "attempts": 1 if s.started_at else 0,
                    "last_returncode": s.returncode,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                }
                for s in self.states.values()
            ]
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(json.dumps(payload, indent=2))

    def run(self):
        max_parallel = int(self.pool.scheduler.get("max_parallel", 1))
        poll_interval = float(self.pool.scheduler.get("poll_interval", 2))
        stall_cycles = 0
        # max_stall = 100000  # after N cycles with no progress, fail remaining

        while True:
            # Assign up to max_parallel - running tasks
            running_now = sum(1 for s in self.states.values() if s.status == "running")
            slots = max(0, max_parallel - running_now)
            assigned = 0
            if slots > 0:
                for s in self.states.values():
                    if slots <= 0:
                        break
                    if s.status != "pending":
                        continue
                    r = self.pool.choose_resource(s.task)
                    if r:
                        remote_dir = r.send_files(s.task)
                        if r.start(s.task, remote_dir=remote_dir):
                            s.status = "running"
                            s.started_at = time.time()
                            slots -= 1
                            assigned += 1
                            self._save()

            # Poll all resources
            finished = self.pool.poll_all()
            progressed = bool(finished) or (assigned > 0)
            for res, run, rc in finished:
                # On transition running->done, fetch outputs now
                try:
                    res.receive_files(run)
                except Exception as e:
                    self.logger.warning(f"Receive files error for {run.task.task_id}: {e}")
                st = self.states.get(run.task.task_id)
                if st and st.status == "running":
                    st.status = "completed" if rc == 0 else "failed"
                    st.returncode = rc
                    st.finished_at = time.time()
                    if rc != 0:
                        job_out = run.task.work_dir / "job.out"
                        if job_out.exists():
                            try:
                                text = job_out.read_text()
                                tail = text[-2000:] if len(text) > 2000 else text
                                self.logger.warning(f"Task {run.task.task_id} failed rc={rc}; job.out tail:\n{tail}")
                            except Exception as ee:
                                self.logger.warning(f"Task {run.task.task_id} failed rc={rc}; job.out read error: {ee}")
                        else:
                            self.logger.warning(f"Task {run.task.task_id} failed rc={rc}; job.out missing")
                    self._save()

            # Termination condition
            total = len(self.states)
            done = sum(1 for s in self.states.values() if s.status in ("completed", "failed"))
            if done >= total:
                break

            time.sleep(poll_interval)
