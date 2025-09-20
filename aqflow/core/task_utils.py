"""
Minimal task result types used by higher-level controllers.

Execution is now handled exclusively by the state_machine; this module
keeps only lightweight dataclasses shared by callers.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    task_id: str
    returncode: int
    stdout_path: Optional[Path]
    stderr_path: Optional[Path]


    # Placeholder for compatibility; execution handled by state_machine
    pass
        return 0

        # Build command via runner
        runner = get_runner(task.software)
        cmd = runner.build_command(shlex.quote(bin_path), shlex.quote(task.input_file.name))
        env = os.environ.copy()
        for k, v in runner.default_environment().items():
            env[str(k)] = str(v)
        sw_cfg = self.software_cfgs.get(task.software, {}).get("execution", {})
        for k, v in sw_cfg.get("environment_vars", {}).items():
            env[str(k)] = str(v)

        pop = subprocess.Popen(cmd, shell=True, cwd=str(task.work_dir), env=env)
        return RunningProc(task_id=task.task_id, popen=pop, work_dir=task.work_dir, stdout_path=out_path, started_at=time.time())

    def poll(self, handle: RunningProc, timeout_seconds: Optional[int] = None) -> Optional[RunResult]:
        """Poll a running process; return RunResult when finished, else None.

        If timeout_seconds is provided and exceeded, kill and return timeout result.
        """
        # Check external timeout
        if timeout_seconds is not None:
            elapsed = time.time() - handle.started_at
            if elapsed > timeout_seconds:
                try:
                    handle.popen.kill()
                except Exception:
                    pass
                text = (handle.stdout_path.read_text() if handle.stdout_path.exists() else "") + f"\n[TIMEOUT] exceeded {timeout_seconds}s\n"
                handle.stdout_path.write_text(text)
                return RunResult(task_id=handle.task_id, returncode=124, stdout_path=handle.stdout_path, stderr_path=None)

        rc = handle.popen.poll()
        if rc is None:
            return None
        return RunResult(task_id=handle.task_id, returncode=rc, stdout_path=handle.stdout_path, stderr_path=None)

    @staticmethod
    def status(task: TaskDef) -> str:
        """Very simple status check: presence of job.out implies completed."""
        return "completed" if (task.work_dir / "job.out").exists() else "pending"
