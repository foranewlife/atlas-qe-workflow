"""
Simplified task processing module.

Responsibilities:
- Execute prepared tasks in their working directories
- Provide local and (pluggable) SSH execution strategies
- Expose basic status checks

Assumptions:
- All inputs already exist; paths/binaries validated by controller
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .configuration import ResourceConfigurationLoader, ResourceConfig, ResourceType
from src.software import get_runner
from .task_creation import TaskDef


@dataclass
class RunResult:
    task_id: str
    returncode: int
    stdout_path: Optional[Path]
    stderr_path: Optional[Path]


@dataclass
class RunningProc:
    """Handle for a running process to support non-blocking execution."""
    task_id: str
    popen: subprocess.Popen
    work_dir: Path
    stdout_path: Path
    started_at: float


class TaskProcessor:
    """Run tasks locally or via SSH with minimal policy logic."""

    def __init__(self, resources_file: Path):
        # Try simplified config first
        import yaml
        self.simple_mode = False
        self.local_paths = {}
        self.max_cores_local = None
        try:
            data = yaml.safe_load(Path(resources_file).read_text())
            if isinstance(data, dict) and 'software_paths' in data:
                self.simple_mode = True
                self.local_paths = data.get('software_paths', {}) or {}
                self.software_cfgs = data.get('software_configs', {}) or {}
                self.exec_settings = data.get('execution', {}) or {}
                self.resource_mgmt = data.get('resource_management', {}) or {}
                self.max_cores_local = self.resource_mgmt.get('local_max_cores')
                self.local = None
                self.remote = None
                return
        except Exception:
            pass

        # Fall back to legacy rich configuration
        loader = ResourceConfigurationLoader(resources_file)
        resource_cfgs, distributed_cfg = loader.load_resource_configuration()
        self.local: Optional[ResourceConfig] = next((r for r in resource_cfgs if r.resource_type == ResourceType.LOCAL), None)
        self.remote: Optional[ResourceConfig] = next((r for r in resource_cfgs if r.resource_type == ResourceType.REMOTE), None)
        self.software_cfgs = (distributed_cfg.software_configs if distributed_cfg else {})
        self.exec_settings = (distributed_cfg.execution_settings if distributed_cfg else {})
        self.resource_mgmt = (distributed_cfg.resource_management if distributed_cfg else {})

    def _required_cores(self, software: str) -> int:
        try:
            return int(self.software_cfgs.get(software, {}).get("execution", {}).get("default_cores", 1))
        except Exception:
            return 1

    def _resolve_binary(self, software: str, prefer_remote: bool = False) -> Tuple[str, bool]:
        """Return (binary_path, is_remote) based on policy and required cores."""
        req = self._required_cores(software)
        # Policy hints
        prefer_remote_policy = False
        if software == 'qe' and req > 1:
            prefer_remote_policy = bool(self.resource_mgmt.get('qe_multicore_prefer_remote', False))
        if software == 'atlas':
            # If atlas_prefer_local is False, prefer remote
            if self.resource_mgmt.get('atlas_prefer_local') is False:
                prefer_remote_policy = True

        # If local lacks required cores, prefer remote
        if self.local and self.local.capability.cores < req:
            prefer_remote_policy = True

        # Final preference
        use_remote = prefer_remote or prefer_remote_policy

        if use_remote and self.remote and software in self.remote.software_paths:
            return self.remote.software_paths[software], True
        if self.simple_mode and software in self.local_paths:
            return self.local_paths[software], False
        if self.local and software in self.local.software_paths:
            return self.local.software_paths[software], False
        # Fallback: simulate
        return f"echo simulate_{software}", False

    def run(self, task: TaskDef, prefer_remote: bool = False) -> RunResult:
        bin_path, is_remote = self._resolve_binary(task.software, prefer_remote)

        if is_remote:
            # Build ssh command string directly (no shlex as requested)
            ssh_host = self.remote.hostname if self.remote else ""
            cmd = f"ssh {ssh_host} \"cd {task.work_dir} && {bin_path} < {task.input_file.name} > job.out 2>&1\""
            proc = subprocess.run(cmd, shell=True, cwd=str(task.work_dir))
            return RunResult(task_id=task.task_id, returncode=proc.returncode, stdout_path=task.work_dir / "job.out", stderr_path=None)

        # Local execution
        out_path = task.work_dir / "job.out"

        # Attempt real execution if binary exists; else fail with clear message
        if not Path(bin_path).exists():
            out_path.write_text(f"[ERROR] binary not found: {bin_path}\n")
            return RunResult(task_id=task.task_id, returncode=127, stdout_path=out_path, stderr_path=None)

        # Compose command via software-specific runner, add MPI if needed
        input_name = task.input_file.name
        runner = get_runner(task.software)
        cmd = runner.build_command(shlex.quote(bin_path), shlex.quote(input_name))

        # MPI: if required cores > 1 and mpi_command available
        req = self._required_cores(task.software)
        mpi_cmd = self.software_cfgs.get(task.software, {}).get("execution", {}).get("mpi_command")
        if req > 1 and mpi_cmd:
            # Prepend mpi command
            cmd = f"{mpi_cmd} -np {req} {cmd}"

        # Environment variables from software config
        env = os.environ.copy()
        # Start with software default env, then overlay config env
        for k, v in runner.default_environment().items():
            env[str(k)] = str(v)
        sw_cfg = self.software_cfgs.get(task.software, {}).get("execution", {})
        for k, v in sw_cfg.get("environment_vars", {}).items():
            env[str(k)] = str(v)

        # Timeout: prefer env override, else config
        timeout_env = os.getenv("ATLAS_QE_MAX_SECONDS")
        timeout = None
        try:
            timeout = int(timeout_env) if timeout_env else None
        except Exception:
            timeout = None
        if timeout is None:
            if task.software == "qe":
                timeout = int(self.exec_settings.get("qe_timeout", 600))
            elif task.software == "atlas":
                timeout = int(self.exec_settings.get("atlas_timeout", 600))
            else:
                timeout = int(self.exec_settings.get("default_timeout", 600))

        try:
            proc = subprocess.run(cmd, shell=True, cwd=str(task.work_dir), env=env, timeout=timeout)
            return RunResult(task_id=task.task_id, returncode=proc.returncode, stdout_path=out_path, stderr_path=None)
        except subprocess.TimeoutExpired:
            out_path.write_text((out_path.read_text() if out_path.exists() else "") + f"\n[TIMEOUT] exceeded {timeout}s\n")
            return RunResult(task_id=task.task_id, returncode=124, stdout_path=out_path, stderr_path=None)

    # -------- Non-blocking API --------
    def start(self, task: TaskDef, prefer_remote: bool = False) -> RunningProc:
        """Start a task asynchronously and return a handle for polling."""
        bin_path, is_remote = self._resolve_binary(task.software, prefer_remote)
        out_path = task.work_dir / "job.out"

        if is_remote:
            # Build ssh command string directly (no shlex)
            ssh_host = self.remote.hostname if self.remote else ""
            cmd = f"ssh {ssh_host} \"cd {task.work_dir} && {bin_path} < {task.input_file.name} > job.out 2>&1\""
            pop = subprocess.Popen(cmd, shell=True, cwd=str(task.work_dir))
            return RunningProc(task_id=task.task_id, popen=pop, work_dir=task.work_dir, stdout_path=out_path, started_at=time.time())

        if not Path(bin_path).exists():
            # Immediately return a failed handle if binary missing
            out_path.write_text(f"[ERROR] binary not found: {bin_path}\n")
            pop = subprocess.Popen("false", shell=True)  # exits non-zero
            return RunningProc(task_id=task.task_id, popen=pop, work_dir=task.work_dir, stdout_path=out_path, started_at=time.time())

    def get_local_max_cores(self) -> int:
        if self.simple_mode:
            try:
                return int(self.max_cores_local or 0) or 0
            except Exception:
                return 0
        if self.local:
            return int(getattr(self.local.capability, 'cores', 0) or 0)
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
