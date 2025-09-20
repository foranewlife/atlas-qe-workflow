"""
Simplified per-resource execution model based on new resources.yaml schema.

Schema (config/resources.yaml):
resources:
  - name: local_workstation
    type: local
    cores: 8
    software:
      atlas: { path: "/abs/path/atlas_mpi.x", cores: 1, env: {OMP_NUM_THREADS: "1"} }
      qe:    { path: "/abs/path/pw.x", cores: 4, mpi: "mpirun", env: {...} }
  - name: server_6101
    type: remote
    host: "6101"
    user: "chenys"
    workdir: "~/atlas_qe_jobs"
    cores: 72
    software: {...}

scheduler: { max_parallel: 2, poll_interval: 2 }
timeouts: { default: 600, atlas: 1800, qe: 7200 }
policy: { prefer: { atlas: local, qe_multicore: remote } }
"""

from __future__ import annotations

import os
import subprocess
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.software import get_runner
from .task_creation import TaskDef


@dataclass
class Running:
    popen: subprocess.Popen
    task: TaskDef
    started_at: float
    cores: int
    remote_dir: Optional[str] = None


class Resource:
    """
    Minimal resource abstraction with clear responsibilities:
    - send_files: push inputs to target (no-op for local)
    - receive_files: pull outputs back to local (no-op for local)
    - check_status: poll process state and handle soft timeout
    """

    def __init__(self, spec: Dict):
        self.logger = logging.getLogger(__name__)
        self.name = spec.get("name")
        self.type = spec.get("type", "local")  # local|remote
        self.cores_total = int(spec.get("cores", 0))
        self.software = spec.get("software", {})  # per software config
        self.host = spec.get("host")
        self.user = spec.get("user")
        self.workdir = spec.get("workdir")
        self.transfer = spec.get("transfer", {}) or {}
        self.pull_all: bool = bool(self.transfer.get("pull_all", False))
        self.running: Dict[str, Running] = {}
        self.max_concurrent_jobs: Optional[int] = spec.get("max_concurrent_jobs")

    @property
    def cores_used(self) -> int:
        return sum(r.cores for r in self.running.values())

    def available_cores(self) -> int:
        return max(self.cores_total - self.cores_used, 0)

    def sw_conf(self, software: str) -> Optional[Dict]:
        return self.software.get(software)

    def can_run(self, task: TaskDef) -> bool:
        conf = self.sw_conf(task.software)
        if not conf:
            return False
        req = int(conf.get("cores", 1))
        if req > self.available_cores():
            return False
        # If local, ensure binary path exists
        if self.type == "local":
            bin_path = conf.get("path")
            if not bin_path or not Path(bin_path).exists():
                return False
        # Remote: assume path exists on remote host (no sync for simplicity)
        return True

    # ---------- File transfer helpers ----------
    def _remote_host(self) -> str:
        host = self.host or ""
        return f"{self.user}@{host}" if self.user else host

    def _remote_task_dir(self, task: TaskDef) -> Optional[str]:
        if self.workdir:
            return f"{self.workdir.rstrip('/')}/{task.task_id}"
        return None

    def send_files(self, task: TaskDef) -> Optional[str]:
        """Send inputs to target. Returns remote task dir if remote; None for local."""
        if self.type == "local":
            return None
        remote_dir = self._remote_task_dir(task)
        remote_host = self._remote_host()
        if remote_dir:
            mk_cmd = f"ssh {remote_host} \"mkdir -p {remote_dir}\""
            rc_mk = subprocess.Popen(mk_cmd, shell=True).wait()
            self.logger.debug(f"[{self.name}] mkdir rc={rc_mk}: {mk_cmd}")
            scp_cmd = f"scp -r {task.work_dir}/* {remote_host}:{remote_dir}/"
            rc_scp = subprocess.Popen(scp_cmd, shell=True).wait()
            self.logger.debug(f"[{self.name}] scp rc={rc_scp}: {scp_cmd}")
        return remote_dir

    def receive_files(self, running: Running) -> None:
        """Pull outputs from target back to local (remote only)."""
        if self.type == "local":
            return
        if not running.remote_dir:
            return
        remote_host = self._remote_host()
        if self.pull_all:
            pull_dir = f"scp -r {remote_host}:{running.remote_dir}/* {running.task.work_dir}/"
            rc = subprocess.Popen(pull_dir, shell=True).wait()
            if rc != 0:
                self.logger.error(f"[{self.name}] pull-all rc={rc}: {pull_dir}")
            else:
                self.logger.info(f"[{self.name}] pull-all ok: {pull_dir}")
            return
        # Selective pull
        expected = running.task.expected_outputs or ["job.out"]
        pulled = 0
        for name in expected:
            pull = f"scp {remote_host}:{running.remote_dir}/{name} {running.task.work_dir}/{name}"
            rc = subprocess.Popen(pull, shell=True).wait()
            if rc == 0:
                self.logger.debug(f"[{self.name}] pull ok: {pull}")
                pulled += 1
            else:
                self.logger.warning(f"[{self.name}] pull rc={rc}: {pull}")
        if pulled == 0:
            pull_dir = f"scp -r {remote_host}:{running.remote_dir}/* {running.task.work_dir}/"
            rc = subprocess.Popen(pull_dir, shell=True).wait()
            if rc != 0:
                self.logger.error(f"[{self.name}] pull dir rc={rc}: {pull_dir}")
            else:
                self.logger.info(f"[{self.name}] pull dir ok: {pull_dir}")

    # ---------- Lifecycle ----------
    def start(self, task: TaskDef, remote_dir: Optional[str] = None) -> bool:
        """Start task on this resource. Orchestrator should call send_files() first if remote."""
        conf = self.sw_conf(task.software)
        if not conf:
            return False
        req = int(conf.get("cores", 1))
        if req > self.available_cores():
            return False

        # Build command with MPI wrapper if needed
        bin_path = conf.get("path")
        runner = get_runner(task.software)
        cmd = runner.build_command(bin_path, task.input_file.name)
        mpi = conf.get("mpi")
        if mpi and req > 1:
            cmd = f"{mpi} -np {req} {cmd}"

        # Environment merge
        env = os.environ.copy()
        for k, v in (conf.get("env") or {}).items():
            env[str(k)] = str(v)

        # Execute
        if self.type == "remote":
            remote_host = self._remote_host()
            if remote_dir:
                run_cmd = f"ssh {remote_host} \"cd {remote_dir} && {cmd}\""
            else:
                run_cmd = f"ssh {remote_host} \"cd {task.work_dir} && {cmd}\""
            self.logger.debug(f"[{self.name}] run: {run_cmd}")
            pop = subprocess.Popen(run_cmd, shell=True, cwd=str(task.work_dir), env=env)
        else:
            pop = subprocess.Popen(cmd, shell=True, cwd=str(task.work_dir), env=env)

        self.logger.info(f"Start {task.task_id} on {self.name} ({self.type}) using {req} cores: {bin_path}")
        self.running[task.task_id] = Running(popen=pop, task=task, started_at=time.time(), cores=req, remote_dir=remote_dir)
        return True

    def check_status(self, r: Running, timeout_soft: Optional[int]) -> Optional[int]:
        """Check a single running process; handle soft timeout; return rc or None if still running."""
        now = time.time()
        if timeout_soft is not None and (now - r.started_at) > timeout_soft:
            try:
                r.popen.kill()
            except Exception:
                pass
            return 124
        rc = r.popen.poll()
        if rc is None:
            return None
        return int(rc)

    def poll(self, timeout_soft: Optional[int] = None) -> List[Tuple[Running, int]]:
        finished: List[Tuple[Running, int]] = []
        for key in list(self.running.keys()):
            r = self.running.get(key)
            if r is None:
                continue
            rc = self.check_status(r, timeout_soft)
            if rc is None:
                continue
            finished.append((r, rc))
            self.running.pop(key, None)
        return finished


class ResourcePool:
    def __init__(self, cfg_path: Path):
        data = yaml.safe_load(Path(cfg_path).read_text())
        self.resources: List[Resource] = [Resource(spec) for spec in data.get("resources", [])]
        self.scheduler = data.get("scheduler", {})
        self.timeouts = data.get("timeouts", {})
        self.policy = data.get("policy", {})

    def choose_resource(self, task: TaskDef) -> Optional[Resource]:
        # Policy: prefer atlas local, qe_multicore remote
        prefer = (self.policy.get("prefer") or {})
        preferred_type: Optional[str] = None
        if task.software == "atlas":
            preferred_type = prefer.get("atlas")
        elif task.software == "qe":
            # Determine if multicore
            is_multi = False
            for r in self.resources:
                conf = r.sw_conf("qe")
                if conf and int(conf.get("cores", 1)) > 1:
                    is_multi = True
                    break
            if is_multi:
                preferred_type = prefer.get("qe_multicore")

        # Iterate by preference first, then fallback
        ordered: List[Resource] = []
        if preferred_type in ("local", "remote"):
            ordered.extend([r for r in self.resources if r.type == preferred_type])
            ordered.extend([r for r in self.resources if r.type != preferred_type])
        else:
            ordered = list(self.resources)

        for r in ordered:
            if r.can_run(task):
                return r
        return None

    def poll_all(self) -> List[Tuple[Resource, Running, int]]:
        results: List[Tuple[Resource, Running, int]] = []
        # Use default soft timeout if provided
        t_soft = self.timeouts.get("default")
        try:
            t_soft = int(t_soft) if t_soft is not None else None
        except Exception:
            t_soft = None
        for r in self.resources:
            for run, rc in r.poll(timeout_soft=t_soft):
                results.append((r, run, rc))
        return results
