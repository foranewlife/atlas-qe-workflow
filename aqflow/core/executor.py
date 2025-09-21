"""
Executor: single-file board orchestrator.

Inputs:
- resources_yaml: config/resources.yaml (schema: resources/scheduler/timeouts)
- board_json: aqflow/board.json (single source of truth)

Behavior:
- Read board.json, schedule queued tasks respecting max_parallel and resource cores
- Build command per software (atlas, qe) and run locally or via ssh
- Poll running processes; on finish, set status and pull outputs for remote
- Save board.json after each tick
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
import shlex
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from .models import BoardModel, BoardMeta, TaskModel
import platform
try:
    import fcntl  # POSIX file locking
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False


BOARD_PATH = Path("aqflow/board.json")
# Global boards home for central board aggregation
GLOBAL_HOME = Path(os.environ.get("AQFLOW_BOARD_HOME", str(Path.home() / ".aqflow" / "boards")))


def load_board(path: Path) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def save_board(path: Path, board: Dict) -> None:
    """Persist board.json atomically; update global symlinks under a file lock (POSIX)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lock_path = p.with_suffix(".lock")
    lock_fh = None
    try:
        if _HAVE_FCNTL:
            lock_fh = open(lock_path, "w")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(board, indent=2))
        tmp.replace(p)
        # Maintain global symlink for centralized board view
        run_id = (board.get("meta") or {}).get("run_id") or p.stem
        gh = GLOBAL_HOME
        gh.mkdir(parents=True, exist_ok=True)
        link = gh / f"{run_id}.json"
        if link.exists() or link.is_symlink():
            try:
                link.unlink()
            except Exception:
                pass
        try:
            link.symlink_to(p.resolve())
        except Exception:
            if not link.exists():
                link.write_bytes(p.read_bytes())
        latest = gh / "latest.json"
        if latest.exists() or latest.is_symlink():
            try:
                latest.unlink()
            except Exception:
                pass
        try:
            latest.symlink_to(link)
        except Exception:
            if not latest.exists() and link.exists():
                latest.write_bytes(link.read_bytes())
    finally:
        if lock_fh is not None:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_fh.close()
            except Exception:
                pass


def ensure_board(board_path: Path, meta: Dict) -> Dict:
    raw = load_board(board_path)
    if not raw:
        now = time.time()
        model = BoardModel(
            meta=BoardMeta(
                run_id=meta.get("run_id") or f"{meta.get('tool','run')}_{time.strftime('%Y%m%d-%H%M%S')}",
                root=meta.get("root") or str(Path.cwd()),
                resources_file=meta.get("resources_file", "config/resources.yaml"),
                tool=meta.get("tool", "custom"),
                args=meta.get("args", []),
                start_time=now,
                last_update=now,
            ),
            tasks={},
        )
        return model.model_dump()
    # Validate with Pydantic but return dict for compatibility
    try:
        _ = BoardModel.model_validate(raw)
    except Exception:
        # If invalid, rebuild minimal meta wrapper
        now = time.time()
        raw = {
            "meta": {
                "run_id": meta.get("run_id") or f"{meta.get('tool','run')}_{time.strftime('%Y%m%d-%H%M%S')}",
                "root": meta.get("root") or str(Path.cwd()),
                "resources_file": meta.get("resources_file", "config/resources.yaml"),
                "tool": meta.get("tool", "custom"),
                "args": meta.get("args", []),
                "start_time": now,
                "last_update": now,
            },
            "tasks": {},
        }
    return raw


def add_tasks(board: Dict, tasks: List[Dict]) -> None:
    tasks_dict = board.setdefault("tasks", {})
    for t in tasks:
        tid = t["id"]
        t.setdefault("status", "queued")
        t.setdefault("resource", None)
        t.setdefault("cmd", None)
        t.setdefault("start_time", None)
        t.setdefault("end_time", None)
        t.setdefault("exit_code", None)
        # Validate single task via Pydantic (best effort)
        try:
            _ = TaskModel.model_validate({
                "id": t.get("id"),
                "name": t.get("name"),
                "type": t.get("type"),
                "workdir": t.get("workdir"),
                "status": t.get("status"),
                "resource": t.get("resource"),
                "cmd": t.get("cmd"),
                "start_time": t.get("start_time"),
                "end_time": t.get("end_time"),
                "exit_code": t.get("exit_code"),
            })
        except Exception:
            # Fall back to raw insert
            pass
        tasks_dict[tid] = t


class Executor:
    """Minimal executor facade around the board.json single-source state.

    Public API:
      - Executor(resources_yaml: Path, board_path: Path|None = None, run_meta: dict|None = None)
      - add_tasks(tasks: list[dict]) -> None
      - run() -> int
      - save() -> None
      - board_path (property)
      - board (property)
    """

    def __init__(self, resources_yaml: Path, board_path: Optional[Path] = None, run_meta: Optional[Dict] = None):
        self.resources_yaml = Path(resources_yaml)
        self._board_path = Path(board_path) if board_path else Path.cwd() / BOARD_PATH
        self._board = ensure_board(self._board_path, meta=run_meta or {"resources_file": str(self.resources_yaml)})

    @property
    def board_path(self) -> Path:
        return self._board_path

    @property
    def board(self) -> Dict:
        return self._board

    def add_tasks(self, tasks: List[Dict]) -> None:
        add_tasks(self._board, tasks)

    def save(self) -> None:
        self._board["meta"]["last_update"] = time.time()
        save_board(self._board_path, self._board)

    def _load_resources(self) -> tuple[list[Dict], Dict, float]:
        cfg = yaml.safe_load(self.resources_yaml.read_text())
        resources: List[Dict] = cfg.get("resources", [])
        scheduler = cfg.get("scheduler", {})
        timeouts = cfg.get("timeouts", {})
        poll_interval = float(scheduler.get("poll_interval", 2))
        self._max_parallel = int(scheduler.get("max_parallel", 1))
        return resources, timeouts, poll_interval

    def _choose_resource(self, sw: str, resources: List[Dict], running: Dict[str, Running]) -> tuple[Optional[Dict], int]:
        for r in resources:
            sw_conf = (r.get("software") or {}).get(sw)
            if not sw_conf:
                continue
            req = int(sw_conf.get("cores", 1))
            used = sum(run.cores for run in running.values() if run.resource is r)
            cap = int(r.get("cores", 0))
            if req <= max(0, cap - used):
                return r, req
        return None, 0

    @dataclass
    class ExecutionPlan:
        is_remote: bool
        host: Optional[str]
        prep_cmds: list[str]
        run_cmd: str
        workdir: Path
        env: Dict[str, str]
        remote_dir: Optional[str]

    def _build_plan(self, tid: str, t: Dict, chosen: Dict, cmd: str, env: Dict[str, str]) -> "Executor.ExecutionPlan":
        workdir = Path(t.get("workdir"))
        if chosen.get("type", "local") == "remote":
            remote_dir = chosen.get("workdir")
            remote_dir = f"{remote_dir.rstrip('/')}/{tid}" if remote_dir else None
            host = _remote_host(chosen)
            prep_cmds = self._plan_remote_prep(host, remote_dir, workdir)
            run_cmd = self._plan_remote_run(host, remote_dir, workdir, cmd)
            return Executor.ExecutionPlan(True, host, prep_cmds, run_cmd, workdir, env, remote_dir)
        else:
            return Executor.ExecutionPlan(False, None, [], cmd, workdir, env, None)

    def _execute_plan(self, plan: "Executor.ExecutionPlan") -> tuple[subprocess.Popen, Optional[str]]:
        if plan.is_remote:
            for c in plan.prep_cmds:
                self._run_shell(c).wait()
            pop = self._run_shell(plan.run_cmd)
            return pop, plan.remote_dir
        else:
            pop = subprocess.Popen(plan.run_cmd, shell=True, cwd=str(plan.workdir), env=plan.env)
            return pop, None

    def _start_process(self, tid: str, t: Dict, chosen: Dict, req_cores: int) -> tuple[subprocess.Popen, Optional[str], str, float]:
        workdir = Path(t.get("workdir"))
        sw = t.get("type")
        sw_conf = (chosen.get("software") or {}).get(sw) or {}
        cmd, cores, extra_env = _build_command(sw, sw_conf, workdir)
        env = os.environ.copy(); env.update(extra_env)
        started_at = time.time()
        plan = self._build_plan(tid, t, chosen, cmd, env)
        pop, remote_dir_actual = self._execute_plan(plan)
        t["status"] = "running"
        t["resource"] = chosen.get("name")
        t["cmd"] = cmd
        t["start_time"] = started_at
        return pop, remote_dir_actual, cmd, started_at

    def _plan_remote_prep(self, host: str, remote_dir: Optional[str], workdir: Path) -> list[str]:
        cmds: list[str] = []
        if remote_dir:
            cmds.append(f"ssh {shlex.quote(host)} {shlex.quote('mkdir -p ' + shlex.quote(remote_dir))}")
            cmds.append(f"scp -r {shlex.quote(str(workdir))}/* {shlex.quote(host)}:{shlex.quote(remote_dir)}/")
        return cmds

    def _plan_remote_run(self, host: str, remote_dir: Optional[str], workdir: Path, cmd: str) -> str:
        if remote_dir:
            inner = f"cd {shlex.quote(remote_dir)} && {cmd}"
        else:
            inner = f"cd {shlex.quote(str(workdir))} && {cmd}"
        return f"ssh {shlex.quote(host)} {shlex.quote(inner)}"

    def _run_shell(self, cmd: str, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> subprocess.Popen:
        return subprocess.Popen(cmd, shell=True, cwd=str(cwd) if cwd else None, env=env)

    def _poll_running(self, running: Dict[str, Running], tasks: Dict[str, Dict], timeouts: Dict) -> bool:
        progressed = False
        for tid, run in list(running.items()):
            rc = run.popen.poll()
            t = tasks.get(tid) or {}
            soft = None
            if t.get("type") and isinstance(timeouts, dict):
                soft = timeouts.get(t.get("type")) or timeouts.get("default")
            if soft and t.get("start_time") and (rc is None):
                if time.time() - float(t["start_time"]) > float(soft):
                    try:
                        run.popen.kill()
                    except Exception:
                        pass
                    rc = 124
            if rc is None:
                continue
            progressed = True
            if run.resource.get("type") == "remote":
                # Plan pull and execute (side effects isolated)
                for cmd in self._plan_pull_outputs(run, tasks[tid]):
                    self._run_shell(cmd).wait()
            self._update_status(tasks[tid], int(rc))
            running.pop(tid, None)
        return progressed

    def _plan_pull_outputs(self, run: Running, task: Dict) -> list[str]:
        """Return a list of shell commands (strings) to collect remote outputs.

        Pure planning: no side effects here.
        """
        cmds: list[str] = []
        res = run.resource
        host = _remote_host(res)
        if not run.remote_dir:
            return cmds
        transfer = res.get("transfer") or {}
        pull_all = bool(transfer.get("pull_all", False))
        if pull_all:
            cmds.append(
                f"scp -r {shlex.quote(host)}:{shlex.quote(run.remote_dir)}/* {shlex.quote(task['workdir'])}/"
            )
        else:
            cmds.append(
                f"scp {shlex.quote(host)}:{shlex.quote(run.remote_dir)}/job.out {shlex.quote(task['workdir'])}/job.out"
            )
        return cmds

    def _update_status(self, task: Dict, rc: int) -> None:
        task["end_time"] = time.time()
        task["exit_code"] = int(rc)
        task["status"] = "succeeded" if rc == 0 else ("timeout" if rc == 124 else "failed")

    def _should_exit(self) -> bool:
        statuses = [t.get("status") for t in (self._board.get("tasks") or {}).values()]
        return not any(s in ("queued", "created", "running") for s in statuses)

    def run(self) -> int:
        resources, timeouts, poll_interval = self._load_resources()
        running: Dict[str, Running] = {}
        try:
            while True:
                tasks: Dict[str, Dict] = self._board.get("tasks", {})
                running_count = sum(1 for t in tasks.values() if t.get("status") == "running")
                slots = max(0, self._max_parallel - running_count)

                while slots > 0:
                    # Find a queued task
                    queued = next((item for item in tasks.items() if item[1].get("status") in ("queued", "created")), None)
                    if not queued:
                        break
                    tid, t = queued
                    chosen, req = self._choose_resource(t.get("type"), resources, running)
                    if not chosen:
                        break
                    pop, remote_dir, cmd, started_at = self._start_process(tid, t, chosen, req)
                    running[tid] = Running(popen=pop, task_id=tid, resource=chosen, remote_dir=remote_dir, cores=req, started_at=started_at)
                    slots -= 1

                progressed = self._poll_running(running, tasks, timeouts)
                self._board["meta"]["last_update"] = time.time()
                save_board(self._board_path, self._board)
                if self._should_exit():
                    return 0
                if not progressed and slots == 0 and running:
                    time.sleep(poll_interval)
                else:
                    time.sleep(0.2)
        except KeyboardInterrupt:
            self._board["meta"]["last_update"] = time.time()
            save_board(self._board_path, self._board)
            return 130


@dataclass
class Running:
    popen: subprocess.Popen
    task_id: str
    resource: Dict
    remote_dir: Optional[str]
    cores: int
    started_at: float


def _build_command(software: str, sw_conf: Dict, workdir: Path) -> Tuple[str, int, Dict[str, str]]:
    """Return (command_string, cores, env_vars). Applies mpi wrapper when needed.

    Important: atlas does NOT use '< atlas.in'; per user, run as 'atlas.x > job.out 2>&1'.
    qe uses '-in <file>'.
    """
    bin_path = sw_conf.get("path")
    if not bin_path:
        raise RuntimeError(f"software path missing for {software}")
    cores = int(sw_conf.get("cores", 1))
    mpi = sw_conf.get("mpi")
    env = {str(k): str(v) for k, v in (sw_conf.get("env") or {}).items()}

    if software == "qe":
        inp = "qe.in" if (workdir / "qe.in").exists() else "job.in"
        base = f"{bin_path} -in {inp}"
    elif software == "atlas":
        # No stdin redirection
        base = f"{bin_path}"
    else:
        # default: just run binary
        base = f"{bin_path}"

    if mpi and cores > 1:
        base = f"{mpi} -np {cores} {base}"

    cmd = f"{base} > job.out 2>&1"
    return cmd, cores, env


def _remote_host(res: Dict) -> str:
    host = res.get("host") or ""
    user = res.get("user")
    return f"{user}@{host}" if user else host


def run(resources_yaml: Path, board_path: Path = BOARD_PATH) -> int:
    """Compatibility wrapper executing through the Executor class."""
    ex = Executor(resources_yaml, board_path=board_path)
    return ex.run()
