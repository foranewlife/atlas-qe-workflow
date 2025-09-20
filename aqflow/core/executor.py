"""
Minimal single-file state machine orchestrator.

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
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


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
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(board, indent=2))
    tmp.replace(p)
    # Maintain global symlink for centralized board view
    try:
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
    except Exception:
        pass


def ensure_board(board_path: Path, meta: Dict) -> Dict:
    board = load_board(board_path)
    if not board:
        now = time.time()
        board = {
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
    return board


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

    def run(self) -> int:
        """Main loop. Returns 0 when all tasks finished (succeeded/failed/timeout)."""
        resources_cfg = yaml.safe_load(self.resources_yaml.read_text())
        resources: List[Dict] = resources_cfg.get("resources", [])
        scheduler = resources_cfg.get("scheduler", {})
        timeouts = resources_cfg.get("timeouts", {})
        max_parallel = int(scheduler.get("max_parallel", 1))
        poll_interval = float(scheduler.get("poll_interval", 2))

        running: Dict[str, Running] = {}

        while True:
            tasks: Dict[str, Dict] = self._board.get("tasks", {})
            running_count = sum(1 for t in tasks.values() if t.get("status") == "running")
            slots = max(0, max_parallel - running_count)

            if slots > 0:
                for tid, t in list(tasks.items()):
                    if slots <= 0:
                        break
                    if t.get("status") not in ("queued", "created"):
                        continue
                    sw = t.get("type")
                    chosen: Optional[Dict] = None
                    chosen_cores = 0
                    for r in resources:
                        sw_conf = (r.get("software") or {}).get(sw)
                        if not sw_conf:
                            continue
                        req = int(sw_conf.get("cores", 1))
                        used = sum(run.cores for run in running.values() if run.resource is r)
                        cap = int(r.get("cores", 0))
                        if req <= max(0, cap - used):
                            chosen = r
                            chosen_cores = req
                            break
                    if not chosen:
                        continue

                    workdir = Path(t.get("workdir"))
                    sw_conf = (chosen.get("software") or {}).get(sw) or {}
                    cmd, req_cores, extra_env = _build_command(sw, sw_conf, workdir)
                    env = os.environ.copy(); env.update(extra_env)
                    started_at = time.time()
                    if chosen.get("type", "local") == "remote":
                        remote_dir = chosen.get("workdir")
                        remote_dir = f"{remote_dir.rstrip('/')}/{tid}" if remote_dir else None
                        host = _remote_host(chosen)
                        if remote_dir:
                            subprocess.Popen(f"ssh {host} \"mkdir -p {remote_dir}\"", shell=True).wait()
                            subprocess.Popen(f"scp -r {workdir}/* {host}:{remote_dir}/", shell=True).wait()
                            run_cmd = f"ssh {host} \"cd {remote_dir} && {cmd}\""
                        else:
                            run_cmd = f"ssh {host} \"cd {workdir} && {cmd}\""
                        pop = subprocess.Popen(run_cmd, shell=True, cwd=str(workdir), env=env)
                        remote_dir_actual = remote_dir
                    else:
                        pop = subprocess.Popen(cmd, shell=True, cwd=str(workdir), env=env)
                        remote_dir_actual = None

                    running[tid] = Running(popen=pop, task_id=tid, resource=chosen, remote_dir=remote_dir_actual, cores=chosen_cores or req_cores, started_at=started_at)
                    t["status"] = "running"
                    t["resource"] = chosen.get("name")
                    t["cmd"] = cmd
                    t["start_time"] = started_at
                    slots -= 1

            progressed = False
            for tid, run in list(running.items()):
                rc = run.popen.poll()
                t = tasks.get(tid) or {}
                soft = None
                if t.get("type") and isinstance(timeouts, dict):
                    soft = timeouts.get(t.get("type")) or timeouts.get("default")
                if soft and t.get("start_time"):
                    if time.time() - float(t["start_time"]) > float(soft):
                        try:
                            run.popen.kill()
                        except Exception:
                            pass
                        rc = 124
                if rc is None:
                    continue
                progressed = True

                res = run.resource
                if res.get("type") == "remote":
                    host = _remote_host(res)
                    if run.remote_dir:
                        transfer = res.get("transfer") or {}
                        pull_all = bool(transfer.get("pull_all", False))
                        if pull_all:
                            subprocess.Popen(f"scp -r {host}:{run.remote_dir}/* {tasks[tid]['workdir']}/", shell=True).wait()
                        else:
                            subprocess.Popen(f"scp {host}:{run.remote_dir}/job.out {tasks[tid]['workdir']}/job.out", shell=True).wait()
                t = tasks.get(tid) or {}
                t["end_time"] = time.time()
                t["exit_code"] = int(rc)
                t["status"] = "succeeded" if int(rc) == 0 else ("timeout" if int(rc) == 124 else "failed")
                running.pop(tid, None)

            self._board["meta"]["last_update"] = time.time()
            save_board(self._board_path, self._board)

            statuses = [t.get("status") for t in (self._board.get("tasks") or {}).values()]
            if not any(s in ("queued", "created", "running") for s in statuses):
                return 0

            if not progressed and slots == 0 and running:
                time.sleep(poll_interval)
            else:
                time.sleep(0.2)


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

