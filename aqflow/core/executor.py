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
from typing import Dict, List, Optional, Tuple, Iterable, Any
import logging

import yaml
from .models import BoardModel, BoardMeta, TaskModel
from aqflow.software.parsers import parse_energy as _sw_parse_energy
import platform
from threading import Event
import sys
import subprocess
try:
    import fcntl  # POSIX file locking
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False

# write log
logger = logging.getLogger(__name__)    

def _is_writable_dir(d: Path) -> bool:
    try:
        d.mkdir(parents=True, exist_ok=True)
        test = d / ".writable"
        test.write_text("ok")
        test.unlink(missing_ok=True)
        return True
    except Exception:
        return False

def _resolve_global_home(path=None) -> Path:
    # Priority: install root -> /tmp
    env = os.environ.get("AQFLOW_BOARD_HOME")
    if env:
        p = Path(env)
        if _is_writable_dir(p):
            return p
        
    candidates = [
        # Path.home() / "aqflow_data" ,
        Path(__file__).resolve().parents[2] / "aqflow_data" ,
        # Path(__file__).resolve().parents[1] / "aqflow_data" ,
        Path("/tmp") / "aqflow_data" ,
    ]

    for c in candidates:
        if path:
            c_path = c / path
        else:
            c_path = c
        if _is_writable_dir(c_path):
            return c_path
    # Fallback: current working dir under aqflow/
    raise RuntimeError("No writable directory found for global boards; please set AQFLOW_BOARD_HOME environment variable.")

BOARD_PATH = Path("aqflow_data/board.json")
# Global boards home for centralized board aggregation
GLOBAL_BOARDS = _resolve_global_home("boards")
GLOBAL_RESOURCES = _resolve_global_home()

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
        gh = GLOBAL_BOARDS
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
    # Validate and normalize board
    try:
        model = BoardModel.model_validate(raw)
        raw = model.model_dump()
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
        # Validate single task via Pydantic
        model = TaskModel.model_validate({
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
        t = model.model_dump()
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
                logger.info(f"Executing remote prep: {c}")
                self._run_shell(c, quiet=True).wait()
            logger.info(f"Executing remotely on {plan.host}: {plan.run_cmd}")
            pop = self._run_shell(plan.run_cmd, quiet=True)
            return pop, plan.remote_dir
        else:
            logger.info(f"Executing locally in {plan.workdir}: {plan.run_cmd}")
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

    def _run_shell(self, cmd: str, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None, *, quiet: bool = False) -> subprocess.Popen:
        """Spawn a shell command.

        If quiet=True, redirect stdout/stderr to DEVNULL to avoid breaking TUI (rich) rendering.
        """
        if quiet:
            return subprocess.Popen(
                cmd,
                shell=True,
                cwd=str(cwd) if cwd else None,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
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
                    self._run_shell(cmd, quiet=True).wait()
            # Prefer energy-parse success over return code
            tdict = tasks[tid]
            sw = tdict.get("type")
            wd = Path(tdict.get("workdir") or ".")
            energy_val = None
            try:
                from .energy import read_energy
                energy_val = read_energy(sw, wd)
            except Exception:
                energy_val = None
            rc_used = 0 if (energy_val is not None) else int(rc)
            # Update status
            self._update_status(tdict, rc_used)
            logger.info(f"Task {tid} finished with exit code {rc} (used={rc_used})")
            # Write per-workdir cache on success
            if rc_used == 0:
                sw_conf = (run.resource.get("software") or {}).get(sw) or {}
                from .cache import write_success_cache
                write_success_cache(
                    software=sw,
                    bin_path=str(sw_conf.get("path")),
                    run_cmd=str(tdict.get("cmd") or ""),
                    workdir=wd,
                    resource=run.resource,
                    energy_eV=energy_val,
                )

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
                    # Cache probe before starting process

                    workdir = Path(t.get("workdir"))
                    sw = t.get("type")
                    sw_conf = (chosen.get("software") or {}).get(sw) or {}
                    cmd_preview, _cores_preview, _env_preview = _build_command(sw, sw_conf, workdir)
                    
                    from .cache import probe_cache
                    pr = probe_cache(
                        software=sw,
                        bin_path=str(sw_conf.get("path")),
                        run_cmd=cmd_preview,
                        workdir=workdir,
                        resource=chosen,
                    )

                    if pr and pr.hit:
                        logger.info(f"Cache hit for {tid} (key={pr.key})")
                        now = time.time()
                        t["status"] = "succeeded"
                        t["resource"] = chosen.get("name")
                        t["cmd"] = cmd_preview
                        t["start_time"] = now
                        t["end_time"] = now
                        t["exit_code"] = 0
                        t["cached"] = True
                        logger.info("cache hit for %s (key=%s)", tid, pr.key)
                        # Do not start a process; leave slot available for next task
                        continue

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


# -----------------------------
# Board (TUI) helper utilities
# Exposed by core so scripts remain thin wrappers.
# -----------------------------

def cleanup_invalid_symlinks(boards_dir: Path) -> int:
    """Remove dangling symlinks under the global boards directory.

    Safe to call repeatedly. Non-symlink files are left untouched.
    """
    removed = 0
    if not boards_dir.exists():
        return 0
    for p in boards_dir.glob("*.json"):
        if p.is_symlink() and not p.resolve().exists():
            try:
                p.unlink()
                removed += 1
            except Exception:
                pass
    latest = boards_dir / "latest.json"
    if latest.is_symlink() and not latest.resolve().exists():
        try:
            latest.unlink()
            removed += 1
        except Exception:
            pass
    return removed


def collect_global_items(boards_dir: Path) -> list[tuple[str, dict]]:
    """Collect (root, board_dict) tuples from all board json files under boards_dir."""
    paths = sorted(
        [p for p in boards_dir.glob("*.json") if p.name != "latest.json"],
        key=lambda p: p.name,
    )
    items: list[tuple[str, dict]] = []
    for p in paths:
        data = load_board(p)
        if not data or not isinstance(data.get("tasks"), dict):
            continue
        items.append((data.get("meta", {}).get("root", str(p)), data))
    return items


def _fmt_elapsed(sec: int) -> str:
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _ensure_rich() -> None:
    """Ensure 'rich' is available; if missing, install via 'uv pip install rich'."""
    try:
        import rich  # noqa: F401
        return
    except Exception:
        # Best-effort install using uv; ignore failure so we raise import error below
        try:
            subprocess.run(["uv", "pip", "install", "rich"], check=False)
        except Exception:
            pass
    # Re-import or fail with clear error
    try:
        import rich  # noqa: F401
    except Exception as e:
        raise RuntimeError("'rich' is required for aqflow board. Please install with: uv pip install rich") from e
    


def _build_rich_tables(
    items: Iterable[tuple[str, dict]],
    *,
    show_all: bool,
    group_by: Optional[str],
    filters: Optional[List[str]],
) -> List[Any]:
    """Return a list of rich renderables representing the board view."""
    from rich.table import Table
    from rich.text import Text
    filters = filters or []
    renderables: List[Any] = []
    now = time.time()

    def match_filters(t: dict) -> bool:
        if not filters:
            return True
        for f in filters:
            if ":" not in f:
                if f.lower() not in (t.get("name", "").lower()):
                    return False
                continue
            k, v = f.split(":", 1)
            v = v.strip().lower()
            if k == "status" and (t.get("status", "").lower() != v):
                return False
            if k == "type" and (t.get("type", "").lower() != v):
                return False
            if k == "resource" and (str(t.get("resource", "")).lower() != v):
                return False
            if k == "name" and (v not in (t.get("name", "").lower())):
                return False
        return True

    for root, data in items:
        tasks = list((data.get("tasks") or {}).values())
        if not show_all:
            tasks = [t for t in tasks if t.get("status") == "running"]
        tasks = [t for t in tasks if match_filters(t)]
        if not tasks:
            continue
        # Grouping
        groups: Dict[Optional[str], List[dict]] = {None: tasks}
        if group_by in ("resource", "type"):
            groups = {}
            for t in tasks:
                key = t.get(group_by) or "-"
                groups.setdefault(key, []).append(t)

        # Root header
        renderables.append(Text(f"root={root}", style="bold cyan"))

        for gk, gtasks in groups.items():
            if gk is not None:
                renderables.append(Text(f"[{group_by}={gk}]", style="magenta"))
            table = Table(expand=True, show_lines=False)
            table.add_column("task_id", style="bold")
            table.add_column("name")
            table.add_column("type")
            table.add_column("status")
            table.add_column("elapsed", justify="right")
            table.add_column("quick")
            # Limit rows to avoid flooding terminal
            limit = _BOARD_ROW_LIMIT
            shown = gtasks if limit is None else gtasks[:limit]
            for t in shown:
                tid = t.get("id", "-")
                name = (t.get("name") or "-")
                typ = (t.get("type") or "-")
                st = (t.get("status") or "-")
                start = t.get("start_time") or 0
                end = t.get("end_time") or None
                sec = int((end or now) - start) if start else 0
                elapsed = _fmt_elapsed(sec)
                quick = f"cd {t.get('workdir', '.')}; tail -n 200 job.out"
                table.add_row(tid, name, typ, st, elapsed, quick)
            renderables.append(table)
            if limit is not None and len(gtasks) > limit:
                renderables.append(Text(f"… and {len(gtasks) - limit} more (use --limit to adjust)", style="dim italic"))
    return renderables or [Text("(no matching tasks)")]


def _collect_status_counts(items: Iterable[tuple[str, dict]]) -> Dict[str, int]:
    counts = {"total": 0, "queued": 0, "created": 0, "running": 0, "succeeded": 0, "failed": 0, "timeout": 0}
    for _, data in items:
        for t in (data.get("tasks") or {}).values():
            s = (t.get("status") or "").lower()
            counts["total"] += 1
            if s in counts:
                counts[s] += 1
    return counts


def _build_progress(items: Iterable[tuple[str, dict]]) -> Any:
    from rich.progress import Progress, BarColumn, TextColumn, TaskProgressColumn
    from rich.text import Text
    from rich.console import Group
    counts = _collect_status_counts(items)
    total = max(1, counts["total"])
    completed = counts["succeeded"] + counts["failed"] + counts["timeout"]
    progress = Progress(
        TextColumn("[bold]Progress[/]"),
        BarColumn(bar_width=None),
        TaskProgressColumn(),
    )
    task_id = progress.add_task("board", total=total)
    progress.update(task_id, completed=completed)
    # Compose counts line
    info = Text(
        f"running {counts['running']} | queued {counts['queued']} | succeeded {counts['succeeded']} | failed {counts['failed']} | timeout {counts['timeout']}",
        style="dim",
    )
    return Group(progress, info)


_BOARD_ROW_LIMIT: Optional[int] = 50


def set_board_row_limit(limit: Optional[int]) -> None:
    """Set global max rows shown per group/root in board tables. None disables the cap."""
    global _BOARD_ROW_LIMIT
    _BOARD_ROW_LIMIT = limit


def watch_global_board(*, show_all: bool, group_by: Optional[str], filters: Optional[List[str]], interval: float = 1.0) -> int:
    """Continuously render the aggregated board view with periodic cleanup (Rich-based)."""
    _ensure_rich()
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
    boards_dir = GLOBAL_BOARDS
    if not boards_dir.exists():
        print("No boards found. Run a workflow to generate aqflow/board.json.")
        return 0
    removed = cleanup_invalid_symlinks(boards_dir)
    last_cleanup = time.time()
    console = Console()
    refresh_per_second = max(1, int(round(1.0 / max(0.01, interval))))
    placeholder = Text("aqflow board initializing...", style="dim")
    try:
        with Live(placeholder, console=console, refresh_per_second=refresh_per_second, screen=True) as live:
            try:
                while True:
                    if time.time() - last_cleanup > 10:
                        removed += cleanup_invalid_symlinks(boards_dir)
                        last_cleanup = time.time()
                    items = collect_global_items(boards_dir)
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    header = Text(f"aqflow board - {ts}  (cleaned {removed} dangling links)", style="bold")
                    progress = _build_progress(items)
                    renderables = _build_rich_tables(items, show_all=show_all, group_by=group_by, filters=filters or [])
                    group = Group(header, progress, *renderables)
                    live.update(group, refresh=True)
                    time.sleep(interval)
            except KeyboardInterrupt:
                return 130
    finally:
        # Best-effort ensure cursor visible even if Live did not exit cleanly
        try:
            sys.stdout.write("\x1b[?25h"); sys.stdout.flush()
        except Exception:
            pass


def render_global_board_once(*, show_all: bool, group_by: Optional[str], filters: Optional[List[str]]) -> str:
    """One-shot rendering of the aggregated board (Rich-based, returns text export)."""
    _ensure_rich()
    from rich.console import Console, Group
    from rich.text import Text
    boards_dir = GLOBAL_BOARDS
    if not boards_dir.exists():
        return "No boards found. Run a workflow to generate aqflow/board.json."
    cleanup_invalid_symlinks(boards_dir)
    items = collect_global_items(boards_dir)
    console = Console(record=True)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    header = Text(f"aqflow board - {ts}", style="bold")
    progress = _build_progress(items)
    renderables = _build_rich_tables(items, show_all=show_all, group_by=group_by, filters=filters or [])
    console.print(Group(header, progress, *renderables))
    return console.export_text(clear=False)


def watch_single_board(board_path: Path, stop: Optional[Event] = None, interval: float = 0.5) -> None:
    """Watch a single board.json and render until completion or stop set (Rich-based)."""
    _ensure_rich()
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
    def done(statuses: List[str]) -> bool:
        return statuses and not any(s in ("queued", "created", "running") for s in statuses)

    console = Console()
    refresh_per_second = max(1, int(round(1.0 / max(0.01, interval))))
    placeholder = Text("aqflow run initializing...", style="dim")
    try:
        with Live(placeholder, console=console, refresh_per_second=refresh_per_second, screen=True) as live:
            try:
                while True:
                    if stop and stop.is_set():
                        break
                    data = load_board(board_path)
                    items: list[tuple[str, dict]] = []
                    if data:
                        items.append(((data.get("meta") or {}).get("root", str(Path.cwd())), data))
                    ts = time.strftime("%Y-%m-%d %H:%M:%S")
                    header = Text(f"aqflow run - {ts}  (watching {board_path})", style="bold")
                    progress = _build_progress(items)
                    renderables = _build_rich_tables(items, show_all=False, group_by=None, filters=[])
                    live.update(Group(header, progress, *renderables), refresh=True)
                    if data:
                        statuses = [t.get("status") for t in (data.get("tasks") or {}).values()]
                        if done(statuses):
                            break
                    time.sleep(interval)
            except KeyboardInterrupt:
                pass
    finally:
        try:
            sys.stdout.write("\x1b[?25h"); sys.stdout.flush()
        except Exception:
            pass
    if stop:
        stop.set()
