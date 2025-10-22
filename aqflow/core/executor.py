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
import threading
import queue
import logging

import yaml
from .models import BoardModel, BoardMeta, TaskModel
from threading import Event
import sys
try:
    import fcntl  # POSIX file locking
    _HAVE_FCNTL = True
except Exception:
    _HAVE_FCNTL = False

# write log
logger = logging.getLogger(__name__)    

# Manual handling markers inside a task working directory.
# If any of these files exist, the executor will not launch the binary
# and will instead parse energy and mark the task as succeeded/failed accordingly.
_MANUAL_MARKERS = (".aqmanual", ".aq_manual")

def _has_manual_marker(workdir: Path) -> bool:
    try:
        wd = Path(workdir)
        return any((wd / m).exists() for m in _MANUAL_MARKERS)
    except Exception:
        return False

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
        self._run_started_at: Optional[float] = None
        self._overall_timeout_sec: Optional[float] = None
        # Event queue for worker threads
        self._events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        # Active task workers
        self._workers: Dict[str, "TaskWorker"] = {}
        # Per-host concurrency gates
        self._host_semaphores: Dict[str, threading.Semaphore] = {}
        # Fatal error flag propagated from workers
        self._fatal_error: Optional[str] = None
        # Probe concurrency gates per host
        self._host_probe_semaphores: Dict[str, threading.Semaphore] = {}

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
        # Diagnostics: remember start time only
        self._run_started_at = time.time()
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
            # For combined async start, use raw cmd; _execute_plan will wrap with ssh/bash
            run_cmd = cmd
            return Executor.ExecutionPlan(True, host, prep_cmds, run_cmd, workdir, env, remote_dir)
        else:
            return Executor.ExecutionPlan(False, None, [], cmd, workdir, env, None)

    def _execute_plan(self, plan: "Executor.ExecutionPlan") -> tuple[Optional[subprocess.Popen], Optional[str], Optional[int]]:
        """Execute the plan.

        Local: spawn long-running process and return Popen.
        Remote: defer submission to TaskWorker; return (None, remote_dir, None).
        """
        if plan.is_remote:
            logger.info(f"Remote task planned for host={plan.host} dir={plan.remote_dir}; submission will be handled by worker thread")
            return None, plan.remote_dir, None
        else:
            logger.info(f"Local task planned for dir={plan.workdir}; submission will be handled by worker thread")
            return None, None, None

    def _ssh_args(self, host: Optional[str], inner: str) -> list[str]:
        """Return argv list for ssh to run a remote command (no bash -lc).

        We pass a single command string; tilde (~) expansion and operators
        like && will be handled by the remote user's default shell.
        """
        return ["ssh", *SSH_OPTS, str(host or ""), inner]

    def _rsync_ssh_e(self) -> str:
        """Return the -e argument content for rsync with our SSH_OPTS."""
        # Join options with spaces; these will be passed as a single string to rsync -e "ssh ..."
        return "ssh " + " ".join(SSH_OPTS)

    def _get_host_semaphore(self, host: str, limit: Optional[int] = None) -> threading.Semaphore:
        """Return a semaphore for a host, creating it with the given limit if missing."""
        if host not in self._host_semaphores:
            self._host_semaphores[host] = threading.Semaphore(int(limit or 1))
        return self._host_semaphores[host]

    def _get_host_probe_semaphore(self, host: str, limit: Optional[int] = None) -> threading.Semaphore:
        """Return a semaphore limiting concurrent probe ssh per host (lighter than rsync/start)."""
        if host not in self._host_probe_semaphores:
            self._host_probe_semaphores[host] = threading.Semaphore(int(limit or 2))
        return self._host_probe_semaphores[host]

    def _remote_path_expr(self, remote_dir: str) -> str:
        """Return a shell-safe path expression for remote bash -lc:

        - If remote_dir starts with '~', return ~/<quoted-rest> so tilde expands.
        - Else return "<escaped-absolute>".
        """
        if remote_dir.startswith("~"):
            rest = remote_dir[2:] if remote_dir.startswith("~/") else remote_dir[1:]
            rest = rest.lstrip("/")
            if rest:
                return f"~/{self._dq_escape(rest)}"
            else:
                return "~"
        return f"\"{self._dq_escape(remote_dir)}\""

    @staticmethod
    def _dq_escape(s: str) -> str:
        """Escape a string for inclusion inside a double-quoted shell string."""
        return s.replace("\\", "\\\\").replace("\"", "\\\"")

    def _write_remote_runner(self, workdir: Path, remote_dir: str, raw_cmd: str) -> Path:
        """Create a runner script in local workdir to be executed on remote side.

        The script will run in the target workdir, write .aq_started, then run the user
        command (without bash -lc) and finally write .aq_exit with rc and end timestamp.
        """
        runner = Path(workdir) / ".aq_launch.sh"
        rd_shell = self._remote_path_expr(remote_dir)
        content = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"cd {rd_shell}\n"
            "date +%s > .aq_started\n"
            f"{raw_cmd}\n"
            "rc=$?\n"
            "sync || true\n"
            "sleep 0.1\n"
            "sync || true\n"
            "printf \"%s %s\\n\" \"$rc\" \"$(date +%s)\" > .aq_exit\n"
        )
        try:
            runner.write_text(content)
            os.chmod(runner, 0o755)
        except Exception:
            pass
        return runner

    def _start_process(self, tid: str, t: Dict, chosen: Dict, req_cores: int) -> tuple[Optional[subprocess.Popen], Optional[str], str, float, Optional[int]]:
        workdir = Path(t.get("workdir"))
        sw = t.get("type")
        sw_conf = (chosen.get("software") or {}).get(sw) or {}
        cmd, cores, extra_env = _build_command(sw, sw_conf, workdir)
        env = os.environ.copy(); env.update(extra_env)
        started_at = time.time()
        plan = self._build_plan(tid, t, chosen, cmd, env)
        # Defer actual submission/spawn to worker thread for both remote and local
        pop, remote_dir_actual, remote_pid = self._execute_plan(plan)
        t["status"] = "created"
        t["resource"] = chosen.get("name")
        t["cmd"] = cmd
        t["start_time"] = started_at
        return pop, remote_dir_actual, cmd, started_at, remote_pid

    def _plan_remote_prep(self, host: str, remote_dir: Optional[str], workdir: Path) -> list[str]:
        cmds: list[str] = []
        if remote_dir:
            mkdir_args = self._ssh_args(host, f"mkdir -p \"{self._dq_escape(remote_dir)}\"")
            cmds.append(" ".join(shlex.quote(a) for a in mkdir_args))
            # Use rsync for efficient, incremental upload (no delete on remote) with SSH_OPTS
            src = str(workdir) + "/"
            dst = f"{host}:{remote_dir}/"
            cmds.append(
                f"rsync -az --timeout=10 -e {shlex.quote(self._rsync_ssh_e())} {shlex.quote(src)} {dst}"
            )
        return cmds

    def _plan_remote_run(self, host: str, remote_dir: Optional[str], workdir: Path, cmd: str) -> str:
        if remote_dir:
            inner = f"cd {remote_dir} && {cmd}"
        else:
            inner = f"cd {str(workdir)} && {cmd}"
        return f"ssh {shlex.quote(host)} {shlex.quote(inner)}"

    def _run_shell(
        self,
        cmd: Any,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        *,
        quiet: bool = False,
        wait: Optional[float | bool] = None,
        capture_output: bool = False,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        detached: bool = False,
        fatal_on_error: bool = True,
    ) -> Any:
        """Spawn or run a shell command with optional async/capture behavior.

        Backward-compatible defaults: returns immediately with Popen; no waiting.

        - quiet=True redirects stdout/stderr to DEVNULL.
        - wait=None: return immediately (async); caller may poll or ignore.
        - wait=True: run synchronously and wait until completion.
        - wait=float: schedule a non-blocking timeout; returns immediately and the
          process will be monitored via the executor's background jobs registry.
        - detached=True: start in a new session (POSIX) so signals affect the whole group.
        """
        popen_kwargs: Dict[str, Any] = {
            "cwd": str(cwd) if cwd else None,
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        cmd_repr = " ".join(shlex.quote(x) for x in cmd) if isinstance(cmd, list) else cmd
        logger.debug(f"_run_shell cmd={cmd_repr} cwd={cwd} quiet={quiet} wait={wait} capture={capture_output} timeout={timeout} detached={detached}")
        if quiet:
            popen_kwargs.update({
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
            })
        if detached:
            # Start in a new session so we can signal the whole group if needed (POSIX)
            popen_kwargs["start_new_session"] = True
        if wait is True:
            # Synchronous execution path
            if capture_output:
                try:
                    if isinstance(cmd, list):
                        res = subprocess.run(
                            cmd,
                            shell=False,
                            cwd=popen_kwargs.get("cwd"),
                            env=popen_kwargs.get("env"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL,
                            timeout=timeout,
                        )
                    else:
                        res = subprocess.run(
                            cmd,
                            shell=True,
                            cwd=popen_kwargs.get("cwd"),
                            env=popen_kwargs.get("env"),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            stdin=subprocess.DEVNULL,
                            timeout=timeout,
                        )
                except Exception as ex:
                    if fatal_on_error:
                        try:
                            self._events.put({"type": "fatal", "message": f"_run_shell exec error: {ex}"})
                        except Exception:
                            pass
                        raise
                    else:
                        # Return a pseudo-result object
                        class R:
                            pass
                        r = R()
                        r.returncode = 127
                        r.stdout = b""
                        r.stderr = str(ex).encode()
                        logger.debug(f"_run_shell exec error tolerated (non-fatal): {ex}")
                        return r
                logger.debug(f"_run_shell done rc={res.returncode}")
                if res.returncode not in (0, None):
                    # Fatal: non-zero return is infrastructure failure for waited commands
                    stderr_txt = ""
                    try:
                        if hasattr(res, "stderr") and res.stderr:
                            stderr_txt = res.stderr.decode(errors="ignore").strip()
                    except Exception:
                        stderr_txt = ""
                    msg = f"_run_shell failed rc={res.returncode}: {cmd_repr}" + (f"; stderr: {stderr_txt}" if stderr_txt else "")
                    logger.error(msg)
                    if fatal_on_error:
                        try:
                            self._events.put({"type": "fatal", "message": msg})
                        except Exception:
                            pass
                        raise RuntimeError(msg)
                    return res
                return res
            else:
                try:
                    if isinstance(cmd, list):
                        res = subprocess.run(
                            cmd,
                            shell=False,
                            cwd=popen_kwargs.get("cwd"),
                            env=popen_kwargs.get("env"),
                            stdout=subprocess.DEVNULL if quiet else None,
                            stderr=subprocess.DEVNULL if quiet else None,
                            stdin=subprocess.DEVNULL,
                            timeout=timeout,
                        )
                    else:
                        res = subprocess.run(
                            cmd,
                            shell=True,
                            cwd=popen_kwargs.get("cwd"),
                            env=popen_kwargs.get("env"),
                            stdout=subprocess.DEVNULL if quiet else None,
                            stderr=subprocess.DEVNULL if quiet else None,
                            stdin=subprocess.DEVNULL,
                            timeout=timeout,
                        )
                except Exception as ex:
                    if fatal_on_error:
                        try:
                            self._events.put({"type": "fatal", "message": f"_run_shell exec error: {ex}"})
                        except Exception:
                            pass
                        raise
                    else:
                        class R:
                            pass
                        r = R()
                        r.returncode = 127
                        r.stdout = b""
                        r.stderr = str(ex).encode()
                        logger.debug(f"_run_shell exec error tolerated (non-fatal): {ex}")
                        return r
                logger.debug(f"_run_shell done rc={res.returncode}")
                if res.returncode not in (0, None):
                    msg = f"_run_shell failed rc={res.returncode}: {cmd_repr}"
                    logger.error(msg)
                    if fatal_on_error:
                        try:
                            self._events.put({"type": "fatal", "message": msg})
                        except Exception:
                            pass
                        raise RuntimeError(msg)
                    return res
                return res
        # Asynchronous spawn
        try:
            if isinstance(cmd, list):
                proc = subprocess.Popen(cmd, shell=False, **popen_kwargs)
            else:
                proc = subprocess.Popen(cmd, shell=True, **popen_kwargs)
        except Exception as ex:
            try:
                self._events.put({"type": "fatal", "message": f"_run_shell spawn error: {ex}"})
            except Exception:
                pass
            raise
        return proc


        return self._drain_events(running, tasks)

    def _poll_running(self, running: Dict[str, Running], tasks: Dict[str, Dict], timeouts: Dict) -> bool:
        """Compatibility entry that drains worker events into the board state.

        Worker threads perform actual submission/polling; here we only merge their events
        and signal whether any progress happened.
        """
        return self._drain_events(running, tasks)

    def _drain_events(self, running: Dict[str, Running], tasks: Dict[str, Dict]) -> bool:
        progressed = False
        while True:
            try:
                ev = self._events.get_nowait()
            except queue.Empty:
                break
            tid = ev.get("tid")
            # fatal events may not have tid
            if ev.get("type") == "fatal":
                self._fatal_error = ev.get("message") or "fatal error from worker"
                logger.error(f"Fatal error reported by worker {tid or ''}: {self._fatal_error}")
                progressed = True
                continue
            if not tid or tid not in tasks:
                continue
            kind = ev.get("type")
            if kind == "status":
                status = ev.get("status")
                meta = ev.get("meta") or {}
                t = tasks[tid]
                if status:
                    t["status"] = status
                for k, v in meta.items():
                    t[k] = v
                progressed = True
                # Finalize
                if status in ("succeeded", "failed", "timeout"):
                    running.pop(tid, None)
                    # Cleanup worker registry
                    wk = self._workers.pop(tid, None)
                    try:
                        if wk:
                            wk.join(timeout=0.1)
                    except Exception:
                        pass
            elif kind == "log":
                logger.info(ev.get("message", ""))
            else:
                # Unknown event; ignore
                pass
        return progressed

    # Removed legacy pull/cleanup helpers; worker threads handle transfers and cleanup now

    def _should_exit(self) -> bool:
        statuses = [t.get("status") for t in (self._board.get("tasks") or {}).values()]
        return not any(s in ("queued", "created", "running") for s in statuses)

    def run(self) -> int:
        resources, timeouts, poll_interval = self._load_resources()
        self._run_started_at = time.time()
        logger.debug(f"Executor started with max_parallel={getattr(self, '_max_parallel', 'N/A')}, resources={len(resources)}")
        running: Dict[str, Running] = {}
        try:
            while True:
                # 1) 更新状态（轮询已运行任务）
                tasks: Dict[str, Dict] = self._board.get("tasks", {})
                logger.debug(f"Tasks: {len(tasks)}, Running: {len(running)}")
                progressed = self._poll_running(running, tasks, timeouts)

                # 2) 提交任务（轮询资源，能提交就提交）
                logger.debug(f"Attempting to submit new tasks; currently running: {len(running)}")
                submitted = self._submit_available(tasks, resources, running, timeouts)

                # 3) 错误处理/收尾（worker事件合并后保存）
                self._board["meta"]["last_update"] = time.time()
                save_board(self._board_path, self._board)
                # Exit on fatal error from any worker
                if self._fatal_error:
                    logger.error(f"Executor exiting due to fatal error: {self._fatal_error}")
                    return 1
                if self._should_exit():
                    return 0
                logger.debug(f"Progressed: {progressed}, Submitted: {submitted}, Running: {len(running)}")
                if not progressed and not submitted and running:
                    # logger.debug(f"Progressed: {progressed}, Submitted: {submitted}, Running: {len(running)}")
                    time.sleep(poll_interval)

        except KeyboardInterrupt:
            self._board["meta"]["last_update"] = time.time()
            save_board(self._board_path, self._board)
            return 130
        except Exception as ex:
            logger.error(f"Executor fatal error: {ex}")
            return 1

    def _submit_available(self, tasks: Dict[str, Dict], resources: List[Dict], running: Dict[str, Running], timeouts: Dict) -> bool:
        """Scan queued tasks and try to submit any that fits current resource availability.

        Returns True if at least one task was marked succeeded by cache or started a process.
        """
        submitted = False
        # Respect global scheduler limit
        try:
            max_par = int(getattr(self, "_max_parallel", 0) or 0)
        except Exception:
            max_par = 0
        if max_par and len(running) >= max_par:
            return False
        queued_items = [(tid, t) for tid, t in tasks.items() if t.get("status") in ("queued", "created")]
        if not queued_items:
            return False
        # Iterate once per loop iteration (no inner loop); caller will call again next tick
        for tid, t in queued_items:
            logger.debug(f"Considering task {tid} for submission")
            # Skip if a worker is already managing this task (avoid duplicate submissions)
            if tid in self._workers:
                logger.debug(f"Worker already active for {tid}; skipping re-submission")
                continue
            if max_par and len(running) >= max_par:
                break
            # If this task type is not supported by any resource, fail fast to avoid infinite waiting
            try:
                sw_type = (t.get("type") or "").lower()
                sup = getattr(self, "_supported_software", set())
                if sw_type and sup and (sw_type not in sup):
                    now = time.time()
                    t["resource"] = None
                    t["cmd"] = None
                    t["start_time"] = now
                    t["end_time"] = now
                    t["exit_code"] = 127
                    t["status"] = "failed"
                    logger.error(f"No resource supports software '{sw_type}' for task {tid}; marking failed.")
                    submitted = True
                    continue
            except Exception:
                pass
            # Manual handling: if marker file exists in workdir, do not launch processes.
            # Parse energy; if present -> mark succeeded; else -> mark failed.
            try:
                wd_manual = Path(t.get("workdir") or ".")
                if _has_manual_marker(wd_manual):
                    logger.debug(f"Manual marker found for {tid} in {wd_manual}; parsing energy without execution")
                    energy_val = None
                    try:
                        from .energy import read_energy
                        sw = t.get("type")
                        energy_val = read_energy(sw, wd_manual)
                    except Exception:
                        energy_val = None
                    now = time.time()
                    t["resource"] = "manual"
                    t["cmd"] = "manual"
                    t["start_time"] = now
                    t["end_time"] = now
                    if energy_val is not None:
                        t["exit_code"] = 0
                        t["status"] = "succeeded"
                        logger.info(f"Manual task {tid} succeeded (energy parsed)")
                    else:
                        t["exit_code"] = 1
                        t["status"] = "failed"
                        logger.warning(f"Manual task {tid} failed (energy not parsed)")
                    # Persist immediately so the board reflects this manual resolution
                    try:
                        self.save()
                    except Exception:
                        pass
                    submitted = True
                    # Do not attempt to allocate resources for this task
                    continue
            except Exception as ex:
                logger.error(f"Manual handling check failed for {tid}: {ex}")
            chosen, req = self._choose_resource(t.get("type"), resources, running)
            if not chosen:
                logger.debug(f"No available resource for task {tid} (type={t.get('type')}); will retry later")
                continue
            logger.info(f"Chosen resource {chosen.get('name')} for task {tid} (requires {req} cores)")
            # Cache probe before starting process
            workdir = Path(t.get("workdir"))
            sw = t.get("type")
            sw_conf = (chosen.get("software") or {}).get(sw) or {}
            cmd_preview, _cores_preview, _env_preview = _build_command(sw, sw_conf, workdir)
            logger.debug(f"Probing cache for task {tid} with command: {cmd_preview}")
            from .cache import probe_cache
            pr = probe_cache(
                software=sw,
                bin_path=str(sw_conf.get("path")),
                run_cmd=cmd_preview,
                workdir=workdir,
                resource=chosen,
            )
            # logger.debug(f"Cache probe result for {tid}: hit={pr.hit if pr else 'N/A'}, key={pr.key if pr else 'N/A'}")
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
                # Persist immediately so the board updates without waiting for the next tick
                try:
                    self.save()
                except Exception:
                    pass
                submitted = True
                # do not consume resource slot when cached; continue to try next queued task
                continue

            # Start worker thread per task
            pop, remote_dir, cmd, started_at, remote_pid = self._start_process(tid, t, chosen, req)
            logger.info(f"Started task {tid} on resource {chosen.get('name')} with cmd: {cmd}")
            worker = TaskWorker(executor=self, tid=tid, task=t.copy(), resource=chosen, req_cores=req, timeouts=timeouts)
            worker.daemon = True
            worker.start()
            self._workers[tid] = worker
            running[tid] = Running(popen=pop, task_id=tid, resource=chosen, remote_dir=remote_dir, cores=req, started_at=started_at, is_remote=(chosen.get("type") == "remote"), remote_pid=remote_pid)
            # Persist immediately
            try:
                self.save()
            except Exception:
                pass
            submitted = True
        return submitted


@dataclass
class Running:
    popen: Optional[subprocess.Popen]
    task_id: str
    resource: Dict
    remote_dir: Optional[str]
    cores: int
    started_at: float
    is_remote: bool = False
    remote_pid: Optional[int] = None


class TaskWorker(threading.Thread):
    def __init__(self, executor: Executor, tid: str, task: Dict[str, Any], resource: Dict[str, Any], req_cores: int, timeouts: Dict[str, Any]):
        super().__init__(name=f"worker-{tid}")
        self.ex = executor
        self.tid = tid
        self.task = task
        self.resource = resource
        self.req_cores = req_cores
        self.timeouts = timeouts or {}
        self.stop_evt = threading.Event()

    def post(self, ev: Dict[str, Any]) -> None:
        ev.setdefault("tid", self.tid)
        try:
            self.ex._events.put(ev)
        except Exception:
            pass

    def run(self) -> None:
        try:
            if (self.resource.get("type") or "local") == "remote":
                self._run_remote()
            else:
                self._run_local()
        except Exception as ex:
            self.post({"type": "status", "status": "failed", "meta": {"exit_code": 1, "end_time": time.time()}, "error": str(ex)})

    def _run_remote(self) -> None:
        host = _remote_host(self.resource)
        workdir = Path(self.task.get("workdir") or ".")
        remote_dir = (self.resource.get("workdir") or "")
        remote_dir = f"{remote_dir.rstrip('/')}/{self.tid}"
        # Submit async: mkdir + rsync + nohup launch
        self.post({"type": "status", "status": "created", "meta": {"resource": self.resource.get("name"), "cmd": self.task.get("cmd"), "start_time": time.time()}})
        # Ensure runner present locally
        self.ex._write_remote_runner(workdir, remote_dir, self.task.get("cmd") or "")
        dir_expr = self.ex._remote_path_expr(remote_dir)
        mkdir_args = self.ex._ssh_args(host, f"mkdir -p {dir_expr}")
        rsync_args = [
            "rsync", "-az", "--timeout=10", "-e", self.ex._rsync_ssh_e(), f"{str(workdir)}/", f"{host}:{remote_dir}/"
        ]
        start_inner = f"cd {dir_expr} && nohup bash .aq_launch.sh </dev/null >/dev/null 2>&1 & echo $! > .aq_pid"
        start_args = self.ex._ssh_args(host, start_inner)
        # Per-host concurrency gate around heavy rsync/start
        sem = self.ex._get_host_semaphore(host, (self.resource.get("connection") or {}).get("ssh_concurrency"))
        with sem:
            # Retry mkdir/rsync/start with backoff on transient network failures
            def _attempt(cmd, desc, timeout_s=None):
                backoff = 1.0
                for i in range(3):
                    res = self.ex._run_shell(cmd, quiet=True, wait=True, capture_output=True, timeout=timeout_s, fatal_on_error=False)
                    rc = getattr(res, "returncode", 0)
                    if rc in (0, None):
                        return True, res
                    err = ""
                    try:
                        if hasattr(res, "stderr") and res.stderr:
                            err = res.stderr.decode(errors="ignore").strip()
                    except Exception:
                        err = ""
                    self.post({"type": "log", "message": f"{self.tid}: {desc} attempt {i+1} failed rc={rc} {err}"})
                    time.sleep(backoff)
                    backoff *= 2
                return False, res

            self.post({"type": "log", "message": f"{self.tid}: creating remote dir {host}:{dir_expr}"})
            ok, res_mk = _attempt(mkdir_args, "mkdir", timeout_s=8)
            if not ok:
                self.post({"type": "status", "status": "failed", "meta": {"exit_code": getattr(res_mk,'returncode',127), "end_time": time.time()}, "message": "mkdir failed after retries"})
                return

            self.post({"type": "log", "message": f"{self.tid}: uploading to {host}:{remote_dir}"})
            ok, res_sync = _attempt(rsync_args, "rsync", timeout_s=None)
            if not ok:
                self.post({"type": "status", "status": "failed", "meta": {"exit_code": getattr(res_sync,'returncode',127), "end_time": time.time()}, "message": "rsync failed after retries"})
                return

            self.post({"type": "log", "message": f"{self.tid}: starting on remote"})
            ok, res_start = _attempt(start_args, "start", timeout_s=8)
            if not ok:
                self.post({"type": "status", "status": "failed", "meta": {"exit_code": getattr(res_start,'returncode',127), "end_time": time.time()}, "message": "start failed after retries"})
                return
        # Probe loop
        soft = float(self.timeouts.get(self.task.get("type"), self.timeouts.get("default", 0)) or 0)
        started = time.time()
        pid_set = False
        while not self.stop_evt.is_set():
            time.sleep(1.0)
            # Flip to running when pid appears
            if not pid_set:
                read_args = self.ex._ssh_args(host, f"cd {dir_expr} && test -s .aq_pid && cat .aq_pid || true")
                res_pid = self.ex._run_shell(read_args, quiet=True, wait=True, capture_output=True, timeout=2)
                pid_txt = (res_pid.stdout.decode(errors="ignore") if hasattr(res_pid, "stdout") and res_pid.stdout else "").strip()
                if pid_txt:
                    try:
                        rpid = int(pid_txt.splitlines()[0])
                    except Exception:
                        rpid = None
                    self.post({"type": "status", "status": "running", "meta": {"remote_pid": rpid, "start_time": started}})
                    pid_set = True
            # Check .aq_exit
            read_exit = self.ex._ssh_args(host, f"cd {dir_expr} && test -f .aq_exit && cat .aq_exit || true")
            res_ex = self.ex._run_shell(read_exit, quiet=True, wait=True, capture_output=True, timeout=2)
            txt = (res_ex.stdout.decode(errors="ignore") if hasattr(res_ex, "stdout") and res_ex.stdout else "").strip()
            if txt:
                try:
                    rc = int((txt.split() or ["0"])[0])
                except Exception:
                    rc = 1
                # Essential pull job.out
                pull_job = [
                    "rsync", "-az", "--timeout=10", "-e", self.ex._rsync_ssh_e(),
                    f"{host}:{remote_dir}/job.out", f"{str(workdir)}/job.out"
                ]
                self.ex._run_shell(pull_job, quiet=True, wait=True, capture_output=True, timeout=20, fatal_on_error=False)
                # Default: full directory incremental pull (required)
                def _attempt_full():
                    backoff = 1.0
                    for i in range(3):
                        full_args = [
                            "rsync", "-az", "--timeout=30", "-e", self.ex._rsync_ssh_e(),
                            f"{host}:{remote_dir}/", f"{str(workdir)}/"
                        ]
                        res_full = self.ex._run_shell(full_args, quiet=True, wait=True, capture_output=True, timeout=None, fatal_on_error=False)
                        rc_full = getattr(res_full, "returncode", 0)
                        if rc_full in (0, None):
                            return True
                        err = ""
                        try:
                            if hasattr(res_full, "stderr") and res_full.stderr:
                                err = res_full.stderr.decode(errors="ignore").strip()
                        except Exception:
                            err = ""
                        self.post({"type": "log", "message": f"{self.tid}: full pull attempt {i+1} failed rc={rc_full} {err}"})
                        time.sleep(backoff)
                        backoff *= 2
                    return False
                full_ok = _attempt_full()
                # Parse energy
                energy_val = None
                try:
                    from .energy import read_energy
                    energy_val = read_energy(self.task.get("type"), workdir)
                except Exception:
                    energy_val = None
                rc_used = 0 if (energy_val is not None) else int(rc)
                # Enforce default full pull: if retrieval failed after retries, mark task failed
                if not full_ok and rc_used == 0:
                    rc_used = 125
                # Write success cache in worker on success
                if rc_used == 0:
                    try:
                        from .cache import write_success_cache
                        sw = self.task.get("type")
                        sw_conf = (self.resource.get("software") or {}).get(sw) or {}
                        write_success_cache(
                            software=sw,
                            bin_path=str(sw_conf.get("path")),
                            run_cmd=str(self.task.get("cmd") or ""),
                            workdir=workdir,
                            resource=self.resource,
                            energy_eV=energy_val,
                        )
                    except Exception:
                        pass
                status = "succeeded" if rc_used == 0 else ("timeout" if rc_used == 124 else "failed")
                self.post({"type": "status", "status": status, "meta": {"exit_code": rc_used, "end_time": time.time(), "energy_eV": energy_val}})
                # Cleanup
                cleanup_args = self.ex._ssh_args(host, f"rm -rf {dir_expr}")
                self.ex._run_shell(cleanup_args, quiet=True, wait=None)
                return
            # Soft timeout
            if soft and (time.time() - started) > soft:
                kill_args = self.ex._ssh_args(host, f"cd {dir_expr} && pid=$(cat .aq_pid 2>/dev/null || echo); if [ -n \"$pid\" ]; then kill -TERM -- -$pid || true; sleep 1; kill -KILL -- -$pid || true; fi; echo 124 $(date +%s) > .aq_exit")
                self.ex._run_shell(kill_args, quiet=True, wait=True, timeout=3)


    def _run_local(self) -> None:
        workdir = Path(self.task.get("workdir") or ".")
        cmd = self.task.get("cmd") or ""
        pop = self.ex._run_shell(cmd, cwd=workdir, env=None, wait=None)
        self.post({"type": "status", "status": "running", "meta": {"start_time": time.time()}})
        # Poll
        while not self.stop_evt.is_set():
            try:
                rc = pop.poll()
            except Exception:
                rc = 1
            if rc is not None:
                # Parse energy
                energy_val = None
                try:
                    from .energy import read_energy
                    energy_val = read_energy(self.task.get("type"), workdir)
                except Exception:
                    energy_val = None
                rc_used = 0 if (energy_val is not None) else int(rc)
                if rc_used == 0:
                    try:
                        from .cache import write_success_cache
                        sw = self.task.get("type")
                        sw_conf = ((self.resource or {}).get("software") or {}).get(sw) or {}
                        write_success_cache(
                            software=sw,
                            bin_path=str(sw_conf.get("path")),
                            run_cmd=str(self.task.get("cmd") or ""),
                            workdir=workdir,
                            resource=self.resource,
                            energy_eV=energy_val,
                        )
                    except Exception:
                        pass
                status = "succeeded" if rc_used == 0 else ("timeout" if rc_used == 124 else "failed")
                self.post({"type": "status", "status": status, "meta": {"exit_code": rc_used, "end_time": time.time(), "energy_eV": energy_val}})
                return
            time.sleep(0.5)


def _build_command(software: str, sw_conf: Dict, workdir: Path) -> Tuple[str, int, Dict[str, str]]:
    """Return (command_string, cores, env_vars). Applies mpi wrapper when needed.

    Important: atlas does NOT use '< atlas.in'; per user, run as 'atlas.x > job.out 2>&1'.
    qe uses '< <file>'.
    """
    bin_path = sw_conf.get("path")
    if not bin_path:
        raise RuntimeError(f"software path missing for {software}")
    cores = int(sw_conf.get("cores", 1))
    mpi = sw_conf.get("mpi")
    env = {str(k): str(v) for k, v in (sw_conf.get("env") or {}).items()}

    if software == "qe":
        inp = "qe.in" if (workdir / "qe.in").exists() else "job.in"
        base = f"{bin_path} < {inp}"
    elif software == "atlas":
        # No stdin redirection
        base = f"{bin_path}"
    else:
        # default: just run binary
        base = f"{bin_path}"

    if mpi and cores > 1:
        base = f"{mpi} -np {cores} {base}"

    cmd = f"OMP_NUM_THREADS=1 {base} > job.out 2>&1"
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
    """Assert that 'rich' is importable; do not attempt auto-install."""
    try:
        import rich  # noqa: F401
    except Exception as e:
        raise RuntimeError("'rich' is required for aqflow board. Please install it in your environment.") from e
    


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
            # Show active tasks by default (running + created)
            tasks = [t for t in tasks if t.get("status") in ("running", "created")]
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
            table.add_column("resource")
            table.add_column("pid", justify="right")
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
                # Colorize status for readability
                st_style = {
                    "running": "green",
                    "created": "yellow",
                    "queued": "cyan",
                    "succeeded": "bold green",
                    "failed": "bold red",
                    "timeout": "magenta",
                }.get(st.lower(), "white")
                st_rich = Text(st, style=st_style)
                start = t.get("start_time") or 0
                end = t.get("end_time") or None
                sec = int((end or now) - start) if start else 0
                elapsed = _fmt_elapsed(sec)
                res_name = str(t.get("resource") or "-")
                pid_txt = str(t.get("remote_pid") or "")
                quick = f"cd {t.get('workdir', '.')}; tail -n 200 job.out"
                table.add_row(tid, name, typ, st_rich, res_name, pid_txt, elapsed, quick)
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


def _read_log_tail(path: Path, max_lines: int = 200) -> str:
    """Read the last ``max_lines`` from a potentially large log file efficiently.

    Avoids loading the entire file into memory by seeking from the end and
    reading in blocks until enough line breaks are collected. Falls back to an
    empty string on errors.
    """
    try:
        p = Path(path)
        if max_lines <= 0:
            max_lines = 1
        if not p.exists():
            return ""
        newline_count_target = max_lines
        block_size = 8192
        with p.open("rb") as fh:
            fh.seek(0, os.SEEK_END)
            file_size = fh.tell()
            if file_size == 0:
                return ""
            data = bytearray()
            remaining = file_size
            newlines = 0
            # Keep reading backwards in blocks until we have enough lines
            while remaining > 0 and newlines <= newline_count_target:
                read_size = block_size if remaining >= block_size else remaining
                remaining -= read_size
                fh.seek(remaining)
                chunk = fh.read(read_size)
                if not chunk:
                    break
                data[:0] = chunk  # prepend
                newlines = data.count(b"\n")
        # Decode and slice the desired tail lines
        text = data.decode(errors="ignore")
        lines = text.splitlines()
        return "\n".join(lines[-max_lines:])
    except Exception:
        # Best-effort fallback
        try:
            text = Path(path).read_text(errors="ignore")
            return "\n".join(text.splitlines()[-max_lines:])
        except Exception:
            return ""


def watch_single_board(
    board_path: Path,
    stop: Optional[Event] = None,
    interval: float = 0.5,
    show_logs: bool = False,
    log_path: Optional[Path] = None,
    log_lines: int = 200,
) -> None:
    """Watch a single board.json and render until completion or stop set (Rich-based).

    When show_logs=True, the screen is split vertically: top = tasks board,
    bottom = tail of the log file.
    """
    _ensure_rich()
    from rich.console import Console, Group
    from rich.live import Live
    from rich.text import Text
    from rich.panel import Panel
    from rich.layout import Layout
    from pathlib import Path as _Path

    if log_path is None:
        log_path = _Path.cwd() / "aqflow_data" / "aqflow.log"
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
                    top = Group(header, progress, *renderables)
                    if show_logs:
                        # Derive how many log lines can fit in the current layout height.
                        try:
                            term_h = console.size.height
                            # Split roughly 50/50; subtract a small overhead for panel borders/title
                            bottom_capacity = max(3, term_h // 2 - 4)
                        except Exception:
                            bottom_capacity = log_lines
                        effective_log_lines = max(1, min(log_lines, bottom_capacity))
                        log_text = _read_log_tail(log_path, max_lines=effective_log_lines)
                        panel = Panel.fit(
                            log_text or "(no logs)",
                            title=f"Logs tail ({effective_log_lines})",
                            border_style="dim",
                        )
                        layout = Layout()
                        # Split 50/50 vertically (top tasks, bottom logs)
                        layout.split_column(
                            Layout(name="top", ratio=1),
                            Layout(name="bottom", ratio=1),
                        )
                        layout["top"].update(top)
                        layout["bottom"].update(panel)
                        live.update(layout, refresh=True)
                    else:
                        live.update(top, refresh=True)
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
SSH_OPTS: List[str] = [
    "-o", "BatchMode=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=8",
]
