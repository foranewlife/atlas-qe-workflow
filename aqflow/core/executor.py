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
        # Background jobs registry for optional non-blocking processes with deadlines
        self._bg_jobs: list[dict] = []

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
        Remote: rsync + launch runner script asynchronously via a single local command; return (popen, remote_dir, None).
        """
        if plan.is_remote:
            # 1) Write runner script locally inside workdir (will be rsynced)
            self._write_remote_runner(plan.workdir, plan.remote_dir or str(plan.workdir), plan.run_cmd)
            # 2) Compose combined async submit: ensure remote dir, rsync, then launch via ssh
            host = plan.host or ""
            remote_dir = plan.remote_dir or str(plan.workdir)
            # For ssh payload use $HOME to avoid ~ issues inside quotes
            rd_shell = "$HOME" + remote_dir[1:] if remote_dir.startswith('~') else remote_dir
            mkdir_args = self._ssh_args(plan.host, f"mkdir -p \"{self._dq_escape(rd_shell)}\"")
            mkdir_cmd = " ".join(shlex.quote(a) for a in mkdir_args)
            # For rsync remote path, keep '~' so it expands on remote shell; do not quote the colon spec
            rsync_cmd = f"rsync -az -e ssh {shlex.quote(str(plan.workdir))}/ {host}:{remote_dir}/"
            start_inner = f"nohup bash \"{rd_shell}/.aq_launch.sh\" </dev/null >/dev/null 2>&1 & echo $! > \"{rd_shell}/.aq_pid\""
            start_args = self._ssh_args(plan.host, start_inner)
            start_cmd = " ".join(shlex.quote(a) for a in start_args)
            combo_cmd = f"{mkdir_cmd} && {rsync_cmd} && {start_cmd}"
            logger.info(f"Remote submit (async): {combo_cmd}")
            pop = self._run_shell(combo_cmd, quiet=True, wait=None)
            return pop, plan.remote_dir, None
        else:
            logger.info(f"Starting local in {plan.workdir}: {plan.run_cmd}")
            pop = self._run_shell(plan.run_cmd, cwd=plan.workdir, env=plan.env, wait=None)
            return pop, None, None

    def _ssh_args(self, host: Optional[str], inner: str) -> list[str]:
        """Return argv list for ssh to run remote bash -lc with given payload string.

        Using argv avoids local shell expansion and quoting pitfalls.
        """
        return ["ssh", str(host or ""), "bash", "-lc", inner]

    @staticmethod
    def _dq_escape(s: str) -> str:
        """Escape a string for inclusion inside a double-quoted shell string."""
        return s.replace("\\", "\\\\").replace("\"", "\\\"")

    def _write_remote_runner(self, workdir: Path, remote_dir: str, raw_cmd: str) -> Path:
        """Create a runner script in local workdir to be executed on remote side.

        The script will cd into the remote_dir (with ~ expanded via $HOME), write .aq_started,
        run the user command under bash -lc, and write .aq_exit with rc and end timestamp.
        """
        runner = Path(workdir) / ".aq_launch.sh"
        rd_shell = "$HOME" + remote_dir[1:] if remote_dir.startswith('~') else remote_dir
        content = (
            "#!/usr/bin/env bash\n"
            "set -e\n"
            f"cd \"{rd_shell}\"\n"
            "date +%s > .aq_started\n"
            f"bash -lc \"{self._dq_escape(raw_cmd)}\"\n"
            "rc=$?\n"
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
        pop, remote_dir_actual, remote_pid = self._execute_plan(plan)
        if plan.is_remote:
            t["status"] = "created"
        else:
            t["status"] = "running"
        t["resource"] = chosen.get("name")
        t["cmd"] = cmd
        t["start_time"] = started_at
        return pop, remote_dir_actual, cmd, started_at, remote_pid

    def _plan_remote_prep(self, host: str, remote_dir: Optional[str], workdir: Path) -> list[str]:
        cmds: list[str] = []
        if remote_dir:
            cmds.append(f"ssh {shlex.quote(host)} {shlex.quote('mkdir -p ' + remote_dir)}")
            # Use rsync for efficient, incremental upload (no delete on remote)
            src = str(workdir) + "/"
            dst = f"{host}:{remote_dir}/"
            cmds.append(
                f"rsync -az -e ssh {shlex.quote(src)} {shlex.quote(dst)}"
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
        cmd: str,
        cwd: Optional[Path] = None,
        env: Optional[Dict[str, str]] = None,
        *,
        quiet: bool = False,
        wait: Optional[float | bool] = None,
        capture_output: bool = False,
        timeout: Optional[float] = None,
        name: Optional[str] = None,
        detached: bool = False,
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
            "shell": True,
            "cwd": str(cwd) if cwd else None,
            "env": env,
            "stdin": subprocess.DEVNULL,
        }
        logger.debug(f"_run_shell cmd={cmd!r} cwd={cwd} quiet={quiet} wait={wait} capture={capture_output} timeout={timeout} detached={detached}")
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
                logger.debug(f"_run_shell done rc={res.returncode}")
                return res
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
                logger.debug(f"_run_shell done rc={res.returncode}")
                return res
        # Asynchronous spawn
        proc = subprocess.Popen(cmd, **popen_kwargs)
        if isinstance(wait, (int, float)) and wait > 0:
            # Record for background monitoring by the executor loop
            try:
                self._bg_jobs.append({
                    "proc": proc,
                    "deadline": time.time() + float(wait),
                    "name": name or cmd,
                    "started_at": time.time(),
                    "detached": bool(detached),
                })
            except Exception:
                pass
        return proc

    def _poll_bg_jobs(self) -> None:
        """Poll and enforce deadlines for background jobs created via _run_shell(wait=<seconds>)."""
        if not getattr(self, "_bg_jobs", None):
            return
        now = time.time()
        remain: list[dict] = []
        for job in self._bg_jobs:
            proc: subprocess.Popen = job.get("proc")
            name = job.get("name")
            dl = job.get("deadline", 0.0) or 0.0
            rc = proc.poll()
            if rc is not None:
                logger.debug(f"Background job finished rc={rc}: {name}")
                continue
            if dl and now > dl:
                # Timed out: best-effort terminate/kill
                try:
                    proc.terminate()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=1.0)
                except Exception:
                    try:
                        proc.kill()
                    except Exception:
                        pass
                logger.warning(f"Background job timed out and was terminated: {name}")
                continue
            remain.append(job)
        self._bg_jobs = remain

    def _poll_running(self, running: Dict[str, Running], tasks: Dict[str, Dict], timeouts: Dict) -> bool:
        progressed = False
        for tid, run in list(running.items()):
            rc = None
            if run.is_remote:
                # If submission is in 'created', flip to 'running' once .aq_pid appears
                tdict_created = tasks.get(tid) or {}
                if (tdict_created.get("status") == "created") and run.remote_dir:
                    try:
                        rd = run.remote_dir
                        rd_shell = "$HOME" + rd[1:] if rd.startswith('~') else rd
                        read_args = self._ssh_args(_remote_host(run.resource), f"test -s \"{self._dq_escape(rd_shell)}/.aq_pid\" && cat \"{self._dq_escape(rd_shell)}/.aq_pid\" || true")
                        res = self._run_shell(" ".join(shlex.quote(a) for a in read_args), quiet=True, wait=True, capture_output=True, timeout=2)
                        out = res.stdout if hasattr(res, "stdout") else b""
                        if out.decode(errors="ignore").strip():
                            tdict_created["status"] = "running"
                            tdict_created["start_time"] = tdict_created.get("start_time") or time.time()
                            self.save()
                    except Exception:
                        # if submit process already ended with failure and no pid, fail fast
                        try:
                            if run.popen and run.popen.poll() is not None and (time.time() - float(tdict_created.get("start_time") or 0)) > 3:
                                tdict_created["status"] = "failed"
                                tdict_created["exit_code"] = 127
                                tdict_created["end_time"] = time.time()
                                self.save()
                                running.pop(tid, None)
                                progressed = True
                                continue
                        except Exception:
                            pass
                # Remote detection via marker file .aq_exit
                try:
                    host = _remote_host(run.resource)
                    if run.remote_dir:
                        rd = run.remote_dir
                        if rd.startswith("~"):
                            rd_shell = "$HOME" + rd[1:]
                        else:
                            rd_shell = rd
                        rd_esc = self._dq_escape(rd_shell)
                        check_args = self._ssh_args(host, f"test -f \"{rd_esc}/.aq_exit\" && cat \"{rd_esc}/.aq_exit\" || true")
                        res2 = self._run_shell(" ".join(shlex.quote(a) for a in check_args), quiet=True, wait=True, capture_output=True, timeout=2)
                        out = res2.stdout if hasattr(res2, "stdout") else b""
                        txt = out.decode(errors="ignore").strip()
                        if txt:
                            parts = txt.split()
                            rc = int(parts[0]) if parts else 0
                except Exception:
                    rc = None
            else:
                rc = run.popen.poll()
            t = tasks.get(tid) or {}
            soft = None
            if t.get("type") and isinstance(timeouts, dict):
                soft = timeouts.get(t.get("type")) or timeouts.get("default")
            if soft and t.get("start_time") and (rc is None):
                if time.time() - float(t["start_time"]) > float(soft):
                    if run.is_remote and run.remote_dir:
                        # Try to terminate remote process group
                        try:
                            host = _remote_host(run.resource)
                            rd = run.remote_dir
                            if rd.startswith("~"):
                                rd_shell = "$HOME" + rd[1:]
                            else:
                                rd_shell = rd
                            rd_esc = self._dq_escape(rd_shell)
                            kill_inner = (
                                f"pid=$(cat \"{rd_esc}/.aq_pid\" 2>/dev/null || echo); "
                                f"if [ -n \"$pid\" ]; then kill -TERM -- -$pid || true; sleep 1; kill -KILL -- -$pid || true; fi; "
                                f"echo 124 $(date +%s) > \"{rd_esc}/.aq_exit\""
                            )
                            kill_args = self._ssh_args(host, kill_inner)
                            self._run_shell(" ".join(shlex.quote(a) for a in kill_args), quiet=True, wait=True, timeout=2)
                            rc = 124
                        except Exception:
                            rc = 124
                    else:
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
                pulls = self._plan_pull_outputs(run, tasks[tid])
                if pulls:
                    # Essential pull (job.out) synchronously
                    self._run_shell(pulls[0], quiet=True, wait=True)
                    # Optional full pull asynchronously
                    for extra in pulls[1:]:
                        logger.info(f"Async full pull: {extra}")
                        self._run_shell(extra, quiet=True, detached=True, wait=None)
                # After pulling outputs, cleanup remote working directory
                for cmd in self._plan_remote_cleanup(run):
                    logger.info(f"Executing remote cleanup (detached): {cmd}")
                    # Fire-and-forget in a detached session; no timeout/poller enforcement
                    self._run_shell(cmd, quiet=True, detached=True, wait=None)
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
            # Persist immediately so the board reflects task completion without delay
            try:
                self.save()
            except Exception:
                pass
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
        # Essential: rsync only job.out (incremental)
        cmds.append(
            f"rsync -az -e ssh {host}:{run.remote_dir}/job.out {task['workdir']}/job.out"
        )
        # Optional: full incremental pull (changed + new files only; no delete)
        if pull_all:
            cmds.append(
                f"rsync -az -e ssh {host}:{run.remote_dir}/ {task['workdir']}/"
            )
        return cmds

    def _plan_remote_cleanup(self, run: Running) -> list[str]:
        """Return commands to delete remote working directory after outputs are pulled.

        Safety: only act if remote_dir is set.
        """

        cmds: list[str] = []
        if not run.remote_dir:
            return cmds
        host = _remote_host(run.resource)
        cmds.append(
            f"ssh {host} 'rm -rf {run.remote_dir}'"
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
                submitted = self._submit_available(tasks, resources, running)

                # 3) 错误处理/收尾
                #    Poll background jobs (non-blocking) created via _run_shell(wait=<seconds>)
                try:
                    self._poll_bg_jobs()
                except Exception:
                    pass
                self._board["meta"]["last_update"] = time.time()
                save_board(self._board_path, self._board)
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

    def _submit_available(self, tasks: Dict[str, Dict], resources: List[Dict], running: Dict[str, Running]) -> bool:
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
            logger.debug(f"Cache probe result for {tid}: hit={pr.hit if pr else 'N/A'}, key={pr.key if pr else 'N/A'}")
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

            pop, remote_dir, cmd, started_at, remote_pid = self._start_process(tid, t, chosen, req)
            running[tid] = Running(
                popen=pop,
                task_id=tid,
                resource=chosen,
                remote_dir=remote_dir,
                cores=req,
                started_at=started_at,
                is_remote=(chosen.get("type") == "remote"),
                remote_pid=remote_pid,
            )
            # Persist immediately after resource assignment and status change to 'running'
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
