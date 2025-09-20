#!/usr/bin/env python3
"""
aqflow: Unified CLI for Atlas–QE workflow

Subcommands:
  aqflow server [--host H --port P]    Start lightweight local task service (dashboard)
  aqflow atlas                         Run atlas in current directory via service
  aqflow qe                            Run qe in current directory via service
  aqflow eos <config.yaml>             Run EOS workflow using unified orchestrator
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.request import Request, urlopen

from aqflow.core.eos_controller import EosController
from aqflow.core.process_orchestrator import ProcessOrchestrator
from aqflow.core.task_creation import TaskDef
from aqflow.utils.logging_config import setup_logging


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


def svc_url(path: str, host: str, port: int) -> str:
    return f"http://{host}:{port}{path}"


def ensure_service(host: str, port: int):
    try:
        with urlopen(svc_url("/health", host, port), timeout=2) as resp:
            if resp.read().decode().strip() == "ok":
                return
    except Exception:
        pass
    print("Starting task service...")
    subprocess.Popen([sys.executable, "-m", "aqflow.scripts.task_service", "--host", host, "--port", str(port)])
    for _ in range(25):
        try:
            with urlopen(svc_url("/health", host, port), timeout=1) as resp:
                if resp.read().decode().strip() == "ok":
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("service failed to start")


def submit_orchestrator(software: str, work_dir: Path, resources: Path) -> int:
    """Submit current directory as a single task to the orchestrator."""
    work_dir = work_dir.resolve()
    # Detect input file
    if software == "qe":
        inp = work_dir / ("qe.in" if (work_dir / "qe.in").exists() else "job.in")
    else:
        inp = work_dir / "atlas.in"
    if not inp.exists():
        print(f"Input file not found: {inp}")
        return 1
    task_id = f"{software}_{work_dir.name}_{int(time.time())}"
    task = TaskDef(
        task_id=task_id,
        software=software,
        work_dir=work_dir,
        input_file=inp,
        expected_outputs=["job.out", "eos_data.json"],
        meta={"mode": "manual", "cwd": str(work_dir)},
    )
    orch = ProcessOrchestrator(resources, Path("results"))
    orch.load_tasks([task])
    orch.run()
    # Check rc by presence of job.out
    rc = 0 if (work_dir / "job.out").exists() else 1
    print(f"submitted {task_id}, rc={rc}")
    return rc


def cmd_server(args: argparse.Namespace) -> int:
    """Start task service. Default: background; use --foreground to run in foreground."""
    if getattr(args, "foreground", False):
        return subprocess.call([sys.executable, "-m", "aqflow.scripts.task_service", "--host", args.host, "--port", str(args.port)])
    # Background mode
    logs_dir = Path("logs"); logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "task_service.log"
    with open(log_file, "a", buffering=1) as lf:
        proc = subprocess.Popen(
            [sys.executable, "-m", "aqflow.scripts.task_service", "--host", args.host, "--port", str(args.port)],
            stdout=lf, stderr=lf, start_new_session=True
        )
    pidfile = logs_dir / "task_service.pid"
    pidfile.write_text(str(proc.pid))
    print(f"Task service started (pid {proc.pid}) at http://{args.host}:{args.port}  logs: {log_file}")
    return 0


def cmd_local(args: argparse.Namespace, software: str) -> int:
    # Submit current directory to orchestrator (not the local service)
    return submit_orchestrator(software, Path(os.getcwd()), Path(args.resources))


def cmd_eos(args: argparse.Namespace) -> int:
    cfg = Path(args.config).resolve()
    setup_logging(level=args.log_level)
    try:
        ctrl = EosController(cfg, Path(args.resources).resolve())
        ctrl.validate_inputs()
        tasks = ctrl.generate_tasks()
        if args.dry_run:
            print(f"Prepared {len(tasks)} tasks (dry-run)")
            return 0
        results = ctrl.execute(tasks)
        failed = [r for r in results if r.returncode != 0]
        print(f"EOS done: {len(results)-len(failed)}/{len(results)} ok")
        return 0 if not failed else 1
    except Exception as e:
        print(f"EOS failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(prog="aqflow", description="Unified CLI for Atlas–QE workflow")
    sub = parser.add_subparsers(dest="cmd")

    p_srv = sub.add_parser("server", help="Start local task service and dashboard")
    p_srv.add_argument("--host", default=DEFAULT_HOST)
    p_srv.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_srv.add_argument("--foreground", action="store_true", help="Run in foreground (default: background)")
    p_srv.set_defaults(func=cmd_server)

    p_atlas = sub.add_parser("atlas", help="Run atlas in current directory via service")
    p_atlas.add_argument("--resources", default="config/resources.yaml")
    p_atlas.set_defaults(func=lambda a: cmd_local(a, "atlas"))

    p_qe = sub.add_parser("qe", help="Run qe in current directory via service")
    p_qe.add_argument("--resources", default="config/resources.yaml")
    p_qe.set_defaults(func=lambda a: cmd_local(a, "qe"))

    p_eos = sub.add_parser("eos", help="Run EOS workflow from config")
    p_eos.add_argument("config", help="Path to workflow YAML config")
    p_eos.add_argument("--resources", default="config/resources.yaml", help="Resource config YAML (default: config/resources.yaml)")
    p_eos.add_argument("--dry-run", action="store_true", help="Only generate tasks and exit")
    p_eos.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p_eos.set_defaults(func=cmd_eos)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
