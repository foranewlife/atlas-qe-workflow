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

from aqflow.eos_controller import EosController
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
    subprocess.Popen([sys.executable, "scripts/task_service.py", "--host", host, "--port", str(port)])
    # wait for health
    for _ in range(25):
        try:
            with urlopen(svc_url("/health", host, port), timeout=1) as resp:
                if resp.read().decode().strip() == "ok":
                    return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("service failed to start")


def submit_local(software: str, host: str, port: int, work_dir: Path):
    ensure_service(host, port)
    payload = json.dumps({"software": software, "work_dir": str(work_dir)}).encode()
    req = Request(svc_url("/run", host, port), data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode())
        print(f"submitted: {data['id']}")
    print(f"dashboard: {svc_url('/dashboard', host, port)}")


def cmd_server(args: argparse.Namespace) -> int:
    # delegate to task_service.py to avoid code duplication
    subprocess.call([sys.executable, "scripts/task_service.py", "--host", args.host, "--port", str(args.port)])
    return 0


def cmd_local(args: argparse.Namespace, software: str) -> int:
    host = args.host or DEFAULT_HOST
    port = args.port or DEFAULT_PORT
    submit_local(software, host, port, Path(os.getcwd()))
    return 0


def cmd_eos(args: argparse.Namespace) -> int:
    cfg = Path(args.config).resolve()
    setup_logging(level=args.log_level)
    try:
        ctrl = EosController(cfg, Path("config/resources.yaml").resolve())
        ctrl.validate_inputs()
        tasks = ctrl.generate_tasks()
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
    p_srv.set_defaults(func=cmd_server)

    p_atlas = sub.add_parser("atlas", help="Run atlas in current directory via service")
    p_atlas.add_argument("--host", default=DEFAULT_HOST)
    p_atlas.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_atlas.set_defaults(func=lambda a: cmd_local(a, "atlas"))

    p_qe = sub.add_parser("qe", help="Run qe in current directory via service")
    p_qe.add_argument("--host", default=DEFAULT_HOST)
    p_qe.add_argument("--port", type=int, default=DEFAULT_PORT)
    p_qe.set_defaults(func=lambda a: cmd_local(a, "qe"))

    p_eos = sub.add_parser("eos", help="Run EOS workflow from config")
    p_eos.add_argument("config", help="Path to workflow YAML config")
    p_eos.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p_eos.set_defaults(func=cmd_eos)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())

