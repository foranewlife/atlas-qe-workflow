#!/usr/bin/env python3
"""
aqflow: Minimal CLI for ATLAS/QE workflow

Commands:
  aqflow board     # show lightweight tasks board (no server)
  aqflow atlas     # run atlas in current directory
  aqflow qe        # run qe in current directory
  aqflow eos CFG   # run EOS workflow from config
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from aqflow.core.eos import EosController
from aqflow.utils.logging_config import setup_logging
from aqflow.core.executor import Executor, BOARD_PATH, load_board, GLOBAL_HOME


def submit_orchestrator(software: str, work_dir: Path, resources: Path) -> int:
    """Submit current directory as a single task to the state machine."""
    work_dir = work_dir.resolve()
    # Detect input file
    if software == "qe":
        inp = work_dir / ("qe.in" if (work_dir / "qe.in").exists() else "job.in")
    else:
        inp = work_dir / "atlas.in"
    if not inp.exists():
        print(f"Input file not found: {inp}")
        return 1
    # Append task to board.json
    resources = Path(resources).resolve()
    ex = Executor(resources, board_path=BOARD_PATH, run_meta={
        "tool": software,
        "args": list(sys.argv),
        "resources_file": str(resources),
        "root": str(Path.cwd()),
    })
    task_id = f"{software}_{work_dir.name}_{int(time.time())}"
    entry = {
        "id": task_id,
        "name": f"{software} {work_dir.name}",
        "type": software,
        "workdir": str(work_dir),
        "status": "queued",
    }
    ex.add_tasks([entry])
    ex.save()
    ex.run()
    rc = 0 if (work_dir / "job.out").exists() else 1
    print(f"submitted {task_id}, rc={rc}")
    return rc


def cmd_board(args: argparse.Namespace) -> int:
    # Aggregate all boards from GLOBAL_HOME (symlinks) and show running tasks by root
    boards_dir = GLOBAL_HOME
    if not boards_dir.exists():
        print("No boards found. Run a workflow to generate aqflow/board.json.")
        return 0
    paths = sorted([p for p in boards_dir.glob("*.json") if p.name != "latest.json"], key=lambda p: p.name)
    items: list[tuple[str, dict]] = []
    for p in paths:
        data = load_board(p)
        if not data or not isinstance(data.get("tasks"), dict):
            continue
        items.append((data.get("meta", {}).get("root", str(p)), data))

    # Group by root, show only running by default (unless --all)
    show_all = getattr(args, "all", False)
    for root, data in items:
        tasks = list((data.get("tasks") or {}).values())
        # Try load resources.yaml for quick ssh tail when remote
        resmap = {}
        try:
            import yaml  # noqa: F401
            rpath = (data.get("meta") or {}).get("resources_file")
            if rpath and Path(rpath).exists():
                rdata = yaml.safe_load(Path(rpath).read_text()) or {}
                for r in (rdata.get("resources") or []):
                    name = r.get("name")
                    if name:
                        resmap[name] = r
        except Exception:
            resmap = {}
        if not show_all:
            tasks = [t for t in tasks if t.get("status") == "running"]
        if not tasks:
            continue
        print(f"root={root}")
        # Header
        print("task_id  name                 type   status   elapsed  quick")
        def fmt_elapsed(sec: int) -> str:
            h = sec // 3600; m = (sec % 3600) // 60; s = sec % 60
            return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        for t in tasks:
            tid = t.get("id", "-")
            name = (t.get("name") or "-")[:20].ljust(20)
            typ = (t.get("type") or "-").ljust(6)
            st = (t.get("status") or "-").ljust(8)
            # elapsed
            start = t.get("start_time") or 0
            end = t.get("end_time") or None
            now = time.time()
            sec = int((end or now) - start) if start else 0
            elapsed = fmt_elapsed(sec)
            quick = f"cd {t.get('workdir', '.')}; tail -n 200 job.out"
            rname = t.get("resource")
            res = resmap.get(rname) if rname else None
            if res and res.get("type") == "remote":
                host = (res.get("user") + "@" if res.get("user") else "") + (res.get("host") or "")
                base = res.get("workdir")
                if host and base:
                    quick = f"ssh {host} 'tail -n 200 {base.rstrip('/')}/{t.get('id','')}/job.out'"
            print(f"{tid:7s} {name} {typ} {st} {elapsed:7s} {quick}")
        print()
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

    p_board = sub.add_parser("board", help="Show tasks board (running by default)")
    p_board.add_argument("--all", action="store_true", help="Show all tasks, not only running")
    p_board.set_defaults(func=cmd_board)

    p_atlas = sub.add_parser("atlas", help="Run atlas in current directory")
    p_atlas.add_argument("--resources", default="config/resources.yaml")
    p_atlas.set_defaults(func=lambda a: cmd_local(a, "atlas"))

    p_qe = sub.add_parser("qe", help="Run qe in current directory")
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
