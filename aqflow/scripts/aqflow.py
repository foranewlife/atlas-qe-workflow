#!/usr/bin/env python3
"""
aqflow: Minimal CLI for ATLAS/QE workflow

Commands:
  aqflow board     # real-time tasks board (no server)
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
from threading import Event, Thread
from typing import List

from aqflow.core.eos import EosController
from aqflow.utils.logging_config import setup_logging
from aqflow.core.executor import (
    Executor,
    BOARD_PATH,
    load_board,
    GLOBAL_BOARDS,
    GLOBAL_RESOURCES,
    cleanup_invalid_symlinks,
    watch_global_board,
    render_global_board_once,
    watch_single_board,
    set_board_row_limit,
)
from aqflow.core.eos_post import EosPostProcessor
 


def _ensure_dep(module: str, package: str) -> None:
    """Ensure a dependency exists; if missing, try to install via 'uv pip install'.

    Only used in CLI edge paths (e.g., optional yaml for board rendering).
    """
    try:
        __import__(module)
    except ImportError:
        # Best-effort install; ignore failures to keep CLI functional
        from subprocess import run
        run(["uv", "pip", "install", package], check=False)


def submit_task(software: str, work_dir: Path, resources: Path) -> int:
    """Submit current directory as a single task to the executor.

    No template or input pre-check here; executor will run with software defaults
    (QE: '-in qe.in' fallback to 'job.in'; ATLAS: binary > job.out)."""
    work_dir = work_dir.resolve()
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
    # If --once, render one-shot and exit; otherwise run in watch mode.
    boards_dir = GLOBAL_BOARDS
    if not boards_dir.exists():
        print("No boards found. Run a workflow to generate aqflow/board.json.")
        return 0
    # apply row limit
    limit = getattr(args, "limit", 50)
    set_board_row_limit(None if (limit is not None and int(limit) <= 0) else int(limit))
    if getattr(args, "clean_only", False):
        removed = cleanup_invalid_symlinks(boards_dir)
        print(f"Removed {removed} dangling symlinks under {boards_dir}")
        return 0
    if getattr(args, "once", False):
        print(
            render_global_board_once(
                show_all=getattr(args, "all", False),
                group_by=getattr(args, "group_by", None),
                filters=getattr(args, "filter", []) or [],
            )
        )
        return 0
    return watch_global_board(
        show_all=getattr(args, "all", False),
        group_by=getattr(args, "group_by", None),
        filters=getattr(args, "filter", []) or [],
        interval=float(getattr(args, "interval", 1.0) or 1.0),
    )


def cmd_local(args: argparse.Namespace, software: str) -> int:
    # Submit current directory to orchestrator (not the local service)
    watch = getattr(args, "watch", True)
    interval = float(getattr(args, "interval", 0.5) or 0.5)
    # apply row limit
    limit = getattr(args, "limit", 50)
    set_board_row_limit(None if (limit is not None and int(limit) <= 0) else int(limit))
    if watch:
        stop = Event()
        t = Thread(target=watch_single_board, args=(BOARD_PATH, stop, interval), daemon=False)
        t.start()
    rc = submit_task(software, Path(os.getcwd()), Path(args.resources))
    if watch:
        # Signal watcher to stop and wait for graceful Live exit to restore cursor
        stop.set()
        try:
            t.join(timeout=2.0)
        except Exception:
            pass
    return rc


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
        # Start watcher thread if requested
        watch = getattr(args, "watch", True)
        interval = float(getattr(args, "interval", 0.5) or 0.5)
        stop = Event()
        # apply row limit for run watch
        limit = getattr(args, "limit", 50)
        set_board_row_limit(None if (limit is not None and int(limit) <= 0) else int(limit))
        if watch:
            t = Thread(target=watch_single_board, args=(BOARD_PATH, stop, interval), daemon=False)
            t.start()
        results = ctrl.execute(tasks)
        stop.set()
        try:
            t.join(timeout=2.0)
        except Exception:
            pass
        failed = [r for r in results if r.returncode != 0]
        print(f"EOS done: {len(results)-len(failed)}/{len(results)} ok")
        return 0 if not failed else 1
    except Exception as e:
        print(f"EOS failed: {e}")
        return 1


def main():
    parser = argparse.ArgumentParser(prog="aqflow", description="Minimal CLI for ATLAS/QE workflow")
    sub = parser.add_subparsers(dest="cmd")

    p_board = sub.add_parser("board", help="Real-time tasks board (running by default)")
    p_board.add_argument("--all", action="store_true", help="Show all tasks, not only running")
    p_board.add_argument(
        "--filter",
        action="append",
        help="Filter tasks: key:value (status/type/resource/name) or plain substring on name; can repeat",
    )
    p_board.add_argument(
        "--group-by", choices=["resource", "type"], help="Group tasks within each root by resource or type"
    )
    p_board.add_argument(
        "--interval", type=float, default=1.0, help="Refresh interval seconds (default: 1.0)"
    )
    p_board.add_argument(
        "--limit", type=int, default=50, help="Max rows per group/root (default: 50). Use 0 to disable"
    )
    p_board.add_argument(
        "--once", action="store_true", help="Render once and exit (no live refresh)"
    )
    p_board.add_argument(
        "--clean-only", action="store_true", help="Only clean dangling symlinks and exit"
    )
    p_board.set_defaults(func=cmd_board)

    p_atlas = sub.add_parser("atlas", help="Run atlas in current directory")
    p_atlas.add_argument("--resources", default=f"{GLOBAL_RESOURCES}/resources.yaml")
    p_atlas.add_argument("--watch", dest="watch", action="store_true", default=True, help="Show live board while running (default)")
    p_atlas.add_argument("--no-watch", dest="watch", action="store_false", help="Disable live board display")
    p_atlas.add_argument("--interval", type=float, default=0.5, help="Watch refresh interval seconds (default: 0.5)")
    p_atlas.add_argument("--limit", type=int, default=50, help="Max rows per view (default: 50). 0 to disable")
    p_atlas.set_defaults(func=lambda a: cmd_local(a, "atlas"))

    p_qe = sub.add_parser("qe", help="Run qe in current directory")
    p_qe.add_argument("--resources", default=f"{GLOBAL_RESOURCES}/resources.yaml")
    p_qe.add_argument("--watch", dest="watch", action="store_true", default=True, help="Show live board while running (default)")
    p_qe.add_argument("--no-watch", dest="watch", action="store_false", help="Disable live board display")
    p_qe.add_argument("--interval", type=float, default=0.5, help="Watch refresh interval seconds (default: 0.5)")
    p_qe.add_argument("--limit", type=int, default=50, help="Max rows per view (default: 50). 0 to disable")
    p_qe.set_defaults(func=lambda a: cmd_local(a, "qe"))

    p_eos = sub.add_parser("eos", help="Run EOS workflow from config")
    p_eos.add_argument("config", help="Path to workflow YAML config")
    p_eos.add_argument("--resources", default=f"{GLOBAL_RESOURCES}/resources.yaml", help="Resource config YAML (default: config/resources.yaml)")
    p_eos.add_argument("--dry-run", default=False, action="store_true", help="Only generate tasks and exit")
    p_eos.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], default="INFO")
    p_eos.add_argument("--watch", dest="watch", action="store_true", default=True, help="Show live board while running (default)")
    p_eos.add_argument("--no-watch", dest="watch", action="store_false", help="Disable live board display")
    p_eos.add_argument("--interval", type=float, default=0.5, help="Watch refresh interval seconds (default: 0.5)")
    p_eos.add_argument("--limit", type=int, default=50, help="Max rows per view (default: 50). 0 to disable")
    p_eos.set_defaults(func=cmd_eos)

    p_eos_post = sub.add_parser("eos-post", help="Post-process EOS results (parse energies, fit)")
    p_eos_post.add_argument("--eos-file", default=str(Path.cwd() / "aqflow_data" / "eos.json"))
    p_eos_post.add_argument("--out-json", default=str(Path.cwd() / "aqflow_data" / "eos_post.json"))
    p_eos_post.add_argument("--out-tsv", default=str(Path.cwd() / "aqflow_data" / "eos_points.tsv"))
    p_eos_post.add_argument("--fit", choices=["none", "quad"], default="quad")
    p_eos_post.set_defaults(func=cmd_eos_post)

    args = parser.parse_args()
def cmd_eos_post(args: argparse.Namespace) -> int:
    proc = EosPostProcessor(
        eos_json=Path(args.eos_file).resolve(),
        out_json=Path(args.out_json).resolve(),
        out_tsv=Path(args.out_tsv).resolve(),
        fit=args.fit,
    )
    out = proc.run()
    print(f"eos post: wrote {proc.out_json} and {proc.out_tsv}")
    if (out.get('fit') or {}).get('vmin') is not None:
        print(f"fit vmin={out['fit']['vmin']:.6f}, emin={out['fit']['emin']:.9f} eV")
    return 0
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
