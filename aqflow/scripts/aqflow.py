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
import logging

from aqflow.core.eos import EosController
from aqflow.utils.logging_config import setup_logging
from aqflow.core.executor import (
    Executor,
    BOARD_PATH,
    GLOBAL_BOARDS,
    GLOBAL_RESOURCES,
    cleanup_invalid_symlinks,
    watch_global_board,
    render_global_board_once,
    watch_single_board,
    set_board_row_limit,
)
from aqflow.core.eos_post import EosPostProcessor

 

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
    logging.getLogger("atlas-qe-workflow").info("submitted %s rc=%d", task_id, rc)
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
            if t.is_alive():
                logging.getLogger("atlas-qe-workflow").warning("board watcher did not exit within 2s; cursor may flicker briefly")
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
            if t.is_alive():
                logging.getLogger("atlas-qe-workflow").warning("board watcher did not exit within 2s; cursor may flicker briefly")
        except Exception:
            pass
        failed = [r for r in results if r.returncode != 0]
        print(f"EOS done: {len(results)-len(failed)}/{len(results)} ok")
        # Auto post-processing (parse energies, fit, plots)
        if getattr(args, "post", True):
            try:
                post = EosPostProcessor(
                    eos_json=Path.cwd() / "aqflow_data" / "eos.json",
                    out_json=Path.cwd() / "aqflow_data" / "eos_post.json",
                    out_tsv=Path.cwd() / "aqflow_data" / "eos_points.tsv",
                    fit=getattr(args, "post_fit", "quad"),
                    eos_model_name=getattr(args, "post_eos_model", "birch_murnaghan"),
                    make_plots=getattr(args, "post_plot", True),
                )
                out = post.run()
                pmg = (out.get("pmg_fit") or {})
                if pmg and not pmg.get("error"):
                    logging.getLogger("atlas-qe-workflow").info(
                        "EOS fit (pymatgen): V0=%.3f A^3, B0=%.2f GPa, E0=%.6f eV",
                        pmg.get('v0'), pmg.get('b0_GPa'), pmg.get('e0')
                    )
                if out.get("plots") and not out["plots"].get("error"):
                    logging.getLogger("atlas-qe-workflow").info(
                        "Plots saved: %s , %s",
                        out['plots'].get('abs_png',''), out['plots'].get('rel_png','')
                    )
            except Exception as e:
                logging.getLogger("atlas-qe-workflow").error("EOS post failed: %s", e)
        return 0 if not failed else 1
    except Exception as e:
        logging.getLogger("atlas-qe-workflow").error("EOS failed: %s", e)
        return 1


def cmd_eos_post(args: argparse.Namespace) -> int:
    # Ensure logs are set (default INFO to file; WARNING on console)
    setup_logging(level="INFO")
    proc = EosPostProcessor(
        eos_json=Path(args.eos_file).resolve(),
        out_json=Path(args.out_json).resolve(),
        out_tsv=Path(args.out_tsv).resolve(),
        fit=args.fit,
        eos_model_name=args.eos_model,
        make_plots=bool(getattr(args, "plot", True)),
        abs_png=Path(getattr(args, "abs_png", Path.cwd() / "aqflow_data" / "eos_curve.png")).resolve(),
        rel_png=Path(getattr(args, "rel_png", Path.cwd() / "aqflow_data" / "eos_curve_relative.png")).resolve(),
    )
    out = proc.run()
    print(f"eos post: wrote {proc.out_json} and {proc.out_tsv}")
    if (out.get('fit') or {}).get('vmin') is not None:
        print(f"fit vmin={out['fit']['vmin']:.6f}, emin={out['fit']['emin']:.9f} eV")
    return 0


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
    # Auto post-processing
    p_eos.add_argument("--no-post", dest="post", action="store_false", default=True, help="Disable EOS post-processing after run")
    p_eos.add_argument("--post", dest="post", action="store_true", help=argparse.SUPPRESS)
    p_eos.add_argument("--post-eos-model", default="birch_murnaghan")
    p_eos.add_argument("--post-fit", choices=["none", "quad"], default="quad")
    p_eos.add_argument("--no-post-plot", dest="post_plot", action="store_false", default=True, help="Disable plot generation in post")
    p_eos.add_argument("--post-plot", dest="post_plot", action="store_true", help=argparse.SUPPRESS)
    p_eos.set_defaults(func=cmd_eos)

    p_eos_post = sub.add_parser("eos-post", help="Post-process EOS results (parse energies, fit, plots)")
    p_eos_post.add_argument("--eos-file", default=str(Path.cwd() / "aqflow_data" / "eos.json"))
    p_eos_post.add_argument("--out-json", default=str(Path.cwd() / "aqflow_data" / "eos_post.json"))
    p_eos_post.add_argument("--out-tsv", default=str(Path.cwd() / "aqflow_data" / "eos_points.tsv"))
    p_eos_post.add_argument("--fit", choices=["none", "quad"], default="quad")
    p_eos_post.add_argument("--eos-model", dest="eos_model", choices=[
        "birch_murnaghan", "murnaghan", "vinet", "pourier_tarantola", "anton_schmidt", "natural_spline"
    ], default="birch_murnaghan")
    p_eos_post.add_argument("--plot", dest="plot", action="store_true", default=True, help="Generate plots (default)")
    p_eos_post.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plot generation")
    p_eos_post.add_argument("--abs-png", default=str(Path.cwd() / "aqflow_data" / "eos_curve.png"))
    p_eos_post.add_argument("--rel-png", default=str(Path.cwd() / "aqflow_data" / "eos_curve_relative.png"))
    p_eos_post.set_defaults(func=cmd_eos_post)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1
    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
