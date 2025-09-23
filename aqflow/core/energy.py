"""
Energy Verification And Retry

Purpose:
- Provide a unified interface to read energies from task workdirs using software parsers
- If energy is missing, optionally re-queue and rerun tasks up to a maximum number of retries

Notes:
- This module does NOT hook into the executor main loop; callers (e.g., EOS) decide when to invoke it
- Unified interface: always call aqflow.software.parsers.parse_energy(software, text)
  (this module does not branch on software behavior beyond picking candidate output file names)
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

from aqflow.software.parsers import parse_energy as sw_parse_energy
from .executor import Executor, BOARD_PATH, ensure_board, save_board


# Candidate output files per software; first that exists is used
ENERGY_FILES: Dict[str, List[str]] = {
    "qe": ["job.out"],
    "atlas": ["atlas.out", "job.out"],
}


def _read_text(path: Path) -> str:
    return Path(path).read_text(errors="ignore") if Path(path).exists() else ""


def read_energy(software: str, workdir: Path) -> Optional[float]:
    """Read energy from workdir using the unified software parser.

    Returns energy in eV, or None if not found.
    """
    software = (software or "").lower()
    for fn in ENERGY_FILES.get(software, ["job.out"]):
        out = Path(workdir) / fn
        if out.exists():
            val = sw_parse_energy(software, _read_text(out))
            if val is not None:
                return float(val)
    return None


def _requeue_tasks(board_path: Path, task_ids: Iterable[str]) -> None:
    """Mark given tasks as queued in board.json (atomic update)."""
    board = ensure_board(board_path, meta={})
    tasks = board.setdefault("tasks", {})
    for tid in task_ids:
        t = tasks.get(tid)
        if not t:
            continue
        t["status"] = "queued"
        t["resource"] = None
        t["cmd"] = None
        t["start_time"] = None
        t["end_time"] = None
        t["exit_code"] = None
    save_board(board_path, board)


def ensure_energies_for_tasks(
    *,
    tasks: List[Dict],
    software_of: Dict[str, str],
    resources_yaml: Path,
    board_path: Path = BOARD_PATH,
    max_retries: int = 3,
) -> Dict[str, Optional[float]]:
    """Ensure energies exist for given tasks by optional reruns.

    Args:
      tasks: list of task dicts (must include id, workdir)
      software_of: mapping task id -> software name used to parse energy
      resources_yaml: resource config path used by Executor for reruns
      board_path: board.json path shared with Executor
      max_retries: maximum rerun attempts when energy is missing (default 3)

    Returns mapping of task id -> energy (None if still missing)
    """
    # First pass read
    energies: Dict[str, Optional[float]] = {}
    for t in tasks:
        tid = t.get("id")
        sw = software_of.get(tid, "")
        wd = Path(t.get("workdir") or ".")
        energies[tid] = read_energy(sw, wd)

    # Retry loop for missing energies
    remaining = [t.get("id") for t in tasks if energies.get(t.get("id")) is None]
    attempt = 0
    while remaining and attempt < max(0, int(max_retries)):
        attempt += 1
        _requeue_tasks(board_path, remaining)
        # Run executor once to process re-queued tasks
        ex = Executor(resources_yaml, board_path=board_path, run_meta={})
        ex.run()
        # Re-read energies for remaining tasks
        still: List[str] = []
        for tid in remaining:
            # find task workdir
            t = next((x for x in tasks if x.get("id") == tid), None)
            if not t:
                continue
            sw = software_of.get(tid, "")
            wd = Path(t.get("workdir") or ".")
            val = read_energy(sw, wd)
            energies[tid] = val
            if val is None:
                still.append(tid)
        remaining = still

    return energies

