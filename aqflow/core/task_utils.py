"""
Minimal task result type used by higher-level controllers.

Execution is handled by aqflow.core.executor; this module keeps only
lightweight dataclasses shared by callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass
class RunResult:
    task_id: str
    returncode: int
    stdout_path: Optional[Path]
    stderr_path: Optional[Path]
