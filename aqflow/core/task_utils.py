"""
Minimal task result model used by higher-level controllers.

Execution is handled by aqflow.core.executor; this module keeps only
lightweight Pydantic BaseModel shared by callers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from pydantic import BaseModel


class RunResult(BaseModel):
    task_id: str
    returncode: int
    stdout_path: Optional[Path] = None
    stderr_path: Optional[Path] = None
