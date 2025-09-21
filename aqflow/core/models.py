"""
Pydantic models for board.json schema (meta + tasks).
"""

from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class BoardMeta(BaseModel):
    run_id: str
    root: str
    resources_file: str
    tool: str
    args: List[str] = Field(default_factory=list)
    start_time: float
    last_update: float


class TaskModel(BaseModel):
    id: str
    name: str
    type: str  # qe|atlas|custom
    workdir: str
    status: str  # created|queued|running|succeeded|failed|timeout
    resource: Optional[str] = None
    cmd: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None


class BoardModel(BaseModel):
    meta: BoardMeta
    tasks: Dict[str, TaskModel] = Field(default_factory=dict)
