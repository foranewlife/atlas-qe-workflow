"""
Pydantic models for board.json schema (meta + tasks).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Literal
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path
import re


class BoardMeta(BaseModel):
    run_id: str
    root: str
    resources_file: str
    tool: Literal["eos", "atlas", "qe", "custom"]
    args: List[str] = Field(default_factory=list)
    start_time: float
    last_update: float

    @field_validator("root")
    @classmethod
    def root_abs(cls, v: str) -> str:
        # must be absolute path
        p = Path(v)
        if not p.is_absolute():
            raise ValueError(f"root must be absolute: {v}")
        return str(p)

    @field_validator("resources_file")
    @classmethod
    def resources_abs(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError(f"resources_file must be absolute: {v}")
        return str(p)

    @model_validator(mode="after")
    def check_times(self) -> "BoardMeta":
        if self.last_update < self.start_time:
            raise ValueError("last_update must be >= start_time")
        return self


class TaskModel(BaseModel):
    id: str
    name: str
    type: Literal["qe", "atlas", "custom"]
    workdir: str
    status: Literal["created", "queued", "running", "succeeded", "failed", "timeout"]
    resource: Optional[str] = None
    cmd: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    exit_code: Optional[int] = None

    @field_validator("id")
    @classmethod
    def id_safe(cls, v: str) -> str:
        # only allow a-zA-Z0-9._- to avoid dangerous shell characters
        if not re.fullmatch(r"[A-Za-z0-9._-]+", v or ""):
            raise ValueError("task id contains invalid characters")
        return v

    @model_validator(mode="after")
    def check_time_status(self) -> "TaskModel":
        # queued/created: no start_time required
        if self.status in ("running", "succeeded", "failed", "timeout"):
            if self.start_time is None:
                raise ValueError("running/completed tasks must have start_time")
        if self.status in ("succeeded", "failed", "timeout"):
            if self.end_time is None or self.exit_code is None:
                raise ValueError("completed tasks must have end_time and exit_code")
            if self.end_time < self.start_time:  # type: ignore[arg-type]
                raise ValueError("end_time must be >= start_time")
        return self


class BoardModel(BaseModel):
    meta: BoardMeta
    tasks: Dict[str, TaskModel] = Field(default_factory=dict)

    @model_validator(mode="after")
    def check_tasks(self) -> "BoardModel":
        # keys must match task ids
        for k, t in (self.tasks or {}).items():
            if k != t.id:
                raise ValueError(f"task key '{k}' does not match id '{t.id}'")
        return self


# ----- EOS data models -----

class EosMeta(BaseModel):
    system: str
    description: str
    config_path: str  # absolute
    created_at: float
    last_update: float

    @field_validator("config_path")
    @classmethod
    def cfg_abs(cls, v: str) -> str:
        p = Path(v)
        if not p.is_absolute():
            raise ValueError("config_path must be absolute")
        return str(p)


class EosTaskEntry(BaseModel):
    id: str
    structure: str
    combination: str
    volume_scale: float
    workdir: str
    status: Literal["queued", "running", "succeeded", "failed", "timeout"]
    exit_code: Optional[int] = None
    job_out: Optional[str] = None
    energy: Optional[float] = None  # placeholder for future parsing
    # Parsed volume in Angstrom^3 (optional; filled by eos-post)
    volume: Optional[float] = None

    @field_validator("workdir")
    @classmethod
    def wd_rel_or_abs(cls, v: str) -> str:
        # allow relative paths (under results) or absolute
        return v


class EosModel(BaseModel):
    meta: EosMeta
    # Schema and units metadata to help downstream tools
    schema_version: int = 1
    units: Dict[str, str] = Field(default_factory=lambda: {"energy": "eV", "volume": "A^3"})
    run: Dict[str, str] = Field(default_factory=dict)

    tasks: List[EosTaskEntry] = Field(default_factory=list)
    # Snapshot of configuration for easier post-processing
    structures_info: Dict[str, Dict] = Field(default_factory=dict)
    combinations_info: Dict[str, Dict] = Field(default_factory=dict)
    curves_index: List[Dict] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_ids(self) -> "EosModel":
        seen = set()
        for t in self.tasks:
            if t.id in seen:
                raise ValueError(f"duplicate task id: {t.id}")
            seen.add(t.id)
        return self
