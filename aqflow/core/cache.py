"""
Simple per-workdir cache for Executor.

Design:
- Each task workdir may contain a '.aqcache.json' describing the last successful run.
- Cache key = sha256(software, binary_signature, run_cmd, inputs_manifest)
- Hit rule = key matches AND file_kinds (input kinds counts) equal (guard against type count drift)
- On hit: caller may mark task succeeded without executing binary.
- On success: write/update .aqcache.json with latest key and metadata.

Remote binaries: use ssh to read mtime; local uses stat + sha256.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


INPUT_PATTERNS = [
    "qe.in", "job.in", "atlas.in", "*.in",
    "POSCAR", "KPOINTS", "*.UPF", "*.POTCAR",
]


def _sha256_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _collect_inputs(workdir: Path) -> Tuple[List[Dict], Dict[str, int]]:
    wd = Path(workdir)
    files: List[Dict] = []
    kinds: Dict[str, int] = {}
    matched: set[Path] = set()
    for pat in INPUT_PATTERNS:
        for p in wd.glob(pat):
            if not p.is_file() or p in matched:
                continue
            rel = p.relative_to(wd)
            stat = p.stat()
            sha = _sha256_file(p)
            entry = {
                "path": str(rel),
                "size": int(stat.st_size),
                "mtime": int(stat.st_mtime),
                "sha256": sha,
            }
            files.append(entry)
            matched.add(p)
            # classify kinds
            name = p.name.lower()
            kind = (
                "qe.in" if name == "qe.in" else
                "job.in" if name == "job.in" else
                "atlas.in" if name == "atlas.in" else
                "poscar" if name == "poscar" else
                "kpoints" if name == "kpoints" else
                "upf" if name.endswith(".upf") else
                "potcar" if name.endswith(".potcar") else
                "in" if name.endswith(".in") else
                "other"
            )
            kinds[kind] = kinds.get(kind, 0) + 1
    files.sort(key=lambda d: d["path"])  # stable ordering for hashing
    return files, kinds


def _local_bin_signature(path: str) -> Optional[Dict]:
    try:
        p = Path(path)
        st = p.stat()
        sha = _sha256_file(p)
        return {"path": str(p), "size": int(st.st_size), "mtime": int(st.st_mtime), "sha256": sha}
    except Exception:
        return None


def _ssh_get_mtime(host: str, path: str, timeout: int = 5) -> Optional[int]:
    # Compatible across GNU/BSD stat; fallback to python
    cmd = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={int(timeout)} {shlex.quote(host)} "
        f"'stat -c %Y {shlex.quote(path)} 2>/dev/null || stat -f %m {shlex.quote(path)} 2>/dev/null || "
        f"python3 -c "\"import os;print(int(os.path.getmtime(r\"{path}\")))\""'"
    )
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout + 2)
        s = out.decode().strip().splitlines()[-1].strip()
        return int(s)
    except Exception:
        return None


def _remote_bin_signature(host: str, path: str, resource: str) -> Optional[Dict]:
    mt = _ssh_get_mtime(host, path)
    if mt is None:
        return None
    return {"resource": resource, "path": path, "mtime": int(mt)}


def _compute_key(software: str, bin_sig: Dict, run_cmd: str, inputs: List[Dict]) -> str:
    obj = {
        "software": software,
        "binary": bin_sig,
        "cmd": run_cmd,
        "inputs": inputs,
    }
    raw = json.dumps(obj, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _read_cache(workdir: Path) -> Optional[Dict]:
    p = Path(workdir) / ".aqcache.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _write_cache(workdir: Path, record: Dict) -> None:
    p = Path(workdir) / ".aqcache.json"
    try:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(record, indent=2))
        tmp.replace(p)
    except Exception:
        pass


@dataclass
class CacheProbe:
    hit: bool
    key: Optional[str]
    record: Optional[Dict]
    inputs: List[Dict]
    kinds: Dict[str, int]
    bin_sig: Optional[Dict]


def probe_cache(*, software: str, bin_path: str, run_cmd: str, workdir: Path, resource: Optional[Dict]) -> CacheProbe:
    """Build current signature and compare with existing .aqcache.json.

    Returns a CacheProbe with hit flag. Does not modify files.
    """
    inputs, kinds = _collect_inputs(workdir)
    if not inputs:
        return CacheProbe(False, None, None, inputs, kinds, None)
    bin_sig: Optional[Dict] = None
    if resource and (resource.get("type") == "remote"):
        host = ((resource.get("user") + "@") if resource.get("user") else "") + (resource.get("host") or "")
        rec = _remote_bin_signature(host, bin_path, resource.get("name") or "")
        bin_sig = rec
    else:
        bin_sig = _local_bin_signature(bin_path)
    if not bin_sig:
        return CacheProbe(False, None, None, inputs, kinds, None)
    key = _compute_key(software, bin_sig, run_cmd, inputs)
    cache = _read_cache(workdir)
    if not cache:
        return CacheProbe(False, key, None, inputs, kinds, bin_sig)
    # file kinds must match
    if (cache.get("file_kinds") or {}) != kinds:
        return CacheProbe(False, key, cache, inputs, kinds, bin_sig)
    if cache.get("key") == key:
        return CacheProbe(True, key, cache, inputs, kinds, bin_sig)
    return CacheProbe(False, key, cache, inputs, kinds, bin_sig)


def write_success_cache(*, software: str, bin_path: str, run_cmd: str, workdir: Path, resource: Optional[Dict], energy_eV: Optional[float]) -> None:
    inputs, kinds = _collect_inputs(workdir)
    if not inputs:
        return
    if resource and (resource.get("type") == "remote"):
        host = ((resource.get("user") + "@") if resource.get("user") else "") + (resource.get("host") or "")
        bin_sig = _remote_bin_signature(host, bin_path, resource.get("name") or "")
    else:
        bin_sig = _local_bin_signature(bin_path)
    if not bin_sig:
        return
    key = _compute_key(software, bin_sig, run_cmd, inputs)
    now = time.time()
    record = {
        "version": 1,
        "software": software,
        "binary_signature": bin_sig,
        "run_cmd": run_cmd,
        "inputs_manifest": inputs,
        "file_kinds": kinds,
        "key": key,
        "energy_eV": energy_eV,
        "created_at": now,
        "updated_at": now,
        "hits": 1,
    }
    _write_cache(workdir, record)

