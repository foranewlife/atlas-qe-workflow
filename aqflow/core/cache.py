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
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


INPUT_PATTERNS = [
    "qe.in",
    "job.in",
    "atlas.in",
    "*.in",
    "POSCAR",
    "KPOINTS",
    "*.UPF",
    "*.POTCAR",
]

# TTL caches for binary signatures to avoid repeated expensive calls during a run
_BIN_SIG_TTL = 1.0  # seconds
_BIN_SIG_LOCAL: Dict[str, Tuple[Optional[Dict], float, Optional[int]]] = {}
_BIN_SIG_REMOTE: Dict[Tuple[str, str], Tuple[Optional[int], float]] = {}


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
            # Do NOT include mtime in inputs_manifest per design (reduce false misses)
            entry = {
                "path": str(rel),
                "size": int(stat.st_size),
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


def _fast_scan_inputs(workdir: Path) -> Tuple[List[Dict], Dict[str, int]]:
    """Fast non-recursive scan to collect candidate input files without hashing.

    Returns list of dicts with keys: path, size, mtime, kind; and a kinds histogram.
    """
    wd = Path(workdir)
    files: List[Dict] = []
    kinds: Dict[str, int] = {}
    try:
        for entry in os.scandir(wd):
            if not entry.is_file():
                continue
            name = entry.name
            lname = name.lower()
            if not (
                lname in ("qe.in", "job.in", "atlas.in", "poscar", "kpoints")
                or lname.endswith(".upf")
                or lname.endswith(".potcar")
                or lname.endswith(".in")
            ):
                continue
            try:
                st = entry.stat()
            except FileNotFoundError:
                continue
            kind = (
                "qe.in"
                if lname == "qe.in"
                else "job.in"
                if lname == "job.in"
                else "atlas.in"
                if lname == "atlas.in"
                else "poscar"
                if lname == "poscar"
                else "kpoints"
                if lname == "kpoints"
                else "upf"
                if lname.endswith(".upf")
                else "potcar"
                if lname.endswith(".potcar")
                else "in"
            )
            files.append(
                {
                    "path": name,
                    "size": int(st.st_size),
                    "mtime": int(st.st_mtime),
                    "kind": kind,
                }
            )
            kinds[kind] = kinds.get(kind, 0) + 1
    except FileNotFoundError:
        pass
    files.sort(key=lambda d: d["path"])  # stable ordering
    return files, kinds


def _local_bin_signature(path: str) -> Optional[Dict]:
    try:
        now = time.time()
        p = Path(path)
        st = p.stat()
        mtime = int(st.st_mtime)
        cached = _BIN_SIG_LOCAL.get(str(p))
        if cached and cached[2] == mtime and (now - cached[1]) < _BIN_SIG_TTL:
            return cached[0]
        sha = _sha256_file(p)
        sig = {"path": str(p), "size": int(st.st_size), "mtime": mtime, "sha256": sha}
        _BIN_SIG_LOCAL[str(p)] = (sig, now, mtime)
        return sig
    except Exception:
        return None


def _ssh_get_mtime(host: str, path: str, timeout: int = 5) -> Optional[int]:
    # Compatible across GNU/BSD stat; fallback to python
    cmd = (
        f"ssh -o BatchMode=yes -o ConnectTimeout={int(timeout)} {host} "
        f"'stat -c %Y {path} 2>/dev/null || stat -f %m {path} 2>/dev/null || "
        f"python3 -c \"import os;print(int(os.path.getmtime(r'{path}')))\"'"
    )
    try:
        out = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL, timeout=timeout + 2)
        s = out.decode().strip().splitlines()[-1].strip()
        return int(s)
    except Exception:
        return None


def _remote_bin_signature(host: str, path: str, resource: str) -> Optional[Dict]:
    now = time.time()
    key = (host, path)
    cached = _BIN_SIG_REMOTE.get(key)
    if cached and (now - cached[1]) < _BIN_SIG_TTL:
        mt = cached[0]
    else:
        mt = _ssh_get_mtime(host, path)
        _BIN_SIG_REMOTE[key] = (mt, now)
    if mt is None:
        return None
    return {"resource": resource, "path": path, "mtime": int(mt)}


def _compute_key(software: str, bin_sig: Dict, run_cmd: str, inputs: List[Dict]) -> str:
    # Fixed-field JSON for speed (inputs must be pre-sorted by path)
    obj = {"software": software, "binary": bin_sig, "cmd": run_cmd, "inputs": inputs}
    raw = json.dumps(obj, separators=(",", ":")).encode()
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

def probe_cache(
    *,
    software: str,
    bin_path: str,
    run_cmd: str,
    workdir: Path,
    resource: Optional[Dict],
) -> CacheProbe:
    """Build current signature and compare with existing .aqcache.json with early exits.

    Steps:
    1) If no cache file, return miss immediately.
    2) Fast scan to compute file kinds (no sha); mismatch -> miss。
    3) Compute binary signature (TTL cache) and only hash changed files when possible.
    """
    if (Path(workdir) / ".aqcache.json").exists():
        return CacheProbe(True, None, None, [], {}, None)
    else:
        return CacheProbe(False, None, None, [], {}, None)

    cache = _read_cache(workdir)
    # Optional fast-path: if configured to assume hit when cache file exists,
    # return a synthetic hit without computing binary signature or hashing inputs.
    try:
        assume_hit = bool(((resource or {}).get("cache") or {}).get("assume_hit_on_presence", True))
    except Exception:
        assume_hit = True
    if cache and assume_hit:
        return CacheProbe(True, cache.get("key"), cache, [], cache.get("file_kinds") or {}, None)
    if not cache:
        return CacheProbe(False, None, None, [], {}, None)

    quick_files, kinds = _fast_scan_inputs(workdir)
    if not quick_files:
        return CacheProbe(False, None, cache, [], kinds, None)

    if (cache.get("file_kinds") or {}) != kinds:
        return CacheProbe(False, None, cache, [], kinds, None)
    logger.info(f"Cache kinds match: {kinds}")
    if resource and (resource.get("type") == "remote"):
        host = ((resource.get("user") + "@") if resource.get("user") else "") + (resource.get("host") or "")
        logger.info(f"Probing remote binary signature on {host}:{bin_path}")
        bin_sig = _remote_bin_signature(host, bin_path, resource.get("name") or "")
    else:
        bin_sig = _local_bin_signature(bin_path)
    if not bin_sig:
        return CacheProbe(False, None, cache, [], kinds, None)
    logger.info(f"Cache binary signature match: {bin_sig}")
    prev_manifest = {d.get("path"): d for d in (cache.get("inputs_manifest") or [])}
    prev_mtimes: Dict[str, int] = (cache.get("inputs_mtime") or {})
    inputs: List[Dict] = []
    wd = Path(workdir)
    for f in quick_files:
        rel = f["path"]
        size = int(f["size"])
        mtime = int(f["mtime"])
        sha: Optional[str]
        if prev_mtimes and rel in prev_mtimes and rel in prev_manifest:
            prev = prev_manifest[rel]
            if int(prev_mtimes.get(rel, -1)) == mtime and int(prev.get("size", -1)) == size:
                sha = prev.get("sha256")
            else:
                sha = _sha256_file(wd / rel)
        else:
            # Old cache without mtimes: compute sha for safety
            sha = _sha256_file(wd / rel)
        inputs.append({"path": rel, "size": size, "sha256": sha})

    inputs.sort(key=lambda d: d["path"])  # stable
    key = _compute_key(software, bin_sig, run_cmd, inputs)
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
    # Collect mtimes for acceleration in future probes (compat with old caches)
    mtimes: Dict[str, int] = {}
    for it in inputs:
        try:
            st = (Path(workdir) / it["path"]).stat()
            mtimes[it["path"]] = int(st.st_mtime)
        except Exception:
            continue
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
        "inputs_mtime": mtimes,
    }
    _write_cache(workdir, record)
