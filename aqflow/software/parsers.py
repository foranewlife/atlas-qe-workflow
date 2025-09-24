"""
Software-specific parsers for energies and volumes.

All returned energies are in eV; volumes in Angstrom^3.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import re

RY_TO_EV = 13.605693009
BOHR_TO_ANG = 0.529177210903
BOHR3_TO_A3 = BOHR_TO_ANG ** 3


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


# ---------------- Energy ----------------

def parse_energy(software: str, job_out_text: str) -> Optional[float]:
    software = (software or "").lower()
    if software == "qe":
        return parse_qe_energy(job_out_text)
    if software == "atlas":
        return parse_atlas_energy(job_out_text)
    # default: try QE-style first then ATLAS
    return parse_qe_energy(job_out_text) or parse_atlas_energy(job_out_text)


def parse_qe_energy(text: str) -> Optional[float]:
    # Prefer '!    total energy =  xxx Ry'
    m = re.search(r"^\s*!\s*total energy\s*=\s*([-+0-9.]+)\s*(Ry|eV)?", text, re.MULTILINE)
    if not m:
        m = re.search(r"total energy\s*=\s*([-+0-9.]+)\s*(Ry|eV)?", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "Ry").strip()
    return val * RY_TO_EV if unit.lower() == "ry" else val


def parse_atlas_energy(text: str) -> Optional[float]:
    """Parse ATLAS per-atom energy strictly from 'Total Energy/atom = ...' (eV/atom)."""
    m = re.search(r"Total\s+Energy/atom\s*[:=]\s*([-+0-9.Ee]+)", text, re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))


# Note: Removed ATLAS detail parser and any 'Ry' handling per user request.


# ---------------- Volume ----------------

def _count_atoms_from_poscar_text(text: str) -> Optional[int]:
    try:
        if not text:
            return None
        lines = text.splitlines()
        if len(lines) < 7:
            return None
        parts = lines[6].split()
        counts = []
        for x in parts:
            try:
                counts.append(int(float(x)))
            except Exception:
                return None
        return sum(counts) if counts else None
    except Exception:
        return None


def parse_volume(software: str, workdir: Path) -> Optional[float]:
    software = (software or "").lower()
    if software == "qe":
        # Prefer qe.in CELL_PARAMETERS (intended geometry); fallback to job.out
        v = parse_qe_volume_from_input(_read_text(Path(workdir) / "qe.in"))
        if v is None:
            v = parse_qe_volume_from_job_out(_read_text(Path(workdir) / "job.out"))
        if v is None:
            return None
        nat = _count_atoms_from_qe_in_text(_read_text(Path(workdir) / "qe.in"))
        if nat and nat > 0:
            return v / float(nat)
        return None
    if software == "atlas":
        # Per-atom volume for ATLAS: POSCAR cell volume divided by natoms
        poscar_text = _read_text(Path(workdir) / "POSCAR")
        vcell = parse_volume_from_poscar(poscar_text)
        if vcell is None:
            return None
        nat = _count_atoms_from_poscar_text(poscar_text)
        if nat and nat > 0:
            return vcell / float(nat)
        return None
    # default: try POSCAR if present
    return parse_volume_from_poscar(_read_text(Path(workdir) / "POSCAR"))


def _count_atoms_from_qe_in_text(text: str) -> Optional[int]:
    try:
        if not text:
            return None
        lines = text.splitlines()
        start = None
        for i, ln in enumerate(lines):
            if ln.strip().lower().startswith("atomic_positions"):
                start = i + 1
                break
        if start is None:
            return None
        n = 0
        for ln in lines[start:]:
            s = ln.strip()
            if not s:
                break
            parts = s.split()
            if len(parts) < 4:
                break
            try:
                float(parts[-1]); float(parts[-2]); float(parts[-3])
            except Exception:
                break
            n += 1
        return n if n > 0 else None
    except Exception:
        return None


def parse_qe_volume_from_job_out(text: str) -> Optional[float]:
    # Quantum ESPRESSO often prints: unit-cell volume = xxx (a.u.)^3 or (Ang^3)
    m = re.search(r"unit-?cell volume\s*=\s*([-+0-9.]+)\s*\(([^)]+)\)\^?3", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = m.group(2).lower()
    if "ang" in unit:
        return val
    # assume atomic units
    return val * BOHR3_TO_A3


def parse_qe_volume_from_input(text: str) -> Optional[float]:
    # Parse CELL_PARAMETERS angstrom followed by 3 lines
    if not text:
        return None
    lines = text.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip().lower().startswith("cell_parameters"):
            # Detect unit
            unit = "angstrom"
            parts = ln.strip().split()
            if len(parts) >= 2:
                unit = parts[1].lower()
            try:
                a = [float(x) for x in lines[i + 1].split()[:3]]
                b = [float(x) for x in lines[i + 2].split()[:3]]
                c = [float(x) for x in lines[i + 3].split()[:3]]
            except Exception:
                return None
            import numpy as np
            lat = np.vstack([a, b, c])
            vol = float(abs(np.linalg.det(lat)))
            if unit.startswith("ang"):
                return vol
            if unit in ("bohr", "a.u.", "au"):
                return vol * BOHR3_TO_A3
            return vol  # fallback
    return None


def parse_volume_from_poscar(poscar_content: str) -> Optional[float]:
    try:
        if not poscar_content:
            return None
        lines = [ln.strip() for ln in poscar_content.splitlines() if ln.strip()]
        if len(lines) < 5:
            return None
        scale = float(lines[1])
        import numpy as np
        a = np.fromstring(lines[2], sep=" ")[:3]
        b = np.fromstring(lines[3], sep=" ")[:3]
        c = np.fromstring(lines[4], sep=" ")[:3]
        lat = np.vstack([a, b, c]) * scale
        vol = float(abs(np.linalg.det(lat)))
        return vol
    except Exception:
        return None
