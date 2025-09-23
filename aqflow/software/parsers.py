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
    """Parse ATLAS energy from job.out text.

    Fallback order:
    - "Total Energy" or generic "energy" style lines if present
    - TN iteration table: use the last row's Energy(eV/atom) and return that value (per-atom)
      Note: callers needing total energy should multiply by atom count.
    """
    # 1) Try explicit labeled energies first
    m = re.search(r"Total\s+Energy\s*[:=]\s*([-+0-9.Ee]+)\s*(eV|Ry)?", text, re.IGNORECASE)
    if not m:
        m = re.search(r"\benergy\s*[:=]\s*([-+0-9.Ee]+)\s*(eV|Ry)?", text, re.IGNORECASE)
    if m:
        val = float(m.group(1))
        unit = (m.group(2) or "eV").strip()
        return val * RY_TO_EV if unit.lower() == "ry" else val

    # 2) Parse TN table rows
    # Header example: Method    N     Energy(Ha)          Energy(eV/atom)           dE(eV/atom)
    # Row example:   TN    :     9     0.181688117400E+01    0.617998128360E+01  -0.89672E-09
    rows = re.findall(r"^[A-Za-z]+\s*:\s*\d+\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)\s+([0-9.+\-Ee]+)", text, re.MULTILINE)
    if rows:
        # Return Energy(eV/atom) from the last iteration
        epa = float(rows[-1][1])
        return epa
    return None


# ---------------- Volume ----------------

def parse_volume(software: str, workdir: Path) -> Optional[float]:
    software = (software or "").lower()
    if software == "qe":
        # Try job.out, then qe.in
        v = parse_qe_volume_from_job_out(_read_text(Path(workdir) / "job.out"))
        if v is None:
            v = parse_qe_volume_from_input(_read_text(Path(workdir) / "qe.in"))
        return v
    if software == "atlas":
        # Prefer POSCAR in workdir
        return parse_volume_from_poscar(_read_text(Path(workdir) / "POSCAR"))
    # default: try POSCAR if present
    return parse_volume_from_poscar(_read_text(Path(workdir) / "POSCAR"))


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
