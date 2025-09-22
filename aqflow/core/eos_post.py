"""
EOS Post-Processing: parse energies from job outputs and build EOS dataset/fit.

Outputs (under aqflow_data/ by default):
- eos_post.json: structured data (points + optional quadratic fit)
- eos_points.tsv: tabular data for quick plotting (volume_scale, energy_eV, status, workdir)

No heavy deps; quadratic fit implemented via normal equations. QE energies (Ry) are
converted to eV (1 Ry = 13.605693009 eV). ATLAS energies assumed to be eV if unit
isn't present.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import re
import time

from .models import EosModel

RY_TO_EV = 13.605693009


def _read_text(path: Path) -> str:
    try:
        return Path(path).read_text(errors="ignore")
    except Exception:
        return ""


def _detect_software_from_combo(combination: str) -> Optional[str]:
    c = (combination or "").lower()
    if "qe" in c:
        return "qe"
    if "atlas" in c:
        return "atlas"
    return None


def _parse_energy_qe(text: str) -> Optional[float]:
    # Prefer line starting with '!    total energy =  xxx Ry'
    m = re.search(r"^\s*!\s*total energy\s*=\s*([-+0-9.]+)\s*(Ry|eV)?", text, re.MULTILINE)
    if not m:
        # Fallback: any 'total energy' line
        m = re.search(r"total energy\s*=\s*([-+0-9.]+)\s*(Ry|eV)?", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "Ry").strip()
    if unit.lower() == "ry":
        return val * RY_TO_EV
    return val


def _parse_energy_atlas(text: str) -> Optional[float]:
    # Try common patterns: 'Total Energy = xxx eV' or 'Total energy: xxx eV'
    m = re.search(r"Total\s+Energy\s*[:=]\s*([-+0-9.]+)\s*(eV|Ry)?", text, re.IGNORECASE)
    if not m:
        # Generic 'energy' fallback
        m = re.search(r"energy\s*[:=]\s*([-+0-9.]+)\s*(eV|Ry)?", text, re.IGNORECASE)
    if not m:
        return None
    val = float(m.group(1))
    unit = (m.group(2) or "eV").strip()
    if unit.lower() == "ry":
        return val * RY_TO_EV
    return val


def _polyfit_quadratic(xs: List[float], ys: List[float]) -> Optional[Tuple[float, float, float]]:
    # Fit y = a x^2 + b x + c using normal equations; stable enough for small N.
    n = len(xs)
    if n < 3:
        return None
    Sx = sum(xs)
    Sx2 = sum(x * x for x in xs)
    Sx3 = sum((x ** 3) for x in xs)
    Sx4 = sum((x ** 4) for x in xs)
    Sy = sum(ys)
    Sxy = sum(x * y for x, y in zip(xs, ys))
    Sx2y = sum((x * x) * y for x, y in zip(xs, ys))
    # Solve the 3x3 linear system
    # [Sx4 Sx3 Sx2][a] = [Sx2y]
    # [Sx3 Sx2 Sx ][b]   [Sxy  ]
    # [Sx2 Sx  n  ][c]   [Sy   ]
    A = [
        [Sx4, Sx3, Sx2],
        [Sx3, Sx2, Sx],
        [Sx2, Sx, n],
    ]
    B = [Sx2y, Sxy, Sy]

    def solve3(a: List[List[float]], b: List[float]) -> Optional[Tuple[float, float, float]]:
        try:
            # Gaussian elimination (no pivoting; small systems typically OK)
            a = [row[:] for row in a]
            b = b[:]
            for i in range(3):
                # Find pivot
                pivot = a[i][i]
                if abs(pivot) < 1e-12:
                    # Try to swap with a lower row
                    for j in range(i + 1, 3):
                        if abs(a[j][i]) > abs(pivot):
                            a[i], a[j] = a[j], a[i]
                            b[i], b[j] = b[j], b[i]
                            pivot = a[i][i]
                            break
                if abs(pivot) < 1e-12:
                    return None
                # Normalize row
                inv = 1.0 / pivot
                for k in range(i, 3):
                    a[i][k] *= inv
                b[i] *= inv
                # Eliminate others
                for j in range(3):
                    if j == i:
                        continue
                    factor = a[j][i]
                    if factor == 0:
                        continue
                    for k in range(i, 3):
                        a[j][k] -= factor * a[i][k]
                    b[j] -= factor * b[i]
            return (b[0], b[1], b[2])
        except Exception:
            return None

    sol = solve3(A, B)
    return sol


@dataclass
class EosPostProcessor:
    eos_json: Path = Path.cwd() / "aqflow_data" / "eos.json"
    out_json: Path = Path.cwd() / "aqflow_data" / "eos_post.json"
    out_tsv: Path = Path.cwd() / "aqflow_data" / "eos_points.tsv"
    fit: str = "quad"  # "none" | "quad"

    def run(self) -> Dict:
        raw = json.loads(Path(self.eos_json).read_text())
        model = EosModel.model_validate(raw)

        points: List[Dict] = []
        xs: List[float] = []
        ys: List[float] = []

        for t in model.tasks:
            job_out = Path(t.job_out) if t.job_out else Path(t.workdir) / "job.out"
            txt = _read_text(job_out)
            software = _detect_software_from_combo(t.combination) or "qe"
            e = _parse_energy_qe(txt) if software == "qe" else _parse_energy_atlas(txt)
            # Update in-memory model for convenience
            t.energy = e
            points.append({
                "structure": t.structure,
                "combination": t.combination,
                "volume_scale": t.volume_scale,
                "energy_eV": e,
                "status": t.status,
                "workdir": t.workdir,
            })
            if e is not None and t.status == "succeeded":
                xs.append(float(t.volume_scale))
                ys.append(float(e))

        fit_result: Dict[str, Optional[float] | str | int] = {"method": self.fit}
        if self.fit == "quad" and len(xs) >= 3:
            sol = _polyfit_quadratic(xs, ys)
            if sol:
                a, b, c = sol
                vmin = None
                emin = None
                if abs(a) > 1e-12:
                    vmin = -b / (2 * a)
                    emin = a * vmin * vmin + b * vmin + c
                fit_result.update({
                    "a": a,
                    "b": b,
                    "c": c,
                    "vmin": vmin,
                    "emin": emin,
                    "n_points": len(xs),
                })
        elif self.fit == "none":
            pass

        out_obj = {
            "meta": {
                "created_at": time.time(),
                "source": str(self.eos_json),
                "fit": self.fit,
            },
            "points": points,
            "fit": fit_result,
        }

        # Persist json (atomic write)
        self.out_json.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.out_json.with_suffix(".tmp")
        tmp.write_text(json.dumps(out_obj, indent=2))
        tmp.replace(self.out_json)

        # Persist tsv
        try:
            with open(self.out_tsv, "w") as fh:
                fh.write("volume_scale\tenergy_eV\tstatus\tworkdir\n")
                for p in points:
                    e = "" if p["energy_eV"] is None else f"{p['energy_eV']:.9f}"
                    fh.write(f"{p['volume_scale']:.6f}\t{e}\t{p['status']}\t{p['workdir']}\n")
        except Exception:
            pass

        # Also update eos.json with parsed energies
        try:
            Path(self.eos_json).write_text(EosModel.model_validate_json(model.model_dump_json()).model_dump_json(indent=2))
        except Exception:
            # best-effort; do not fail post-processing on write-back
            pass

        return out_obj

