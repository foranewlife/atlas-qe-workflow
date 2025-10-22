#!/usr/bin/env python3
"""
Parameter fitting driver for ATLAS vs QE EOS targets (standalone).

Stages (iterative loop until convergence):
- Stage 0: Extract QE targets (V0_ref, B0_ref) for selected structures using a named QE
  combination from the base YAML (e.g., qe_ecut60). Will run `aqflow eos` if needed.
- Stage 1: Generate a neighborhood grid of ATLAS parameters around a center with steps,
  using the selected structures from the base YAML. Run `aqflow eos` to evaluate.
- Stage 2: Evaluate grid, compute aggregate score across structures, pick best -> update
  center, shrink steps. Then loop back to Stage 1 until improvement < threshold.

Score per structure s:
  dv_pct = 100 * abs((V0_s - V0_ref_s) / V0_ref_s)
  db_pct = 100 * abs((B0_s - B0_ref_s) / B0_ref_s)
  score_s = 10 * dv_pct + 1 * db_pct
Aggregate score = average of score_s across selected structures with valid fits.

Outputs per iteration under an experiment directory:
- Derived YAML that was run
- Archived eos_post.json
- Summary TSV of evaluated points and metrics
- Plotly HTML (and PNG if kaleido available) visualizations for progress

This script is standalone and does not modify core modules.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
try:
    from aqflow.core.configuration import ConfigurationLoader  # type: ignore
except Exception:
    ConfigurationLoader = None  # fallback if not available


# ----------------------- Data models and config parsing -----------------------


@dataclass
class ParamCenter:
    paraA: float
    Gammax: float
    rHC_A: float
    rHC_B: float


@dataclass
class ParamStep:
    paraA: float
    Gammax: float
    rHC_A: float
    rHC_B: float


@dataclass
class Bounds:
    paraA: Tuple[float, float] | None = None
    Gammax: Tuple[float, float] | None = None
    rHC_A: Tuple[float, float] | None = None
    rHC_B: Tuple[float, float] | None = None


@dataclass
class OptimConfig:
    base_yaml: Path
    qe_ref_name: str
    structures: List[str] | None  # None means use all from base_yaml
    atlas_template: str
    pseudopotential_set: str
    params_center: ParamCenter
    params_step: ParamStep
    neighborhood: str  # "full_grid" | "axis_cross"
    shrink_factor: float
    weight_v0: float
    weight_b0: float
    aggregate: str  # "mean" | "median"
    optimize_params: Tuple[str, ...]
    min_improvement: float
    max_iterations: int
    resources: Optional[Path]
    outdir: Path
    save_png: bool
    save_html: bool


def _cfg_tuple(dct: Dict[str, Any] | None) -> Tuple[float, float] | None:
    if not dct:
        return None
    return float(dct[0]), float(dct[1])  # type: ignore[index]


ALL_PARAM_NAMES: Tuple[str, ...] = ("paraA", "Gammax", "rHC_A", "rHC_B")


def _normalize_optimize_params(raw: Any) -> Tuple[str, ...]:
    if raw is None:
        return ALL_PARAM_NAMES
    allowed = {name.lower(): name for name in ALL_PARAM_NAMES}
    items: List[str] = []
    if isinstance(raw, str):
        text = raw.strip()
        if not text or text.lower() == "all":
            return ALL_PARAM_NAMES
        items = [seg for seg in text.replace(",", " ").split() if seg]
    else:
        for entry in raw:
            if isinstance(entry, str):
                parts = entry.replace(",", " ").split()
            else:
                parts = [str(entry)]
            for part in parts:
                part = part.strip()
                if part:
                    items.append(part)
    seen: set[str] = set()
    normalized: List[str] = []
    for item in items:
        key = item.lower()
        if key not in allowed:
            raise ValueError(f"Unsupported optimize param '{item}'. Allowed: {', '.join(ALL_PARAM_NAMES)}")
        name = allowed[key]
        if name not in seen:
            seen.add(name)
            normalized.append(name)
    if not normalized:
        return ALL_PARAM_NAMES
    return tuple(normalized)


def load_optim_config(path: Path) -> OptimConfig:
    raw = yaml.safe_load(Path(path).read_text())
    params = raw.get("params", {})
    scoring = raw.get("scoring", {})
    stopping = raw.get("stopping", {})
    runtime = raw.get("runtime", {})
    plotting = raw.get("plotting", {})
    return OptimConfig(
        base_yaml=Path(raw["base_yaml"]).resolve(),
        qe_ref_name=str(raw["qe_ref_name"]).strip(),
        structures=(raw.get("structures") if raw.get("structures") != "all" else None),
        atlas_template=str(params.get("atlas_template") or "atlas.in.oftau.parameters.template"),
        pseudopotential_set=str(params.get("pseudopotential_set") or "C"),
        params_center=ParamCenter(
            paraA=float(params["center"]["paraA"]),
            Gammax=float(params["center"]["Gammax"]),
            rHC_A=float(params["center"]["rHC_A"]),
            rHC_B=float(params["center"]["rHC_B"]),
        ),
        params_step=ParamStep(
            paraA=float(params["step"]["paraA"]),
            Gammax=float(params["step"]["Gammax"]),
            rHC_A=float(params["step"]["rHC_A"]),
            rHC_B=float(params["step"]["rHC_B"]),
        ),
        neighborhood=str(params.get("neighborhood") or "axis_cross"),
        shrink_factor=float(params.get("shrink_factor", 0.5)),
        weight_v0=float(scoring.get("weight_v0", 10.0)),
        weight_b0=float(scoring.get("weight_b0", 1.0)),
        aggregate=str(scoring.get("aggregate", "mean")),
        optimize_params=_normalize_optimize_params(params.get("optimize")),
        min_improvement=float(stopping.get("min_improvement", 0.5)),
        max_iterations=int(stopping.get("max_iterations", 8)),
        resources=(Path(runtime["resources"]).resolve() if runtime.get("resources") else None),
        outdir=Path(runtime.get("outdir") or (Path.cwd() / "results" / "param_fit")).resolve(),
        save_png=bool(plotting.get("save_png", True)),
        save_html=bool(plotting.get("save_html", True)),
    )


# ---------------------------- YAML helpers (in-place) ----------------------------


def load_base_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def select_structures(base: Dict[str, Any], names: Optional[List[str]]) -> List[Dict[str, Any]]:
    structs: List[Dict[str, Any]] = base.get("structures") or []
    if not names:
        return structs
    wanted = set(names)
    return [s for s in structs if s.get("name") in wanted]

def append_combos_inplace(yaml_path: Path, combos_to_add: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Append new parameter_combinations in-place at the end of the YAML.

    Skips any combo whose name already exists. Returns the list of combos
    actually appended.
    """
    data = load_base_yaml(yaml_path)
    pcs: List[Dict[str, Any]] = data.get("parameter_combinations") or []
    existing = {c.get("name") for c in pcs}
    added: List[Dict[str, Any]] = []
    for c in combos_to_add:
        nm = c.get("name")
        if nm in existing:
            continue
        pcs.append(c)
        added.append(c)
        existing.add(nm)
    data["parameter_combinations"] = pcs
    yaml_path.write_text(yaml.safe_dump(data, sort_keys=False))
    return added


def make_qe_only_combos(base: Dict[str, Any], qe_ref_name: str, structures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    pcs: List[Dict[str, Any]] = base.get("parameter_combinations") or []
    ref = next((c for c in pcs if c.get("name") == qe_ref_name), None)
    if not ref:
        raise ValueError(f"QE combo '{qe_ref_name}' not found in base YAML")
    out = [{
        "name": ref.get("name"),
        "software": ref.get("software"),
        "template": ref.get("template") or ref.get("template_file"),
        "applies_to_structures": [s.get("name") for s in structures],
        "pseudopotential_set": ref.get("pseudopotential_set"),
        "template_substitutions": ref.get("template_substitutions", {}),
    }]
    return out


def _clamp(x: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None:
        x = max(x, lo)
    if hi is not None:
        x = min(x, hi)
    return x


def build_grid(
    center: ParamCenter,
    step: ParamStep,
    bounds: Optional[Bounds],
    neighborhood: str,
    active_params: Iterable[str],
) -> List[Tuple[float, float, float, float]]:
    """Return list of parameter tuples (paraA, Gammax, rHC_A, rHC_B)."""
    c = center
    st = step
    active = {name for name in active_params}
    base = (c.paraA, c.Gammax, c.rHC_A, c.rHC_B)
    pts: List[Tuple[float, float, float, float]] = []
    if neighborhood == "full_grid":
        def _grid_vals(name: str, center_val: float, step_val: float) -> List[float]:
            if name in active and step_val != 0:
                return [center_val - step_val, center_val, center_val + step_val]
            return [center_val]

        vals = [
            _grid_vals("paraA", c.paraA, st.paraA),
            _grid_vals("Gammax", c.Gammax, st.Gammax),
            _grid_vals("rHC_A", c.rHC_A, st.rHC_A),
            _grid_vals("rHC_B", c.rHC_B, st.rHC_B),
        ]
        for a in vals[0]:
            for g in vals[1]:
                for ra in vals[2]:
                    for rb in vals[3]:
                        pts.append((a, g, ra, rb))
    else:  # axis_cross (default)
        pts = [base]
        if "paraA" in active and st.paraA != 0:
            pts.append((c.paraA - st.paraA, c.Gammax, c.rHC_A, c.rHC_B))
            pts.append((c.paraA + st.paraA, c.Gammax, c.rHC_A, c.rHC_B))
        if "Gammax" in active and st.Gammax != 0:
            pts.append((c.paraA, c.Gammax - st.Gammax, c.rHC_A, c.rHC_B))
            pts.append((c.paraA, c.Gammax + st.Gammax, c.rHC_A, c.rHC_B))
        if "rHC_A" in active and st.rHC_A != 0:
            pts.append((c.paraA, c.Gammax, c.rHC_A - st.rHC_A, c.rHC_B))
            pts.append((c.paraA, c.Gammax, c.rHC_A + st.rHC_A, c.rHC_B))
        if "rHC_B" in active and st.rHC_B != 0:
            pts.append((c.paraA, c.Gammax, c.rHC_A, c.rHC_B - st.rHC_B))
            pts.append((c.paraA, c.Gammax, c.rHC_A, c.rHC_B + st.rHC_B))
        if len(pts) == 1:
            pts = [base]
    # Clamp within bounds if provided
    out: List[Tuple[float, float, float, float]] = []
    for a, g, ra, rb in pts:
        aa = _clamp(a, bounds.paraA[0] if bounds and bounds.paraA else None, bounds.paraA[1] if bounds and bounds.paraA else None)
        gg = _clamp(g, bounds.Gammax[0] if bounds and bounds.Gammax else None, bounds.Gammax[1] if bounds and bounds.Gammax else None)
        rra = _clamp(ra, bounds.rHC_A[0] if bounds and bounds.rHC_A else None, bounds.rHC_A[1] if bounds and bounds.rHC_A else None)
        rrb = _clamp(rb, bounds.rHC_B[0] if bounds and bounds.rHC_B else None, bounds.rHC_B[1] if bounds and bounds.rHC_B else None)
        out.append((aa, gg, rra, rrb))
    # Deduplicate (float tuples) while preserving order
    seen = set()
    uniq: List[Tuple[float, float, float, float]] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return uniq


def make_atlas_combos(
    params_list: List[Tuple[float, float, float, float]],
    applies_to: List[str],
    atlas_template: str,
    pseudopotential_set: str,
) -> List[Dict[str, Any]]:
    combos: List[Dict[str, Any]] = []
    for a, g, ra, rb in params_list:
        name = f"atlas_{a:.3f}_{g:.3f}_{ra:.3f}_{rb:.3f}"
        combos.append({
            "name": name,
            "software": "atlas",
            "template": atlas_template,
            "applies_to_structures": applies_to,
            "pseudopotential_set": pseudopotential_set,
            "template_substitutions": {
                "paraA": float(a),
                "Gammax": float(g),
                "rHC_A": float(ra),
                "rHC_B": float(rb),
            },
        })
    return combos


# ------------------------------ EOS parsing & score ------------------------------


def _load_eos_post(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text())


def extract_ref_targets(eos_post: Dict[str, Any], structures: List[str], ref_name: Optional[str] = None) -> Dict[str, Tuple[float, float]]:
    """Return mapping structure -> (V0_ref, B0_ref)."""
    targets: Dict[str, Tuple[float, float]] = {}
    for c in eos_post.get("curves", []):
        s = c.get("structure")
        if s not in structures:
            continue
        sw = (c.get("software") or "").lower()
        name = (c.get("combination") or "")
        if sw != "qe" and ("qe" not in name.lower()):
            continue
        if ref_name and name != ref_name:
            continue
        pmg = c.get("pmg_fit") or {}
        if pmg and not pmg.get("error") and (pmg.get("v0") is not None) and (pmg.get("b0_GPa") is not None):
            try:
                targets[s] = (float(pmg["v0"]), float(pmg["b0_GPa"]))
            except Exception:
                pass
    return targets


def extract_atlas_fits(eos_post: Dict[str, Any], structures: List[str]) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Return mapping structure -> { combo_name: (V0, B0_GPa) } for ATLAS curves."""
    out: Dict[str, Dict[str, Tuple[float, float]]] = {s: {} for s in structures}
    for c in eos_post.get("curves", []):
        s = c.get("structure")
        if s not in structures:
            continue
        sw = (c.get("software") or "").lower()
        if sw != "atlas":
            continue
        name = c.get("combination") or c.get("key") or "atlas"
        pmg = c.get("pmg_fit") or {}
        if pmg and not pmg.get("error") and (pmg.get("v0") is not None) and (pmg.get("b0_GPa") is not None):
            try:
                out[s][name] = (float(pmg["v0"]), float(pmg["b0_GPa"]))
            except Exception:
                pass
    return out


def compute_scores(
    ref: Dict[str, Tuple[float, float]],
    atlas: Dict[str, Dict[str, Tuple[float, float]]],
    weight_v0: float,
    weight_b0: float,
) -> Dict[str, Dict[str, Any]]:
    """Return mapping combo_name -> { per-structure metrics, aggregate score }.

    Per-structure metrics: dv_pct, db_pct, V0, B0, and score_s.
    Aggregate is mean over available structures.
    """
    combos: Dict[str, Dict[str, Any]] = {}
    for s, ref_vals in ref.items():
        v0_ref, b0_ref = ref_vals
        for combo_name, vals in (atlas.get(s) or {}).items():
            v0, b0 = vals
            try:
                dv_pct = abs((v0 - v0_ref) / v0_ref) * 100.0
                db_pct = abs((b0 - b0_ref) / b0_ref) * 100.0
                score_s = weight_v0 * dv_pct + weight_b0 * db_pct
            except Exception:
                continue
            rec = combos.setdefault(combo_name, {"by_structure": {}, "scores": []})
            rec["by_structure"][s] = {
                "V0": v0,
                "B0_GPa": b0,
                "dv_pct": dv_pct,
                "db_pct": db_pct,
                "score_s": score_s,
            }
            rec["scores"].append(score_s)
    # Aggregate
    for combo_name, rec in combos.items():
        scores = rec.get("scores", [])
        if scores:
            rec["score"] = float(sum(scores) / len(scores))
        else:
            rec["score"] = float("inf")
    return combos


# ----------------------------------- plotting -----------------------------------


def _try_write_plotly(fig, path_html: Optional[Path], path_png: Optional[Path]) -> None:
    try:
        import plotly.io as pio
    except Exception:
        return
    if path_html:
        try:
            pio.write_html(fig, file=str(path_html), full_html=True, include_plotlyjs="inline")
        except Exception:
            pass
    if path_png:
        try:
            pio.write_image(fig, str(path_png), scale=2)  # requires kaleido installed
        except Exception:
            pass


def plot_iteration(
    iterdir: Path,
    grid_points: List[Tuple[float, float, float, float]],
    combos_scores: Dict[str, Dict[str, Any]],
    structures: List[str],
    *,
    save_png: bool,
    save_html: bool,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    # Scatter of dv_pct vs db_pct colored by aggregate score (one point per combo per structure)
    xs: List[float] = []
    ys: List[float] = []
    cols: List[float] = []
    texts: List[str] = []
    marker_symbols: List[str] = []
    symbol_sequence = [
        "circle",
        "square",
        "diamond",
        "x",
        "triangle-up",
        "triangle-down",
        "triangle-left",
        "triangle-right",
        "star",
        "hexagon",
    ]
    if len(structures) > len(symbol_sequence):
        print(
            f"plot_iteration: more than {len(symbol_sequence)} structures; markers will repeat.",
            file=sys.stderr,
        )
    symbol_map = {s: symbol_sequence[idx % len(symbol_sequence)] for idx, s in enumerate(structures)}
    best_combo_name = None
    best_combo_score = float("inf")
    for combo, rec in combos_scores.items():
        combo_score = float(rec.get("score", float("inf")))
        if combo_score < best_combo_score:
            best_combo_score = combo_score
            best_combo_name = combo
        agg = float(rec.get("score", float("inf")))
        for s in structures:
            m = (rec.get("by_structure") or {}).get(s)
            if not m:
                continue
            xs.append(float(m["dv_pct"]))
            ys.append(float(m["db_pct"]))
            cols.append(agg)
            texts.append(f"{combo}<br>{s}")
            marker_symbols.append(symbol_map[s])
    best_points: List[Tuple[float, float, str, str, float]] = []
    if best_combo_name:
        best_rec = combos_scores.get(best_combo_name, {})
        bys = best_rec.get("by_structure") or {}
        for s in structures:
            m = bys.get(s)
            if not m:
                continue
            best_points.append(
                (
                    float(m["dv_pct"]),
                    float(m["db_pct"]),
                    best_combo_name,
                    s,
                    float(m.get("score_s", float("inf"))),
                )
            )
    fig_sc = go.Figure(
        data=[
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    color=cols,
                    colorscale="Viridis",
                    showscale=True,
                    size=9,
                    symbol=marker_symbols,
                ),
                text=texts,
                hovertemplate="dv%%=%{x:.2f}%%<br>db%%=%{y:.2f}%%<br>%{text}<extra>score</extra>",
            )
        ]
    )
    if best_points:
        best_xs = [p[0] for p in best_points]
        best_ys = [p[1] for p in best_points]
        best_combo = best_points[0][2]
        best_texts = [f"{best_combo}<br>{p[3]}" for p in best_points]
        best_scores = [p[4] for p in best_points]
        best_symbols = [symbol_map[p[3]] for p in best_points]
        fig_sc.add_trace(
            go.Scatter(
                x=best_xs,
                y=best_ys,
                mode="markers+text",
                marker=dict(
                    size=16,
                    color="rgba(220,20,60,0.9)",
                    line=dict(color="black", width=2),
                    symbol=best_symbols,
                ),
                text=best_texts,
                textposition="top center",
                hovertemplate=(
                    "Best combo<br>"
                    "dv%%=%{x:.2f}%%<br>"
                    "db%%=%{y:.2f}%%<br>"
                    "%{text}<br>"
                    "score_s=%{customdata:.6f}"
                    "<extra></extra>"
                ),
                customdata=best_scores,
                name="Best combo",
            )
        )
    fig_sc.update_layout(
        title="Per-structure error scatter (colored by aggregate score)",
        xaxis_title="dv% (V0)",
        yaxis_title="db% (B0)",
        template="plotly_white",
    )
    out_html = iterdir / "scatter_errors.html" if save_html else None
    out_png = iterdir / "scatter_errors.png" if save_png else None
    _try_write_plotly(fig_sc, out_html, out_png)


# ----------------------------------- runner -----------------------------------


def run_aqflow_eos(yaml_path: Path, workdir: Path, resources: Optional[Path]) -> int:
    cmd = ["aqflow", "eos", str(yaml_path)]
    if resources:
        cmd += ["--resources", str(resources)]
    env = os.environ.copy()
    try:
        proc = subprocess.run(cmd, cwd=str(workdir), env=env, check=False)
        return int(proc.returncode)
    except FileNotFoundError:
        print("aqflow CLI not found in PATH; please install or adjust PATH", file=sys.stderr)
        return 127


def archive_eos_post(src_dir: Path, dst_path: Path) -> Optional[Path]:
    p = src_dir / "aqflow_data" / "eos_post.json"
    if p.exists():
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst_path)
        return dst_path
    return None


def stage0_targets(cfg: OptimConfig, base_dir: Path, exp_dir: Path, structures: List[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
    # Run eos on the user YAML as-is (to refresh eos_post.json), then extract QE targets
    rc = run_aqflow_eos(cfg.base_yaml, workdir=base_dir, resources=cfg.resources)
    if rc != 0:
        print(f"Stage0: aqflow eos exited with rc={rc}")
    archived = archive_eos_post(base_dir, exp_dir / "iter_00" / "eos_post.json")
    if not archived:
        raise RuntimeError("Stage0: eos_post.json not found after running base YAML")
    eos_post = _load_eos_post(archived)
    names = [s.get("name") for s in structures]
    targets = extract_ref_targets(eos_post, names, ref_name=cfg.qe_ref_name)
    if not targets:
        raise RuntimeError("Stage0: no QE targets extracted (check qe_ref_name and structures)")
    return targets


def summarize_to_tsv(path: Path, combos_scores: Dict[str, Dict[str, Any]], structures: List[str]) -> None:
    cols = [
        "name", "paraA", "Gammax", "rHC_A", "rHC_B", "score",
    ]
    for s in structures:
        cols.extend([f"{s}_V0", f"{s}_B0_GPa", f"{s}_dv_pct", f"{s}_db_pct", f"{s}_score_s"])
    lines = ["\t".join(cols)]
    for name, rec in sorted(combos_scores.items(), key=lambda kv: float(kv[1].get("score", float("inf")))):
        # parse params back from name atlas_<a>_<g>_<ra>_<rb>
        try:
            _, a, g, ra, rb = name.split("_")
        except ValueError:
            a = g = ra = rb = ""
        row = [
            name,
            a, g, ra, rb,
            f"{float(rec.get('score', float('inf'))):.6f}",
        ]
        bys = rec.get("by_structure") or {}
        for s in structures:
            m = bys.get(s)
            if not m:
                row.extend(["", "", "", "", ""])
            else:
                row.extend([
                    f"{m['V0']:.6f}", f"{m['B0_GPa']:.6f}", f"{m['dv_pct']:.4f}", f"{m['db_pct']:.4f}", f"{m['score_s']:.6f}",
                ])
        lines.append("\t".join(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description="Iterative parameter fitting for EOS (ATLAS vs QE targets)")
    ap.add_argument("--config", required=True, help="Path to optimizer config YAML")
    ap.add_argument(
        "--optimize-paraA-gammax-only",
        action="store_true",
        help="Restrict optimization loop to paraA and Gammax (freeze rHC_A and rHC_B).",
    )
    args = ap.parse_args()

    cfg = load_optim_config(Path(args.config))
    if args.optimize_paraA_gammax_only:
        cfg = dataclasses.replace(cfg, optimize_params=("paraA", "Gammax"))
    base = load_base_yaml(cfg.base_yaml)
    # Optional validation/read via AQFLOW API
    try:
        if ConfigurationLoader is not None:
            _ = ConfigurationLoader(cfg.base_yaml).load_configuration()
    except Exception:
        pass
    base_dir = cfg.base_yaml.parent
    structures = select_structures(base, cfg.structures)
    if not structures:
        print("No structures selected.")
        return 2
    structure_names = [s.get("name") for s in structures]

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_dir = cfg.outdir / f"param_fit_{ts}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Stage 0: QE targets
    targets = stage0_targets(cfg, base_dir, exp_dir, structures)
    print("Targets (V0_ref, B0_ref):", targets)

    # Iterative loop
    prev_best = float("inf")
    center = dataclasses.replace(cfg.params_center)
    step = dataclasses.replace(cfg.params_step)
    bounds = Bounds()  # can be expanded to parse from config if needed later

    for it in range(1, cfg.max_iterations + 1):
        iterdir = exp_dir / f"iter_{it:02d}"
        iterdir.mkdir(parents=True, exist_ok=True)

        params_list = build_grid(center, step, bounds, cfg.neighborhood, cfg.optimize_params)
        combos = make_atlas_combos(params_list, structure_names, cfg.atlas_template, cfg.pseudopotential_set)
        added = append_combos_inplace(cfg.base_yaml, combos)
        # Keep a record of what was appended this iteration
        (iterdir / "added_combos.yaml").write_text(yaml.safe_dump(added, sort_keys=False))

        rc = run_aqflow_eos(cfg.base_yaml, workdir=base_dir, resources=cfg.resources)
        if rc != 0:
            print(f"Iteration {it}: aqflow eos exited with rc={rc}")

        archived = archive_eos_post(base_dir, iterdir / "eos_post.json")
        if not archived:
            print(f"Iteration {it}: eos_post.json not found; skipping evaluation")
            continue
        eos_post = _load_eos_post(archived)
        atlas = extract_atlas_fits(eos_post, structure_names)
        combos_scores = compute_scores(targets, atlas, cfg.weight_v0, cfg.weight_b0)

        # Summaries and plots
        summarize_to_tsv(iterdir / "summary.tsv", combos_scores, structure_names)
        plot_iteration(iterdir, params_list, combos_scores, structure_names, save_png=cfg.save_png, save_html=cfg.save_html)

        # Pick best and check convergence
        best_name, best_rec = min(
            combos_scores.items(), key=lambda kv: float(kv[1].get("score", float("inf")))
        ) if combos_scores else (None, None)
        if not best_name:
            print(f"Iteration {it}: no valid combos; stopping")
            break
        best_score = float(best_rec.get("score", float("inf")))
        print(f"Iteration {it}: best {best_name} score={best_score:.6f}")
        # Update center from best_name
        try:
            _, a, g, ra, rb = best_name.split("_")
            center = ParamCenter(paraA=float(a), Gammax=float(g), rHC_A=float(ra), rHC_B=float(rb))
        except Exception:
            pass

        # Convergence
        if prev_best < float("inf") and abs(prev_best - best_score) < cfg.min_improvement:
            print(f"Converged: improvement {abs(prev_best - best_score):.6f} < {cfg.min_improvement}")
            break
        prev_best = best_score
        # Shrink steps
        step = ParamStep(
            paraA=step.paraA * cfg.shrink_factor if "paraA" in cfg.optimize_params else step.paraA,
            Gammax=step.Gammax * cfg.shrink_factor if "Gammax" in cfg.optimize_params else step.Gammax,
            rHC_A=step.rHC_A * cfg.shrink_factor if "rHC_A" in cfg.optimize_params else step.rHC_A,
            rHC_B=step.rHC_B * cfg.shrink_factor if "rHC_B" in cfg.optimize_params else step.rHC_B,
        )

    # Write final best.json
    try:
        best = {
            "center": dataclasses.asdict(center),
            "last_best_score": prev_best,
            "targets": {k: {"V0": v0, "B0_GPa": b0} for k, (v0, b0) in targets.items()},
        }
        (exp_dir / "best.json").write_text(json.dumps(best, indent=2))
    except Exception:
        pass

    print(f"Experiment directory: {exp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
