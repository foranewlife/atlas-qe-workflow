#!/usr/bin/env python3
"""
Plot EOS curves across varying parameter A from eos_post.json, with a QE reference.

Inputs:
  - eos_post.json produced by `aqflow eos` + post (aqflow/core/eos_post.py)

Behavior:
  - For each structure in the file, overlay all curves (combinations) on one plot.
  - Use `--ref-substr` (default: 'qe_ecut60') to identify the reference curve and style it prominently.
  - Legend is human-friendly (e.g., 'QE ecut=60', 'ATLAS KEDF=801') and includes B0 in GPa when available.
  - Mark v0 points (from pmg_fit) on the plot for each curve.

Output:
  - One PNG per structure: <out_dir>/eos_paraA_compare_<structure>.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


def _load_eos_post(path: Path) -> Dict:
    return json.loads(Path(path).read_text())


def _group_by_structure(curves: List[Dict]) -> Dict[str, List[Dict]]:
    by = {}
    for c in curves:
        s = c.get("structure") or "unknown"
        by.setdefault(s, []).append(c)
    return by


def _extract_A_from_name(comb: str) -> Optional[float]:
    import re
    s = (comb or "").lower()
    # Pattern 1: atlas_kedfNNN
    m = re.search(r"atlas[_-]?kedf(\d+)", s)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    # Pattern 2: atlas_<A>_... (e.g., atlas_2.0_-1.0_0.45_0.10)
    m2 = re.match(r"^atlas_([+-]?[0-9]+(?:\.[0-9]+)?)_", s)
    if m2:
        try:
            return float(m2.group(1))
        except Exception:
            return None
    return None


def _is_atlas_param_name(name: str) -> Tuple[bool, Optional[float]]:
    """Return (True, A) if name is of form atlas_<A>_... and not containing 'origin'/'kstau'."""
    s = (name or "").lower()
    if ("origin" in s) or ("kstau" in s):
        return False, None
    a = _extract_A_from_name(s)
    if a is not None:
        return True, a
    return False, None


def _load_eos_json_optional(eos_json: Optional[Path]) -> Dict[str, Dict]:
    """Return combinations_info mapping if eos.json is available, else {}.

    Structure matches EosModel.combinations_info: { name: {template_substitutions, software, ...} }
    """
    if not eos_json:
        return {}
    try:
        raw = json.loads(Path(eos_json).read_text())
        return raw.get("combinations_info") or {}
    except Exception:
        return {}


def _extract_A(comb: str, combos_info: Optional[Dict[str, Dict]] = None) -> Optional[float]:
    # 1) from name pattern (atlas_kedfNNN)
    a = _extract_A_from_name(comb)
    if a is not None:
        return a
    # 2) from eos.json combinations_info -> template_substitutions
    if combos_info and comb in combos_info:
        subs = (combos_info.get(comb) or {}).get("template_substitutions") or {}
        for k in ("A", "a", "KEDF", "kedf"):
            val = subs.get(k)
            try:
                if val is not None:
                    return float(val)
            except Exception:
                continue
    return None


def _label_from_combination(comb: str, pmg_fit: Optional[Dict], *, ref: bool = False, A: Optional[float] = None, software: Optional[str] = None) -> str:
    import re
    s = (comb or "").lower()
    # QE reference label
    m = re.search(r"qe[_-]?ecut(\d+)", s)
    if m:
        return f"QE ecut={m.group(1)}" + (" [REF]" if ref else "")
    # ATLAS compact label: atlas A=XX(XXGPa)
    if A is None:
        A = _extract_A(comb)
    if A is not None or (software or "").lower() == "atlas":
        b0_txt = ""
        if pmg_fit and isinstance(pmg_fit, dict) and (pmg_fit.get("b0_GPa") is not None) and not pmg_fit.get("error"):
            try:
                b0_txt = f"({float(pmg_fit['b0_GPa']):.2f}GPa)"
            except Exception:
                b0_txt = ""
        if A is not None:
            return f"ATLAS A={int(A)}{b0_txt}"
        return f"ATLAS {b0_txt}" if b0_txt else "ATLAS"
    # fallback
    return (comb or "").replace("_", " ") + (" [REF]" if ref else "")


def _extract_xy_points(curve: Dict) -> Tuple[np.ndarray, np.ndarray]:
    pts = [
        (p.get("volume_A3"), p.get("energy_eV"))
        for p in (curve.get("points") or [])
        if (p.get("volume_A3") is not None and p.get("energy_eV") is not None)
    ]
    if not pts:
        return np.array([]), np.array([])
    v = np.array([float(x[0]) for x in pts], dtype=float)
    e = np.array([float(x[1]) for x in pts], dtype=float)
    # sort by volume for smooth lines
    order = np.argsort(v)
    return v[order], e[order]


def _match_ref(name: str, ref_substr: str) -> bool:
    s = (name or "").lower()
    ref = (ref_substr or "").lower()
    if ref and ref in s:
        return True
    # tolerant QE pattern: qe + ecut + number found in ref_substr
    import re
    m = re.search(r"(\d+)", ref)
    if m and "qe" in s and "ecut" in s and m.group(1) in s:
        return True
    return False


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    r = tuple(int(hex_color[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))
    return (r[0] / 255.0, r[1] / 255.0, r[2] / 255.0)


def _mix_with_white(rgb: Tuple[float, float, float], t: float) -> Tuple[float, float, float]:
    # t in [0,1]; t=0 -> original color (deep), t=1 -> white (light)
    return (rgb[0] + (1.0 - rgb[0]) * t, rgb[1] + (1.0 - rgb[1]) * t, rgb[2] + (1.0 - rgb[2]) * t)


def plot_structure(structure: str, curves: List[Dict], out_dir: Path, ref_substr: str,
                   exclude_substr: Optional[List[str]] = None,
                   mono_color: str = "#4169E1",
                   small_color: str = "#4169E1", large_color: str = "#DC143C",
                   combos_info: Optional[Dict[str, Dict]] = None) -> Tuple[Optional[Path], Optional[Path]]:
    if not curves:
        return None
    # filter curves: remove unrelated (e.g., 'kstau'), and keep only ATLAS A-curves + ref
    exclude_substr = [x.lower() for x in (exclude_substr or [])]
    kept: List[Dict] = []
    ref_candidates: List[Dict] = []
    for c in curves:
        name = (c.get("combination") or "").lower()
        if any(x in name for x in exclude_substr):
            continue
        if _match_ref(name, ref_substr):
            ref_candidates.append(c)
            kept.append(c)
            continue
        # keep only atlas_<A>_... style
        ok, _ = _is_atlas_param_name(name)
        if ok:
            kept.append(c)
    curves = kept

    # pick reference
    ref = None
    if ref_candidates:
        ref = ref_candidates[0]
    else:
        # fallback: pick a QE curve if present
        for c in curves:
            if "qe" in (c.get("software") or "").lower() or "qe" in (c.get("combination") or "").lower():
                ref = c
                break
        # last fallback: first curve
        if ref is None and curves:
            ref = curves[0]

    fig, ax = plt.subplots()

    # Build two-color gradient by A value (large->small)
    from matplotlib.colors import LinearSegmentedColormap
    atlas_nonref: List[Dict] = [c for c in curves if not (ref is not None and c is ref) and ((c.get("software") or "").lower() == "atlas")]
    A_vals: List[Tuple[Optional[float], Dict]] = []
    for c in atlas_nonref:
        # Prefer parsing A from name like atlas_<A>_... ; fallback to combos_info
        A = _extract_A_from_name(c.get("combination") or "")
        if A is None:
            A = _extract_A(c.get("combination") or "", combos_info)
        A_vals.append((A, c))
    # Determine range; if all A missing or identical, fallback to index spacing
    As_num = [a for a, _ in A_vals if a is not None]
    if As_num:
        amax = max(As_num); amin = min(As_num)
    else:
        amax = amin = 1.0
    # Small A -> first color (assumed deeper), Large A -> second color (assumed lighter)
    # Monochrome gradient base color (deep for small A -> lighter for large A)
    base_rgb = _hex_to_rgb(mono_color)
    # Precompute color position per curve
    t_map: Dict[int, float] = {}
    if As_num and amax > amin:
        for a, c in A_vals:
            if a is None:
                t_map[id(c)] = 0.5
            else:
                t_map[id(c)] = (a - amin) / (amax - amin)
    else:
        n = max(1, len(A_vals))
        # Ascending order so smaller A maps to lower t (darker color)
        for idx, (a, c) in enumerate(sorted(A_vals, key=lambda x: (x[0] is None, x[0] if x[0] is not None else 0), reverse=False)):
            t_map[id(c)] = 0.0 if n == 1 else idx / (n - 1)

    # First draw non-reference
    plotted_nonref = 0
    v_all: List[float] = []
    for c in curves:
        if ref is not None and c is ref:
            continue
        v, e = _extract_xy_points(c)
        if len(v) == 0:
            continue
        A = _extract_A_from_name(c.get("combination") or "")
        if A is None:
            A = _extract_A(c.get("combination") or "", combos_info)
        t = t_map.get(id(c), 0.5)
        color = _mix_with_white(base_rgb, t)
        lbl = _label_from_combination(c.get("combination") or "", c.get("pmg_fit"), ref=False, A=A, software=c.get("software"))
        ax.plot(v, e, marker="o", linestyle="-", linewidth=1.5, markersize=4, color=color, alpha=0.95, label=lbl)
        v_all.extend(list(v))
        plotted_nonref += 1
        # v0 dot
        pmg = c.get("pmg_fit") or {}
        try:
            if pmg and not pmg.get("error"):
                vv = float(pmg.get("v0"))
                ee = float(pmg.get("e0"))
                ax.plot(vv, ee, "o", color=color, markersize=6, markeredgecolor="black", markeredgewidth=0.6)
        except Exception:
            pass

    # Draw reference on top with distinct style
    if ref is not None:
        v, e = _extract_xy_points(ref)
        if len(v) > 0:
            lbl = _label_from_combination(ref.get("combination") or "", ref.get("pmg_fit"), ref=True, software=ref.get("software"))
            ax.plot(v, e, marker="o", linestyle="-", linewidth=3.0, markersize=5, color="black", alpha=0.98, label=lbl)
            pmg = ref.get("pmg_fit") or {}
            try:
                if pmg and not pmg.get("error"):
                    vv = float(pmg.get("v0"))
                    ee = float(pmg.get("e0"))
                    ax.plot(vv, ee, "o", color="black", markersize=7, markeredgecolor="white", markeredgewidth=0.8)
            except Exception:
                pass
            v_all.extend(list(v))
            # leave y-limits to autoscale; only adjust x-range below

    ax.set_xlabel("Volume (A^3/atom)", fontsize=13)
    ax.set_ylabel("Energy (eV/atom)", fontsize=13)
    ax.set_title(f"EOS Curves — {structure}", fontsize=15)
    ax.grid(True, alpha=0.3)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(fontsize=9)
    # x-range padding
    if v_all:
        vmin = float(np.min(v_all)); vmax = float(np.max(v_all))
        pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.1
        ax.set_xlim(vmin - pad, vmax + pad)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path_abs = out_dir / f"eos_paraA_compare_abs_{structure}.png"
    fig.savefig(out_path_abs, dpi=300)
    plt.close(fig)

    # Relative to reference (E - E_ref) plot
    out_path_rel: Optional[Path] = None
    # Relative-to-own-min plot (each curve subtracts its own E_min or pmg_fit.e0)
    if curves:
        fig2, ax2 = plt.subplots()
        v_all2: List[float] = []
        for c in curves:
            v, e = _extract_xy_points(c)
            if len(v) == 0:
                continue
            pmg = c.get("pmg_fit") or {}
            if pmg and not pmg.get("error") and (pmg.get("e0") is not None):
                e0 = float(pmg.get("e0"))
            else:
                e0 = float(np.min(e))
            e_rel = e - e0
            A = _extract_A_from_name(c.get("combination") or "")
            if A is None:
                A = _extract_A(c.get("combination") or "", combos_info)
            t = t_map.get(id(c), 0.5)
            color = _mix_with_white(base_rgb, t) if (c.get("software") or "").lower() == "atlas" else ("black" if c is ref else "gray")
            lbl = _label_from_combination(c.get("combination") or "", c.get("pmg_fit"), ref=(c is ref), A=A, software=c.get("software"))
            lw = 3.0 if (c is ref) else 1.5
            ax2.plot(v, e_rel, marker="o", linestyle="-", linewidth=lw, markersize=4, color=color, alpha=0.95, label=lbl)
            # mark v0 at 0 if available
            try:
                if pmg and not pmg.get("error") and (pmg.get("v0") is not None):
                    vv = float(pmg.get("v0"))
                    ax2.plot(vv, 0.0, "o", color=color, markersize=6, markeredgecolor="white", markeredgewidth=0.8)
            except Exception:
                pass
            v_all2.extend(list(v))
            # y-limits will autoscale for relative plot; we adjust x-range later
        ax2.axhline(y=0.0, color="black", linestyle="-", linewidth=1.0, alpha=0.5)
        ax2.set_xlabel("Volume (A^3/atom)", fontsize=13)
        ax2.set_ylabel("Energy - E_min (eV/atom)", fontsize=13)
        ax2.set_title(f"EOS Curves Relative to Own Min — {structure}", fontsize=15)
        ax2.grid(True, alpha=0.3)
        handles2, labels2 = ax2.get_legend_handles_labels()
        if handles2:
            ax2.legend(fontsize=9)
        if v_all2:
            vmin = float(np.min(v_all2)); vmax = float(np.max(v_all2))
            pad = 0.05 * (vmax - vmin) if vmax > vmin else 0.1
            ax2.set_xlim(vmin - pad, vmax + pad)
        fig2.tight_layout()
        out_path_rel = out_dir / f"eos_paraA_compare_rel_{structure}.png"
        fig2.savefig(out_path_rel, dpi=300)
        plt.close(fig2)

    return out_path_abs, out_path_rel


def main():
    ap = argparse.ArgumentParser(description="Plot EOS curves under parameter-A variations with a QE reference")
    ap.add_argument("eos_post", help="Path to eos_post.json")
    ap.add_argument("--ref-substr", default="qe_ecut60", help="Substring to identify reference combination (default: qe_ecut60)")
    ap.add_argument("--exclude", action="append", default=["kstau", "origin"], help="Exclude curves containing this substring (can repeat). Default: kstau, origin")
    ap.add_argument("--mono-color", default="#4169E1", help="Monochrome base color for ATLAS gradient (default royalblue)")
    # Backward-compatible options (ignored when mono-color is set)
    ap.add_argument("--small-color", default="#4169E1", help=argparse.SUPPRESS)
    ap.add_argument("--large-color", default="#DC143C", help=argparse.SUPPRESS)
    ap.add_argument("--eos-json", default=None, help="Optional path to eos.json to derive parameter A from combinations_info")
    ap.add_argument("--out-dir", default=None, help="Output directory (default: same directory as eos_post.json)")
    args = ap.parse_args()

    eos_post_path = Path(args.eos_post).resolve()
    data = _load_eos_post(eos_post_path)
    curves = data.get("curves") or []
    by_struct = _group_by_structure(curves)

    out_dir = Path(args.out_dir).resolve() if args.out_dir else eos_post_path.parent
    eos_json_path = Path(args.eos_json).resolve() if args.eos_json else (eos_post_path.parent / "eos.json")
    combos_info = _load_eos_json_optional(eos_json_path if eos_json_path.exists() else None)
    saved: List[Path] = []
    for struct, items in by_struct.items():
        p_abs, p_rel = plot_structure(
            struct, items, out_dir, ref_substr=args.ref_substr,
            exclude_substr=args.exclude,
            mono_color=args.mono_color,
            combos_info=combos_info,
        )
        if p_abs is not None:
            saved.append(p_abs)
        if p_rel is not None:
            saved.append(p_rel)

    if saved:
        print("Saved:")
        for p in saved:
            print(str(p))
    else:
        print("No plots generated (no curves)")


if __name__ == "__main__":
    main()
