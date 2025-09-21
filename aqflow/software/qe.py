"""
QE software adapter: input generation and command construction.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .base import SoftwareInputGenerator, SoftwareRunner
from aqflow.core.configuration import StructureConfig, ParameterCombination, WorkflowConfiguration
from aqflow.core.template_engine import TemplateProcessor, StructureProcessor


class QEInputGenerator(SoftwareInputGenerator):
    def create_inputs(
        self,
        work_dir: Path,
        config: WorkflowConfiguration,
        config_dir: Path,
        structure: StructureConfig,
        combination: ParameterCombination,
        volume_scale: float,
        template_proc: TemplateProcessor,
        struct_proc: StructureProcessor,
    ) -> Path:
        work_dir.mkdir(parents=True, exist_ok=True)

        # Generate QE input content from template
        text = template_proc.process_template(
            template_file=combination.template,
            substitutions=combination.template_substitutions,
            structure=structure,
            combination=combination,
        )

        # Append structure sections derived from POSCAR
        poscar = struct_proc.generate_structure_content(structure=structure, volume_scale=volume_scale)
        text = self._append_structure_sections(text, poscar)

        input_path = work_dir / "qe.in"
        input_path.write_text(text)
        return input_path

    def _append_structure_sections(self, input_content: str, poscar_content: str) -> str:
        lines = poscar_content.strip().split('\n')
        if len(lines) < 8:
            return input_content

        try:
            scaling = float(lines[1])
        except Exception:
            scaling = 1.0

        a1 = [float(x) for x in lines[2].split()]
        a2 = [float(x) for x in lines[3].split()]
        a3 = [float(x) for x in lines[4].split()]
        elements = lines[5].split()
        counts = [int(x) for x in lines[6].split()]
        direct = lines[7].strip().lower().startswith('d')

        total = sum(counts)
        positions: List[List[float]] = []
        idx = 8
        for _ in range(total):
            if idx < len(lines):
                parts = lines[idx].split()
                if len(parts) >= 3:
                    positions.append([float(parts[0]), float(parts[1]), float(parts[2])])
            idx += 1

        # CELL_PARAMETERS
        cell = ["", "CELL_PARAMETERS angstrom"]
        for v in (a1, a2, a3):
            s = [vv * scaling for vv in v]
            cell.append(f"{s[0]:.14f} {s[1]:.14f} {s[2]:.14f}")

        # ATOMIC_POSITIONS
        apos = ["", "ATOMIC_POSITIONS angstrom"]
        labels: List[str] = []
        for el, n in zip(elements, counts):
            labels.extend([el] * n)

        def frac_to_cart(frac):
            return [
                frac[0] * a1[0] * scaling + frac[1] * a2[0] * scaling + frac[2] * a3[0] * scaling,
                frac[0] * a1[1] * scaling + frac[1] * a2[1] * scaling + frac[2] * a3[1] * scaling,
                frac[0] * a1[2] * scaling + frac[1] * a2[2] * scaling + frac[2] * a3[2] * scaling,
            ]

        for label, coord in zip(labels, positions):
            cart = frac_to_cart(coord) if direct else [c * scaling for c in coord]
            apos.append(f"{label} {cart[0]:.10f} {cart[1]:.10f} {cart[2]:.10f}")

        return input_content + "\n" + "\n".join(cell) + "\n" + "\n".join(apos) + "\n"


class QERunner(SoftwareRunner):
    def build_command(self, binary_path: str, input_filename: str) -> str:
        # QE prefers '-in file'
        return f"{binary_path} -in {input_filename} > job.out 2>&1"
