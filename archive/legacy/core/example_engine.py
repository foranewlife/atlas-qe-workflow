"""
Example-based input generation engine.

This module implements a simple approach where users provide example input files
and the system modifies specific parameters to generate parameter sweeps.
"""

import re
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging
import numpy as np

logger = logging.getLogger(__name__)


class ExampleEngine:
    """
    Simple engine that takes example input files and modifies parameters.

    Users provide concrete example files, system modifies specific values
    for parameter sweeps like EOS calculations.
    """

    def __init__(self):
        """Initialize the example engine."""
        self.parameter_patterns = self._init_parameter_patterns()

    def _init_parameter_patterns(self) -> Dict[str, Dict[str, str]]:
        """Initialize regex patterns for different software parameters."""
        return {
            'atlas': {
                'gap': r'(GAP\s*=\s*)([0-9.]+)',
                'kedf': r'(KEDF\s*=\s*)(\d+)',
                'etol': r'(ETOL\s*=\s*)([0-9.e-]+)',
                'maxiter': r'(MAXITER\s*=\s*)(\d+)',
                'ppfile': r'(ppfile\s*=\s*)(.*)',
                'elements': r'(ELEMENTS\s*=\s*)(.*)',
            },
            'qe': {
                'ecutwfc': r'(ecutwfc\s*=\s*)([0-9.]+)',
                'ecutrho': r'(ecutrho\s*=\s*)([0-9.]+)',
                'conv_thr': r'(conv_thr\s*=\s*)([0-9.e-]+)',
                'mixing_beta': r'(mixing_beta\s*=\s*)([0-9.]+)',
                'electron_maxstep': r'(electron_maxstep\s*=\s*)(\d+)',
                'degauss': r'(degauss\s*=\s*)([0-9.]+)',
                'k_points': r'(K_POINTS.*\n\s*)(\d+\s+\d+\s+\d+)',
            },
            'poscar': {
                'lattice_scale': r'^(\s*)([-+]?[0-9]*\.?[0-9]+)(\s*)$',
                'lattice_vectors': r'^(\s*)([-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+)(\s*)$',
            }
        }

    def modify_parameter(
        self,
        input_content: str,
        parameter_name: str,
        new_value: Union[str, float, int],
        software: str = 'atlas'
    ) -> str:
        """
        Modify a single parameter in input file content.

        Args:
            input_content: Original input file content
            parameter_name: Name of parameter to modify
            new_value: New value for the parameter
            software: Software type (atlas, qe, poscar)

        Returns:
            Modified input content
        """
        if software not in self.parameter_patterns:
            raise ValueError(f"Unknown software: {software}")

        patterns = self.parameter_patterns[software]
        if parameter_name not in patterns:
            logger.warning(f"Unknown parameter {parameter_name} for {software}")
            return input_content

        pattern = patterns[parameter_name]

        # Special handling for different parameter types
        if parameter_name == 'k_points':
            # Handle K_POINTS section
            if isinstance(new_value, (list, tuple)) and len(new_value) == 3:
                replacement = f"\\g<1>{new_value[0]} {new_value[1]} {new_value[2]} 0 0 0"
            else:
                replacement = f"\\g<1>{new_value}"
        else:
            replacement = f"\\g<1>{new_value}"

        modified_content = re.sub(pattern, replacement, input_content, flags=re.MULTILINE)

        if modified_content == input_content:
            logger.warning(f"Parameter {parameter_name} not found or not modified")

        return modified_content

    def scale_structure_volume(
        self,
        poscar_content: str,
        volume_scale_factor: float
    ) -> str:
        """
        Scale structure volume in POSCAR content.

        Args:
            poscar_content: Original POSCAR content
            volume_scale_factor: Volume scaling factor

        Returns:
            POSCAR content with scaled lattice
        """
        lines = poscar_content.strip().split('\n')
        if len(lines) < 5:
            raise ValueError("Invalid POSCAR format")

        # Volume scales as cube root of linear scaling
        linear_scale = volume_scale_factor ** (1.0/3.0)

        # Scale lattice constant (line 1, 0-indexed)
        scale_line = lines[1].strip()
        try:
            current_scale = float(scale_line)
            new_scale = current_scale * linear_scale
            lines[1] = f"   {new_scale:.8f}"
        except ValueError:
            logger.warning("Could not parse lattice scale factor")

        return '\n'.join(lines)

    def generate_volume_series(
        self,
        poscar_content: str,
        volume_range: tuple = (0.8, 1.2),
        num_points: int = 11
    ) -> List[str]:
        """
        Generate series of POSCAR files with different volumes.

        Args:
            poscar_content: Base POSCAR content
            volume_range: (min_scale, max_scale) for volume
            num_points: Number of volume points

        Returns:
            List of POSCAR contents with scaled volumes
        """
        volume_scales = np.linspace(volume_range[0], volume_range[1], num_points)

        scaled_poscars = []
        for scale in volume_scales:
            scaled_poscar = self.scale_structure_volume(poscar_content, scale)
            scaled_poscars.append(scaled_poscar)

        return scaled_poscars

    def create_eos_inputs(
        self,
        atlas_example: Optional[str] = None,
        qe_example: Optional[str] = None,
        poscar_example: Optional[str] = None,
        volume_range: tuple = (0.8, 1.2),
        num_points: int = 11,
        output_dir: Path = None
    ) -> Dict[str, List[Path]]:
        """
        Create EOS calculation inputs from example files.

        Args:
            atlas_example: Path to ATLAS example input
            qe_example: Path to QE example input
            poscar_example: Path to POSCAR example
            volume_range: Volume scaling range
            num_points: Number of volume points
            output_dir: Output directory

        Returns:
            Dictionary of generated file paths
        """
        if output_dir is None:
            output_dir = Path.cwd() / "eos_calculations"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generated_files = {'atlas': [], 'qe': [], 'poscar': []}

        # Load POSCAR if provided
        poscar_content = None
        if poscar_example:
            with open(poscar_example, 'r') as f:
                poscar_content = f.read()

        # Generate volume series
        volume_scales = np.linspace(volume_range[0], volume_range[1], num_points)

        for i, volume_scale in enumerate(volume_scales):
            volume_dir = output_dir / f"volume_{i:02d}_{volume_scale:.5f}"
            volume_dir.mkdir(exist_ok=True)

            # Generate scaled POSCAR
            if poscar_content:
                scaled_poscar = self.scale_structure_volume(poscar_content, volume_scale)
                poscar_file = volume_dir / "POSCAR"
                with open(poscar_file, 'w') as f:
                    f.write(scaled_poscar)
                generated_files['poscar'].append(poscar_file)

            # Copy ATLAS input if provided
            if atlas_example:
                atlas_file = volume_dir / "atlas.in"
                shutil.copy2(atlas_example, atlas_file)
                generated_files['atlas'].append(atlas_file)

            # Copy QE input if provided
            if qe_example:
                qe_file = volume_dir / "job.in"

                # Load and potentially modify QE input for this volume
                with open(qe_example, 'r') as f:
                    qe_content = f.read()

                # If we have lattice vectors, we need to update QE input
                if poscar_content:
                    qe_content = self._update_qe_cell_parameters(qe_content, scaled_poscar)

                with open(qe_file, 'w') as f:
                    f.write(qe_content)
                generated_files['qe'].append(qe_file)

        logger.info(f"Generated EOS inputs for {len(volume_scales)} volumes in {output_dir}")
        return generated_files

    def _update_qe_cell_parameters(self, qe_content: str, poscar_content: str) -> str:
        """Update QE CELL_PARAMETERS section from POSCAR."""
        # Extract lattice vectors from POSCAR
        lines = poscar_content.strip().split('\n')
        if len(lines) < 5:
            return qe_content

        try:
            scale = float(lines[1].strip())
            lattice_vectors = []
            for i in range(2, 5):
                vec_line = lines[i].strip().split()
                if len(vec_line) >= 3:
                    vec = [float(x) * scale for x in vec_line[:3]]
                    lattice_vectors.append(vec)

            if len(lattice_vectors) == 3:
                # Replace CELL_PARAMETERS section
                cell_section = "CELL_PARAMETERS { angstrom }\n"
                for vec in lattice_vectors:
                    cell_section += f"   {vec[0]:12.8f}  {vec[1]:12.8f}  {vec[2]:12.8f}\n"

                # Find and replace existing CELL_PARAMETERS
                pattern = r'CELL_PARAMETERS.*?\n(?:\s*[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s+[-+]?[0-9]*\.?[0-9]+\s*\n){3}'
                qe_content = re.sub(pattern, cell_section, qe_content, flags=re.MULTILINE)

        except (ValueError, IndexError):
            logger.warning("Could not update QE cell parameters from POSCAR")

        return qe_content

    def batch_modify_parameters(
        self,
        input_file: Union[str, Path],
        parameter_modifications: List[Dict[str, Any]],
        output_dir: Path,
        software: str = 'atlas'
    ) -> List[Path]:
        """
        Create multiple input files with different parameter values.

        Args:
            input_file: Base input file
            parameter_modifications: List of parameter changes
            output_dir: Output directory
            software: Software type

        Returns:
            List of generated input file paths
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Read base input file
        with open(input_file, 'r') as f:
            base_content = f.read()

        generated_files = []

        for i, modifications in enumerate(parameter_modifications):
            # Start with base content
            modified_content = base_content

            # Apply all modifications
            for param_name, new_value in modifications.items():
                modified_content = self.modify_parameter(
                    modified_content, param_name, new_value, software
                )

            # Create descriptive filename
            param_desc = "_".join([f"{k}{v}" for k, v in modifications.items()])
            output_file = output_dir / f"{input_file.stem}_{param_desc}{input_file.suffix}"

            # Write modified file
            with open(output_file, 'w') as f:
                f.write(modified_content)

            generated_files.append(output_file)

        logger.info(f"Generated {len(generated_files)} modified input files")
        return generated_files