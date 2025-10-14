"""
Template processing engine for ATLAS-QE workflow system.

This module handles template file processing, parameter substitution,
and automatic generation of software-specific input sections.
"""

import re
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import numpy as np

from .configuration import StructureConfig, ParameterCombination, WorkflowConfiguration

logger = logging.getLogger(__name__)


class TemplateProcessor:
    """
    Template file processor with parameter substitution and auto-generation.

    Handles loading template files, performing parameter substitutions,
    and generating software-specific sections automatically.
    """

    def __init__(self, config: WorkflowConfiguration, config_dir: Path):
        """Initialize template processor."""
        self.config = config
        self.config_dir = Path(config_dir)

    def process_template(
        self,
        template_file: str,
        substitutions: Dict[str, Any],
        structure: StructureConfig,
        combination: ParameterCombination
    ) -> str:
        """
        Process template file with substitutions and auto-generation.

        Args:
            template_file: Template file path (relative to config directory)
            substitutions: Parameter substitutions to perform
            structure: Structure configuration
            combination: Parameter combination configuration

        Returns:
            Processed template content
        """
        # Load template file
        template_path = self._resolve_template_path(template_file)
        with open(template_path, 'r') as f:
            template_content = f.read()

        # Perform basic substitutions
        processed_content = self._substitute_parameters(template_content, substitutions)

        # Generate and insert software-specific sections
        if combination.software == 'atlas':
            processed_content = self._insert_atlas_sections(
                processed_content, structure, combination
            )
        elif combination.software == 'qe':
            processed_content = self._insert_qe_sections(
                processed_content, structure, combination
            )

        return processed_content

    def _resolve_template_path(self, template_file: str) -> Path:
        """Resolve template file path relative to configuration directory."""
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = self.config_dir / template_path

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        return template_path

    def _substitute_parameters(self, content: str, substitutions: Dict[str, Any]) -> str:
        """Perform parameter substitutions in template content."""
        for key, value in substitutions.items():
            placeholder = f"{{{key}}}"
            content = content.replace(placeholder, str(value))

        return content

    def _insert_atlas_sections(
        self,
        content: str,
        structure: StructureConfig,
        combination: ParameterCombination
    ) -> str:
        """Insert ATLAS-specific auto-generated sections."""
        # Generate CELLFILE line (ATLAS needs this to read structure file)
        cellfile_line = "CELLFILE = POSCAR"

        # Generate ELEMENTS line
        elements_line = f"ELEMENTS = {' '.join(structure.elements)}"

        # Generate ppfile line
        pp_set = self.config.pseudopotential_sets[combination.pseudopotential_set]
        pp_files = [pp_set[elem] for elem in structure.elements]
        ppfile_line = f"ppfile = {' '.join(pp_files)}"

        # Insert lines after comment header
        lines = content.split('\n')
        insert_pos = 1  # After first comment line

        # Insert in reverse order to maintain positions
        lines.insert(insert_pos, ppfile_line)
        lines.insert(insert_pos, elements_line)
        lines.insert(insert_pos, cellfile_line)

        return '\n'.join(lines)

    def _insert_qe_sections(
        self,
        content: str,
        structure: StructureConfig,
        combination: ParameterCombination
    ) -> str:
        """Insert QE-specific auto-generated sections."""
        # Generate all QE sections
        pseudopotentials_section = self._generate_qe_pseudopotentials(structure, combination)
        atomic_species_section = self._generate_qe_atomic_species(structure, combination)

        # Add sections at the end
        if "&PSEUDOPOTENTIALS" not in content:
            content += f"\n{pseudopotentials_section}\n"

        if "ATOMIC_SPECIES" not in content:
            content += f"{atomic_species_section}\n"

        return content

    def _generate_qe_atomic_species(
        self,
        structure: StructureConfig,
        combination: ParameterCombination
    ) -> str:
        """Generate QE ATOMIC_SPECIES section."""
        pp_set = self.config.pseudopotential_sets[combination.pseudopotential_set]

        # Approximate atomic masses (should be looked up from proper database)
        atomic_masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.941, 'Be': 9.012, 'B': 10.811,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.086, 'P': 30.974,
            'S': 32.065, 'Cl': 35.453, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Ga': 69.723, 'As': 74.922
        }

        species_lines = ["ATOMIC_SPECIES"]
        for element in structure.elements:
            pp_file = pp_set[element]
            mass = atomic_masses.get(element, 1.0)
            species_lines.append(f"   {element}  {mass:.3f}  {pp_file}")

        return '\n'.join(species_lines)

    def _generate_qe_pseudopotentials(
        self,
        structure: StructureConfig,
        combination: ParameterCombination
    ) -> str:
        """Generate QE &PSEUDOPOTENTIALS section."""
        pp_set = self.config.pseudopotential_sets[combination.pseudopotential_set]

        pp_lines = ["&PSEUDOPOTENTIALS"]
        for element in structure.elements:
            pp_file = pp_set[element]
            # QE pseudopotentials section uses lowercase element names
            pp_lines.append(f"   {element.lower():12} = {pp_file}")
        pp_lines.append("/")

        return '\n'.join(pp_lines)


class StructureProcessor:
    """
    Structure file processor for volume scaling and format conversion.

    Handles POSCAR file loading, volume scaling, and structure generation.
    """

    def __init__(self, config_dir: Path):
        """Initialize structure processor."""
        self.config_dir = Path(config_dir)

    def generate_structure_content(
        self,
        structure: StructureConfig,
        volume_scale: float
    ) -> str:
        """
        Generate or scale structure content (POSCAR format).

        Args:
            structure: Structure configuration
            volume_scale: Volume scaling factor

        Returns:
            POSCAR content as string
        """
        if structure.file:
            return self._load_and_scale_structure_file(structure, volume_scale)
        elif structure.structure_type:
            return self._generate_structure(structure, volume_scale)
        else:
            raise ValueError(f"Structure '{structure.name}' has neither file nor structure_type")

    def _load_and_scale_structure_file(
        self,
        structure: StructureConfig,
        volume_scale: float
    ) -> str:
        """Load structure using ASE, scale volume, and return POSCAR content.

        Uses ASE to ensure robust parsing of various structure formats and
        consistent coordinate/cell handling across the workflow.
        """
        from ase.io import read, write  # lazy import
        import io

        structure_path = self._resolve_structure_path(structure.file)

        atoms = read(str(structure_path))
        # Volume scales with the cube of linear dimensions
        s = float(volume_scale) ** (1.0 / 3.0)
        # Scale cell and atomic positions accordingly
        atoms.set_cell(atoms.cell * s, scale_atoms=True)

        buf = io.StringIO()
        write(buf, atoms, format="vasp")  # POSCAR format
        return buf.getvalue()

    def _generate_structure(
        self,
        structure: StructureConfig,
        volume_scale: float
    ) -> str:
        """Generate structure from structure_type specification."""
        if structure.structure_type == 'fcc':
            return self._generate_fcc_structure(structure, volume_scale)
        elif structure.structure_type == 'bcc':
            return self._generate_bcc_structure(structure, volume_scale)
        else:
            raise ValueError(f"Unknown structure_type: {structure.structure_type}")

    def _generate_fcc_structure(
        self,
        structure: StructureConfig,
        volume_scale: float
    ) -> str:
        """Generate FCC structure POSCAR."""
        if not structure.lattice_parameter:
            raise ValueError(f"FCC structure '{structure.name}' requires lattice_parameter")

        lattice_param = structure.lattice_parameter
        scaled_lattice = lattice_param * (volume_scale ** (1.0/3.0))

        # Simple FCC for single element
        if len(structure.elements) != 1:
            raise ValueError("FCC generation currently supports single element only")

        element = structure.elements[0]

        poscar_lines = [
            f"FCC {element} - volume scale {volume_scale:.5f}",
            f"   {scaled_lattice:.8f}",
            "   0.000000   0.500000   0.500000",
            "   0.500000   0.000000   0.500000",
            "   0.500000   0.500000   0.000000",
            f" {element}",
            "   1",
            "Direct",
            "   0.000000   0.000000   0.000000"
        ]

        return '\n'.join(poscar_lines)

    def _generate_bcc_structure(
        self,
        structure: StructureConfig,
        volume_scale: float
    ) -> str:
        """Generate BCC structure POSCAR."""
        if not structure.lattice_parameter:
            raise ValueError(f"BCC structure '{structure.name}' requires lattice_parameter")

        lattice_param = structure.lattice_parameter
        scaled_lattice = lattice_param * (volume_scale ** (1.0/3.0))

        if len(structure.elements) != 1:
            raise ValueError("BCC generation currently supports single element only")

        element = structure.elements[0]

        poscar_lines = [
            f"BCC {element} - volume scale {volume_scale:.5f}",
            f"   {scaled_lattice:.8f}",
            "   1.000000   0.000000   0.000000",
            "   0.000000   1.000000   0.000000",
            "   0.000000   0.000000   1.000000",
            f" {element}",
            "   2",
            "Direct",
            "   0.000000   0.000000   0.000000",
            "   0.500000   0.500000   0.500000"
        ]

        return '\n'.join(poscar_lines)

    def _scale_poscar_volume(self, poscar_content: str, volume_scale_factor: float) -> str:
        """Scale volume in POSCAR content by scaling lattice vectors."""
        lines = poscar_content.strip().split('\n')
        if len(lines) < 6:
            raise ValueError("Invalid POSCAR format")

        # Volume scales as cube root of linear scaling
        linear_scale = volume_scale_factor ** (1.0/3.0)

        # Keep line 1 (scaling factor) as 1.0 or original value
        # Scale the lattice vectors (lines 2-4, 0-indexed)
        for i in range(2, 5):
            if i < len(lines):
                vector_line = lines[i].strip()
                components = vector_line.split()
                if len(components) >= 3:
                    try:
                        # Scale each component of the lattice vector
                        scaled_components = [float(comp) * linear_scale for comp in components]
                        # Format with high precision to match reference
                        formatted_line = "    " + "    ".join(f"{comp:.16f}" for comp in scaled_components)
                        lines[i] = formatted_line
                    except ValueError:
                        logger.warning(f"Could not parse lattice vector on line {i+1}")

        return '\n'.join(lines)

    def _resolve_structure_path(self, structure_file: str) -> Path:
        """Resolve structure file path relative to configuration directory."""
        structure_path = Path(structure_file)
        if not structure_path.is_absolute():
            structure_path = self.config_dir / structure_path

        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        return structure_path





# Note: InputFileGenerator was removed; TaskCreator/software adapters generate inputs.
