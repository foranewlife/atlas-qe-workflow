"""
Crystal structure generation utilities.

This module provides functionality to generate common crystal structures
(FCC, BCC, diamond, etc.) for materials calculations.
"""

import numpy as np
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class StructureGenerator:
    """
    Generator for common crystal structures.

    Provides methods to create crystal structures for materials calculations,
    including standard structures like FCC, BCC, and diamond.
    """

    def __init__(self):
        """Initialize the structure generator."""
        self.structure_templates = self._init_structure_templates()

    def _init_structure_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize templates for common crystal structures."""
        return {
            "fcc": {
                "lattice_vectors": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                "basis_positions": np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.0],
                    [0.5, 0.0, 0.5],
                    [0.0, 0.5, 0.5]
                ])
            },
            "bcc": {
                "lattice_vectors": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                "basis_positions": np.array([
                    [0.0, 0.0, 0.0],
                    [0.5, 0.5, 0.5]
                ])
            },
            "diamond": {
                "lattice_vectors": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                "basis_positions": np.array([
                    [0.0, 0.0, 0.0],
                    [0.25, 0.25, 0.25],
                    [0.5, 0.5, 0.0],
                    [0.75, 0.75, 0.25],
                    [0.5, 0.0, 0.5],
                    [0.75, 0.25, 0.75],
                    [0.0, 0.5, 0.5],
                    [0.25, 0.75, 0.75]
                ])
            },
            "sc": {  # Simple cubic
                "lattice_vectors": np.array([
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]
                ]),
                "basis_positions": np.array([
                    [0.0, 0.0, 0.0]
                ])
            }
        }

    def generate_structure(
        self,
        structure_type: str,
        elements: List[str],
        lattice_parameter: float = 4.0,
        supercell: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Generate a crystal structure.

        Args:
            structure_type: Type of crystal structure (fcc, bcc, diamond, etc.)
            elements: List of element symbols
            lattice_parameter: Lattice parameter in Angstroms
            supercell: Supercell dimensions [nx, ny, nz]

        Returns:
            Dictionary containing lattice vectors, species, and coordinates
        """
        if structure_type not in self.structure_templates:
            raise ValueError(f"Unknown structure type: {structure_type}")

        template = self.structure_templates[structure_type]

        # Scale lattice vectors by lattice parameter
        lattice_vectors = template["lattice_vectors"] * lattice_parameter
        basis_positions = template["basis_positions"]

        # Handle supercell if specified
        if supercell is None:
            supercell = [1, 1, 1]

        # Generate atomic positions
        species = []
        coordinates = []

        for nx in range(supercell[0]):
            for ny in range(supercell[1]):
                for nz in range(supercell[2]):
                    for i, basis_pos in enumerate(basis_positions):
                        # Add supercell translation
                        position = basis_pos + np.array([nx, ny, nz])

                        # Assign elements cyclically
                        element = elements[i % len(elements)]

                        species.append(element)
                        coordinates.append(position.tolist())

        # Scale lattice vectors for supercell
        final_lattice = lattice_vectors * np.array(supercell).reshape(3, 1)

        structure = {
            "lattice": final_lattice.tolist(),
            "species": species,
            "coords": coordinates,
            "structure_type": structure_type,
            "lattice_parameter": lattice_parameter,
            "supercell": supercell
        }

        logger.debug(f"Generated {structure_type} structure with {len(species)} atoms")

        return structure

    def scale_structure_volume(
        self,
        structure: Dict[str, Any],
        volume_scale_factor: float
    ) -> Dict[str, Any]:
        """
        Scale structure volume by a given factor.

        Args:
            structure: Original structure dictionary
            volume_scale_factor: Volume scaling factor

        Returns:
            Structure with scaled lattice vectors
        """
        # Volume scales as the cube of linear dimension
        linear_scale = volume_scale_factor ** (1.0/3.0)

        scaled_structure = structure.copy()
        scaled_structure["lattice"] = [
            [vec[0] * linear_scale, vec[1] * linear_scale, vec[2] * linear_scale]
            for vec in structure["lattice"]
        ]

        if "lattice_parameter" in structure:
            scaled_structure["lattice_parameter"] = structure["lattice_parameter"] * linear_scale

        return scaled_structure

    def generate_volume_series(
        self,
        base_structure: Dict[str, Any],
        volume_range: tuple = (0.8, 1.2),
        num_points: int = 11
    ) -> List[Dict[str, Any]]:
        """
        Generate a series of structures with different volumes for EOS calculation.

        Args:
            base_structure: Base crystal structure
            volume_range: Tuple of (min_scale, max_scale) for volume
            num_points: Number of volume points to generate

        Returns:
            List of structures with scaled volumes
        """
        volume_scales = np.linspace(volume_range[0], volume_range[1], num_points)

        structures = []
        for scale in volume_scales:
            scaled_structure = self.scale_structure_volume(base_structure, scale)
            scaled_structure["volume_scale"] = scale
            structures.append(scaled_structure)

        logger.info(f"Generated {len(structures)} volume-scaled structures")

        return structures

    def get_structure_info(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get information about a crystal structure.

        Args:
            structure: Structure dictionary

        Returns:
            Dictionary with structure information
        """
        lattice = np.array(structure["lattice"])
        volume = np.abs(np.linalg.det(lattice))

        unique_species = list(set(structure["species"]))
        composition = {species: structure["species"].count(species) for species in unique_species}

        return {
            "volume": volume,
            "num_atoms": len(structure["species"]),
            "unique_species": unique_species,
            "composition": composition,
            "structure_type": structure.get("structure_type", "unknown"),
            "lattice_parameter": structure.get("lattice_parameter"),
            "density": len(structure["species"]) / volume
        }

    def validate_structure(self, structure: Dict[str, Any]) -> tuple[bool, str]:
        """
        Validate a crystal structure.

        Args:
            structure: Structure to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_keys = ["lattice", "species", "coords"]

        for key in required_keys:
            if key not in structure:
                return False, f"Missing required key: {key}"

        # Check dimensions
        if len(structure["lattice"]) != 3:
            return False, "Lattice must have 3 vectors"

        for i, vec in enumerate(structure["lattice"]):
            if len(vec) != 3:
                return False, f"Lattice vector {i} must have 3 components"

        # Check species and coordinates match
        if len(structure["species"]) != len(structure["coords"]):
            return False, "Number of species and coordinates must match"

        # Check coordinate dimensions
        for i, coord in enumerate(structure["coords"]):
            if len(coord) != 3:
                return False, f"Coordinate {i} must have 3 components"

        # Check lattice determinant (non-zero volume)
        lattice = np.array(structure["lattice"])
        if np.abs(np.linalg.det(lattice)) < 1e-10:
            return False, "Lattice vectors are linearly dependent (zero volume)"

        return True, ""

    def get_available_structures(self) -> List[str]:
        """Get list of available structure types."""
        return list(self.structure_templates.keys())