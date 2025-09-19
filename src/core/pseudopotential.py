"""
Pseudopotential management for multi-element systems.

This module handles:
- Pseudopotential combination validation
- Element-to-file mapping
- ATLAS and QE input file generation
- Pseudopotential file existence checking
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class PseudopotentialManager:
    """
    Manages pseudopotential combinations for multi-element systems.

    Ensures proper mapping between elements and pseudopotential files,
    validates combinations, and generates appropriate input file sections.
    """

    def __init__(self, pseudopotential_base_path: Optional[Path] = None):
        """
        Initialize pseudopotential manager.

        Args:
            pseudopotential_base_path: Base directory containing pseudopotential files
        """
        self.base_path = pseudopotential_base_path or Path("pseudopotentials")

    def resolve_pseudopotential_combination(
        self,
        elements: List[str],
        pp_set_definition: Dict[str, Any]
    ) -> Dict[str, str]:
        """
        Resolve pseudopotential combination to element-file mapping.

        Args:
            elements: List of elements in the system (e.g., ['Mg', 'Al'])
            pp_set_definition: Pseudopotential set definition from config

        Returns:
            Dict mapping element to pseudopotential file

        Example:
            Input: elements=['Mg', 'Al'], pp_set='lda_combo'
            Output: {'Mg': 'Mg_lda.recpot', 'Al': 'Al_lda.recpot'}
        """
        if "files" not in pp_set_definition:
            raise ValueError(f"Pseudopotential set definition missing 'files' key: {pp_set_definition}")

        pp_files = pp_set_definition["files"]

        if len(pp_files) != len(elements):
            raise ValueError(
                f"Number of pseudopotential files ({len(pp_files)}) "
                f"does not match number of elements ({len(elements)})"
            )

        # Create element-to-file mapping
        element_pp_mapping = {}
        for i, element in enumerate(elements):
            element_pp_mapping[element] = pp_files[i]

        logger.debug(f"Resolved pseudopotential mapping: {element_pp_mapping}")
        return element_pp_mapping

    def validate_pseudopotential_set(
        self,
        elements: List[str],
        pp_set_definition: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """
        Validate a pseudopotential set definition.

        Args:
            elements: List of elements in the system
            pp_set_definition: Pseudopotential set definition

        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check basic structure
            if "name" not in pp_set_definition:
                return False, "Missing 'name' field in pseudopotential set"

            if "files" not in pp_set_definition:
                return False, "Missing 'files' field in pseudopotential set"

            # Check file count matches element count
            pp_files = pp_set_definition["files"]
            if len(pp_files) != len(elements):
                return False, (
                    f"Number of files ({len(pp_files)}) does not match "
                    f"number of elements ({len(elements)})"
                )

            # Check file existence (if base path is set)
            if self.base_path and self.base_path.exists():
                for pp_file in pp_files:
                    full_path = self.base_path / pp_file
                    if not full_path.exists():
                        return False, f"Pseudopotential file not found: {full_path}"

            return True, ""

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def validate_all_pseudopotential_sets(
        self,
        config: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """
        Validate all pseudopotential sets in a configuration.

        Args:
            config: Complete system configuration

        Returns:
            Tuple of (all_valid, error_messages)
        """
        elements = config.get("elements", [])
        errors = []

        # Validate ATLAS pseudopotential sets
        if "atlas" in config:
            atlas_config = config["atlas"]

            # Single element systems (direct pseudopotentials list)
            if "pseudopotentials" in atlas_config:
                # For single elements, each pseudopotential is independent
                if len(elements) == 1:
                    for pp_file in atlas_config["pseudopotentials"]:
                        if self.base_path and self.base_path.exists():
                            full_path = self.base_path / pp_file
                            if not full_path.exists():
                                errors.append(f"ATLAS pseudopotential not found: {full_path}")

            # Multi-element systems (pseudopotential sets)
            if "pseudopotential_sets" in atlas_config:
                for pp_set in atlas_config["pseudopotential_sets"]:
                    is_valid, error = self.validate_pseudopotential_set(elements, pp_set)
                    if not is_valid:
                        errors.append(f"ATLAS pseudopotential set '{pp_set.get('name', 'unnamed')}': {error}")

        # Validate QE pseudopotential sets
        if "qe" in config:
            qe_config = config["qe"]

            # Single element systems
            if "pseudopotentials" in qe_config:
                if len(elements) == 1:
                    for pp_file in qe_config["pseudopotentials"]:
                        if self.base_path and self.base_path.exists():
                            full_path = self.base_path / pp_file
                            if not full_path.exists():
                                errors.append(f"QE pseudopotential not found: {full_path}")

            # Multi-element systems
            if "pseudopotential_sets" in qe_config:
                for pp_set in qe_config["pseudopotential_sets"]:
                    is_valid, error = self.validate_pseudopotential_set(elements, pp_set)
                    if not is_valid:
                        errors.append(f"QE pseudopotential set '{pp_set.get('name', 'unnamed')}': {error}")

        return len(errors) == 0, errors

    def generate_atlas_ppfile_line(self, element_pp_mapping: Dict[str, str]) -> str:
        """
        Generate ATLAS PPFILE line for input file.

        Args:
            element_pp_mapping: Dictionary mapping elements to pseudopotential files

        Returns:
            PPFILE line for ATLAS input

        Example:
            Input: {'Mg': 'Mg_lda.recpot', 'Al': 'Al_lda.recpot'}
            Output: "PPFILE = Mg_lda.recpot Al_lda.recpot"
        """
        # Order files by the element order provided
        pp_files = []
        for element in sorted(element_pp_mapping.keys()):
            pp_files.append(element_pp_mapping[element])

        return f"PPFILE = {' '.join(pp_files)}"

    def generate_qe_atomic_species(
        self,
        element_pp_mapping: Dict[str, str],
        atomic_masses: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Generate QE ATOMIC_SPECIES section.

        Args:
            element_pp_mapping: Dictionary mapping elements to pseudopotential files
            atomic_masses: Optional custom atomic masses

        Returns:
            ATOMIC_SPECIES section for QE input

        Example:
            Output:
            ATOMIC_SPECIES
            Mg 24.305 Mg_lda.recpot
            Al 26.982 Al_lda.recpot
        """
        # Default atomic masses
        default_masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
            'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
            'Ga': 69.723, 'Ge': 72.630, 'As': 74.922, 'Se': 78.971, 'Br': 79.904,
            'Kr': 83.798, 'Rb': 85.468, 'Sr': 87.62, 'Y': 88.906, 'Zr': 91.224,
        }

        if atomic_masses is None:
            atomic_masses = default_masses

        lines = ["ATOMIC_SPECIES"]
        for element in sorted(element_pp_mapping.keys()):
            mass = atomic_masses.get(element, 1.0)
            pp_file = element_pp_mapping[element]
            lines.append(f"{element} {mass:.3f} {pp_file}")

        return "\n".join(lines)

    def enumerate_pseudopotential_combinations(
        self,
        elements: List[str],
        software_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Enumerate all valid pseudopotential combinations for a software.

        Args:
            elements: List of elements in the system
            software_config: ATLAS or QE configuration section

        Returns:
            List of pseudopotential combination definitions
        """
        combinations = []

        # Single element systems
        if len(elements) == 1 and "pseudopotentials" in software_config:
            for pp_file in software_config["pseudopotentials"]:
                combinations.append({
                    "name": Path(pp_file).stem,  # Use filename without extension
                    "files": [pp_file]
                })

        # Multi-element systems
        if "pseudopotential_sets" in software_config:
            combinations.extend(software_config["pseudopotential_sets"])

        return combinations

    def get_pseudopotential_info(self, pp_file: str) -> Dict[str, Any]:
        """
        Extract information about a pseudopotential file.

        Args:
            pp_file: Pseudopotential filename

        Returns:
            Dictionary with pseudopotential metadata
        """
        pp_path = self.base_path / pp_file if self.base_path else Path(pp_file)

        info = {
            "filename": pp_file,
            "exists": pp_path.exists() if self.base_path else None,
            "size": pp_path.stat().st_size if pp_path.exists() else None,
            "type": self._infer_pseudopotential_type(pp_file)
        }

        return info

    def _infer_pseudopotential_type(self, pp_file: str) -> str:
        """
        Infer pseudopotential type from filename.

        Args:
            pp_file: Pseudopotential filename

        Returns:
            Inferred type (e.g., 'lda', 'pbe', 'paw', 'ncpp')
        """
        filename_lower = pp_file.lower()

        # ATLAS pseudopotentials
        if ".recpot" in filename_lower:
            if "lda" in filename_lower:
                return "lda"
            elif "pbe" in filename_lower:
                return "pbe"
            elif "pbesol" in filename_lower:
                return "pbesol"
            else:
                return "unknown_atlas"

        # QE pseudopotentials
        elif ".upf" in filename_lower:
            if "paw" in filename_lower:
                return "paw"
            elif "ncpp" in filename_lower or "nc" in filename_lower:
                return "ncpp"
            elif "uspp" in filename_lower or "us" in filename_lower:
                return "uspp"
            else:
                return "unknown_qe"

        return "unknown"