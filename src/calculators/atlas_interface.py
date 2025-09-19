"""
ATLAS (OFDFT) calculator interface.

This module provides a complete interface to the ATLAS orbital-free density
functional theory software, including input file generation, execution
management, and output parsing.
"""

import re
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .base_calculator import BaseCalculator, CalculationResult, CalculationStatus

logger = logging.getLogger(__name__)


class ATLASCalculator(BaseCalculator):
    """
    Interface for ATLAS orbital-free density functional theory calculations.

    ATLAS specializes in large-scale materials simulations using orbital-free DFT,
    making it suitable for systems with thousands of atoms while maintaining
    reasonable computational cost.
    """

    def __init__(self, **kwargs):
        """Initialize ATLAS calculator."""
        super().__init__(**kwargs)

    def generate_input_files(
        self,
        structure: Dict[str, Any],
        parameters: Dict[str, Any],
        workspace: Path
    ) -> List[str]:
        """
        Generate ATLAS input files.

        ATLAS uses a single input file (atlas.in) containing all calculation
        parameters, structure definition, and pseudopotential information.

        Args:
            structure: Crystal structure information
            parameters: ATLAS calculation parameters
            workspace: Directory where input files should be created

        Returns:
            List of created input file names
        """
        workspace = self.prepare_workspace(workspace)

        # Validate inputs
        is_valid, error = self.validate_structure(structure)
        if not is_valid:
            raise ValueError(f"Invalid structure: {error}")

        is_valid, error = self.validate_parameters(parameters)
        if not is_valid:
            raise ValueError(f"Invalid parameters: {error}")

        # Generate POSCAR file (ATLAS uses VASP format)
        poscar_file = workspace / "POSCAR"
        self._write_poscar_file(poscar_file, structure, parameters)

        # Generate main input file
        input_file = workspace / "atlas.in"
        self._write_atlas_input(input_file, structure, parameters)

        # Copy pseudopotential files
        pp_files = parameters.get("pseudopotential_files", [])
        if pp_files:
            self.copy_pseudopotentials(pp_files, workspace)

        created_files = ["POSCAR", "atlas.in"] + pp_files
        logger.info(f"Generated ATLAS input files: {created_files}")

        return created_files

    def _write_atlas_input(self, input_file: Path, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write the main ATLAS input file."""
        with open(input_file, 'w') as f:
            # Write POSCAR file reference and elements
            self._write_poscar_reference(f, structure, parameters)

            # Basic calculation settings
            self._write_atlas_calculation_parameters(f, parameters)

    def _write_poscar_reference(self, f, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write POSCAR file reference and basic setup."""
        f.write("CELLFILE = POSCAR\n")

        # Elements list
        if "element_pp_mapping" in parameters:
            elements = list(parameters["element_pp_mapping"].keys())
        else:
            elements = structure.get("species", [])
            elements = list(dict.fromkeys(elements))  # Remove duplicates, preserve order

        f.write(f"ELEMENTS = {' '.join(elements)}\n")

        # Pseudopotential files
        pp_files = parameters.get("pseudopotential_files", [])
        if pp_files:
            f.write(f"ppfile = {' '.join(pp_files)}\n")

        f.write("\n# above gen by atlas-qe-workflow\n")

    def _write_atlas_calculation_parameters(self, f, parameters: Dict[str, Any]):
        """Write ATLAS-specific calculation parameters based on real examples."""
        # System identifier
        f.write("iSYSTEM = ATLAS_JOB\n")

        # Grid spacing (GAP parameter)
        gap = parameters.get("gap", parameters.get("grid_spacing", 0.20))
        f.write(f"GAP = {gap:.2f}\n")

        # KEDF functional
        functional = parameters.get("functional", "kedf701")
        if functional.lower().startswith("kedf"):
            kedf_num = functional.replace("kedf", "")
            f.write(f"KEDF = {kedf_num}\n")
        else:
            f.write("KEDF = 701\n")  # Default

        # Exchange-correlation
        f.write("exc = 1\n")

        # Standard ATLAS parameters from real examples
        f.write("mgp_type = MGPA\n")
        f.write("ScfIter = 300\n")
        f.write("L_NLPS_EDF = .false.\n")
        f.write("paraA = -1.8\n")
        f.write("Gammax = -1.0\n")
        f.write("nlpsf_ke = 8\n")
        f.write("hc_type = 2\n")
        f.write("HC_RR = 1.02\n")
        f.write("rHC_A = 0.45\n")
        f.write("rHC_B = 0.10\n")
        f.write("Lpbc = .true.\n")
        f.write("LCELL = 14\n")
        f.write("max_kfr = 3.0\n")
        f.write("min_kfr = 1E-5\n")
        f.write("LDAK_KF_NUM = 60\n")
        f.write("ildak = 9\n")
        f.write("LMSCF = .false.\n")
        f.write("LMIXXX = .true.\n")
        f.write("rz_rat = 10\n")
        f.write("NORDER = 10\n")
        f.write("LVW = F\n")
        f.write("lforce = .false.\n")
        f.write("lstress = .false.\n")
        f.write("lpden = .true.\n")
        f.write("i_guess = 0\n")
        f.write("IGOAL = 0\n")
        f.write("iopm = 1\n")
        f.write("nssp = 0\n")
        f.write("sfreq = 1\n")
        f.write("lpdebug = F\n")
        f.write("lsplineii = .true.\n")
        f.write("lspline = .true.\n")

        # Convergence tolerance
        scf_tol = parameters.get("scf_tolerance", 1e-7)
        f.write(f"scftol = {scf_tol:.0e}\n")

    def _write_poscar_file(self, poscar_file: Path, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write POSCAR file in VASP format for ATLAS."""
        with open(poscar_file, 'w') as f:
            # Get unique elements and their counts
            species = structure["species"]
            coords = structure["coords"]
            lattice = structure["lattice"]

            # Get unique elements in order of appearance
            unique_elements = []
            element_counts = {}
            for element in species:
                if element not in unique_elements:
                    unique_elements.append(element)
                    element_counts[element] = 0
                element_counts[element] += 1

            # Header: element names
            f.write(f"{' '.join(unique_elements)}\n")

            # Scaling factor
            f.write(" 1.0000000000000000\n")

            # Lattice vectors
            for vector in lattice:
                f.write(f"     {vector[0]:16.10f}    {vector[1]:16.10f}    {vector[2]:16.10f}\n")

            # Element names (again)
            f.write(f" {' '.join(unique_elements)}\n")

            # Element counts
            counts_str = '   '.join(str(element_counts[elem]) for elem in unique_elements)
            f.write(f"   {counts_str}\n")

            # Coordinate system (Direct = fractional)
            f.write("Direct\n")

            # Atomic positions grouped by element type
            for element in unique_elements:
                for i, (spec, coord) in enumerate(zip(species, coords)):
                    if spec == element:
                        f.write(f"  {coord[0]:16.13f}  {coord[1]:16.13f}  {coord[2]:16.13f}\n")

    def _write_calculation_section(self, f, parameters: Dict[str, Any]):
        """Write basic calculation settings."""
        f.write("# Basic Calculation Settings\n")
        f.write(f"TASK = {parameters.get('task', 'SCF')}\n")
        f.write(f"FUNCTIONAL = {parameters.get('functional', 'KEDF701')}\n")

        # Grid settings
        grid_spacing = parameters.get('grid_spacing', 0.20)
        f.write(f"GAP = {grid_spacing}\n")

        # Periodic boundary conditions
        f.write("PERIODIC = T T T\n")

        f.write("\n")

    def _write_structure_section(self, f, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write crystal structure definition."""
        f.write("# Crystal Structure\n")

        # Lattice vectors
        lattice = structure["lattice"]
        f.write("LATTICE\n")
        for i, vector in enumerate(lattice):
            f.write(f"{vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")

        # Atomic species and positions
        species = structure["species"]
        coords = structure["coords"]

        f.write(f"\nNATOM = {len(species)}\n")
        f.write("COORDS\n")

        for i, (spec, coord) in enumerate(zip(species, coords)):
            # Convert species to atomic number if needed
            atomic_num = self._get_atomic_number(spec)
            f.write(f"{atomic_num} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")

        f.write("\n")

    def _write_pseudopotential_section(self, f, parameters: Dict[str, Any]):
        """Write pseudopotential file specification."""
        f.write("# Pseudopotentials\n")

        pp_files = parameters.get("pseudopotential_files", [])
        if pp_files:
            f.write(f"PPFILE = {' '.join(pp_files)}\n")
        else:
            logger.warning("No pseudopotential files specified")

        f.write("\n")

    def _write_kedf_section(self, f, parameters: Dict[str, Any]):
        """Write kinetic energy density functional settings."""
        f.write("# Kinetic Energy Density Functional\n")

        functional = parameters.get('functional', 'KEDF701')
        f.write(f"KEDF = {functional}\n")

        # Additional KEDF parameters if specified
        if 'kedf_alpha' in parameters:
            f.write(f"KEDF_ALPHA = {parameters['kedf_alpha']}\n")

        if 'kedf_beta' in parameters:
            f.write(f"KEDF_BETA = {parameters['kedf_beta']}\n")

        f.write("\n")

    def _write_convergence_section(self, f, parameters: Dict[str, Any]):
        """Write convergence criteria and optimization settings."""
        f.write("# Convergence Settings\n")

        # SCF convergence
        energy_tol = parameters.get('energy_tolerance', 1e-6)
        f.write(f"ETOL = {energy_tol}\n")

        density_tol = parameters.get('density_tolerance', 1e-5)
        f.write(f"DTOL = {density_tol}\n")

        max_scf = parameters.get('max_scf_iterations', 100)
        f.write(f"MAXSCF = {max_scf}\n")

        # Mixing parameters
        mixing_alpha = parameters.get('mixing_alpha', 0.3)
        f.write(f"MIXALPHA = {mixing_alpha}\n")

        f.write("\n")

    def _write_output_section(self, f, parameters: Dict[str, Any]):
        """Write output file specifications."""
        f.write("# Output Settings\n")
        f.write("PRINT_DENSITY = T\n")
        f.write("PRINT_POTENTIAL = T\n")
        f.write("DENSFILE = DENSFILE\n")
        f.write("POTFILE = POTFILE\n")
        f.write("\n")

    def run_calculation(
        self,
        workspace: Path,
        timeout: Optional[float] = None
    ) -> CalculationResult:
        """
        Execute ATLAS calculation.

        Args:
            workspace: Directory containing input files
            timeout: Maximum execution time in seconds

        Returns:
            CalculationResult with calculation status and data
        """
        if timeout is None:
            timeout = 3600  # Default 1 hour timeout

        # Check for required input files
        input_file = workspace / "atlas.in"
        if not input_file.exists():
            return CalculationResult(
                status=CalculationStatus.FAILED,
                error_message="ATLAS input file not found"
            )

        # Execute ATLAS
        command = [str(self.executable_path)]
        return_code, stdout, stderr = self.execute_command(command, workspace, timeout)

        # Determine calculation status
        if return_code == 0:
            converged, message = self.check_convergence(workspace)
            status = CalculationStatus.COMPLETED if converged else CalculationStatus.FAILED
        else:
            status = CalculationStatus.FAILED
            message = stderr or "ATLAS execution failed"

        # Parse output if successful
        energy_data = None
        computational_details = None

        if status == CalculationStatus.COMPLETED:
            try:
                parsed_results = self.parse_output(workspace)
                energy_data = parsed_results.get("energy_data")
                computational_details = parsed_results.get("computational_details")
            except Exception as e:
                logger.warning(f"Failed to parse ATLAS output: {e}")
                status = CalculationStatus.FAILED
                message = str(e)

        return CalculationResult(
            status=status,
            energy_data=energy_data,
            computational_details=computational_details,
            error_message=message if status == CalculationStatus.FAILED else None,
            output_files=["atlas.out", "DENSFILE", "POTFILE"]
        )

    def parse_output(self, workspace: Path) -> Dict[str, Any]:
        """
        Parse ATLAS output files.

        Args:
            workspace: Directory containing output files

        Returns:
            Dictionary with parsed calculation results
        """
        results = {}

        # Parse main output file
        output_file = workspace / "atlas.out"
        if output_file.exists():
            results.update(self._parse_main_output(output_file))

        # Parse density file if available
        density_file = workspace / "DENSFILE"
        if density_file.exists():
            results["density_available"] = True
            results["density_file"] = str(density_file)

        return results

    def _parse_main_output(self, output_file: Path) -> Dict[str, Any]:
        """Parse the main ATLAS output file based on real format."""
        results = {
            "computational_details": {},
            "scf_history": [],
            "final_energy": None,
            "final_energy_per_atom": None,
            "convergence_achieved": False,
            "energy_components": {}
        }

        with open(output_file, 'r') as f:
            content = f.read()

        # Parse SCF convergence history - ATLAS format: TN : iteration energy energy_per_atom delta_e
        scf_pattern = r"TN\s*:\s*(\d+)\s+([-\d\.E\+\-]+)\s+([-\d\.E\+\-]+)\s+([-\d\.E\+\-]+)"
        scf_matches = re.findall(scf_pattern, content)

        for match in scf_matches:
            iteration, energy, energy_per_atom, delta_e = match
            results["scf_history"].append({
                "iteration": int(iteration),
                "energy": float(energy),
                "energy_per_atom": float(energy_per_atom),
                "delta_energy": float(delta_e)
            })

        # Extract final total energy from energy summary
        total_energy_pattern = r"Total Energy\s*=\s*([-\d\.E\+\-]+)"
        total_energy_match = re.search(total_energy_pattern, content)
        if total_energy_match:
            results["final_energy"] = float(total_energy_match.group(1))

        # Extract final energy per atom
        energy_per_atom_pattern = r"Total Energy/atom\s*=\s*([-\d\.E\+\-]+)"
        energy_per_atom_match = re.search(energy_per_atom_pattern, content)
        if energy_per_atom_match:
            results["final_energy_per_atom"] = float(energy_per_atom_match.group(1))

        # Parse energy components
        self._parse_energy_components(content, results["energy_components"])

        # Check convergence - ATLAS typically converges when delta_e is small
        if results["scf_history"]:
            last_delta = abs(results["scf_history"][-1]["delta_energy"])
            results["convergence_achieved"] = last_delta < 1e-6  # Default convergence criterion

        # Parse computational details
        self._parse_computational_details(content, results["computational_details"])

        return results

    def _parse_energy_components(self, content: str, components: Dict[str, float]):
        """Parse energy components from ATLAS output."""
        component_patterns = {
            "exchange_correlation": r"Exchange-Correlation Energy\s*=\s*([-\d\.E\+\-]+)",
            "coulomb": r"Coulomb Energy\s*=\s*([-\d\.E\+\-]+)",
            "ion_electron_local": r"Ion-Electron local Energy\s*=\s*([-\d\.E\+\-]+)",
            "ewald": r"Ewald Energy\s*=\s*([-\d\.E\+\-]+)",
            "ion_electron_nonlocal": r"Ion-Electron Nonlocal local En\s*=\s*([-\d\.E\+\-]+)",
            "pv": r"PV\s*=\s*([-\d\.E\+\-]+)",
            "enthalpy": r"Enthalpy\s*=\s*([-\d\.E\+\-]+)"
        }

        for component, pattern in component_patterns.items():
            match = re.search(pattern, content)
            if match:
                components[component] = float(match.group(1))

    def _parse_computational_details(self, content: str, details: Dict[str, Any]):
        """Extract computational details from output."""
        # Grid information
        grid_pattern = r"Grid dimensions:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)"
        grid_match = re.search(grid_pattern, content)
        if grid_match:
            details["grid_dimensions"] = [int(x) for x in grid_match.groups()]

        # Memory usage
        memory_pattern = r"Memory usage:\s*([\d\.]+)\s*GB"
        memory_match = re.search(memory_pattern, content)
        if memory_match:
            details["memory_usage_gb"] = float(memory_match.group(1))

        # Timing information
        timing_pattern = r"Total time:\s*([\d\.]+)\s*seconds"
        timing_match = re.search(timing_pattern, content)
        if timing_match:
            details["total_time_seconds"] = float(timing_match.group(1))

    def check_convergence(self, workspace: Path) -> Tuple[bool, str]:
        """
        Check if ATLAS calculation converged successfully.

        Args:
            workspace: Directory containing output files

        Returns:
            Tuple of (converged, status_message)
        """
        output_file = workspace / "atlas.out"
        if not output_file.exists():
            return False, "Output file not found"

        try:
            with open(output_file, 'r') as f:
                content = f.read()

            # Check for successful convergence
            if "SCF Converged" in content:
                return True, "SCF converged successfully"

            # Check for specific error conditions
            if "Maximum SCF iterations exceeded" in content:
                return False, "SCF convergence not achieved within maximum iterations"

            if "Energy not decreasing" in content:
                return False, "SCF energy not decreasing"

            if "Error" in content or "FATAL" in content:
                return False, "Error detected in calculation"

            return False, "Convergence status unclear"

        except Exception as e:
            return False, f"Error checking convergence: {e}"

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate ATLAS-specific parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        required_params = ["functional", "pseudopotential_files"]
        for param in required_params:
            if param not in parameters:
                return False, f"Missing required parameter: {param}"

        # Validate functional
        valid_functionals = ["KEDF701", "KEDF801", "WGC", "TF"]
        functional = parameters.get("functional", "")
        if functional not in valid_functionals:
            return False, f"Invalid functional: {functional}. Valid options: {valid_functionals}"

        # Validate grid spacing
        gap = parameters.get("grid_spacing", parameters.get("gap", 0.20))
        if not (0.1 <= gap <= 0.5):
            return False, f"Grid spacing (gap) should be between 0.1 and 0.5 Å, got {gap}"

        # Validate pseudopotential files
        pp_files = parameters.get("pseudopotential_files", [])
        if not pp_files:
            return False, "No pseudopotential files specified"

        for pp_file in pp_files:
            if not pp_file.endswith(".recpot"):
                return False, f"ATLAS pseudopotential file should end with .recpot: {pp_file}"

        return True, ""

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default ATLAS calculation parameters."""
        return {
            "task": "SCF",
            "functional": "KEDF701",
            "grid_spacing": 0.20,
            "energy_tolerance": 1e-6,
            "density_tolerance": 1e-5,
            "max_scf_iterations": 100,
            "mixing_alpha": 0.3,
        }

    def estimate_memory_usage(self, structure: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """
        Estimate memory usage for ATLAS calculation.

        Args:
            structure: Crystal structure
            parameters: Calculation parameters

        Returns:
            Estimated memory usage in GB
        """
        # Get lattice volume
        lattice = np.array(structure["lattice"])
        volume = np.abs(np.linalg.det(lattice))

        # Grid spacing determines memory requirements
        gap = parameters.get("grid_spacing", parameters.get("gap", 0.20))
        grid_points_per_dimension = int(np.ceil((volume ** (1/3)) / gap))
        total_grid_points = grid_points_per_dimension ** 3

        # Memory per grid point (density, potential, etc.)
        memory_per_point = 8 * 8  # 8 bytes per double, ~8 arrays
        total_memory_bytes = total_grid_points * memory_per_point

        # Convert to GB and add overhead
        base_memory_gb = total_memory_bytes / (1024**3)
        overhead_factor = 1.5  # 50% overhead for other data structures

        return base_memory_gb * overhead_factor

    def _get_atomic_number(self, species: str) -> int:
        """Convert element symbol to atomic number."""
        element_map = {
            'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8,
            'F': 9, 'Ne': 10, 'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15,
            'S': 16, 'Cl': 17, 'Ar': 18, 'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22,
            'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29,
            'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34, 'Br': 35, 'Kr': 36,
            'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Tc': 43,
            'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
        }

        return element_map.get(species, 1)  # Default to hydrogen if unknown