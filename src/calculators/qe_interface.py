"""
Quantum ESPRESSO (QE) calculator interface.

This module provides a complete interface to Quantum ESPRESSO for DFT
calculations, including SCF, structure optimization, and property calculations.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging

from .base_calculator import BaseCalculator, CalculationResult, CalculationStatus

logger = logging.getLogger(__name__)


class QECalculator(BaseCalculator):
    """
    Interface for Quantum ESPRESSO density functional theory calculations.

    QE provides highly accurate DFT calculations with excellent convergence
    properties and extensive functionality for materials property calculations.
    """

    def __init__(self, **kwargs):
        """Initialize QE calculator."""
        super().__init__(**kwargs)

    def generate_input_files(
        self,
        structure: Dict[str, Any],
        parameters: Dict[str, Any],
        workspace: Path
    ) -> List[str]:
        """
        Generate QE input files.

        QE uses separate input files for different calculation types:
        - scf.in: Self-consistent field calculation
        - relax.in: Structure optimization (if needed)

        Args:
            structure: Crystal structure information
            parameters: QE calculation parameters
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

        created_files = []

        # Generate SCF input
        scf_input = workspace / "scf.in"
        self._write_scf_input(scf_input, structure, parameters)
        created_files.append("scf.in")

        # Generate structure optimization input if needed
        if parameters.get("calculate_forces", False):
            relax_input = workspace / "relax.in"
            self._write_relax_input(relax_input, structure, parameters)
            created_files.append("relax.in")

        # Copy pseudopotential files
        pp_files = parameters.get("pseudopotential_files", [])
        if pp_files:
            self.copy_pseudopotentials(pp_files, workspace)
            created_files.extend(pp_files)

        logger.info(f"Generated QE input files: {created_files}")
        return created_files

    def _write_scf_input(self, input_file: Path, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write QE SCF input file based on real format."""
        with open(input_file, 'w') as f:
            # Control section
            f.write("&CONTROL\n")
            f.write("   calculation      = 'scf',\n")
            f.write("   verbosity        = 'high',\n")
            f.write("   restart_mode     = 'from_scratch',\n")
            f.write("   tstress          = .t.,\n")
            f.write("   tprnfor          = .t.\n")
            f.write("   outdir           = './',\n")
            f.write(f"   prefix           = 'C',\n")  # Using 'C' as in real example
            f.write("   etot_conv_thr    = 1.0d-6\n")
            f.write("   forc_conv_thr    = 1.0d-5\n")
            f.write("   disk_io          = 'nowf',\n")
            f.write("   pseudo_dir       = '.'\n")
            f.write("/\n")

            # System section
            f.write("&SYSTEM\n")
            self._write_system_section(f, structure, parameters)
            f.write("/\n")

            # Electrons section
            f.write("&ELECTRONS\n")
            self._write_electrons_section(f, parameters)
            f.write("/\n")

            # Empty sections that appear in real files
            f.write("&IONS\n")
            f.write("/\n")
            f.write("&CELL\n")
            f.write("/\n")
            f.write("&FCP\n")
            f.write("/\n")
            f.write("&RISM\n")
            f.write("/\n")

            # Pseudopotentials section (optional, might be ignored)
            if "element_pp_mapping" in parameters:
                f.write("&PSEUDOPOTENTIALS\n")
                for element, pp_file in parameters["element_pp_mapping"].items():
                    f.write(f"   {element.lower()}               = {pp_file}\n")
                f.write("/\n")

            # Atomic species
            self._write_atomic_species(f, structure, parameters)

            # K-points
            self._write_k_points(f, parameters)

            # Cell parameters
            self._write_cell_parameters(f, structure)

            # Atomic positions
            self._write_atomic_positions(f, structure)

    def _write_relax_input(self, input_file: Path, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write QE structure optimization input file."""
        with open(input_file, 'w') as f:
            # Control section for relaxation
            f.write("&CONTROL\n")
            f.write(f"  calculation = 'vc-relax'\n")
            f.write(f"  prefix = 'qe_relax'\n")
            f.write(f"  outdir = './outdir'\n")
            f.write(f"  pseudo_dir = './'\n")
            f.write(f"  verbosity = 'high'\n")
            f.write("/\n\n")

            # System parameters
            f.write("&SYSTEM\n")
            self._write_system_section(f, structure, parameters)
            f.write("/\n\n")

            # Electronic parameters
            f.write("&ELECTRONS\n")
            self._write_electrons_section(f, parameters)
            f.write("/\n\n")

            # Ion optimization parameters
            f.write("&IONS\n")
            f.write("  ion_dynamics = 'bfgs'\n")
            f.write("/\n\n")

            # Cell optimization parameters
            f.write("&CELL\n")
            f.write("  cell_dynamics = 'bfgs'\n")
            f.write("  press = 0.0\n")
            f.write("/\n\n")

            # Rest of the input (same as SCF)
            self._write_atomic_species(f, structure, parameters)
            self._write_atomic_positions(f, structure)
            self._write_k_points(f, parameters)
            if structure.get("lattice"):
                self._write_cell_parameters(f, structure)

    def _write_system_section(self, f, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write the SYSTEM section based on real format."""
        # Basic system parameters
        num_atoms = len(structure["species"])
        unique_species = list(dict.fromkeys(structure["species"]))  # Preserve order

        f.write(f"   ibrav            = 0\n")  # Use CELL_PARAMETERS
        f.write(f"   nat              = {num_atoms}\n")
        f.write(f"   ntyp             = {len(unique_species)}\n")

        # Electronic structure parameters
        ecutwfc = parameters.get("ecutwfc", 60)
        f.write(f"   ecutwfc          = {ecutwfc}\n")

        # ecutrho from real example
        ecutrho = parameters.get("ecutrho", ecutwfc * 4)  # Default is 4x ecutwfc
        f.write(f"   ecutrho          = {ecutrho}\n")

        # Occupations (from real example)
        f.write("   occupations      = 'smearing'\n")
        f.write("   degauss          = 0.01\n")
        f.write("   smearing         = 'mp'\n")  # Methfessel-Paxton as in real example

        # Exchange-correlation functional (from real example using LDA)
        xc_functional = parameters.get("xc_functional", "LDA")
        if xc_functional == "LDA":
            f.write("   input_dft        = 'pz',\n")  # PZ = Perdew-Zunger LDA
        elif xc_functional == "PBE":
            f.write("   input_dft        = 'pbe',\n")
        elif xc_functional == "PBEsol":
            f.write("   input_dft        = 'pbesol',\n")

    def _write_electrons_section(self, f, parameters: Dict[str, Any]):
        """Write the ELECTRONS section based on real format."""
        # From real example
        electron_maxstep = parameters.get("electron_maxstep", 500)
        f.write(f"   electron_maxstep = {electron_maxstep}\n")

        conv_thr = parameters.get("conv_thr", 1e-7)  # Real example uses 1e-7
        f.write(f"   conv_thr         = {conv_thr:.1e}\n")

        # Mixing parameters from real example
        f.write("   mixing_mode      = 'local-TF'\n")
        mixing_beta = parameters.get("mixing_beta", 0.2)  # Real example uses 0.2
        f.write(f"   mixing_beta      = {mixing_beta}\n")

    def _write_atomic_species(self, f, structure: Dict[str, Any], parameters: Dict[str, Any]):
        """Write ATOMIC_SPECIES section."""
        f.write("ATOMIC_SPECIES\n")

        unique_species = list(set(structure["species"]))
        pp_files = parameters.get("pseudopotential_files", [])
        element_pp_mapping = parameters.get("element_pp_mapping", {})

        for i, species in enumerate(unique_species):
            mass = self._get_atomic_mass(species)

            # Get pseudopotential file for this species
            if element_pp_mapping and species in element_pp_mapping:
                pp_file = element_pp_mapping[species]
            elif i < len(pp_files):
                pp_file = pp_files[i]
            else:
                pp_file = f"{species}.upf"

            f.write(f"{species} {mass:.3f} {pp_file}\n")

        f.write("\n")

    def _write_atomic_positions(self, f, structure: Dict[str, Any]):
        """Write ATOMIC_POSITIONS section based on real format."""
        f.write("ATOMIC_POSITIONS angstrom\n")  # Real example uses angstrom

        # Convert fractional coordinates to Cartesian if needed
        lattice = structure["lattice"]
        coords = structure["coords"]
        species = structure["species"]

        import numpy as np
        lattice_matrix = np.array(lattice)

        for spec, coord in zip(species, coords):
            # Convert fractional to Cartesian coordinates
            cart_coord = np.dot(coord, lattice_matrix)
            f.write(f"{spec} {cart_coord[0]:.10f} {cart_coord[1]:.10f} {cart_coord[2]:.10f}  \n")

        f.write("\n")

    def _write_k_points(self, f, parameters: Dict[str, Any]):
        """Write K_POINTS section based on real format."""
        k_points = parameters.get("k_points", [9, 9, 9])  # Real example uses 9x9x9

        f.write("K_POINTS automatic\n")
        f.write(f"{k_points[0]} {k_points[1]} {k_points[2]}  0 0 0\n")
        f.write("\n")

    def _write_cell_parameters(self, f, structure: Dict[str, Any]):
        """Write CELL_PARAMETERS section."""
        f.write("CELL_PARAMETERS angstrom\n")

        lattice = structure["lattice"]
        for vector in lattice:
            f.write(f"{vector[0]:.6f} {vector[1]:.6f} {vector[2]:.6f}\n")

        f.write("\n")

    def run_calculation(
        self,
        workspace: Path,
        timeout: Optional[float] = None
    ) -> CalculationResult:
        """
        Execute QE calculation.

        Args:
            workspace: Directory containing input files
            timeout: Maximum execution time in seconds

        Returns:
            CalculationResult with calculation status and data
        """
        if timeout is None:
            timeout = 7200  # Default 2 hour timeout

        # Run SCF calculation first
        scf_result = self._run_scf_calculation(workspace, timeout)

        if scf_result.status != CalculationStatus.COMPLETED:
            return scf_result

        # Run structure optimization if requested
        relax_input = workspace / "relax.in"
        if relax_input.exists():
            relax_result = self._run_relax_calculation(workspace, timeout)
            if relax_result.status == CalculationStatus.COMPLETED:
                # Combine results
                scf_result.optimized_structure = relax_result.optimized_structure

        return scf_result

    def _run_scf_calculation(self, workspace: Path, timeout: float) -> CalculationResult:
        """Run SCF calculation."""
        input_file = workspace / "scf.in"
        if not input_file.exists():
            return CalculationResult(
                status=CalculationStatus.FAILED,
                error_message="SCF input file not found"
            )

        # Execute pw.x
        command = [str(self.executable_path)]
        return_code, stdout, stderr = self.execute_command(
            command, workspace, timeout, input_file="scf.in"
        )

        # Check results
        if return_code == 0:
            converged, message = self.check_convergence(workspace)
            status = CalculationStatus.COMPLETED if converged else CalculationStatus.FAILED
        else:
            status = CalculationStatus.FAILED
            message = stderr or "QE execution failed"

        # Parse output
        energy_data = None
        computational_details = None

        if status == CalculationStatus.COMPLETED:
            try:
                parsed_results = self.parse_output(workspace)
                energy_data = parsed_results.get("energy_data")
                computational_details = parsed_results.get("computational_details")
            except Exception as e:
                logger.warning(f"Failed to parse QE output: {e}")

        return CalculationResult(
            status=status,
            energy_data=energy_data,
            computational_details=computational_details,
            error_message=message if status == CalculationStatus.FAILED else None,
            output_files=["scf.out", "outdir/"]
        )

    def _run_relax_calculation(self, workspace: Path, timeout: float) -> CalculationResult:
        """Run structure optimization calculation."""
        input_file = workspace / "relax.in"
        command = [str(self.executable_path)]
        return_code, stdout, stderr = self.execute_command(
            command, workspace, timeout, input_file="relax.in"
        )

        if return_code == 0:
            # Parse optimized structure
            optimized_structure = self._parse_optimized_structure(workspace)
            return CalculationResult(
                status=CalculationStatus.COMPLETED,
                optimized_structure=optimized_structure
            )
        else:
            return CalculationResult(
                status=CalculationStatus.FAILED,
                error_message="Structure optimization failed"
            )

    def execute_command(self, command: List[str], workspace: Path, timeout: float, input_file: str = None) -> Tuple[int, str, str]:
        """Execute QE command with input redirection."""
        if input_file:
            # QE reads from stdin
            full_command = command + [f"< {input_file}"]
        else:
            full_command = command

        return super().execute_command(full_command, workspace, timeout)

    def parse_output(self, workspace: Path) -> Dict[str, Any]:
        """Parse QE output files."""
        results = {}

        # Parse SCF output
        scf_output = workspace / "scf.out"
        if scf_output.exists():
            results.update(self._parse_scf_output(scf_output))

        # Parse XML output if available
        xml_file = workspace / "outdir" / "qe_calc.save" / "data-file-schema.xml"
        if xml_file.exists():
            results.update(self._parse_xml_output(xml_file))

        return results

    def _parse_scf_output(self, output_file: Path) -> Dict[str, Any]:
        """Parse SCF output file based on real QE format."""
        results = {
            "scf_history": [],
            "final_energy": None,
            "final_energy_ry": None,
            "convergence_achieved": False,
            "computational_details": {},
            "energy_components": {}
        }

        with open(output_file, 'r') as f:
            content = f.read()

        # Parse SCF iterations - QE format: "total energy = -XX.XXXXX Ry"
        scf_pattern = r"total energy\s*=\s*([-\d\.]+)\s*Ry"
        energies = re.findall(scf_pattern, content)

        for i, energy in enumerate(energies):
            energy_ry = float(energy)
            results["scf_history"].append({
                "iteration": i + 1,
                "energy_ry": energy_ry,
                "energy_ev": energy_ry * 13.6057  # Convert Ry to eV
            })

        # Final energy from the exclamation mark line: "!    total energy = -XX.XXXXX Ry"
        final_energy_pattern = r"!\s*total energy\s*=\s*([-\d\.]+)\s*Ry"
        final_energy_match = re.search(final_energy_pattern, content)
        if final_energy_match:
            energy_ry = float(final_energy_match.group(1))
            results["final_energy_ry"] = energy_ry
            results["final_energy"] = energy_ry * 13.6057  # Convert to eV

        # Parse energy components
        self._parse_qe_energy_components(content, results["energy_components"])

        # Check convergence
        if "convergence has been achieved" in content or final_energy_match:
            results["convergence_achieved"] = True

        # Parse computational details
        self._parse_qe_computational_details(content, results["computational_details"])

        return results

    def _parse_qe_energy_components(self, content: str, components: Dict[str, float]):
        """Parse QE energy components."""
        component_patterns = {
            "one_electron": r"one-electron contribution\s*=\s*([-\d\.]+)\s*Ry",
            "hartree": r"hartree contribution\s*=\s*([-\d\.]+)\s*Ry",
            "xc": r"xc contribution\s*=\s*([-\d\.]+)\s*Ry",
            "ewald": r"ewald contribution\s*=\s*([-\d\.]+)\s*Ry",
            "smearing": r"smearing contrib\. \(-TS\)\s*=\s*([-\d\.]+)\s*Ry",
            "internal_energy": r"internal energy E=F\+TS\s*=\s*([-\d\.]+)\s*Ry"
        }

        for component, pattern in component_patterns.items():
            match = re.search(pattern, content)
            if match:
                energy_ry = float(match.group(1))
                components[component + "_ry"] = energy_ry
                components[component + "_ev"] = energy_ry * 13.6057

    def _parse_xml_output(self, xml_file: Path) -> Dict[str, Any]:
        """Parse QE XML output for additional data."""
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            results = {}

            # Extract final energy from XML
            energy_elem = root.find(".//energy")
            if energy_elem is not None:
                total_energy = energy_elem.find("total")
                if total_energy is not None:
                    results["total_energy_xml"] = float(total_energy.text)

            return results

        except Exception as e:
            logger.warning(f"Failed to parse XML output: {e}")
            return {}

    def _parse_optimized_structure(self, workspace: Path) -> Optional[Dict[str, Any]]:
        """Parse optimized structure from relax calculation."""
        relax_output = workspace / "relax.out"
        if not relax_output.exists():
            return None

        # Implementation would parse final atomic positions and cell parameters
        # This is a simplified version
        return {"optimized": True, "source": "relax calculation"}

    def _parse_qe_computational_details(self, content: str, details: Dict[str, Any]):
        """Extract computational details from QE output."""
        # CPU time
        time_pattern = r"PWSCF\s+:\s*([\d\.]+)s CPU"
        time_match = re.search(time_pattern, content)
        if time_match:
            details["cpu_time_seconds"] = float(time_match.group(1))

        # Memory usage
        memory_pattern = r"Memory usage:\s*([\d\.]+)\s*MB"
        memory_match = re.search(memory_pattern, content)
        if memory_match:
            details["memory_usage_mb"] = float(memory_match.group(1))

    def check_convergence(self, workspace: Path) -> Tuple[bool, str]:
        """Check QE calculation convergence."""
        output_file = workspace / "scf.out"
        if not output_file.exists():
            return False, "Output file not found"

        try:
            with open(output_file, 'r') as f:
                content = f.read()

            if "convergence has been achieved" in content:
                return True, "SCF converged successfully"

            if "convergence NOT achieved" in content:
                return False, "SCF convergence not achieved"

            if "Maximum CPU time exceeded" in content:
                return False, "Calculation timed out"

            return False, "Convergence status unclear"

        except Exception as e:
            return False, f"Error checking convergence: {e}"

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate QE parameters."""
        # Check energy cutoff
        ecutwfc = parameters.get("ecutwfc", 60.0)
        if not (20.0 <= ecutwfc <= 200.0):
            return False, f"ecutwfc should be between 20 and 200 Ry, got {ecutwfc}"

        # Check k-points
        k_points = parameters.get("k_points", [8, 8, 8])
        if len(k_points) != 3 or any(k < 1 for k in k_points):
            return False, f"k_points should be 3 positive integers, got {k_points}"

        return True, ""

    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default QE parameters."""
        return {
            "ecutwfc": 60.0,
            "conv_thr": 1e-6,
            "mixing_beta": 0.7,
            "electron_maxstep": 100,
            "k_points": [8, 8, 8],
            "xc_functional": "PBE"
        }

    def _get_atomic_mass(self, species: str) -> float:
        """Get atomic mass for species."""
        masses = {
            'H': 1.008, 'He': 4.003, 'Li': 6.94, 'Be': 9.012, 'B': 10.81,
            'C': 12.011, 'N': 14.007, 'O': 15.999, 'F': 18.998, 'Ne': 20.180,
            'Na': 22.990, 'Mg': 24.305, 'Al': 26.982, 'Si': 28.085, 'P': 30.974,
            'S': 32.06, 'Cl': 35.45, 'Ar': 39.948, 'K': 39.098, 'Ca': 40.078,
            'Sc': 44.956, 'Ti': 47.867, 'V': 50.942, 'Cr': 51.996, 'Mn': 54.938,
            'Fe': 55.845, 'Co': 58.933, 'Ni': 58.693, 'Cu': 63.546, 'Zn': 65.38,
        }
        return masses.get(species, 1.0)