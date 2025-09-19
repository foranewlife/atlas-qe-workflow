"""
Base calculator interface for materials calculation software.

This module defines the abstract interface that all calculator implementations
must follow, ensuring consistent behavior across different software packages.
"""

import os
import time
import shutil
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class CalculationStatus(Enum):
    """Status of a calculation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class CalculationResult:
    """
    Result of a materials calculation.

    Contains both the numerical results and metadata about the calculation.
    """
    status: CalculationStatus
    energy_data: Optional[Dict[str, Any]] = None  # E vs V data
    optimized_structure: Optional[Dict[str, Any]] = None
    eos_parameters: Optional[Dict[str, float]] = None  # E0, V0, B0, etc.
    computational_details: Optional[Dict[str, Any]] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    output_files: Optional[List[str]] = None


class BaseCalculator(ABC):
    """
    Abstract base class for materials calculation software interfaces.

    All calculator implementations (ATLAS, QE, etc.) must inherit from this
    class and implement the required abstract methods.
    """

    def __init__(
        self,
        executable_path: str,
        working_directory: Optional[Path] = None,
        mpi_command: Optional[str] = None,
        num_cores: int = 1
    ):
        """
        Initialize the calculator.

        Args:
            executable_path: Path to the software executable
            working_directory: Base directory for calculations
            mpi_command: MPI command for parallel execution (e.g., "mpirun")
            num_cores: Number of CPU cores to use
        """
        self.executable_path = Path(executable_path)
        self.working_directory = working_directory or Path.cwd()
        self.mpi_command = mpi_command
        self.num_cores = num_cores

        # Verify executable exists
        if not self.executable_path.exists():
            raise FileNotFoundError(f"Executable not found: {self.executable_path}")

        logger.info(f"Initialized {self.__class__.__name__} with executable: {self.executable_path}")

    @abstractmethod
    def generate_input_files(
        self,
        structure: Dict[str, Any],
        parameters: Dict[str, Any],
        workspace: Path
    ) -> List[str]:
        """
        Generate input files for the calculation.

        Args:
            structure: Crystal structure information
            parameters: Calculation parameters
            workspace: Directory where input files should be created

        Returns:
            List of created input file names
        """
        pass

    @abstractmethod
    def run_calculation(
        self,
        workspace: Path,
        timeout: Optional[float] = None
    ) -> CalculationResult:
        """
        Execute the calculation.

        Args:
            workspace: Directory containing input files
            timeout: Maximum execution time in seconds

        Returns:
            CalculationResult with status and data
        """
        pass

    @abstractmethod
    def parse_output(self, workspace: Path) -> Dict[str, Any]:
        """
        Parse calculation output files.

        Args:
            workspace: Directory containing output files

        Returns:
            Dictionary with parsed results
        """
        pass

    @abstractmethod
    def check_convergence(self, workspace: Path) -> Tuple[bool, str]:
        """
        Check if the calculation converged successfully.

        Args:
            workspace: Directory containing output files

        Returns:
            Tuple of (converged, status_message)
        """
        pass

    def prepare_workspace(self, workspace: Path) -> Path:
        """
        Prepare calculation workspace.

        Args:
            workspace: Target workspace directory

        Returns:
            Path to the prepared workspace
        """
        workspace = Path(workspace)
        workspace.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Prepared workspace: {workspace}")
        return workspace

    def cleanup_workspace(self, workspace: Path, keep_files: Optional[List[str]] = None):
        """
        Clean up workspace after calculation.

        Args:
            workspace: Workspace directory to clean
            keep_files: List of files/patterns to keep
        """
        if not workspace.exists():
            return

        if keep_files is None:
            keep_files = ["*.out", "*.log", "*.xml", "*.dat"]

        all_files = list(workspace.rglob("*"))
        files_to_remove = []

        for file_path in all_files:
            if file_path.is_file():
                keep_file = False
                for pattern in keep_files:
                    if file_path.match(pattern):
                        keep_file = True
                        break

                if not keep_file:
                    files_to_remove.append(file_path)

        for file_path in files_to_remove:
            try:
                file_path.unlink()
            except Exception as e:
                logger.warning(f"Failed to remove {file_path}: {e}")

        logger.debug(f"Cleaned {len(files_to_remove)} files from workspace")

    def execute_command(
        self,
        command: List[str],
        workspace: Path,
        timeout: Optional[float] = None
    ) -> Tuple[int, str, str]:
        """
        Execute a system command in the workspace.

        Args:
            command: Command and arguments to execute
            workspace: Working directory for execution
            timeout: Maximum execution time in seconds

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        # Prepare full command with MPI if needed
        if self.mpi_command and self.num_cores > 1:
            full_command = [self.mpi_command, "-np", str(self.num_cores)] + command
        else:
            full_command = command

        logger.info(f"Executing command: {' '.join(full_command)}")
        logger.debug(f"Working directory: {workspace}")

        try:
            start_time = time.time()
            result = subprocess.run(
                full_command,
                cwd=workspace,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            execution_time = time.time() - start_time

            logger.info(f"Command completed in {execution_time:.2f}s with return code {result.returncode}")

            return result.returncode, result.stdout, result.stderr

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout}s")
            return -1, "", "Command timed out"

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return -1, "", str(e)

    def copy_pseudopotentials(
        self,
        pseudopotential_files: List[str],
        workspace: Path,
        pseudopotential_base_path: Optional[Path] = None
    ):
        """
        Copy pseudopotential files to workspace.

        Args:
            pseudopotential_files: List of pseudopotential files to copy
            workspace: Target workspace directory
            pseudopotential_base_path: Source directory for pseudopotentials
        """
        if pseudopotential_base_path is None:
            pseudopotential_base_path = Path("pseudopotentials")

        for pp_file in pseudopotential_files:
            source_path = pseudopotential_base_path / pp_file
            target_path = workspace / pp_file

            if source_path.exists():
                shutil.copy2(source_path, target_path)
                logger.debug(f"Copied pseudopotential: {pp_file}")
            else:
                logger.warning(f"Pseudopotential file not found: {source_path}")

    def validate_structure(self, structure: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate crystal structure data.

        Args:
            structure: Structure dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["lattice", "species", "coords"]

        for field in required_fields:
            if field not in structure:
                return False, f"Missing required field: {field}"

        # Basic validation
        if not structure["species"]:
            return False, "No atomic species defined"

        if not structure["coords"]:
            return False, "No atomic coordinates defined"

        if len(structure["species"]) != len(structure["coords"]):
            return False, "Mismatch between number of species and coordinates"

        return True, ""

    def get_calculation_summary(self, workspace: Path) -> Dict[str, Any]:
        """
        Get a summary of the calculation in the workspace.

        Args:
            workspace: Workspace directory

        Returns:
            Dictionary with calculation summary
        """
        summary = {
            "workspace": str(workspace),
            "calculator": self.__class__.__name__,
            "executable": str(self.executable_path),
            "num_cores": self.num_cores,
            "files": []
        }

        if workspace.exists():
            summary["files"] = [f.name for f in workspace.iterdir() if f.is_file()]

        return summary

    def estimate_memory_usage(self, structure: Dict[str, Any], parameters: Dict[str, Any]) -> float:
        """
        Estimate memory usage for the calculation.

        Args:
            structure: Crystal structure
            parameters: Calculation parameters

        Returns:
            Estimated memory usage in GB
        """
        # Base implementation provides conservative estimate
        # Subclasses should override with software-specific estimates
        num_atoms = len(structure.get("species", []))
        base_memory = 0.5  # GB base memory

        # Scale with system size
        memory_per_atom = 0.1  # GB per atom (conservative)
        total_memory = base_memory + num_atoms * memory_per_atom

        return total_memory

    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get default calculation parameters.

        Returns:
            Dictionary with default parameters for this calculator
        """
        # Base implementation - subclasses should override
        return {}

    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate calculation parameters.

        Args:
            parameters: Parameters to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Base implementation - subclasses should provide specific validation
        return True, ""