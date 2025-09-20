"""
Software execution engine for ATLAS-QE workflow system.

This module provides the core execution capabilities for running
ATLAS and QE calculations, including process management, monitoring,
and result collection.
"""

import os
import time
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import threading
import queue
import signal

logger = logging.getLogger(__name__)


class ExecutionStatus(Enum):
    """Status of calculation execution."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class ExecutionResult:
    """Result of a calculation execution."""
    status: ExecutionStatus
    return_code: int
    stdout: str
    stderr: str
    execution_time: float
    energy: Optional[float] = None
    convergence_achieved: bool = False
    error_message: Optional[str] = None


class SoftwareExecutor:
    """
    Base software execution interface.

    Provides process management, monitoring, and result collection
    for materials calculation software.
    """

    def __init__(
        self,
        executable_path: str,
        environment: Optional[Dict[str, str]] = None,
        timeout: float = 3600.0
    ):
        """
        Initialize software executor.

        Args:
            executable_path: Path to software executable
            environment: Environment variables for execution
            timeout: Default timeout for calculations (seconds)
        """
        self.executable_path = Path(executable_path)
        self.environment = environment or {}
        self.timeout = timeout

        # Verify executable exists and is executable
        if not self.executable_path.exists():
            raise FileNotFoundError(f"Executable not found: {self.executable_path}")

        if not os.access(self.executable_path, os.X_OK):
            raise PermissionError(f"Executable not executable: {self.executable_path}")

        logger.info(f"Initialized {self.__class__.__name__} with {self.executable_path}")

    def execute(
        self,
        input_file: Path,
        working_dir: Path,
        output_file: Optional[Path] = None,
        timeout: Optional[float] = None
    ) -> ExecutionResult:
        """
        Execute software calculation.

        Args:
            input_file: Input file for calculation
            working_dir: Working directory for execution
            output_file: Output file path (optional)
            timeout: Execution timeout (uses default if None)

        Returns:
            ExecutionResult with execution details and results
        """
        timeout = timeout or self.timeout

        # Prepare execution environment
        exec_env = os.environ.copy()
        exec_env.update(self.environment)

        # Prepare command
        command = self._prepare_command(input_file, output_file)

        logger.info(f"Executing: {' '.join(map(str, command))}")
        logger.info(f"Working directory: {working_dir}")
        logger.info(f"Timeout: {timeout}s")

        start_time = time.time()

        try:
            # Execute process
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                env=exec_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )

            # Monitor execution with timeout
            try:
                stdout, stderr = process.communicate(timeout=timeout)
                return_code = process.returncode
                execution_time = time.time() - start_time

                # Determine status
                if return_code == 0:
                    status = ExecutionStatus.COMPLETED
                else:
                    status = ExecutionStatus.FAILED

                # Parse results
                energy, converged = self._parse_output(stdout, stderr, working_dir)

                return ExecutionResult(
                    status=status,
                    return_code=return_code,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    energy=energy,
                    convergence_achieved=converged
                )

            except subprocess.TimeoutExpired:
                # Kill process on timeout
                if os.name != 'nt':
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()

                stdout, stderr = process.communicate()
                execution_time = time.time() - start_time

                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    return_code=-1,
                    stdout=stdout,
                    stderr=stderr,
                    execution_time=execution_time,
                    error_message=f"Execution timed out after {timeout}s"
                )

        except Exception as e:
            execution_time = time.time() - start_time
            error_msg = f"Execution failed: {str(e)}"
            logger.error(error_msg)

            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                return_code=-1,
                stdout="",
                stderr=error_msg,
                execution_time=execution_time,
                error_message=error_msg
            )

    def _prepare_command(self, input_file: Path, output_file: Optional[Path]) -> List[str]:
        """
        Prepare execution command. Override in subclasses.

        Args:
            input_file: Input file path
            output_file: Output file path

        Returns:
            Command as list of strings
        """
        return [str(self.executable_path), str(input_file)]

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        working_dir: Path
    ) -> Tuple[Optional[float], bool]:
        """
        Parse execution output for energy and convergence.
        Override in subclasses.

        Args:
            stdout: Standard output
            stderr: Standard error
            working_dir: Working directory

        Returns:
            Tuple of (energy, convergence_achieved)
        """
        return None, False


class ATLASExecutor(SoftwareExecutor):
    """ATLAS software executor."""

    def _prepare_command(self, input_file: Path, output_file: Optional[Path]) -> List[str]:
        """Prepare ATLAS execution command."""
        command = [str(self.executable_path)]

        if input_file.name != 'atlas.in':
            # If input file is not atlas.in, specify it
            command.extend(['-i', str(input_file)])

        if output_file:
            command.extend(['-o', str(output_file)])

        return command

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        working_dir: Path
    ) -> Tuple[Optional[float], bool]:
        """Parse ATLAS output for energy and convergence."""
        energy = None
        converged = False

        # Look for final energy in stdout
        energy_patterns = [
            r'Final\s+energy:\s*([+-]?\d+\.?\d*)',
            r'Total\s+energy:\s*([+-]?\d+\.?\d*)',
            r'E_total\s*=\s*([+-]?\d+\.?\d*)'
        ]

        for pattern in energy_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    energy = float(matches[-1])  # Take the last match
                    break
                except ValueError:
                    continue

        # Check for convergence indicators
        convergence_patterns = [
            r'convergence\s+achieved',
            r'calculation\s+converged',
            r'SCF\s+converged'
        ]

        for pattern in convergence_patterns:
            if re.search(pattern, stdout, re.IGNORECASE):
                converged = True
                break

        # Also check output files if they exist
        output_file = working_dir / 'atlas.out'
        if output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    file_content = f.read()

                # Try to extract energy from output file
                if energy is None:
                    for pattern in energy_patterns:
                        matches = re.findall(pattern, file_content, re.IGNORECASE)
                        if matches:
                            try:
                                energy = float(matches[-1])
                                break
                            except ValueError:
                                continue

                # Check convergence in output file
                if not converged:
                    for pattern in convergence_patterns:
                        if re.search(pattern, file_content, re.IGNORECASE):
                            converged = True
                            break

            except Exception as e:
                logger.warning(f"Failed to read ATLAS output file: {e}")

        logger.debug(f"ATLAS parsing result: energy={energy}, converged={converged}")
        return energy, converged


class QEExecutor(SoftwareExecutor):
    """Quantum ESPRESSO executor."""

    def __init__(self, **kwargs):
        """Initialize QE executor."""
        # QE often requires MPI
        self.mpi_command = kwargs.pop('mpi_command', 'mpirun')
        self.num_cores = kwargs.pop('num_cores', 1)
        super().__init__(**kwargs)

    def _prepare_command(self, input_file: Path, output_file: Optional[Path]) -> List[str]:
        """Prepare QE execution command."""
        command = []

        # Add MPI if more than 1 core
        if self.num_cores > 1:
            command.extend([self.mpi_command, '-np', str(self.num_cores)])

        command.append(str(self.executable_path))

        # QE reads from stdin, writes to stdout
        command.extend(['-input', str(input_file)])

        return command

    def _parse_output(
        self,
        stdout: str,
        stderr: str,
        working_dir: Path
    ) -> Tuple[Optional[float], bool]:
        """Parse QE output for energy and convergence."""
        energy = None
        converged = False

        # QE energy patterns
        energy_patterns = [
            r'!\s*total\s+energy\s*=\s*([+-]?\d+\.?\d*)\s*Ry',
            r'total\s+energy\s*=\s*([+-]?\d+\.?\d*)\s*Ry',
            r'Final\s+energy\s*=\s*([+-]?\d+\.?\d*)\s*Ry'
        ]

        for pattern in energy_patterns:
            matches = re.findall(pattern, stdout, re.IGNORECASE)
            if matches:
                try:
                    energy = float(matches[-1])
                    break
                except ValueError:
                    continue

        # QE convergence patterns
        convergence_patterns = [
            r'convergence\s+has\s+been\s+achieved',
            r'End\s+of\s+self-consistent\s+calculation',
            r'JOB\s+DONE'
        ]

        for pattern in convergence_patterns:
            if re.search(pattern, stdout, re.IGNORECASE):
                converged = True
                break

        logger.debug(f"QE parsing result: energy={energy}, converged={converged}")
        return energy, converged


class ExecutionManager:
    """
    Manages parallel execution of multiple calculations.

    Provides task scheduling, monitoring, and resource management
    for running multiple calculations concurrently.
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        atlas_config: Optional[Dict[str, Any]] = None,
        qe_config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize execution manager.

        Args:
            max_concurrent: Maximum number of concurrent calculations
            atlas_config: ATLAS executor configuration
            qe_config: QE executor configuration
        """
        self.max_concurrent = max_concurrent
        self.atlas_config = atlas_config or {}
        self.qe_config = qe_config or {}

        # Initialize executors
        self.executors = {}
        self._initialize_executors()

        # Task management
        self.active_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []

    def _initialize_executors(self):
        """Initialize software executors."""
        # Initialize ATLAS executor if configured
        if 'executable_path' in self.atlas_config:
            try:
                self.executors['atlas'] = ATLASExecutor(**self.atlas_config)
                logger.info("ATLAS executor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize ATLAS executor: {e}")

        # Initialize QE executor if configured
        if 'executable_path' in self.qe_config:
            try:
                self.executors['qe'] = QEExecutor(**self.qe_config)
                logger.info("QE executor initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize QE executor: {e}")

    def execute_calculation(
        self,
        software: str,
        input_file: Path,
        working_dir: Path,
        task_id: Optional[str] = None
    ) -> ExecutionResult:
        """
        Execute a single calculation.

        Args:
            software: Software name ('atlas' or 'qe')
            input_file: Input file path
            working_dir: Working directory
            task_id: Optional task identifier

        Returns:
            ExecutionResult
        """
        if software not in self.executors:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                return_code=-1,
                stdout="",
                stderr=f"No executor available for software: {software}",
                execution_time=0.0,
                error_message=f"No executor available for software: {software}"
            )

        executor = self.executors[software]
        task_id = task_id or f"{software}_{time.time()}"

        logger.info(f"Starting calculation {task_id} with {software}")

        result = executor.execute(input_file, working_dir)

        if result.status == ExecutionStatus.COMPLETED:
            self.completed_tasks.append((task_id, result))
            logger.info(f"Calculation {task_id} completed successfully")
        else:
            self.failed_tasks.append((task_id, result))
            logger.warning(f"Calculation {task_id} failed: {result.error_message}")

        return result

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of execution manager state."""
        return {
            'max_concurrent': self.max_concurrent,
            'available_executors': list(self.executors.keys()),
            'active_tasks': len(self.active_tasks),
            'completed_tasks': len(self.completed_tasks),
            'failed_tasks': len(self.failed_tasks)
        }