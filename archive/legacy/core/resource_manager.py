"""
Resource management system for distributed computing.

This module provides comprehensive resource management capabilities including
local and remote computational resources, load balancing, and intelligent
task allocation.
"""

import os
import time
import subprocess
import threading
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import psutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of computational resources."""
    LOCAL = "local"
    REMOTE = "remote"


class ResourceStatus(Enum):
    """Status of computational resources."""
    AVAILABLE = "available"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"


@dataclass
class ResourceCapability:
    """Capability description for a computational resource."""
    cores: int
    memory_gb: float
    supports_mpi: bool = True
    max_concurrent_jobs: int = 1
    preferred_software: List[str] = field(default_factory=list)


@dataclass
class ResourceMetrics:
    """Real-time metrics for a computational resource."""
    cpu_usage_percent: float
    memory_usage_percent: float
    active_jobs: int
    total_jobs_completed: int
    total_jobs_failed: int
    last_update: float = field(default_factory=time.time)


class ComputationalResource(ABC):
    """
    Abstract base class for computational resources.

    Defines the interface for both local and remote computational resources.
    """

    def __init__(
        self,
        name: str,
        resource_type: ResourceType,
        capability: ResourceCapability,
        software_paths: Dict[str, str]
    ):
        """
        Initialize computational resource.

        Args:
            name: Unique name for the resource
            resource_type: Type of resource (local/remote)
            capability: Resource capabilities
            software_paths: Paths to software executables
        """
        self.name = name
        self.resource_type = resource_type
        self.capability = capability
        self.software_paths = software_paths
        self.status = ResourceStatus.OFFLINE
        self.metrics = ResourceMetrics(0.0, 0.0, 0, 0, 0)
        self.active_jobs = {}
        self._lock = threading.Lock()

        logger.info(f"Initialized {resource_type.value} resource: {name}")

    @abstractmethod
    def check_availability(self) -> bool:
        """Check if resource is available for computation."""
        pass

    @abstractmethod
    def get_software_path(self, software: str) -> Optional[str]:
        """Get path to software executable."""
        pass

    @abstractmethod
    def submit_job(
        self,
        job_id: str,
        software: str,
        input_files: List[str],
        working_dir: str,
        num_cores: int = 1
    ) -> bool:
        """Submit a computational job."""
        pass

    @abstractmethod
    def check_job_status(self, job_id: str) -> Optional[str]:
        """Check status of a submitted job."""
        pass

    @abstractmethod
    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        pass

    @abstractmethod
    def collect_results(self, job_id: str, output_dir: str) -> bool:
        """Collect results from a completed job."""
        pass

    def update_metrics(self) -> ResourceMetrics:
        """Update and return current resource metrics."""
        with self._lock:
            self.metrics.active_jobs = len(self.active_jobs)
            self.metrics.last_update = time.time()
            return self.metrics

    def can_accept_job(self, software: str, cores_needed: int = 1) -> bool:
        """Check if resource can accept a new job."""
        with self._lock:
            # Check basic availability
            if self.status != ResourceStatus.AVAILABLE:
                return False

            # Check concurrent job limit
            if len(self.active_jobs) >= self.capability.max_concurrent_jobs:
                return False

            # Check core availability
            cores_in_use = sum(job.get('cores', 1) for job in self.active_jobs.values())
            if cores_in_use + cores_needed > self.capability.cores:
                return False

            # Check software support
            if software not in self.software_paths:
                return False

            return True

    def get_load_score(self) -> float:
        """Calculate current load score (0.0 = idle, 1.0 = fully loaded)."""
        with self._lock:
            job_load = len(self.active_jobs) / self.capability.max_concurrent_jobs
            cpu_load = self.metrics.cpu_usage_percent / 100.0
            return max(job_load, cpu_load)


class LocalResource(ComputationalResource):
    """Local computational resource implementation."""

    def __init__(
        self,
        name: str,
        capability: ResourceCapability,
        software_paths: Dict[str, str]
    ):
        """Initialize local resource."""
        super().__init__(name, ResourceType.LOCAL, capability, software_paths)
        self.processes = {}  # job_id -> subprocess.Popen

    def check_availability(self) -> bool:
        """Check local system availability."""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()

            # Update metrics
            with self._lock:
                self.metrics.cpu_usage_percent = cpu_percent
                self.metrics.memory_usage_percent = memory.percent

            # Check if system is overloaded
            if cpu_percent > 90 or memory.percent > 95:
                self.status = ResourceStatus.BUSY
                return False

            # Verify software executables exist
            for software, path in self.software_paths.items():
                if not Path(path).exists():
                    logger.warning(f"Software executable not found: {software} at {path}")
                    self.status = ResourceStatus.ERROR
                    return False

            self.status = ResourceStatus.AVAILABLE
            return True

        except Exception as e:
            logger.error(f"Error checking local resource availability: {e}")
            self.status = ResourceStatus.ERROR
            return False

    def get_software_path(self, software: str) -> Optional[str]:
        """Get local software path."""
        path = self.software_paths.get(software)
        if path and Path(path).exists():
            return path
        return None

    def submit_job(
        self,
        job_id: str,
        software: str,
        input_files: List[str],
        working_dir: str,
        num_cores: int = 1
    ) -> bool:
        """Submit job to local resource."""
        try:
            software_path = self.get_software_path(software)
            if not software_path:
                logger.error(f"Software not available: {software}")
                return False

            # Prepare command
            command = self._prepare_command(software, software_path, input_files, num_cores)

            # Submit job
            process = subprocess.Popen(
                command,
                cwd=working_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Track job
            with self._lock:
                self.active_jobs[job_id] = {
                    'software': software,
                    'cores': num_cores,
                    'working_dir': working_dir,
                    'start_time': time.time()
                }
                self.processes[job_id] = process

            logger.info(f"Submitted local job {job_id} with {software}")
            return True

        except Exception as e:
            logger.error(f"Failed to submit local job {job_id}: {e}")
            return False

    def check_job_status(self, job_id: str) -> Optional[str]:
        """Check local job status."""
        with self._lock:
            if job_id not in self.processes:
                return None

            process = self.processes[job_id]

            if process.poll() is None:
                return "running"
            elif process.returncode == 0:
                return "completed"
            else:
                return "failed"

    def cancel_job(self, job_id: str) -> bool:
        """Cancel local job."""
        try:
            with self._lock:
                if job_id in self.processes:
                    process = self.processes[job_id]
                    process.terminate()

                    # Wait for termination, then kill if necessary
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()

                    # Clean up
                    del self.processes[job_id]
                    if job_id in self.active_jobs:
                        del self.active_jobs[job_id]

                    logger.info(f"Cancelled local job {job_id}")
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel local job {job_id}: {e}")
            return False

    def collect_results(self, job_id: str, output_dir: str) -> bool:
        """Collect results from local job."""
        try:
            with self._lock:
                if job_id not in self.active_jobs:
                    return False

                job_info = self.active_jobs[job_id]
                working_dir = job_info['working_dir']

            # For local jobs, results are already in the working directory
            # Just verify the expected output files exist
            working_path = Path(working_dir)
            output_files = list(working_path.glob("*.out")) + list(working_path.glob("*.log"))

            if output_files:
                logger.info(f"Results collected for local job {job_id}")
                return True
            else:
                logger.warning(f"No output files found for local job {job_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to collect results for local job {job_id}: {e}")
            return False

    def _prepare_command(
        self,
        software: str,
        software_path: str,
        input_files: List[str],
        num_cores: int
    ) -> List[str]:
        """Prepare execution command for local software."""
        command = []

        if software.lower() == 'qe' and num_cores > 1:
            # QE with MPI
            command.extend(['mpirun', '-np', str(num_cores)])

        command.append(software_path)

        if software.lower() == 'qe':
            # QE reads from stdin
            if input_files:
                command.extend(['-input', input_files[0]])
        elif software.lower() == 'atlas':
            # ATLAS typically reads from atlas.in
            pass

        return command


class RemoteResource(ComputationalResource):
    """Remote computational resource implementation using SSH."""

    def __init__(
        self,
        name: str,
        hostname: str,
        capability: ResourceCapability,
        software_paths: Dict[str, str],
        username: Optional[str] = None,
        ssh_key: Optional[str] = None,
        remote_work_dir: str = "/tmp/atlas_qe_jobs"
    ):
        """Initialize remote resource."""
        super().__init__(name, ResourceType.REMOTE, capability, software_paths)
        self.hostname = hostname
        self.username = username or os.getenv('USER')
        self.ssh_key = ssh_key
        # Expand tilde in remote work directory
        self.remote_work_dir = self._expand_remote_path(remote_work_dir)
        self.ssh_base_cmd = self._build_ssh_command()

    def _expand_remote_path(self, path: str) -> str:
        """Expand tilde in remote path to absolute path."""
        if not path.startswith('~'):
            return path

        try:
            # Build temporary SSH command for path expansion
            ssh_cmd = ['ssh']
            if self.ssh_key:
                ssh_cmd.extend(['-i', self.ssh_key])
            ssh_cmd.extend([
                '-o', 'ConnectTimeout=10',
                '-o', 'StrictHostKeyChecking=no',
                f'{self.username}@{self.hostname}',
                'echo', path
            ])

            result = subprocess.run(ssh_cmd, capture_output=True, timeout=15, text=True)

            if result.returncode == 0:
                expanded_path = result.stdout.strip()
                logger.debug(f"Expanded remote path {path} to {expanded_path}")
                return expanded_path
            else:
                logger.warning(f"Failed to expand remote path {path}, using as-is")
                return path

        except Exception as e:
            logger.warning(f"Error expanding remote path {path}: {e}, using as-is")
            return path

    def _build_ssh_command(self) -> List[str]:
        """Build base SSH command."""
        cmd = ['ssh']

        if self.ssh_key:
            cmd.extend(['-i', self.ssh_key])

        # SSH options for automation
        cmd.extend([
            '-o', 'BatchMode=yes',
            '-o', 'ConnectTimeout=10',
            '-o', 'ServerAliveInterval=60'
        ])

        cmd.append(f"{self.username}@{self.hostname}")

        return cmd

    def check_availability(self) -> bool:
        """Check remote resource availability."""
        try:
            # Test SSH connection
            cmd = self.ssh_base_cmd + ['echo', 'test']
            result = subprocess.run(cmd, capture_output=True, timeout=15)

            if result.returncode != 0:
                logger.warning(f"SSH connection failed to {self.hostname}")
                self.status = ResourceStatus.OFFLINE
                return False

            # Check remote system resources
            cmd = self.ssh_base_cmd + ['uptime']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                # Parse load average from uptime output
                output = result.stdout.decode().strip()
                if 'load average:' in output:
                    load_part = output.split('load average:')[1].strip()
                    load_1min = float(load_part.split(',')[0].strip())

                    # Estimate CPU usage (load average / cores)
                    cpu_usage = min(100.0, (load_1min / self.capability.cores) * 100)

                    with self._lock:
                        self.metrics.cpu_usage_percent = cpu_usage

            # Verify software paths
            for software, path in self.software_paths.items():
                cmd = self.ssh_base_cmd + ['test', '-f', path]
                result = subprocess.run(cmd, capture_output=True, timeout=10)

                if result.returncode != 0:
                    logger.warning(f"Remote software not found: {software} at {path}")
                    self.status = ResourceStatus.ERROR
                    return False

            self.status = ResourceStatus.AVAILABLE
            return True

        except Exception as e:
            logger.error(f"Error checking remote resource {self.hostname}: {e}")
            self.status = ResourceStatus.ERROR
            return False

    def get_software_path(self, software: str) -> Optional[str]:
        """Get remote software path."""
        return self.software_paths.get(software)

    def submit_job(
        self,
        job_id: str,
        software: str,
        input_files: List[str],
        remote_work_dir: str,
        num_cores: int = 1
    ) -> bool:
        """Submit job to remote resource."""
        try:
            # The remote_work_dir parameter is already the full remote path
            logger.debug(f"Submitting job {job_id} to remote directory: {remote_work_dir}")

            # Create remote working directory
            cmd = self.ssh_base_cmd + ['mkdir', '-p', remote_work_dir]
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode != 0:
                logger.error(f"Failed to create remote directory for job {job_id}: {result.stderr.decode()}")
                return False

            # Determine local working directory from job_id
            # job_id format: {structure_name}_{combination_name}_{volume_point:.5f}_{timestamp}
            # e.g. gaas_zincblende_atlas_kedf701_gaas_lda_lpp_density_1.05000_1758267263722
            parts = job_id.split('_')
            if len(parts) >= 4:
                # Last part is timestamp, second to last is volume (5 decimal places)
                timestamp = parts[-1]
                volume = parts[-2]

                # The rest is structure_combination: structure_name + combination_name
                structure_combination_parts = parts[:-2]

                # Find where structure ends and combination begins
                # Look for software indicators (atlas, qe)
                structure_end = -1
                for i, part in enumerate(structure_combination_parts):
                    if part in ['atlas', 'qe']:
                        structure_end = i
                        break

                if structure_end > 0:
                    structure = '_'.join(structure_combination_parts[:structure_end])
                    combination = '_'.join(structure_combination_parts[structure_end:])
                else:
                    # Fallback: assume first two parts form structure name
                    if len(structure_combination_parts) >= 2:
                        structure = '_'.join(structure_combination_parts[:2])
                        combination = '_'.join(structure_combination_parts[2:])
                    else:
                        # Single part - treat as structure
                        structure = structure_combination_parts[0] if structure_combination_parts else "unknown"
                        combination = ""

                local_work_dir = f"results/{structure}/{combination}/{volume}"
                logger.debug(f"Parsed job_id {job_id} -> structure: {structure}, combination: {combination}, volume: {volume}")
            else:
                logger.error(f"Invalid job_id format: {job_id} - expected at least 4 parts")
                return False

            # Transfer input files
            if not self._transfer_files_to_remote(local_work_dir, remote_work_dir):
                logger.error(f"Failed to transfer files for job {job_id}")
                return False

            # Prepare and submit job
            software_path = self.get_software_path(software)
            if not software_path:
                logger.error(f"Software not available on remote: {software}")
                return False

            # Create job script
            job_script = self._create_job_script(
                job_id, software, software_path, num_cores, remote_work_dir
            )

            # Submit job script
            cmd = self.ssh_base_cmd + [f'cd {remote_work_dir} && nohup bash job.sh > job.log 2>&1 &']
            result = subprocess.run(cmd, capture_output=True, timeout=30)

            if result.returncode == 0:
                with self._lock:
                    self.active_jobs[job_id] = {
                        'software': software,
                        'cores': num_cores,
                        'working_dir': remote_work_dir,
                        'start_time': time.time()
                    }

                logger.info(f"Submitted remote job {job_id} to {self.hostname}")
                return True
            else:
                logger.error(f"Failed to submit remote job {job_id}")
                return False

        except Exception as e:
            logger.error(f"Failed to submit remote job {job_id}: {e}")
            return False

    def check_job_status(self, job_id: str) -> Optional[str]:
        """Check remote job status."""
        try:
            with self._lock:
                if job_id not in self.active_jobs:
                    return None

                job_info = self.active_jobs[job_id]
                remote_dir = job_info['working_dir']

            # Check if job is still running
            cmd = self.ssh_base_cmd + [f'pgrep -f {job_id}']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                return "running"

            # Check for completion status
            cmd = self.ssh_base_cmd + [f'test -f {remote_dir}/job.done']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                return "completed"

            # Check for error status
            cmd = self.ssh_base_cmd + [f'test -f {remote_dir}/job.error']
            result = subprocess.run(cmd, capture_output=True, timeout=10)

            if result.returncode == 0:
                return "failed"

            return "unknown"

        except Exception as e:
            logger.error(f"Failed to check remote job status {job_id}: {e}")
            return "error"

    def cancel_job(self, job_id: str) -> bool:
        """Cancel remote job."""
        try:
            # Kill processes associated with job
            cmd = self.ssh_base_cmd + [f'pkill -f {job_id}']
            subprocess.run(cmd, capture_output=True, timeout=10)

            # Clean up
            with self._lock:
                if job_id in self.active_jobs:
                    del self.active_jobs[job_id]

            logger.info(f"Cancelled remote job {job_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to cancel remote job {job_id}: {e}")
            return False

    def collect_results(self, job_id: str, output_dir: str) -> bool:
        """Collect results from remote job."""
        try:
            with self._lock:
                if job_id not in self.active_jobs:
                    return False

                job_info = self.active_jobs[job_id]
                remote_dir = job_info['working_dir']

            # Transfer results back
            return self._transfer_files_from_remote(remote_dir, output_dir)

        except Exception as e:
            logger.error(f"Failed to collect results for remote job {job_id}: {e}")
            return False

    def _transfer_files_to_remote(self, local_dir: str, remote_dir: str) -> bool:
        """Transfer files to remote resource."""
        try:
            # Use rsync for efficient transfer (directory should already be created)
            cmd = [
                'rsync', '-avz', '--timeout=60',
                f"{local_dir}/",
                f"{self.username}@{self.hostname}:{remote_dir}/"
            ]

            if self.ssh_key:
                cmd.extend(['-e', f'ssh -i {self.ssh_key}'])

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0:
                logger.debug(f"Files transferred to {self.hostname}:{remote_dir}")
                return True
            else:
                logger.error(f"rsync failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"File transfer to remote failed: {e}")
            return False

    def _transfer_files_from_remote(self, remote_dir: str, local_dir: str) -> bool:
        """Transfer files from remote resource."""
        try:
            # Ensure local directory exists
            Path(local_dir).mkdir(parents=True, exist_ok=True)

            # Use rsync for efficient transfer
            cmd = [
                'rsync', '-avz', '--timeout=60',
                f"{self.username}@{self.hostname}:{remote_dir}/",
                f"{local_dir}/"
            ]

            if self.ssh_key:
                cmd.extend(['-e', f'ssh -i {self.ssh_key}'])

            result = subprocess.run(cmd, capture_output=True, timeout=120)

            if result.returncode == 0:
                logger.debug(f"Files transferred from {self.hostname}:{remote_dir}")
                return True
            else:
                logger.error(f"rsync failed: {result.stderr.decode()}")
                return False

        except Exception as e:
            logger.error(f"File transfer from remote failed: {e}")
            return False

    def _create_job_script(
        self,
        job_id: str,
        software: str,
        software_path: str,
        num_cores: int,
        remote_dir: str
    ) -> str:
        """Create job execution script for remote submission."""
        script_lines = [
            "#!/bin/bash",
            f"# Job script for {job_id}",
            f"cd {remote_dir}",
            "",
            "# Set up environment",
            "export OMP_NUM_THREADS=1",
            "",
            "# Job execution",
        ]

        if software.lower() == 'qe' and num_cores > 1:
            script_lines.append(f"mpirun -np {num_cores} {software_path} < job.in > job.out 2>&1")
        elif software.lower() == 'atlas':
            script_lines.append(f"{software_path} atlas.in > job.out 2>&1")
        else:
            script_lines.append(f"{software_path} > job.out 2>&1")

        script_lines.extend([
            "",
            "# Check exit status",
            "if [ $? -eq 0 ]; then",
            "    touch job.done",
            "else",
            "    touch job.error",
            "fi"
        ])

        script_content = "\n".join(script_lines)

        # Write script to remote
        cmd = self.ssh_base_cmd + [f'cat > {remote_dir}/job.sh << "EOF"\n{script_content}\nEOF']
        subprocess.run(cmd, capture_output=True, timeout=30)

        # Make script executable
        cmd = self.ssh_base_cmd + [f'chmod +x {remote_dir}/job.sh']
        subprocess.run(cmd, capture_output=True, timeout=10)

        return script_content


class ResourceManager:
    """
    Central resource management system.

    Manages multiple computational resources, provides load balancing,
    and intelligent task allocation.
    """

    def __init__(self):
        """Initialize resource manager."""
        self.resources: Dict[str, ComputationalResource] = {}
        self._lock = threading.Lock()

        logger.info("ResourceManager initialized")

    def add_resource(self, resource: ComputationalResource):
        """Add a computational resource."""
        with self._lock:
            self.resources[resource.name] = resource
            logger.info(f"Added resource: {resource.name} ({resource.resource_type.value})")

    def remove_resource(self, name: str):
        """Remove a computational resource."""
        with self._lock:
            if name in self.resources:
                del self.resources[name]
                logger.info(f"Removed resource: {name}")

    def get_available_resources(
        self,
        software: str,
        cores_needed: int = 1
    ) -> List[ComputationalResource]:
        """Get list of resources that can run the specified software."""
        available = []

        with self._lock:
            for resource in self.resources.values():
                if resource.can_accept_job(software, cores_needed):
                    available.append(resource)

        # Sort by load score (prefer less loaded resources)
        available.sort(key=lambda r: r.get_load_score())

        return available

    def select_best_resource(
        self,
        software: str,
        cores_needed: int = 1,
        prefer_local: bool = False
    ) -> Optional[ComputationalResource]:
        """Select the best resource for a job."""
        available = self.get_available_resources(software, cores_needed)

        if not available:
            return None

        if prefer_local:
            # Prefer local resources if available
            local_resources = [r for r in available if r.resource_type == ResourceType.LOCAL]
            if local_resources:
                return local_resources[0]

        # For ATLAS, prefer local (single core)
        if software.lower() == 'atlas':
            local_resources = [r for r in available if r.resource_type == ResourceType.LOCAL]
            if local_resources:
                return local_resources[0]

        # For QE with multiple cores, prefer remote
        if software.lower() == 'qe' and cores_needed > 1:
            remote_resources = [r for r in available if r.resource_type == ResourceType.REMOTE]
            if remote_resources:
                return remote_resources[0]

        # Default: return least loaded resource
        return available[0]

    def update_all_resources(self) -> Dict[str, ResourceMetrics]:
        """Update metrics for all resources."""
        metrics = {}

        with self._lock:
            for name, resource in self.resources.items():
                try:
                    resource.check_availability()
                    metrics[name] = resource.update_metrics()
                except Exception as e:
                    logger.error(f"Failed to update resource {name}: {e}")

        return metrics

    def get_resource_summary(self) -> Dict[str, Any]:
        """Get summary of all resources."""
        summary = {
            'total_resources': len(self.resources),
            'local_resources': 0,
            'remote_resources': 0,
            'available_resources': 0,
            'total_cores': 0,
            'resources': {}
        }

        with self._lock:
            for name, resource in self.resources.items():
                if resource.resource_type == ResourceType.LOCAL:
                    summary['local_resources'] += 1
                else:
                    summary['remote_resources'] += 1

                if resource.status == ResourceStatus.AVAILABLE:
                    summary['available_resources'] += 1

                summary['total_cores'] += resource.capability.cores

                summary['resources'][name] = {
                    'type': resource.resource_type.value,
                    'status': resource.status.value,
                    'cores': resource.capability.cores,
                    'active_jobs': len(resource.active_jobs),
                    'load_score': resource.get_load_score(),
                    'software': list(resource.software_paths.keys())
                }

        return summary