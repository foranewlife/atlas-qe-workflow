"""
Remote execution engine for distributed computing.

This module provides comprehensive remote execution capabilities including
SSH-based file transfer, job submission, monitoring, and result collection.
"""

import os
import time
import threading
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml

from .resource_manager import ResourceManager, ComputationalResource, ResourceType
from .task_manager import TaskManager, TaskStatus

logger = logging.getLogger(__name__)


class JobStatus(Enum):
    """Status of remote jobs."""
    PENDING = "pending"
    TRANSFERRING = "transferring"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    COLLECTING = "collecting"
    TIMEOUT = "timeout"


@dataclass
class RemoteJob:
    """Remote job information."""
    job_id: str
    task_id: str
    resource_name: str
    software: str
    local_work_dir: str
    remote_work_dir: str
    status: JobStatus = JobStatus.PENDING
    start_time: float = field(default_factory=time.time)
    completion_time: Optional[float] = None
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    execution_log: List[str] = field(default_factory=list)


class RemoteExecutionEngine:
    """
    Engine for executing computational jobs on remote resources.

    Handles the complete lifecycle of remote job execution including:
    - File transfer to remote resources
    - Job submission and monitoring
    - Result collection and cleanup
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        task_manager: TaskManager,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize remote execution engine.

        Args:
            resource_manager: Manager for computational resources
            task_manager: Manager for task state tracking
            config: Configuration dictionary
        """
        self.resource_manager = resource_manager
        self.task_manager = task_manager
        self.config = config or {}

        # Job tracking
        self.active_jobs: Dict[str, RemoteJob] = {}
        self.completed_jobs: List[RemoteJob] = []
        self._lock = threading.Lock()

        # Background monitoring
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()

        logger.info("RemoteExecutionEngine initialized")

    def start_monitoring(self):
        """Start background job monitoring."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitor_thread.start()
            logger.info("Started remote job monitoring")

    def stop_monitoring(self):
        """Stop background job monitoring."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._stop_monitoring.set()
            self._monitor_thread.join(timeout=10)
            logger.info("Stopped remote job monitoring")

    def submit_job(
        self,
        task_id: str,
        software: str,
        local_work_dir: str,
        cores_needed: int = 1,
        priority: str = "normal"
    ) -> Optional[str]:
        """
        Submit a job for remote execution.

        Args:
            task_id: Task identifier
            software: Software to execute ('atlas' or 'qe')
            local_work_dir: Local directory containing input files
            cores_needed: Number of cores required
            priority: Job priority

        Returns:
            Job ID if submission successful, None otherwise
        """
        try:
            # Select best resource
            resource = self.resource_manager.select_best_resource(
                software, cores_needed
            )

            if not resource:
                logger.warning(f"No suitable resource available for task {task_id}")
                return None

            # Use task_id as job_id (task_id already includes timestamp)
            job_id = task_id

            # Create remote job
            remote_job = RemoteJob(
                job_id=job_id,
                task_id=task_id,
                resource_name=resource.name,
                software=software,
                local_work_dir=local_work_dir,
                remote_work_dir=self._get_remote_work_dir(resource, job_id)
            )

            # Start job execution
            if self._execute_remote_job(remote_job, resource, cores_needed):
                with self._lock:
                    self.active_jobs[job_id] = remote_job

                logger.info(f"Submitted remote job {job_id} for task {task_id}")
                return job_id
            else:
                logger.error(f"Failed to submit remote job for task {task_id}")
                return None

        except Exception as e:
            logger.error(f"Error submitting remote job for task {task_id}: {e}")
            return None

    def get_job_status(self, job_id: str) -> Optional[JobStatus]:
        """Get status of a remote job."""
        with self._lock:
            if job_id in self.active_jobs:
                return self.active_jobs[job_id].status

        # Check completed jobs
        for job in self.completed_jobs:
            if job.job_id == job_id:
                return job.status

        return None

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a remote job."""
        try:
            with self._lock:
                if job_id not in self.active_jobs:
                    logger.warning(f"Job {job_id} not found in active jobs")
                    return False

                job = self.active_jobs[job_id]

            # Get resource
            resource = self.resource_manager.resources.get(job.resource_name)
            if not resource:
                logger.error(f"Resource {job.resource_name} not found")
                return False

            # Cancel job on resource
            success = resource.cancel_job(job_id)

            if success:
                job.status = JobStatus.FAILED
                job.error_message = "Job cancelled by user"
                job.completion_time = time.time()

                with self._lock:
                    del self.active_jobs[job_id]
                    self.completed_jobs.append(job)

                # Update task status
                self.task_manager.update_task_status(
                    job.task_id, TaskStatus.CANCELLED
                )

                logger.info(f"Cancelled remote job {job_id}")

            return success

        except Exception as e:
            logger.error(f"Error cancelling job {job_id}: {e}")
            return False

    def _execute_remote_job(
        self,
        job: RemoteJob,
        resource: ComputationalResource,
        cores_needed: int
    ) -> bool:
        """Execute a remote job through its complete lifecycle."""
        try:
            # Phase 1: Transfer files to remote
            job.status = JobStatus.TRANSFERRING
            job.execution_log.append(f"Starting file transfer to {resource.name}")

            if not self._transfer_files_to_remote(job, resource):
                job.status = JobStatus.FAILED
                job.error_message = "File transfer to remote failed"
                return False

            # Phase 2: Submit job on remote resource
            job.status = JobStatus.QUEUED
            job.execution_log.append("Submitting job to remote resource")

            input_files = self._get_input_files(job.local_work_dir, job.software)
            success = resource.submit_job(
                job.job_id,
                job.software,
                input_files,
                job.remote_work_dir,
                cores_needed
            )

            if success:
                job.status = JobStatus.RUNNING
                job.execution_log.append("Job submitted successfully")

                # Update task status
                self.task_manager.update_task_status(
                    job.task_id, TaskStatus.RUNNING
                )

                return True
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Failed to submit job to remote resource"
                return False

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Remote job execution failed: {str(e)}"
            logger.error(f"Error executing remote job {job.job_id}: {e}")
            return False

    def _transfer_files_to_remote(
        self,
        job: RemoteJob,
        resource: ComputationalResource
    ) -> bool:
        """Transfer input files to remote resource."""
        try:
            if resource.resource_type != ResourceType.REMOTE:
                # For local resources, no transfer needed
                job.remote_work_dir = job.local_work_dir
                return True

            # Use resource's file transfer method
            return resource._transfer_files_to_remote(
                job.local_work_dir,
                job.remote_work_dir
            )

        except Exception as e:
            logger.error(f"File transfer failed for job {job.job_id}: {e}")
            return False

    def _collect_results(self, job: RemoteJob, resource: ComputationalResource) -> bool:
        """Collect results from completed remote job."""
        try:
            job.status = JobStatus.COLLECTING
            job.execution_log.append("Collecting results from remote")

            if resource.resource_type == ResourceType.REMOTE:
                # Transfer results back for remote resources
                success = resource.collect_results(job.job_id, job.local_work_dir)
            else:
                # For local resources, results are already local
                success = True

            if success:
                # Verify output files exist
                output_files = self._verify_output_files(job.local_work_dir, job.software)
                job.output_files = output_files

                if output_files:
                    job.status = JobStatus.COMPLETED
                    job.execution_log.append("Results collected successfully")

                    # Update task status with results
                    self._update_task_with_results(job)
                    return True
                else:
                    job.status = JobStatus.FAILED
                    job.error_message = "No output files found"
                    return False
            else:
                job.status = JobStatus.FAILED
                job.error_message = "Failed to collect results"
                return False

        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = f"Result collection failed: {str(e)}"
            logger.error(f"Error collecting results for job {job.job_id}: {e}")
            return False

    def _update_task_with_results(self, job: RemoteJob):
        """Update task with job results."""
        try:
            # Parse output for energy and convergence
            energy, converged = self._parse_output_files(job)

            # Update task in database
            self.task_manager.update_task_completion(
                job.task_id,
                energy=energy,
                converged=converged,
                execution_time=job.completion_time - job.start_time,
                output_files=job.output_files
            )

            logger.debug(f"Updated task {job.task_id} with results: E={energy}, converged={converged}")

        except Exception as e:
            logger.error(f"Error updating task results for {job.task_id}: {e}")

    def _parse_output_files(self, job: RemoteJob) -> "Tuple[Optional[float], bool]":
        """Parse output files for energy and convergence information."""
        try:
            energy = None
            converged = False

            if job.software.lower() == 'atlas':
                energy, converged = self._parse_atlas_output(job.local_work_dir)
            elif job.software.lower() == 'qe':
                energy, converged = self._parse_qe_output(job.local_work_dir)

            return energy, converged

        except Exception as e:
            logger.error(f"Error parsing output files for job {job.job_id}: {e}")
            return None, False

    def _parse_atlas_output(self, work_dir: str) -> "Tuple[Optional[float], bool]":
        """Parse ATLAS output files."""
        import re

        try:
            output_file = Path(work_dir) / "atlas.out"
            if not output_file.exists():
                return None, False

            with open(output_file, 'r') as f:
                content = f.read()

            # Look for final energy
            energy_pattern = r"Total Energy\s*=\s*([-\d\.E\+\-]+)"
            energy_match = re.search(energy_pattern, content)
            energy = float(energy_match.group(1)) if energy_match else None

            # Check convergence
            convergence_patterns = [
                r"SCF Converged",
                r"convergence achieved",
                r"calculation converged"
            ]

            converged = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in convergence_patterns
            )

            return energy, converged

        except Exception as e:
            logger.error(f"Error parsing ATLAS output: {e}")
            return None, False

    def _parse_qe_output(self, work_dir: str) -> "Tuple[Optional[float], bool]":
        """Parse Quantum ESPRESSO output files."""
        import re

        try:
            output_file = Path(work_dir) / "job.out"
            if not output_file.exists():
                # Try other common QE output names
                for name in ["pw.out", "scf.out"]:
                    alt_file = Path(work_dir) / name
                    if alt_file.exists():
                        output_file = alt_file
                        break
                else:
                    return None, False

            with open(output_file, 'r') as f:
                content = f.read()

            # Look for final energy (in Ry)
            energy_patterns = [
                r"!\s*total\s+energy\s*=\s*([-\d\.E\+\-]+)\s*Ry",
                r"Final\s+energy\s*=\s*([-\d\.E\+\-]+)\s*Ry"
            ]

            energy = None
            for pattern in energy_patterns:
                energy_match = re.search(pattern, content)
                if energy_match:
                    energy = float(energy_match.group(1))
                    break

            # Check convergence
            convergence_patterns = [
                r"JOB DONE",
                r"convergence has been achieved",
                r"End of self-consistent calculation"
            ]

            converged = any(
                re.search(pattern, content, re.IGNORECASE)
                for pattern in convergence_patterns
            )

            return energy, converged

        except Exception as e:
            logger.error(f"Error parsing QE output: {e}")
            return None, False

    def _get_input_files(self, work_dir: str, software: str) -> List[str]:
        """Get list of input files for software."""
        work_path = Path(work_dir)

        if software.lower() == 'atlas':
            files = ["atlas.in"]
            # Add POSCAR if it exists
            if (work_path / "POSCAR").exists():
                files.append("POSCAR")
        elif software.lower() == 'qe':
            # Look for QE input files
            qe_files = list(work_path.glob("*.in")) + list(work_path.glob("job.*"))
            files = [f.name for f in qe_files if f.is_file()]

        return files

    def _verify_output_files(self, work_dir: str, software: str) -> List[str]:
        """Verify and return list of output files."""
        work_path = Path(work_dir)
        output_files = []

        if software.lower() == 'atlas':
            expected_files = ["atlas.out"]
            optional_files = ["DENSFILE", "POTFILE"]
        elif software.lower() == 'qe':
            expected_files = ["job.out"]
            optional_files = ["*.xml", "*.save"]
        else:
            expected_files = []
            optional_files = []

        # Check expected files
        for filename in expected_files:
            file_path = work_path / filename
            if file_path.exists():
                output_files.append(filename)

        # Check optional files
        for pattern in optional_files:
            if '*' in pattern:
                matches = list(work_path.glob(pattern))
                output_files.extend([f.name for f in matches])
            else:
                file_path = work_path / pattern
                if file_path.exists():
                    output_files.append(pattern)

        return output_files

    def _get_remote_work_dir(self, resource: ComputationalResource, job_id: str) -> str:
        """Get remote working directory for job."""
        if hasattr(resource, 'remote_work_dir'):
            base_dir = resource.remote_work_dir
        else:
            base_dir = "/tmp/atlas_qe_jobs"

        return f"{base_dir}/{job_id}"

    def _monitoring_loop(self):
        """Background monitoring loop for active jobs."""
        monitor_interval = self.config.get('monitor_interval', 30)

        while not self._stop_monitoring.is_set():
            try:
                self._check_all_jobs()
                time.sleep(monitor_interval)
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(monitor_interval)

    def _check_all_jobs(self):
        """Check status of all active jobs."""
        with self._lock:
            active_job_ids = list(self.active_jobs.keys())

        for job_id in active_job_ids:
            try:
                self._check_job_status(job_id)
            except Exception as e:
                logger.error(f"Error checking job {job_id}: {e}")

    def _check_job_status(self, job_id: str):
        """Check status of a specific job."""
        with self._lock:
            if job_id not in self.active_jobs:
                return

            job = self.active_jobs[job_id]

        # Get resource
        resource = self.resource_manager.resources.get(job.resource_name)
        if not resource:
            logger.error(f"Resource {job.resource_name} not found for job {job_id}")
            return

        # Check job status on resource
        remote_status = resource.check_job_status(job_id)

        if remote_status == "completed":
            # Job completed, collect results
            job.completion_time = time.time()
            success = self._collect_results(job, resource)

            if success:
                job.execution_log.append("Job completed successfully")
            else:
                job.execution_log.append("Job completed but result collection failed")

            # Move to completed jobs
            with self._lock:
                del self.active_jobs[job_id]
                self.completed_jobs.append(job)

        elif remote_status == "failed":
            # Job failed
            job.status = JobStatus.FAILED
            job.completion_time = time.time()
            job.error_message = "Job failed on remote resource"
            job.execution_log.append("Job failed")

            # Update task status
            self.task_manager.update_task_status(job.task_id, TaskStatus.FAILED)

            # Move to completed jobs
            with self._lock:
                del self.active_jobs[job_id]
                self.completed_jobs.append(job)

        elif remote_status is None:
            # Job not found, might have been cleaned up
            logger.warning(f"Job {job_id} not found on resource {job.resource_name}")

        # Check for timeout
        current_time = time.time()
        timeout = self.config.get('job_timeout', 3600)
        if current_time - job.start_time > timeout:
            logger.warning(f"Job {job_id} timed out after {timeout} seconds")

            # Cancel timed out job
            resource.cancel_job(job_id)

            job.status = JobStatus.TIMEOUT
            job.completion_time = current_time
            job.error_message = f"Job timed out after {timeout} seconds"

            # Update task status
            self.task_manager.update_task_status(job.task_id, TaskStatus.TIMEOUT)

            # Move to completed jobs
            with self._lock:
                del self.active_jobs[job_id]
                self.completed_jobs.append(job)

    def get_execution_summary(self) -> Dict[str, Any]:
        """Get summary of remote execution engine state."""
        with self._lock:
            active_count = len(self.active_jobs)
            completed_count = len(self.completed_jobs)

            # Count by status
            status_counts = {}
            for job in self.active_jobs.values():
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

            for job in self.completed_jobs[-100:]:  # Last 100 completed jobs
                status = job.status.value
                status_counts[status] = status_counts.get(status, 0) + 1

        return {
            'active_jobs': active_count,
            'completed_jobs': completed_count,
            'status_breakdown': status_counts,
            'monitoring_active': self._monitor_thread and self._monitor_thread.is_alive()
        }