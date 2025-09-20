"""
Distributed task scheduler for ATLAS-QE workflow system.

This module provides intelligent task scheduling across local and remote
computational resources with load balancing, priority handling, and
failure recovery.
"""

import time
import threading
import logging
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import yaml

from .resource_manager import ResourceManager, ComputationalResource, ResourceType
from .remote_executor import RemoteExecutionEngine, JobStatus
from .task_manager import TaskManager, TaskStatus, TaskPriority

logger = logging.getLogger(__name__)


class SchedulingStrategy(Enum):
    """Task scheduling strategies."""
    LOAD_BALANCED = "load_balanced"
    RESOURCE_OPTIMIZED = "resource_optimized"
    PRIORITY_FIRST = "priority_first"


@dataclass
class ScheduledTask:
    """Task scheduled for execution."""
    task_id: str
    software: str
    cores_needed: int
    priority: TaskPriority
    local_work_dir: str
    submit_time: float = field(default_factory=time.time)
    scheduled_time: Optional[float] = None
    resource_name: Optional[str] = None
    job_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 2


class DistributedScheduler:
    """
    Intelligent distributed task scheduler.

    Manages task queues, resource allocation, load balancing,
    and failure recovery across multiple computational resources.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        task_manager: TaskManager,
        remote_executor: RemoteExecutionEngine,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize distributed scheduler.

        Args:
            resource_manager: Manager for computational resources
            task_manager: Manager for task state tracking
            remote_executor: Engine for remote job execution
            config: Configuration dictionary
        """
        self.resource_manager = resource_manager
        self.task_manager = task_manager
        self.remote_executor = remote_executor
        self.config = config or {}

        # Task queues by priority
        self.task_queues = {
            TaskPriority.URGENT: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue()
        }

        # Tracking
        self.scheduled_tasks: Dict[str, ScheduledTask] = {}
        self.running_tasks: Dict[str, ScheduledTask] = {}
        self.completed_tasks: List[ScheduledTask] = []
        self.failed_tasks: List[ScheduledTask] = []

        # Scheduler control
        self._scheduler_thread = None
        self._stop_scheduler = threading.Event()
        self._lock = threading.Lock()

        # Configuration
        self.strategy = SchedulingStrategy(
            self.config.get('strategy', 'load_balanced')
        )
        self.max_concurrent_tasks = self.config.get('max_concurrent_tasks', 20)
        self.scheduling_interval = self.config.get('scheduling_interval', 5)

        logger.info("DistributedScheduler initialized")

    def start_scheduler(self):
        """Start the background scheduler."""
        if self._scheduler_thread is None or not self._scheduler_thread.is_alive():
            self._stop_scheduler.clear()
            self._scheduler_thread = threading.Thread(
                target=self._scheduling_loop,
                daemon=True
            )
            self._scheduler_thread.start()

            # Start remote executor monitoring
            self.remote_executor.start_monitoring()

            logger.info("Started distributed task scheduler")

    def stop_scheduler(self):
        """Stop the background scheduler."""
        if self._scheduler_thread and self._scheduler_thread.is_alive():
            self._stop_scheduler.set()
            self._scheduler_thread.join(timeout=30)

            # Stop remote executor monitoring
            self.remote_executor.stop_monitoring()

            logger.info("Stopped distributed task scheduler")

    def submit_task(
        self,
        task_id: str,
        software: str,
        local_work_dir: str,
        cores_needed: int = 1,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> bool:
        """
        Submit a task for distributed execution.

        Args:
            task_id: Unique task identifier
            software: Software to execute ('atlas' or 'qe')
            local_work_dir: Local directory containing input files
            cores_needed: Number of cores required
            priority: Task priority

        Returns:
            True if task submitted successfully
        """
        try:
            # Create scheduled task
            scheduled_task = ScheduledTask(
                task_id=task_id,
                software=software,
                cores_needed=cores_needed,
                priority=priority,
                local_work_dir=local_work_dir
            )

            # Add to appropriate queue
            priority_queue = self.task_queues[priority]
            priority_queue.put((time.time(), scheduled_task))

            # Track task
            with self._lock:
                self.scheduled_tasks[task_id] = scheduled_task

            # Update task status
            self.task_manager.update_task_status(task_id, TaskStatus.QUEUED)

            logger.info(f"Submitted task {task_id} with {software} (priority: {priority.value})")
            return True

        except Exception as e:
            logger.error(f"Error submitting task {task_id}: {e}")
            return False

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a submitted or running task."""
        try:
            with self._lock:
                # Check if task is scheduled
                if task_id in self.scheduled_tasks:
                    task = self.scheduled_tasks[task_id]

                    # If task has a job_id, cancel the remote job
                    if task.job_id:
                        self.remote_executor.cancel_job(task.job_id)

                    # Move to failed tasks
                    del self.scheduled_tasks[task_id]
                    if task_id in self.running_tasks:
                        del self.running_tasks[task_id]

                    task.retry_count = task.max_retries  # Prevent retries
                    self.failed_tasks.append(task)

                    # Update task status
                    self.task_manager.update_task_status(task_id, TaskStatus.CANCELLED)

                    logger.info(f"Cancelled task {task_id}")
                    return True

            logger.warning(f"Task {task_id} not found for cancellation")
            return False

        except Exception as e:
            logger.error(f"Error cancelling task {task_id}: {e}")
            return False

    def get_task_status(self, task_id: str) -> Optional[str]:
        """Get current status of a task."""
        with self._lock:
            if task_id in self.running_tasks:
                task = self.running_tasks[task_id]
                if task.job_id:
                    job_status = self.remote_executor.get_job_status(task.job_id)
                    if job_status:
                        return job_status.value
                return "running"

            if task_id in self.scheduled_tasks:
                return "queued"

        # Check completed/failed tasks
        for task in self.completed_tasks:
            if task.task_id == task_id:
                return "completed"

        for task in self.failed_tasks:
            if task.task_id == task_id:
                return "failed"

        return None

    def _scheduling_loop(self):
        """Main scheduling loop."""
        while not self._stop_scheduler.is_set():
            try:
                self._process_task_queues()
                self._check_running_tasks()
                self._update_resource_metrics()

                time.sleep(self.scheduling_interval)

            except Exception as e:
                logger.error(f"Error in scheduling loop: {e}")
                time.sleep(self.scheduling_interval)

    def _process_task_queues(self):
        """Process tasks from priority queues."""
        # Check current load
        with self._lock:
            current_running = len(self.running_tasks)

        if current_running >= self.max_concurrent_tasks:
            return

        # Process queues in priority order
        for priority in [TaskPriority.URGENT, TaskPriority.HIGH, TaskPriority.NORMAL, TaskPriority.LOW]:
            queue_obj = self.task_queues[priority]

            while not queue_obj.empty() and current_running < self.max_concurrent_tasks:
                try:
                    submit_time, task = queue_obj.get_nowait()

                    if self._schedule_task(task):
                        current_running += 1
                    else:
                        # Put task back in queue if scheduling failed
                        queue_obj.put((submit_time, task))
                        break

                except queue.Empty:
                    break

    def _schedule_task(self, task: ScheduledTask) -> bool:
        """Schedule a specific task for execution."""
        try:
            # Select best resource
            resource = self._select_resource_for_task(task)

            if not resource:
                logger.debug(f"No suitable resource available for task {task.task_id}")
                return False

            # Submit job to remote executor
            job_id = self.remote_executor.submit_job(
                task.task_id,
                task.software,
                task.local_work_dir,
                task.cores_needed,
                task.priority.value
            )

            if job_id:
                task.job_id = job_id
                task.resource_name = resource.name
                task.scheduled_time = time.time()

                # Move to running tasks
                with self._lock:
                    if task.task_id in self.scheduled_tasks:
                        del self.scheduled_tasks[task.task_id]
                    self.running_tasks[task.task_id] = task

                logger.info(f"Scheduled task {task.task_id} on {resource.name}")
                return True
            else:
                logger.warning(f"Failed to submit job for task {task.task_id}")
                return False

        except Exception as e:
            logger.error(f"Error scheduling task {task.task_id}: {e}")
            return False

    def _select_resource_for_task(self, task: ScheduledTask) -> Optional[ComputationalResource]:
        """Select the best resource for a task based on strategy."""
        if self.strategy == SchedulingStrategy.LOAD_BALANCED:
            return self._select_load_balanced_resource(task)
        elif self.strategy == SchedulingStrategy.RESOURCE_OPTIMIZED:
            return self._select_optimized_resource(task)
        elif self.strategy == SchedulingStrategy.PRIORITY_FIRST:
            return self._select_priority_resource(task)
        else:
            return self.resource_manager.select_best_resource(
                task.software, task.cores_needed
            )

    def _select_load_balanced_resource(self, task: ScheduledTask) -> Optional[ComputationalResource]:
        """Select resource based on load balancing."""
        available = self.resource_manager.get_available_resources(
            task.software, task.cores_needed
        )

        if not available:
            return None

        # Calculate weighted load scores
        best_resource = None
        best_score = float('inf')

        for resource in available:
            # Get current metrics
            metrics = resource.update_metrics()

            # Calculate weighted score
            load_score = resource.get_load_score()
            active_jobs_ratio = metrics.active_jobs / resource.capability.max_concurrent_jobs

            # Weight factors from config
            weights = self.config.get('load_balance_weights', {
                'cpu_usage': 0.4,
                'memory_usage': 0.2,
                'active_jobs': 0.4
            })

            weighted_score = (
                weights.get('cpu_usage', 0.4) * (metrics.cpu_usage_percent / 100.0) +
                weights.get('memory_usage', 0.2) * (metrics.memory_usage_percent / 100.0) +
                weights.get('active_jobs', 0.4) * active_jobs_ratio
            )

            if weighted_score < best_score:
                best_score = weighted_score
                best_resource = resource

        return best_resource

    def _select_optimized_resource(self, task: ScheduledTask) -> Optional[ComputationalResource]:
        """Select resource optimized for specific software."""
        available = self.resource_manager.get_available_resources(
            task.software, task.cores_needed
        )

        if not available:
            return None

        # Software-specific preferences
        if task.software.lower() == 'atlas':
            # Prefer local resources for ATLAS (single core)
            local_resources = [r for r in available if r.resource_type == ResourceType.LOCAL]
            if local_resources:
                return min(local_resources, key=lambda r: r.get_load_score())

        elif task.software.lower() == 'qe' and task.cores_needed > 1:
            # Prefer remote resources for multi-core QE
            remote_resources = [r for r in available if r.resource_type == ResourceType.REMOTE]
            if remote_resources:
                return min(remote_resources, key=lambda r: r.get_load_score())

        # Fallback to least loaded resource
        return min(available, key=lambda r: r.get_load_score())

    def _select_priority_resource(self, task: ScheduledTask) -> Optional[ComputationalResource]:
        """Select resource based on task priority."""
        available = self.resource_manager.get_available_resources(
            task.software, task.cores_needed
        )

        if not available:
            return None

        # High priority tasks get best resources
        if task.priority in [TaskPriority.URGENT, TaskPriority.HIGH]:
            # Prefer resources with more cores and lower load
            return min(available, key=lambda r: (r.get_load_score(), -r.capability.cores))
        else:
            # Normal/low priority tasks get any available resource
            return min(available, key=lambda r: r.get_load_score())

    def _check_running_tasks(self):
        """Check status of running tasks and handle completion/failure."""
        with self._lock:
            running_task_ids = list(self.running_tasks.keys())

        for task_id in running_task_ids:
            try:
                with self._lock:
                    task = self.running_tasks.get(task_id)
                    if not task:
                        continue

                # Check job status
                if task.job_id:
                    job_status = self.remote_executor.get_job_status(task.job_id)

                    if job_status == JobStatus.COMPLETED:
                        self._handle_task_completion(task)
                    elif job_status in [JobStatus.FAILED, JobStatus.TIMEOUT]:
                        self._handle_task_failure(task, job_status.value)

            except Exception as e:
                logger.error(f"Error checking running task {task_id}: {e}")

    def _handle_task_completion(self, task: ScheduledTask):
        """Handle successful task completion."""
        try:
            # Move to completed tasks
            with self._lock:
                if task.task_id in self.running_tasks:
                    del self.running_tasks[task.task_id]
                self.completed_tasks.append(task)

            # Update task status
            self.task_manager.update_task_status(task.task_id, TaskStatus.COMPLETED)

            logger.info(f"Task {task.task_id} completed successfully")

        except Exception as e:
            logger.error(f"Error handling task completion for {task.task_id}: {e}")

    def _handle_task_failure(self, task: ScheduledTask, failure_reason: str):
        """Handle task failure with retry logic."""
        try:
            task.retry_count += 1

            if task.retry_count <= task.max_retries:
                # Retry task
                logger.info(f"Retrying task {task.task_id} (attempt {task.retry_count}/{task.max_retries})")

                # Reset job info
                task.job_id = None
                task.resource_name = None
                task.scheduled_time = None

                # Move back to queue
                with self._lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.scheduled_tasks[task.task_id] = task

                # Add back to priority queue
                priority_queue = self.task_queues[task.priority]
                priority_queue.put((time.time(), task))

                # Update task status
                self.task_manager.update_task_status(task.task_id, TaskStatus.QUEUED)

            else:
                # Max retries exceeded
                with self._lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.failed_tasks.append(task)

                # Update task status
                self.task_manager.update_task_status(
                    task.task_id,
                    TaskStatus.FAILED,
                    error_message=f"Max retries exceeded: {failure_reason}"
                )

                logger.error(f"Task {task.task_id} failed after {task.max_retries} retries: {failure_reason}")

        except Exception as e:
            logger.error(f"Error handling task failure for {task.task_id}: {e}")

    def _update_resource_metrics(self):
        """Update metrics for all resources."""
        try:
            self.resource_manager.update_all_resources()
        except Exception as e:
            logger.error(f"Error updating resource metrics: {e}")

    def get_scheduler_summary(self) -> Dict[str, Any]:
        """Get comprehensive scheduler status summary."""
        with self._lock:
            # Count tasks by status
            queued_count = len(self.scheduled_tasks)
            running_count = len(self.running_tasks)
            completed_count = len(self.completed_tasks)
            failed_count = len(self.failed_tasks)

            # Count by priority in queues
            queue_counts = {}
            for priority, queue_obj in self.task_queues.items():
                queue_counts[priority.value] = queue_obj.qsize()

            # Resource utilization
            resource_summary = self.resource_manager.get_resource_summary()

        return {
            'scheduler_active': self._scheduler_thread and self._scheduler_thread.is_alive(),
            'strategy': self.strategy.value,
            'task_counts': {
                'queued': queued_count,
                'running': running_count,
                'completed': completed_count,
                'failed': failed_count,
                'total': queued_count + running_count + completed_count + failed_count
            },
            'queue_breakdown': queue_counts,
            'resource_summary': resource_summary,
            'remote_executor': self.remote_executor.get_execution_summary(),
            'performance_metrics': {
                'max_concurrent_tasks': self.max_concurrent_tasks,
                'current_utilization': running_count / self.max_concurrent_tasks,
                'scheduling_interval': self.scheduling_interval
            }
        }

    def submit_workflow_tasks(
        self,
        task_definitions: List[Dict[str, Any]],
        base_priority: TaskPriority = TaskPriority.NORMAL
    ) -> List[str]:
        """
        Submit multiple tasks from a workflow.

        Args:
            task_definitions: List of task definition dictionaries
            base_priority: Base priority for tasks

        Returns:
            List of successfully submitted task IDs
        """
        submitted_tasks = []

        for task_def in task_definitions:
            try:
                task_id = task_def['task_id']
                software = task_def['software']
                work_dir = task_def['working_directory']
                cores = task_def.get('cores_needed', 1)

                # Determine priority (can be overridden per task)
                priority = TaskPriority(task_def.get('priority', base_priority.value))

                if self.submit_task(task_id, software, work_dir, cores, priority):
                    submitted_tasks.append(task_id)
                else:
                    logger.error(f"Failed to submit task {task_id}")

            except Exception as e:
                logger.error(f"Error submitting task from workflow: {e}")

        logger.info(f"Submitted {len(submitted_tasks)}/{len(task_definitions)} workflow tasks")
        return submitted_tasks