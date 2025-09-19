#!/usr/bin/env python3
"""
Distributed ATLAS-QE Workflow Executor

Comprehensive distributed workflow execution system that integrates all components:
- Resource management for local and remote computational resources
- Intelligent task scheduling with load balancing and priority handling
- Real-time monitoring and alerting
- Comprehensive error handling and recovery
- Distributed execution across multiple resources

Usage:
    python run_distributed_workflow.py workflow_config.yaml --resources resources.yaml
    python run_distributed_workflow.py workflow_config.yaml --dry-run
    python run_distributed_workflow.py workflow_config.yaml --monitor-only
"""

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any

# Import distributed system components
from src.core.configuration import ConfigurationManager
from src.core.resource_manager import (
    ResourceManager, LocalResource, RemoteResource,
    ResourceCapability, ResourceType
)
from src.core.task_manager import TaskManager, TaskStatus, TaskPriority
from src.core.remote_executor import RemoteExecutionEngine
from src.core.distributed_scheduler import DistributedScheduler, SchedulingStrategy
from src.core.monitoring_system import MonitoringSystem, Alert
from src.core.error_handler import ErrorHandler
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


class DistributedWorkflowExecutor:
    """
    Main distributed workflow execution controller.

    Orchestrates the entire distributed computing system including
    resource management, task scheduling, monitoring, and error handling.
    """

    def __init__(
        self,
        workflow_config_file: str,
        resource_config_file: Optional[str] = None,
        dry_run: bool = False,
        monitor_only: bool = False
    ):
        """
        Initialize distributed workflow executor.

        Args:
            workflow_config_file: Path to workflow configuration file
            resource_config_file: Path to resource configuration file
            dry_run: If True, only generate tasks without execution
            monitor_only: If True, only start monitoring without execution
        """
        self.workflow_config_file = workflow_config_file
        self.resource_config_file = resource_config_file or "config/resources.yaml"
        self.dry_run = dry_run
        self.monitor_only = monitor_only

        # System components
        self.config_manager: Optional[ConfigurationManager] = None
        self.resource_manager: Optional[ResourceManager] = None
        self.task_manager: Optional[TaskManager] = None
        self.remote_executor: Optional[RemoteExecutionEngine] = None
        self.scheduler: Optional[DistributedScheduler] = None
        self.monitoring_system: Optional[MonitoringSystem] = None
        self.error_handler: Optional[ErrorHandler] = None

        # Execution state
        self.execution_start_time: Optional[float] = None
        self.total_tasks: int = 0
        self.submitted_tasks: int = 0
        self.completed_tasks: int = 0
        self.failed_tasks: int = 0

        # Shutdown handling
        self._shutdown_requested = False

        logger.info("DistributedWorkflowExecutor initialized")

    def initialize_system(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing distributed workflow system...")

            # 1. Load configurations
            self.config_manager = ConfigurationManager(
                workflow_config_file=self.workflow_config_file,
                resource_config_file=self.resource_config_file
            )

            if not self.config_manager.validate_configurations():
                logger.error("Configuration validation failed")
                return False

            # 2. Initialize task manager
            workspace_dir = self.config_manager.workflow_config.data_paths.get(
                'base_directory', 'results'
            )
            self.task_manager = TaskManager(workspace_dir=workspace_dir)

            # 3. Initialize resource manager
            self.resource_manager = ResourceManager()
            self._setup_computational_resources()

            # 4. Initialize error handler
            error_config = self.config_manager.get_execution_settings()
            self.error_handler = ErrorHandler(config=error_config)

            # 5. Initialize remote executor
            executor_config = self.config_manager.get_execution_settings()
            self.remote_executor = RemoteExecutionEngine(
                resource_manager=self.resource_manager,
                task_manager=self.task_manager,
                config=executor_config
            )

            # 6. Initialize scheduler
            scheduler_config = self.config_manager.get_resource_management_settings()
            self.scheduler = DistributedScheduler(
                resource_manager=self.resource_manager,
                task_manager=self.task_manager,
                remote_executor=self.remote_executor,
                config=scheduler_config
            )

            # 7. Initialize monitoring system
            monitoring_config = self.config_manager.get_monitoring_settings()
            self.monitoring_system = MonitoringSystem(
                resource_manager=self.resource_manager,
                task_manager=self.task_manager,
                scheduler=self.scheduler,
                remote_executor=self.remote_executor,
                config=monitoring_config
            )

            # Setup alert callbacks
            self.monitoring_system.add_alert_callback(self._handle_alert)

            logger.info("System initialization completed successfully")
            return True

        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False

    def _setup_computational_resources(self):
        """Setup computational resources from configuration."""
        try:
            resource_configs = self.config_manager.resource_configs

            for config in resource_configs:
                if config.resource_type == ResourceType.LOCAL:
                    resource = LocalResource(
                        name=config.name,
                        capability=config.capability,
                        software_paths=config.software_paths
                    )
                else:
                    resource = RemoteResource(
                        name=config.name,
                        hostname=config.hostname,
                        capability=config.capability,
                        software_paths=config.software_paths,
                        username=config.username,
                        ssh_key=config.ssh_key,
                        remote_work_dir=config.remote_work_dir
                    )

                self.resource_manager.add_resource(resource)

            logger.info(f"Setup {len(resource_configs)} computational resources")

            # Check resource availability
            self.resource_manager.update_all_resources()

        except Exception as e:
            logger.error(f"Error setting up computational resources: {e}")
            raise

    def execute_workflow(self) -> bool:
        """Execute the complete distributed workflow."""
        try:
            self.execution_start_time = time.time()

            if self.monitor_only:
                return self._run_monitoring_only()

            logger.info("Starting distributed workflow execution...")

            # 1. Generate tasks
            if not self._generate_workflow_tasks():
                logger.error("Task generation failed")
                return False

            if self.dry_run:
                logger.info("Dry run completed - tasks generated but not executed")
                return True

            # 2. Start system components
            if not self._start_system_components():
                logger.error("Failed to start system components")
                return False

            # 3. Submit tasks for execution
            if not self._submit_workflow_tasks():
                logger.error("Task submission failed")
                return False

            # 4. Monitor execution
            return self._monitor_execution()

        except KeyboardInterrupt:
            logger.info("Workflow execution interrupted by user")
            self._shutdown_system()
            return False
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            self._shutdown_system()
            return False

    def _generate_workflow_tasks(self) -> bool:
        """Generate tasks from workflow configuration."""
        try:
            from src.core.configuration import ParameterSpaceEnumerator
            from src.core.template_engine import TemplateProcessor, StructureProcessor

            # Enumerate parameter space
            enumerator = ParameterSpaceEnumerator(self.config_manager.workflow_config)
            parameter_space = enumerator.discover_parameter_space()
            self.total_tasks = enumerator.count_total_calculations()

            logger.info(f"Generating {self.total_tasks} workflow tasks...")

            # Generate input files and create tasks using direct components
            config_dir = Path(self.workflow_config_file).parent  # Use config directory
            template_processor = TemplateProcessor(self.config_manager.workflow_config, config_dir)
            structure_processor = StructureProcessor(config_dir)
            task_count = 0

            for structure, combination in parameter_space:
                # Generate volume points
                volume_points = self._generate_volume_points(
                    structure.volume_range,
                    structure.volume_points
                )

                for volume_factor in volume_points:
                    task_id = f"{structure.name}_{combination.name}_{volume_factor:.5f}_{int(time.time() * 1000)}"

                    # Create working directory
                    work_dir = Path(self.config_manager.workflow_config.data_paths['base_directory'])
                    work_dir = work_dir / structure.name / combination.name / f"{volume_factor:.5f}"
                    work_dir.mkdir(parents=True, exist_ok=True)

                    # Generate input files using direct components
                    try:
                        # Get pseudopotential set
                        pseudopotential_set = self.config_manager.workflow_config.pseudopotential_sets.get(
                            combination.pseudopotential_set, {}
                        )

                        # Process template file
                        template_content = template_processor.process_template(
                            template_file=combination.template_file,
                            substitutions=combination.template_substitutions,
                            structure=structure,
                            combination=combination
                        )

                        # Generate structure content
                        structure_content = structure_processor.generate_structure_content(
                            structure=structure,
                            volume_scale=volume_factor
                        )

                        # Write main input file
                        input_filename = f"{combination.software}.in"
                        input_path = work_dir / input_filename
                        with open(input_path, 'w') as f:
                            f.write(template_content)

                        # Write structure file if needed
                        if combination.software == 'atlas':
                            poscar_path = work_dir / "POSCAR"
                            with open(poscar_path, 'w') as f:
                                f.write(structure_content)

                        # Copy pseudopotential files if needed
                        for element, pp_file in pseudopotential_set.items():
                            pp_source = config_dir / "data" / "pseudopotentials" / pp_file
                            pp_dest = work_dir / pp_file
                            if pp_source.exists() and not pp_dest.exists():
                                import shutil
                                shutil.copy2(pp_source, pp_dest)
                                logger.debug(f"Copied {pp_source} -> {pp_dest}")
                            elif pp_dest.exists():
                                logger.debug(f"Pseudopotential file already exists: {pp_dest}")
                            else:
                                logger.warning(f"Pseudopotential file not found: {pp_source}")

                        # Create task in database
                        task_def = self.task_manager.create_task(
                            structure_name=structure.name,
                            combination_name=combination.name,
                            software=combination.software,
                            volume_point=volume_factor,
                            input_file=str(work_dir / f"{combination.software}.in"),
                            working_directory=str(work_dir)
                        )

                        task_count += 1

                    except Exception as e:
                        logger.error(f"Failed to generate task {task_id}: {e}")

            logger.info(f"Generated {task_count}/{self.total_tasks} tasks successfully")
            return task_count > 0

        except Exception as e:
            logger.error(f"Error generating workflow tasks: {e}")
            return False

    def _generate_volume_points(self, volume_range: tuple, num_points: int) -> List[float]:
        """Generate volume scaling factors."""
        import numpy as np
        return np.linspace(volume_range[0], volume_range[1], num_points).tolist()

    def _start_system_components(self) -> bool:
        """Start all system components."""
        try:
            logger.info("Starting system components...")

            # Start monitoring system
            self.monitoring_system.start_monitoring()

            # Start scheduler
            self.scheduler.start_scheduler()

            logger.info("System components started successfully")
            return True

        except Exception as e:
            logger.error(f"Error starting system components: {e}")
            return False

    def _submit_workflow_tasks(self) -> bool:
        """Submit workflow tasks to the scheduler."""
        try:
            # Get pending tasks from database
            pending_tasks = self.task_manager.get_tasks_by_status(TaskStatus.PENDING)

            logger.info(f"Submitting {len(pending_tasks)} tasks to scheduler...")

            # Submit tasks to scheduler
            submitted_count = 0
            for task in pending_tasks:
                # Determine cores needed based on software
                cores_needed = 1
                if task.software.lower() == 'qe':
                    cores_needed = 4  # Default for QE

                # Determine priority (could be based on task properties)
                priority = TaskPriority.NORMAL

                success = self.scheduler.submit_task(
                    task_id=task.task_id,
                    software=task.software,
                    local_work_dir=task.working_directory,
                    cores_needed=cores_needed,
                    priority=priority
                )

                if success:
                    submitted_count += 1
                else:
                    logger.warning(f"Failed to submit task {task.task_id}")

            self.submitted_tasks = submitted_count
            logger.info(f"Submitted {submitted_count}/{len(pending_tasks)} tasks successfully")

            return submitted_count > 0

        except Exception as e:
            logger.error(f"Error submitting workflow tasks: {e}")
            return False

    def _monitor_execution(self) -> bool:
        """Monitor workflow execution until completion."""
        try:
            logger.info("Monitoring workflow execution...")

            start_time = time.time()
            last_status_time = 0
            status_interval = 30  # Print status every 30 seconds

            while not self._shutdown_requested:
                current_time = time.time()

                # Print status periodically
                if current_time - last_status_time >= status_interval:
                    self._print_execution_status()
                    last_status_time = current_time

                # Check if execution is complete
                if self._is_execution_complete():
                    logger.info("Workflow execution completed")
                    break

                # Sleep for a short interval
                time.sleep(5)

            # Final status
            self._print_final_summary()
            return True

        except Exception as e:
            logger.error(f"Error monitoring execution: {e}")
            return False

    def _run_monitoring_only(self) -> bool:
        """Run only monitoring without executing new tasks."""
        try:
            logger.info("Starting monitoring-only mode...")

            if not self._start_system_components():
                return False

            logger.info("Monitoring system active - press Ctrl+C to stop")

            # Monitor indefinitely
            while not self._shutdown_requested:
                time.sleep(10)
                self._print_execution_status()

            return True

        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
            return True

    def _is_execution_complete(self) -> bool:
        """Check if workflow execution is complete."""
        try:
            # Get current task counts
            scheduler_summary = self.scheduler.get_scheduler_summary()
            task_counts = scheduler_summary['task_counts']

            queued = task_counts['queued']
            running = task_counts['running']

            # Execution is complete when no tasks are queued or running
            return queued == 0 and running == 0

        except Exception as e:
            logger.error(f"Error checking execution completion: {e}")
            return False

    def _print_execution_status(self):
        """Print current execution status."""
        try:
            scheduler_summary = self.scheduler.get_scheduler_summary()
            resource_summary = self.resource_manager.get_resource_summary()
            monitoring_status = self.monitoring_system.get_current_status()

            task_counts = scheduler_summary['task_counts']
            current_time = time.time()
            elapsed_time = current_time - self.execution_start_time if self.execution_start_time else 0

            print("\n" + "="*80)
            print("DISTRIBUTED WORKFLOW EXECUTION STATUS")
            print("="*80)
            print(f"Elapsed Time: {elapsed_time:.1f} seconds")
            print(f"System Load: {monitoring_status['metrics']['system_load']:.2f}")
            print(f"Memory Usage: {monitoring_status['metrics']['memory_usage']:.1f}%")

            print("\nTask Status:")
            print(f"  Total: {task_counts['total']}")
            print(f"  Queued: {task_counts['queued']}")
            print(f"  Running: {task_counts['running']}")
            print(f"  Completed: {task_counts['completed']}")
            print(f"  Failed: {task_counts['failed']}")

            print("\nResource Status:")
            print(f"  Total Resources: {resource_summary['total_resources']}")
            print(f"  Available: {resource_summary['available_resources']}")
            print(f"  Total Cores: {resource_summary['total_cores']}")

            for name, details in resource_summary['resources'].items():
                status_icon = "✓" if details['status'] == 'available' else "✗"
                print(f"    {status_icon} {name}: {details['active_jobs']}/{details['cores']} jobs, "
                      f"load: {details['load_score']:.2f}")

            # Show active alerts
            active_alerts = [a for a in monitoring_status['active_alerts'] if not a.get('resolved', False)]
            if active_alerts:
                print(f"\nActive Alerts ({len(active_alerts)}):")
                for alert in active_alerts[-3:]:  # Show last 3 alerts
                    print(f"  [{alert['severity'].upper()}] {alert['message']}")

            print("="*80)

        except Exception as e:
            logger.error(f"Error printing execution status: {e}")

    def _print_final_summary(self):
        """Print final execution summary."""
        try:
            if not self.execution_start_time:
                return

            total_time = time.time() - self.execution_start_time
            scheduler_summary = self.scheduler.get_scheduler_summary()
            task_counts = scheduler_summary['task_counts']

            print("\n" + "="*80)
            print("DISTRIBUTED WORKFLOW EXECUTION SUMMARY")
            print("="*80)
            print(f"Total Execution Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

            print("\nTask Results:")
            print(f"  Total Tasks: {task_counts['total']}")
            print(f"  Completed: {task_counts['completed']}")
            print(f"  Failed: {task_counts['failed']}")
            print(f"  Success Rate: {task_counts['completed']/max(task_counts['total'], 1)*100:.1f}%")

            if task_counts['completed'] > 0:
                avg_time = total_time / task_counts['completed']
                print(f"  Average Time per Task: {avg_time:.1f} seconds")
                print(f"  Throughput: {task_counts['completed']/total_time*3600:.1f} tasks/hour")

            # Error statistics
            if hasattr(self, 'error_handler') and self.error_handler:
                error_stats = self.error_handler.get_error_statistics()
                if error_stats.get('total_errors', 0) > 0:
                    print("\nError Analysis:")
                    print(f"  Total Errors: {error_stats['total_errors']}")
                    print(f"  Resolution Rate: {error_stats['resolution_rate']:.1f}%")
                    print(f"  Average Retries: {error_stats['avg_retry_count']:.1f}")

            print("="*80)

        except Exception as e:
            logger.error(f"Error printing final summary: {e}")

    def _handle_alert(self, alert: Alert):
        """Handle system alerts."""
        try:
            severity_icons = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'error': '❌',
                'critical': '🚨'
            }

            icon = severity_icons.get(alert.severity, '❓')
            timestamp = time.strftime('%H:%M:%S', time.localtime(alert.timestamp))

            print(f"\n{icon} ALERT [{timestamp}] {alert.severity.upper()}: {alert.message}")

            # Take automated actions for critical alerts
            if alert.severity == 'critical':
                logger.critical(f"Critical alert: {alert.message}")
                # Could implement automated shutdown or resource reallocation here

        except Exception as e:
            logger.error(f"Error handling alert: {e}")

    def _shutdown_system(self):
        """Gracefully shutdown the system."""
        try:
            logger.info("Shutting down distributed workflow system...")
            self._shutdown_requested = True

            # Stop scheduler
            if self.scheduler:
                self.scheduler.stop_scheduler()

            # Stop monitoring
            if self.monitoring_system:
                self.monitoring_system.stop_monitoring()

            # Stop remote executor
            if self.remote_executor:
                self.remote_executor.stop_monitoring()

            logger.info("System shutdown completed")

        except Exception as e:
            logger.error(f"Error during system shutdown: {e}")

    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_system()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def main():
    """Main entry point for distributed workflow execution."""
    parser = argparse.ArgumentParser(
        description="Distributed ATLAS-QE Workflow Executor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete distributed workflow
  python run_distributed_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml

  # Dry run (generate tasks only)
  python run_distributed_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml --dry-run

  # Monitor existing workflow
  python run_distributed_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml --monitor-only

  # Use custom resource configuration
  python run_distributed_workflow.py workflow.yaml --resources my_resources.yaml
        """
    )

    parser.add_argument(
        "workflow_config",
        help="Path to workflow configuration YAML file"
    )

    parser.add_argument(
        "--resources",
        default="config/resources.yaml",
        help="Path to resource configuration YAML file (default: config/resources.yaml)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate tasks and input files without executing calculations"
    )

    parser.add_argument(
        "--monitor-only",
        action="store_true",
        help="Only start monitoring system without submitting new tasks"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        help="Log file path (default: logs/distributed_workflow.log)"
    )

    args = parser.parse_args()

    # Setup logging
    log_file = args.log_file or "logs/distributed_workflow.log"
    setup_logging(level=args.log_level, log_file=log_file)

    try:
        # Create and initialize executor
        executor = DistributedWorkflowExecutor(
            workflow_config_file=args.workflow_config,
            resource_config_file=args.resources,
            dry_run=args.dry_run,
            monitor_only=args.monitor_only
        )

        # Setup signal handlers
        executor._setup_signal_handlers()

        # Initialize system
        if not executor.initialize_system():
            logger.error("System initialization failed")
            return 1

        # Execute workflow
        success = executor.execute_workflow()

        # Shutdown system
        executor._shutdown_system()

        return 0 if success else 1

    except Exception as e:
        logger.error(f"Distributed workflow execution failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())