#!/usr/bin/env python3
"""
Distributed Workflow Monitoring and Management Tool

Provides real-time monitoring, resource status, task management,
and system administration capabilities for the distributed workflow system.

Usage:
    python monitor_distributed_workflow.py --status
    python monitor_distributed_workflow.py --resources
    python monitor_distributed_workflow.py --tasks --structure gaas_zincblende
    python monitor_distributed_workflow.py --dashboard
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from aqflow.core.configuration import ConfigurationManager
from aqflow.core.resource_manager import ResourceManager, LocalResource, RemoteResource
from aqflow.core.task_manager import TaskManager, TaskStatus
from aqflow.core.monitoring_system import MonitoringSystem
from aqflow.utils.logging_config import setup_logging

import logging
logger = logging.getLogger(__name__)


class DistributedWorkflowMonitor:
    """
    Monitoring and management interface for distributed workflow system.

    Provides command-line tools for monitoring system status, managing resources,
    viewing task progress, and controlling the workflow execution.
    """

    def __init__(
        self,
        resource_config_file: str = "config/resources.yaml",
        workflow_config_file: Optional[str] = None
    ):
        """Initialize workflow monitor."""
        self.config_manager = ConfigurationManager(
            workflow_config_file=workflow_config_file,
            resource_config_file=resource_config_file
        )

        # Initialize components
        self.task_manager = TaskManager(workspace_dir=Path("results"))
        self.resource_manager = ResourceManager()

        # Setup resources
        self._setup_resources()

        logger.info("DistributedWorkflowMonitor initialized")

    def _setup_resources(self):
        """Setup computational resources."""
        for config in self.config_manager.resource_configs:
            if config.resource_type.value == 'local':
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

    def show_system_status(self):
        """Display comprehensive system status."""
        print("\n" + "="*80)
        print("DISTRIBUTED WORKFLOW SYSTEM STATUS")
        print("="*80)

        # Configuration summary
        config_summary = self.config_manager.get_configuration_summary()
        print(f"Configuration Status:")
        print(f"  Workflow Config: {'✓' if config_summary['workflow_config_loaded'] else '✗'}")
        print(f"  Resource Configs: {config_summary['resource_configs_loaded']}")
        print(f"  Distributed Config: {'✓' if config_summary['distributed_config_loaded'] else '✗'}")

        # Task summary
        task_summary = self.task_manager.get_task_summary()
        print(f"\nTask Summary:")
        print(f"  Total Tasks: {task_summary.get('total_tasks', 0)}")
        print(f"  Completed: {task_summary.get('completed', 0)}")
        print(f"  Failed: {task_summary.get('failed', 0)}")
        print(f"  Running: {task_summary.get('running', 0)}")
        print(f"  Pending: {task_summary.get('pending', 0)}")
        print(f"  Completion Rate: {task_summary.get('completion_rate', 0):.1f}%")

        # Resource summary
        resource_summary = self.resource_manager.get_resource_summary()
        print(f"\nResource Summary:")
        print(f"  Total Resources: {resource_summary['total_resources']}")
        print(f"  Available: {resource_summary['available_resources']}")
        print(f"  Total Cores: {resource_summary['total_cores']}")

        print("="*80)

    def show_resource_details(self):
        """Display detailed resource information."""
        print("\n" + "="*80)
        print("COMPUTATIONAL RESOURCES")
        print("="*80)

        # Update resource metrics
        self.resource_manager.update_all_resources()

        for name, resource in self.resource_manager.resources.items():
            metrics = resource.metrics
            status_icon = "✓" if resource.status.value == 'available' else "✗"

            print(f"\n{status_icon} {name} ({resource.resource_type.value.upper()})")
            print(f"  Status: {resource.status.value}")
            print(f"  Cores: {resource.capability.cores}")
            print(f"  Memory: {resource.capability.memory_gb} GB")
            print(f"  Max Jobs: {resource.capability.max_concurrent_jobs}")
            print(f"  Active Jobs: {metrics.active_jobs}")
            print(f"  Load Score: {resource.get_load_score():.2f}")

            if hasattr(resource, 'hostname'):
                print(f"  Hostname: {resource.hostname}")

            print(f"  Software: {', '.join(resource.software_paths.keys())}")

            # Check resource availability
            available = resource.check_availability()
            print(f"  Availability Check: {'✓ Pass' if available else '✗ Fail'}")

        print("="*80)

    def show_task_details(
        self,
        structure: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20
    ):
        """Display detailed task information."""
        print("\n" + "="*80)
        print("TASK DETAILS")
        print("="*80)

        # Get structure progress
        structure_progress = self.task_manager.get_structure_progress()

        if structure:
            if structure in structure_progress:
                self._print_structure_progress(structure, structure_progress[structure])
            else:
                print(f"Structure '{structure}' not found")
                print(f"Available structures: {', '.join(structure_progress.keys())}")
        else:
            # Show progress for all structures
            for struct_name, progress in structure_progress.items():
                self._print_structure_progress(struct_name, progress)

        # Show recent tasks if status filter is specified
        if status:
            print(f"\nRecent tasks with status '{status}' (limit {limit}):")
            try:
                task_status = TaskStatus(status)
                tasks = self.task_manager.get_tasks_by_status(task_status)[:limit]

                for task in tasks:
                    print(f"  {task.task_id}")
                    print(f"    Structure: {task.structure_name}")
                    print(f"    Software: {task.software}")
                    print(f"    Volume: {task.volume_point}")
                    print(f"    Directory: {task.working_directory}")

            except ValueError:
                print(f"Invalid status: {status}")
                print(f"Valid statuses: {[s.value for s in TaskStatus]}")

        print("="*80)

    def _print_structure_progress(self, structure_name: str, progress: Dict[str, Any]):
        """Print progress for a specific structure."""
        print(f"\n{structure_name}:")

        for combination_name, stats in progress.items():
            total = stats['total']
            completed = stats.get('completed', 0)
            failed = stats.get('failed', 0)
            running = stats.get('running', 0)
            pending = stats.get('pending', 0)

            completion_rate = (completed / total * 100) if total > 0 else 0

            print(f"  {combination_name}: {completed}/{total} ({completion_rate:.1f}%)")
            if running > 0:
                print(f"    Running: {running}")
            if pending > 0:
                print(f"    Pending: {pending}")
            if failed > 0:
                print(f"    Failed: {failed}")

    def run_dashboard(self, update_interval: int = 30):
        """Run interactive dashboard."""
        print("\n" + "="*80)
        print("DISTRIBUTED WORKFLOW DASHBOARD")
        print("Press Ctrl+C to exit")
        print("="*80)

        try:
            while True:
                # Clear screen (works on most terminals)
                print("\033[2J\033[H")

                # Show current timestamp
                current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                print(f"Dashboard - {current_time}")
                print("="*80)

                # System status
                self.show_system_status()

                # Resource status
                print("\nResource Status:")
                self.resource_manager.update_all_resources()
                for name, resource in self.resource_manager.resources.items():
                    status_icon = "🟢" if resource.status.value == 'available' else "🔴"
                    load = resource.get_load_score()
                    load_icon = "🟢" if load < 0.5 else "🟡" if load < 0.8 else "🔴"

                    print(f"  {status_icon} {name}: {resource.metrics.active_jobs} jobs, "
                          f"load {load_icon} {load:.2f}")

                print(f"\nNext update in {update_interval} seconds...")
                time.sleep(update_interval)

        except KeyboardInterrupt:
            print("\nDashboard stopped.")

    def export_results(self, output_file: str, format: str = 'json'):
        """Export task results to file."""
        output_path = Path(output_file)
        self.task_manager.export_results(output_path, format)
        print(f"Results exported to {output_file}")

    def cleanup_old_tasks(self, days: int = 30):
        """Clean up old completed tasks."""
        print(f"Cleaning up tasks older than {days} days...")
        self.task_manager.cleanup_completed_tasks(days)
        print("Cleanup completed.")

    def check_system_health(self):
        """Perform comprehensive system health check."""
        print("\n" + "="*80)
        print("SYSTEM HEALTH CHECK")
        print("="*80)

        health_issues = []

        # Check configurations
        if not self.config_manager.validate_configurations():
            health_issues.append("❌ Configuration validation failed")
        else:
            print("✅ Configuration validation passed")

        # Check resource availability
        available_resources = 0
        total_resources = len(self.resource_manager.resources)

        for name, resource in self.resource_manager.resources.items():
            if resource.check_availability():
                available_resources += 1
                print(f"✅ Resource {name} is available")
            else:
                health_issues.append(f"❌ Resource {name} is not available")

        if available_resources == 0:
            health_issues.append("❌ No resources are available")

        # Check database connectivity
        try:
            task_summary = self.task_manager.get_task_summary()
            print("✅ Task database is accessible")
        except Exception as e:
            health_issues.append(f"❌ Task database error: {e}")

        # Check disk space
        try:
            import shutil
            workspace = Path("results")
            if workspace.exists():
                total, used, free = shutil.disk_usage(workspace)
                free_percent = (free / total) * 100

                if free_percent > 10:
                    print(f"✅ Disk space: {free_percent:.1f}% free")
                else:
                    health_issues.append(f"⚠️  Low disk space: {free_percent:.1f}% free")
        except Exception as e:
            health_issues.append(f"❌ Could not check disk space: {e}")

        # Summary
        if health_issues:
            print(f"\n❌ HEALTH CHECK FAILED ({len(health_issues)} issues)")
            for issue in health_issues:
                print(f"  {issue}")
        else:
            print("\n✅ SYSTEM HEALTH: ALL CHECKS PASSED")

        print("="*80)

    def cancel_tasks(self, task_ids: List[str]):
        """Cancel specified tasks."""
        print(f"Cancelling {len(task_ids)} tasks...")

        cancelled_count = 0
        for task_id in task_ids:
            try:
                self.task_manager.update_task_status(task_id, TaskStatus.CANCELLED)
                cancelled_count += 1
                print(f"✅ Cancelled task {task_id}")
            except Exception as e:
                print(f"❌ Failed to cancel task {task_id}: {e}")

        print(f"Cancelled {cancelled_count}/{len(task_ids)} tasks")

    def restart_failed_tasks(self):
        """Restart all failed tasks."""
        failed_tasks = self.task_manager.get_tasks_by_status(TaskStatus.FAILED)
        print(f"Restarting {len(failed_tasks)} failed tasks...")

        restarted_count = 0
        for task in failed_tasks:
            try:
                self.task_manager.update_task_status(task.task_id, TaskStatus.PENDING)
                restarted_count += 1
            except Exception as e:
                print(f"❌ Failed to restart task {task.task_id}: {e}")

        print(f"Restarted {restarted_count}/{len(failed_tasks)} tasks")


def main():
    """Main entry point for monitoring tool."""
    parser = argparse.ArgumentParser(
        description="Distributed Workflow Monitoring and Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--resources",
        default="config/resources.yaml",
        help="Resource configuration file"
    )

    parser.add_argument(
        "--workflow",
        help="Workflow configuration file"
    )

    # Action arguments
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show system status"
    )

    parser.add_argument(
        "--resource-details",
        action="store_true",
        help="Show detailed resource information"
    )

    parser.add_argument(
        "--tasks",
        action="store_true",
        help="Show task details"
    )

    parser.add_argument(
        "--structure",
        help="Filter tasks by structure name"
    )

    parser.add_argument(
        "--task-status",
        choices=[s.value for s in TaskStatus],
        help="Filter tasks by status"
    )

    parser.add_argument(
        "--dashboard",
        action="store_true",
        help="Run interactive dashboard"
    )

    parser.add_argument(
        "--update-interval",
        type=int,
        default=30,
        help="Dashboard update interval in seconds"
    )

    parser.add_argument(
        "--export",
        help="Export results to file"
    )

    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format"
    )

    parser.add_argument(
        "--cleanup-days",
        type=int,
        help="Clean up tasks older than specified days"
    )

    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Perform system health check"
    )

    parser.add_argument(
        "--cancel-tasks",
        nargs="+",
        help="Cancel specified task IDs"
    )

    parser.add_argument(
        "--restart-failed",
        action="store_true",
        help="Restart all failed tasks"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)

    try:
        # Initialize monitor
        monitor = DistributedWorkflowMonitor(
            resource_config_file=args.resources,
            workflow_config_file=args.workflow
        )

        # Execute requested actions
        if args.status:
            monitor.show_system_status()

        if args.resource_details:
            monitor.show_resource_details()

        if args.tasks:
            monitor.show_task_details(
                structure=args.structure,
                status=args.task_status
            )

        if args.dashboard:
            monitor.run_dashboard(args.update_interval)

        if args.export:
            monitor.export_results(args.export, args.format)

        if args.cleanup_days:
            monitor.cleanup_old_tasks(args.cleanup_days)

        if args.health_check:
            monitor.check_system_health()

        if args.cancel_tasks:
            monitor.cancel_tasks(args.cancel_tasks)

        if args.restart_failed:
            monitor.restart_failed_tasks()

        # Default action if no specific action specified
        if not any([
            args.status, args.resource_details, args.tasks, args.dashboard,
            args.export, args.cleanup_days, args.health_check,
            args.cancel_tasks, args.restart_failed
        ]):
            monitor.show_system_status()

    except Exception as e:
        logger.error(f"Monitor execution failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())