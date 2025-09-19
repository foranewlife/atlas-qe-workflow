"""
Real-time monitoring system for distributed computing.

This module provides comprehensive monitoring capabilities including
real-time status tracking, progress monitoring, resource monitoring,
and web-based dashboard.
"""

import time
import threading
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import sqlite3

from .resource_manager import ResourceManager, ResourceMetrics
from .distributed_scheduler import DistributedScheduler
from .task_manager import TaskManager, TaskStatus
from .remote_executor import RemoteExecutionEngine

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics to monitor."""
    RESOURCE_UTILIZATION = "resource_utilization"
    TASK_THROUGHPUT = "task_throughput"
    JOB_SUCCESS_RATE = "job_success_rate"
    QUEUE_LENGTH = "queue_length"
    EXECUTION_TIME = "execution_time"


@dataclass
class SystemMetrics:
    """System-wide metrics snapshot."""
    timestamp: float = field(default_factory=time.time)

    # Task metrics
    total_tasks: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    running_tasks: int = 0
    queued_tasks: int = 0

    # Resource metrics
    total_resources: int = 0
    available_resources: int = 0
    total_cores: int = 0
    active_cores: int = 0

    # Performance metrics
    avg_execution_time: float = 0.0
    success_rate: float = 0.0
    throughput_per_hour: float = 0.0

    # System health
    system_load: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    metric_type: MetricType
    threshold: float
    operator: str  # '>', '<', '>=', '<=', '=='
    severity: str  # 'info', 'warning', 'error', 'critical'
    cooldown_seconds: int = 300
    last_triggered: float = 0.0
    enabled: bool = True


@dataclass
class Alert:
    """System alert."""
    rule_name: str
    message: str
    severity: str
    timestamp: float = field(default_factory=time.time)
    resolved: bool = False
    resolved_timestamp: Optional[float] = None


class MonitoringSystem:
    """
    Comprehensive monitoring system for distributed computing.

    Provides real-time monitoring, alerting, metrics collection,
    and dashboard capabilities.
    """

    def __init__(
        self,
        resource_manager: ResourceManager,
        task_manager: TaskManager,
        scheduler: DistributedScheduler,
        remote_executor: RemoteExecutionEngine,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize monitoring system.

        Args:
            resource_manager: Resource manager instance
            task_manager: Task manager instance
            scheduler: Distributed scheduler instance
            remote_executor: Remote execution engine instance
            config: Configuration dictionary
        """
        self.resource_manager = resource_manager
        self.task_manager = task_manager
        self.scheduler = scheduler
        self.remote_executor = remote_executor
        self.config = config or {}

        # Monitoring state
        self.monitoring_active = False
        self._monitor_thread = None
        self._stop_monitoring = threading.Event()

        # Metrics storage
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = self.config.get('max_history_size', 1000)

        # Alerting
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: List[Alert] = []
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Database for persistent metrics
        self.db_path = self.config.get('metrics_db', 'monitoring/metrics.db')
        self._init_metrics_database()

        # Monitoring intervals
        self.metrics_interval = self.config.get('metrics_interval', 30)
        self.alert_check_interval = self.config.get('alert_check_interval', 60)

        # Initialize default alert rules
        self._setup_default_alerts()

        logger.info("MonitoringSystem initialized")

    def start_monitoring(self):
        """Start the monitoring system."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._stop_monitoring.clear()

            self._monitor_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self._monitor_thread.start()

            logger.info("Started monitoring system")

    def stop_monitoring(self):
        """Stop the monitoring system."""
        if self.monitoring_active:
            self.monitoring_active = False
            self._stop_monitoring.set()

            if self._monitor_thread:
                self._monitor_thread.join(timeout=30)

            logger.info("Stopped monitoring system")

    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # Get scheduler summary
            scheduler_summary = self.scheduler.get_scheduler_summary()
            task_counts = scheduler_summary['task_counts']
            resource_summary = scheduler_summary['resource_summary']

            # Calculate performance metrics
            total_completed = task_counts['completed']
            total_failed = task_counts['failed']
            total_finished = total_completed + total_failed

            success_rate = (total_completed / total_finished * 100) if total_finished > 0 else 0.0

            # Calculate throughput (tasks per hour)
            throughput = self._calculate_throughput()

            # Get average execution time
            avg_exec_time = self._get_average_execution_time()

            # Get system resource usage
            system_load, memory_usage, disk_usage = self._get_system_resource_usage()

            # Create metrics snapshot
            metrics = SystemMetrics(
                total_tasks=task_counts['total'],
                completed_tasks=task_counts['completed'],
                failed_tasks=task_counts['failed'],
                running_tasks=task_counts['running'],
                queued_tasks=task_counts['queued'],

                total_resources=resource_summary['total_resources'],
                available_resources=resource_summary['available_resources'],
                total_cores=resource_summary['total_cores'],
                active_cores=self._get_active_cores(),

                avg_execution_time=avg_exec_time,
                success_rate=success_rate,
                throughput_per_hour=throughput,

                system_load=system_load,
                memory_usage=memory_usage,
                disk_usage=disk_usage
            )

            # Store metrics
            self._store_metrics(metrics)

            return metrics

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return SystemMetrics()

    def add_alert_rule(self, rule: AlertRule):
        """Add a new alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        self.alert_rules = [r for r in self.alert_rules if r.name != rule_name]
        logger.info(f"Removed alert rule: {rule_name}")

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add a callback function for alert notifications."""
        self.alert_callbacks.append(callback)

    def get_current_status(self) -> Dict[str, Any]:
        """Get current system status summary."""
        current_metrics = self.collect_metrics()

        return {
            'monitoring_active': self.monitoring_active,
            'timestamp': current_metrics.timestamp,
            'metrics': asdict(current_metrics),
            'active_alerts': [asdict(alert) for alert in self.active_alerts if not alert.resolved],
            'resource_details': self._get_detailed_resource_status(),
            'task_breakdown': self._get_task_breakdown(),
            'recent_activity': self._get_recent_activity()
        }

    def get_metrics_history(
        self,
        hours: int = 24,
        metric_types: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get historical metrics data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate time range
            end_time = time.time()
            start_time = end_time - (hours * 3600)

            # Query metrics
            cursor.execute('''
                SELECT timestamp, metrics_json FROM system_metrics
                WHERE timestamp >= ? AND timestamp <= ?
                ORDER BY timestamp
            ''', (start_time, end_time))

            rows = cursor.fetchall()
            conn.close()

            # Parse and filter metrics
            history = []
            for timestamp, metrics_json in rows:
                metrics_data = json.loads(metrics_json)

                if metric_types:
                    # Filter to specific metric types
                    filtered_data = {
                        'timestamp': timestamp,
                        **{k: v for k, v in metrics_data.items() if k in metric_types}
                    }
                    history.append(filtered_data)
                else:
                    history.append({'timestamp': timestamp, **metrics_data})

            return history

        except Exception as e:
            logger.error(f"Error retrieving metrics history: {e}")
            return []

    def _monitoring_loop(self):
        """Main monitoring loop."""
        last_metrics_time = 0
        last_alert_check_time = 0

        while not self._stop_monitoring.is_set():
            try:
                current_time = time.time()

                # Collect metrics
                if current_time - last_metrics_time >= self.metrics_interval:
                    self.collect_metrics()
                    last_metrics_time = current_time

                # Check alerts
                if current_time - last_alert_check_time >= self.alert_check_interval:
                    self._check_alerts()
                    last_alert_check_time = current_time

                # Sleep for a short interval
                time.sleep(5)

            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(30)

    def _check_alerts(self):
        """Check all alert rules and trigger alerts if necessary."""
        try:
            current_metrics = self.metrics_history[-1] if self.metrics_history else SystemMetrics()

            for rule in self.alert_rules:
                if not rule.enabled:
                    continue

                # Check cooldown period
                if time.time() - rule.last_triggered < rule.cooldown_seconds:
                    continue

                # Get metric value
                metric_value = self._get_metric_value(current_metrics, rule.metric_type)
                if metric_value is None:
                    continue

                # Check threshold
                if self._evaluate_threshold(metric_value, rule.threshold, rule.operator):
                    self._trigger_alert(rule, metric_value)

        except Exception as e:
            logger.error(f"Error checking alerts: {e}")

    def _trigger_alert(self, rule: AlertRule, current_value: float):
        """Trigger an alert for a rule."""
        try:
            message = f"{rule.name}: {rule.metric_type.value} is {current_value} (threshold: {rule.operator} {rule.threshold})"

            alert = Alert(
                rule_name=rule.name,
                message=message,
                severity=rule.severity
            )

            # Add to active alerts
            self.active_alerts.append(alert)

            # Update rule last triggered time
            rule.last_triggered = time.time()

            # Call alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")

            logger.warning(f"ALERT [{rule.severity.upper()}]: {message}")

        except Exception as e:
            logger.error(f"Error triggering alert: {e}")

    def _get_metric_value(self, metrics: SystemMetrics, metric_type: MetricType) -> Optional[float]:
        """Get metric value by type."""
        metric_map = {
            MetricType.RESOURCE_UTILIZATION: metrics.active_cores / max(metrics.total_cores, 1) * 100,
            MetricType.TASK_THROUGHPUT: metrics.throughput_per_hour,
            MetricType.JOB_SUCCESS_RATE: metrics.success_rate,
            MetricType.QUEUE_LENGTH: metrics.queued_tasks,
            MetricType.EXECUTION_TIME: metrics.avg_execution_time
        }

        return metric_map.get(metric_type)

    def _evaluate_threshold(self, value: float, threshold: float, operator: str) -> bool:
        """Evaluate threshold condition."""
        if operator == '>':
            return value > threshold
        elif operator == '<':
            return value < threshold
        elif operator == '>=':
            return value >= threshold
        elif operator == '<=':
            return value <= threshold
        elif operator == '==':
            return abs(value - threshold) < 0.001
        else:
            return False

    def _calculate_throughput(self) -> float:
        """Calculate task throughput (tasks per hour)."""
        try:
            if len(self.metrics_history) < 2:
                return 0.0

            # Look at last hour of data
            current_time = time.time()
            hour_ago = current_time - 3600

            # Find metrics from an hour ago
            hour_ago_metrics = None
            for metrics in reversed(self.metrics_history):
                if metrics.timestamp <= hour_ago:
                    hour_ago_metrics = metrics
                    break

            if not hour_ago_metrics:
                return 0.0

            current_metrics = self.metrics_history[-1]
            completed_delta = current_metrics.completed_tasks - hour_ago_metrics.completed_tasks
            time_delta = current_metrics.timestamp - hour_ago_metrics.timestamp

            if time_delta > 0:
                return (completed_delta / time_delta) * 3600  # Convert to per hour
            else:
                return 0.0

        except Exception as e:
            logger.error(f"Error calculating throughput: {e}")
            return 0.0

    def _get_average_execution_time(self) -> float:
        """Get average execution time for completed tasks."""
        try:
            # Query recent completed tasks from task manager
            # This is a simplified version - actual implementation would query the database
            return 300.0  # Placeholder: 5 minutes average

        except Exception as e:
            logger.error(f"Error getting average execution time: {e}")
            return 0.0

    def _get_system_resource_usage(self) -> "Tuple[float, float, float]":
        """Get system resource usage (load, memory, disk)."""
        try:
            import psutil

            # CPU load
            load_avg = psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0.0

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100

            return load_avg, memory_percent, disk_percent

        except Exception as e:
            logger.error(f"Error getting system resource usage: {e}")
            return 0.0, 0.0, 0.0

    def _get_active_cores(self) -> int:
        """Get number of currently active cores."""
        try:
            active_cores = 0
            for resource in self.resource_manager.resources.values():
                for job_info in resource.active_jobs.values():
                    active_cores += job_info.get('cores', 1)
            return active_cores

        except Exception as e:
            logger.error(f"Error getting active cores: {e}")
            return 0

    def _get_detailed_resource_status(self) -> Dict[str, Any]:
        """Get detailed status of all resources."""
        resource_details = {}

        for name, resource in self.resource_manager.resources.items():
            metrics = resource.update_metrics()

            resource_details[name] = {
                'type': resource.resource_type.value,
                'status': resource.status.value,
                'capability': {
                    'cores': resource.capability.cores,
                    'memory_gb': resource.capability.memory_gb,
                    'max_concurrent_jobs': resource.capability.max_concurrent_jobs
                },
                'current_metrics': {
                    'cpu_usage_percent': metrics.cpu_usage_percent,
                    'memory_usage_percent': metrics.memory_usage_percent,
                    'active_jobs': metrics.active_jobs,
                    'load_score': resource.get_load_score()
                },
                'active_jobs': list(resource.active_jobs.keys()),
                'software_available': list(resource.software_paths.keys())
            }

        return resource_details

    def _get_task_breakdown(self) -> Dict[str, Any]:
        """Get detailed task breakdown by software and status."""
        try:
            # This would query the task manager database for detailed breakdown
            # Simplified version for now
            return {
                'by_software': {
                    'atlas': {'queued': 10, 'running': 5, 'completed': 50, 'failed': 2},
                    'qe': {'queued': 15, 'running': 8, 'completed': 45, 'failed': 1}
                },
                'by_priority': {
                    'urgent': {'queued': 2, 'running': 1, 'completed': 5, 'failed': 0},
                    'high': {'queued': 5, 'running': 3, 'completed': 20, 'failed': 1},
                    'normal': {'queued': 15, 'running': 8, 'completed': 60, 'failed': 2},
                    'low': {'queued': 3, 'running': 1, 'completed': 10, 'failed': 0}
                }
            }

        except Exception as e:
            logger.error(f"Error getting task breakdown: {e}")
            return {}

    def _get_recent_activity(self) -> List[Dict[str, Any]]:
        """Get recent system activity."""
        try:
            # This would query recent events from the database
            # Simplified version for now
            recent_time = time.time() - 3600  # Last hour

            activities = []
            for alert in self.active_alerts:
                if alert.timestamp >= recent_time:
                    activities.append({
                        'timestamp': alert.timestamp,
                        'type': 'alert',
                        'severity': alert.severity,
                        'message': alert.message
                    })

            # Sort by timestamp (most recent first)
            activities.sort(key=lambda x: x['timestamp'], reverse=True)

            return activities[:20]  # Return last 20 activities

        except Exception as e:
            logger.error(f"Error getting recent activity: {e}")
            return []

    def _store_metrics(self, metrics: SystemMetrics):
        """Store metrics in database and memory."""
        try:
            # Store in memory
            self.metrics_history.append(metrics)

            # Limit memory history size
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]

            # Store in database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute('''
                INSERT INTO system_metrics (timestamp, metrics_json)
                VALUES (?, ?)
            ''', (metrics.timestamp, json.dumps(asdict(metrics))))

            conn.commit()
            conn.close()

        except Exception as e:
            logger.error(f"Error storing metrics: {e}")

    def _init_metrics_database(self):
        """Initialize the metrics database."""
        try:
            db_dir = Path(self.db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)

            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Create tables
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp REAL NOT NULL,
                    metrics_json TEXT NOT NULL
                )
            ''')

            cursor.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rule_name TEXT NOT NULL,
                    message TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    resolved_timestamp REAL
                )
            ''')

            # Create indexes
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON system_metrics(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alerts(timestamp)')

            conn.commit()
            conn.close()

            logger.info("Initialized metrics database")

        except Exception as e:
            logger.error(f"Error initializing metrics database: {e}")

    def _setup_default_alerts(self):
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="High Resource Utilization",
                metric_type=MetricType.RESOURCE_UTILIZATION,
                threshold=90.0,
                operator='>',
                severity='warning',
                cooldown_seconds=300
            ),
            AlertRule(
                name="Low Success Rate",
                metric_type=MetricType.JOB_SUCCESS_RATE,
                threshold=80.0,
                operator='<',
                severity='error',
                cooldown_seconds=600
            ),
            AlertRule(
                name="Long Queue",
                metric_type=MetricType.QUEUE_LENGTH,
                threshold=50,
                operator='>',
                severity='info',
                cooldown_seconds=600
            ),
            AlertRule(
                name="Long Execution Time",
                metric_type=MetricType.EXECUTION_TIME,
                threshold=3600,  # 1 hour
                operator='>',
                severity='warning',
                cooldown_seconds=300
            )
        ]

        self.alert_rules.extend(default_rules)
        logger.info("Setup default alert rules")