"""
Task management and status tracking system.

This module provides comprehensive task management capabilities including
status tracking, progress monitoring, and persistence of task states.
"""

import json
import sqlite3
import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a calculation task."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TaskPriority(Enum):
    """Priority levels for tasks."""
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


@dataclass
class TaskDefinition:
    """Definition of a calculation task."""
    task_id: str
    structure_name: str
    combination_name: str
    software: str
    volume_point: float
    input_file: str
    working_directory: str
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: datetime = None
    parameters_fingerprint: str = ""

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

    def generate_fingerprint(self) -> str:
        """Generate unique fingerprint for task parameters."""
        content = f"{self.structure_name}_{self.combination_name}_{self.software}_{self.volume_point}"
        return hashlib.md5(content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat() if self.created_at else None
        data['priority'] = self.priority.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskDefinition':
        """Create from dictionary."""
        if 'created_at' in data and data['created_at']:
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if 'priority' in data:
            data['priority'] = TaskPriority(data['priority'])
        return cls(**data)


@dataclass
class TaskResult:
    """Result of a completed task."""
    task_id: str
    status: TaskStatus
    energy: Optional[float] = None
    execution_time: float = 0.0
    return_code: int = 0
    error_message: Optional[str] = None
    output_files: List[str] = None
    converged: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.output_files is None:
            self.output_files = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TaskResult':
        """Create from dictionary."""
        if 'status' in data:
            data['status'] = TaskStatus(data['status'])
        if 'started_at' in data and data['started_at']:
            data['started_at'] = datetime.fromisoformat(data['started_at'])
        if 'completed_at' in data and data['completed_at']:
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        return cls(**data)


class TaskDatabase:
    """SQLite-based task database for persistence."""

    def __init__(self, db_path: Path):
        """Initialize task database."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_database()

    def _initialize_database(self):
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Tasks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    structure_name TEXT NOT NULL,
                    combination_name TEXT NOT NULL,
                    software TEXT NOT NULL,
                    volume_point REAL NOT NULL,
                    input_file TEXT NOT NULL,
                    working_directory TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    parameters_fingerprint TEXT NOT NULL,
                    task_data TEXT NOT NULL
                )
            ''')

            # Results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS results (
                    task_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    energy REAL,
                    execution_time REAL NOT NULL,
                    return_code INTEGER NOT NULL,
                    error_message TEXT,
                    output_files TEXT,
                    converged BOOLEAN NOT NULL,
                    started_at TEXT,
                    completed_at TEXT,
                    metadata TEXT,
                    result_data TEXT NOT NULL,
                    FOREIGN KEY (task_id) REFERENCES tasks (task_id)
                )
            ''')

            # Indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_status ON results (status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_tasks_fingerprint ON tasks (parameters_fingerprint)')

            conn.commit()

    def save_task(self, task: TaskDefinition):
        """Save task to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO tasks (
                    task_id, structure_name, combination_name, software,
                    volume_point, input_file, working_directory, priority,
                    created_at, parameters_fingerprint, task_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                task.task_id,
                task.structure_name,
                task.combination_name,
                task.software,
                task.volume_point,
                task.input_file,
                task.working_directory,
                task.priority.value,
                task.created_at.isoformat(),
                task.parameters_fingerprint,
                json.dumps(task.to_dict())
            ))
            conn.commit()

    def save_result(self, result: TaskResult):
        """Save task result to database."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO results (
                    task_id, status, energy, execution_time, return_code,
                    error_message, output_files, converged, started_at,
                    completed_at, metadata, result_data
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                result.task_id,
                result.status.value,
                result.energy,
                result.execution_time,
                result.return_code,
                result.error_message,
                json.dumps(result.output_files),
                result.converged,
                result.started_at.isoformat() if result.started_at else None,
                result.completed_at.isoformat() if result.completed_at else None,
                json.dumps(result.metadata),
                json.dumps(result.to_dict())
            ))
            conn.commit()

    def get_task(self, task_id: str) -> Optional[TaskDefinition]:
        """Get task by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT task_data FROM tasks WHERE task_id = ?', (task_id,))
            row = cursor.fetchone()
            if row:
                return TaskDefinition.from_dict(json.loads(row[0]))
        return None

    def get_result(self, task_id: str) -> Optional[TaskResult]:
        """Get task result by ID."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT result_data FROM results WHERE task_id = ?', (task_id,))
            row = cursor.fetchone()
            if row:
                return TaskResult.from_dict(json.loads(row[0]))
        return None

    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskDefinition]:
        """Get all tasks with given status."""
        tasks = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT t.task_data FROM tasks t
                LEFT JOIN results r ON t.task_id = r.task_id
                WHERE r.status = ? OR (r.status IS NULL AND ? = 'pending')
            ''', (status.value, status.value))

            for row in cursor.fetchall():
                tasks.append(TaskDefinition.from_dict(json.loads(row[0])))

        return tasks

    def get_task_summary(self) -> Dict[str, int]:
        """Get summary of task counts by status."""
        summary = {status.value: 0 for status in TaskStatus}

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT COALESCE(r.status, 'pending') as status, COUNT(*) as count
                FROM tasks t
                LEFT JOIN results r ON t.task_id = r.task_id
                GROUP BY COALESCE(r.status, 'pending')
            ''')

            for row in cursor.fetchall():
                summary[row[0]] = row[1]

        return summary


class TaskManager:
    """
    Comprehensive task management system.

    Provides high-level interface for task creation, tracking,
    and status management with persistence.
    """

    def __init__(self, workspace_dir: Path, database_path: Optional[Path] = None):
        """
        Initialize task manager.

        Args:
            workspace_dir: Base directory for task workspaces
            database_path: Path to SQLite database file
        """
        self.workspace_dir = Path(workspace_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        if database_path is None:
            database_path = self.workspace_dir / 'tasks.db'

        self.db = TaskDatabase(database_path)
        self.active_tasks = {}  # task_id -> TaskResult (for running tasks)

        logger.info(f"TaskManager initialized with workspace: {self.workspace_dir}")

    def create_task(
        self,
        structure_name: str,
        combination_name: str,
        software: str,
        volume_point: float,
        input_file: str,
        working_directory: str,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> TaskDefinition:
        """
        Create a new task.

        Args:
            structure_name: Name of the structure
            combination_name: Name of the parameter combination
            software: Software to use ('atlas' or 'qe')
            volume_point: Volume scaling factor
            input_file: Path to input file
            working_directory: Working directory for calculation
            priority: Task priority

        Returns:
            Created TaskDefinition
        """
        # Generate unique task ID
        timestamp = int(time.time() * 1000)
        task_id = f"{structure_name}_{combination_name}_{volume_point:.5f}_{timestamp}"

        # Create task definition
        task = TaskDefinition(
            task_id=task_id,
            structure_name=structure_name,
            combination_name=combination_name,
            software=software,
            volume_point=volume_point,
            input_file=input_file,
            working_directory=working_directory,
            priority=priority
        )

        # Generate fingerprint
        task.parameters_fingerprint = task.generate_fingerprint()

        # Save to database
        self.db.save_task(task)

        logger.info(f"Created task: {task_id}")
        return task

    def update_task_status(
        self,
        task_id: str,
        status: TaskStatus,
        energy: Optional[float] = None,
        execution_time: float = 0.0,
        return_code: int = 0,
        error_message: Optional[str] = None,
        converged: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> TaskResult:
        """
        Update task status and result.

        Args:
            task_id: Task identifier
            status: New task status
            energy: Calculated energy (if available)
            execution_time: Time taken for execution
            return_code: Process return code
            error_message: Error message (if failed)
            converged: Whether calculation converged
            metadata: Additional metadata

        Returns:
            Updated TaskResult
        """
        # Get or create result
        result = self.db.get_result(task_id)
        if result is None:
            result = TaskResult(task_id=task_id, status=status)

        # Update fields
        result.status = status
        if energy is not None:
            result.energy = energy
        result.execution_time = execution_time
        result.return_code = return_code
        if error_message is not None:
            result.error_message = error_message
        result.converged = converged
        if metadata is not None:
            result.metadata.update(metadata)

        # Set timestamps
        now = datetime.now()
        if status == TaskStatus.RUNNING and result.started_at is None:
            result.started_at = now
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
            result.completed_at = now

        # Update active tasks tracking
        if status == TaskStatus.RUNNING:
            self.active_tasks[task_id] = result
        elif task_id in self.active_tasks:
            del self.active_tasks[task_id]

        # Save to database
        self.db.save_result(result)

        logger.debug(f"Updated task {task_id}: {status.value}")
        return result

    def get_pending_tasks(self, limit: Optional[int] = None) -> List[TaskDefinition]:
        """Get pending tasks, optionally limited by count."""
        tasks = self.db.get_tasks_by_status(TaskStatus.PENDING)

        # Sort by priority and creation time
        priority_order = {
            TaskPriority.URGENT: 0,
            TaskPriority.HIGH: 1,
            TaskPriority.NORMAL: 2,
            TaskPriority.LOW: 3
        }

        tasks.sort(key=lambda t: (priority_order.get(t.priority, 2), t.created_at))

        if limit:
            tasks = tasks[:limit]

        return tasks

    def get_task_summary(self) -> Dict[str, Any]:
        """Get comprehensive task summary."""
        summary = self.db.get_task_summary()

        # Add runtime information
        summary['active_tasks'] = len(self.active_tasks)
        summary['total_tasks'] = sum(summary.values())

        # Calculate completion rate
        completed = summary.get('completed', 0)
        total = summary['total_tasks']
        completion_rate = (completed / total * 100) if total > 0 else 0

        summary['completion_rate'] = completion_rate

        return summary

    def get_structure_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get progress summary by structure."""
        progress = {}

        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    t.structure_name,
                    t.combination_name,
                    COALESCE(r.status, 'pending') as status,
                    COUNT(*) as count
                FROM tasks t
                LEFT JOIN results r ON t.task_id = r.task_id
                GROUP BY t.structure_name, t.combination_name, COALESCE(r.status, 'pending')
                ORDER BY t.structure_name, t.combination_name
            ''')

            for row in cursor.fetchall():
                structure_name, combination_name, status, count = row

                if structure_name not in progress:
                    progress[structure_name] = {}

                if combination_name not in progress[structure_name]:
                    progress[structure_name][combination_name] = {
                        'total': 0,
                        'completed': 0,
                        'failed': 0,
                        'running': 0,
                        'pending': 0
                    }

                progress[structure_name][combination_name]['total'] += count
                progress[structure_name][combination_name][status] = count

        return progress

    def cleanup_completed_tasks(self, keep_days: int = 30):
        """
        Clean up old completed tasks.

        Args:
            keep_days: Number of days to keep completed tasks
        """
        cutoff_date = datetime.now().timestamp() - (keep_days * 24 * 3600)

        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                DELETE FROM results
                WHERE status = 'completed'
                AND completed_at < datetime(?, 'unixepoch')
            ''', (cutoff_date,))

            cursor.execute('''
                DELETE FROM tasks
                WHERE task_id NOT IN (SELECT task_id FROM results)
            ''')

            conn.commit()
            removed_count = cursor.rowcount

        logger.info(f"Cleaned up {removed_count} completed tasks older than {keep_days} days")

    def export_results(self, output_file: Path, format: str = 'json'):
        """
        Export task results to file.

        Args:
            output_file: Output file path
            format: Export format ('json' or 'csv')
        """
        with sqlite3.connect(self.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    t.task_id, t.structure_name, t.combination_name, t.software,
                    t.volume_point, r.status, r.energy, r.execution_time,
                    r.converged, r.error_message
                FROM tasks t
                LEFT JOIN results r ON t.task_id = r.task_id
                ORDER BY t.structure_name, t.combination_name, t.volume_point
            ''')

            results = cursor.fetchall()

        if format == 'json':
            data = []
            for row in results:
                data.append({
                    'task_id': row[0],
                    'structure_name': row[1],
                    'combination_name': row[2],
                    'software': row[3],
                    'volume_point': row[4],
                    'status': row[5],
                    'energy': row[6],
                    'execution_time': row[7],
                    'converged': bool(row[8]) if row[8] is not None else None,
                    'error_message': row[9]
                })

            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)

        elif format == 'csv':
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'task_id', 'structure_name', 'combination_name', 'software',
                    'volume_point', 'status', 'energy', 'execution_time',
                    'converged', 'error_message'
                ])
                writer.writerows(results)

        logger.info(f"Exported {len(results)} results to {output_file}")

    def get_tasks_by_status(self, status: TaskStatus) -> List[TaskDefinition]:
        """Get all tasks with a specific status."""
        return self.db.get_tasks_by_status(status)

    def update_task_completion(
        self,
        task_id: str,
        energy: Optional[float] = None,
        converged: bool = False,
        execution_time: Optional[float] = None,
        output_files: Optional[List[str]] = None
    ):
        """Update task with completion results."""
        metadata = {}
        if output_files:
            metadata['output_files'] = output_files

        self.update_task_status(
            task_id=task_id,
            status=TaskStatus.COMPLETED,
            energy=energy,
            converged=converged,
            metadata=metadata
        )