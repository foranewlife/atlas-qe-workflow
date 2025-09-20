#!/usr/bin/env python3
"""
Task Query Tool for ATLAS-QE Workflow

Simple command-line tool to query and inspect task status and results.

Usage:
    python query_tasks.py --summary
    python query_tasks.py --structure gaas_zincblende
    python query_tasks.py --status pending
"""

import argparse
import json
from pathlib import Path

from aqflow.core.task_manager import TaskManager, TaskStatus
from aqflow.utils.logging_config import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Task Query Tool")
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("results/workflow_tasks.db"),
        help="Path to task database"
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show task summary"
    )
    parser.add_argument(
        "--structure",
        help="Filter by structure name"
    )
    parser.add_argument(
        "--status",
        choices=[s.value for s in TaskStatus],
        help="Filter by task status"
    )
    parser.add_argument(
        "--export",
        type=Path,
        help="Export results to file"
    )
    parser.add_argument(
        "--format",
        choices=["json", "csv"],
        default="json",
        help="Export format"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    # Initialize task manager
    if not args.database.exists():
        print(f"Task database not found: {args.database}")
        print("Run the workflow first to create tasks.")
        return

    task_manager = TaskManager(
        workspace_dir=args.database.parent,
        database_path=args.database
    )

    if args.summary:
        # Show task summary
        summary = task_manager.get_task_summary()
        print("\n" + "="*50)
        print("TASK SUMMARY")
        print("="*50)
        print(f"Total tasks: {summary['total_tasks']}")
        print(f"Active tasks: {summary['active_tasks']}")
        print(f"Completion rate: {summary['completion_rate']:.1f}%")
        print("\nStatus breakdown:")
        for status, count in summary.items():
            if status not in ['active_tasks', 'total_tasks', 'completion_rate']:
                print(f"  {status}: {count}")

        # Show structure progress
        print("\n" + "="*50)
        print("STRUCTURE PROGRESS")
        print("="*50)
        structure_progress = task_manager.get_structure_progress()
        for structure_name, structure_data in structure_progress.items():
            print(f"\n{structure_name}:")
            for combination_name, combo_data in structure_data.items():
                total = combo_data['total']
                completed = combo_data.get('completed', 0)
                failed = combo_data.get('failed', 0)
                pending = combo_data.get('pending', 0)
                running = combo_data.get('running', 0)

                completion_rate = (completed / total * 100) if total > 0 else 0
                print(f"  {combination_name}: {completed}/{total} ({completion_rate:.0f}%)")
                if failed > 0:
                    print(f"    Failed: {failed}")
                if running > 0:
                    print(f"    Running: {running}")
                if pending > 0:
                    print(f"    Pending: {pending}")

    elif args.status:
        # Show tasks by status
        status = TaskStatus(args.status)
        tasks = task_manager.db.get_tasks_by_status(status)
        print(f"\nTasks with status '{args.status}': {len(tasks)}")
        print("-" * 60)
        for task in tasks[:10]:  # Show first 10
            print(f"{task.task_id}")
            print(f"  Structure: {task.structure_name}")
            print(f"  Combination: {task.combination_name}")
            print(f"  Volume point: {task.volume_point}")
            print(f"  Created: {task.created_at}")
            print()
        if len(tasks) > 10:
            print(f"... and {len(tasks) - 10} more tasks")

    elif args.structure:
        # Show tasks for specific structure
        print(f"\nTasks for structure '{args.structure}':")
        print("-" * 60)

        # Query database directly for structure-specific tasks
        import sqlite3
        with sqlite3.connect(task_manager.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT
                    t.task_id, t.combination_name, t.volume_point,
                    COALESCE(r.status, 'pending') as status,
                    r.energy, r.execution_time
                FROM tasks t
                LEFT JOIN results r ON t.task_id = r.task_id
                WHERE t.structure_name = ?
                ORDER BY t.combination_name, t.volume_point
            ''', (args.structure,))

            results = cursor.fetchall()

        if not results:
            print(f"No tasks found for structure '{args.structure}'")
        else:
            current_combination = None
            for row in results:
                task_id, combination_name, volume_point, status, energy, exec_time = row

                if combination_name != current_combination:
                    current_combination = combination_name
                    print(f"\n{combination_name}:")

                energy_str = f"{energy:.6f}" if energy is not None else "N/A"
                time_str = f"{exec_time:.2f}s" if exec_time is not None else "N/A"
                print(f"  V={volume_point:.5f}: {status:>10} E={energy_str:>12} t={time_str:>8}")

    if args.export:
        # Export results
        print(f"\nExporting results to {args.export} in {args.format} format...")
        task_manager.export_results(args.export, args.format)
        print("Export completed.")


if __name__ == "__main__":
    main()