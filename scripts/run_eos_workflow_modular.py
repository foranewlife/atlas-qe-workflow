#!/usr/bin/env python3
"""
ATLAS-QE EOS Workflow Runner (Modular Version)

Multi-structure, multi-method equation of state calculation workflow.
Modular implementation using the src/ architecture.

Usage:
    python run_eos_workflow_modular.py examples/gaas_eos_study/gaas_eos_study.yaml
"""

import sys
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import numpy as np
import json
from datetime import datetime

# Import modular components
from src.core.configuration import ConfigurationLoader, ParameterSpaceEnumerator
from src.core.template_engine import InputFileGenerator
from src.core.execution_engine import ExecutionManager, ExecutionStatus
from src.core.task_manager import TaskManager, TaskStatus, TaskPriority
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


@dataclass
class CalculationTask:
    """Individual calculation task definition."""
    structure_name: str
    combination_name: str
    volume_point: float
    volume_index: int
    input_files: Dict[str, Path]
    output_dir: Path
    software: str


@dataclass
class WorkflowResults:
    """Complete workflow execution results."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time: float
    results_by_structure: Dict[str, Dict[str, Any]]


class ModularEOSWorkflowRunner:
    """
    Modular EOS workflow orchestrator using src/ architecture.

    This is a transitional implementation that uses the new modular components
    while maintaining compatibility with the current functionality.
    """

    def __init__(self, config_path: Path):
        """Initialize workflow runner with modular configuration."""
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent

        # Load configuration using modular system
        self.config_loader = ConfigurationLoader(config_path)
        self.config = self.config_loader.load_configuration()

        # Initialize parameter space enumerator
        self.param_enumerator = ParameterSpaceEnumerator(self.config)

        # Initialize input file generator
        self.input_generator = InputFileGenerator(self.config, self.config_dir)

        # Setup results directory
        self.results_dir = Path(self.config.data_paths.get('base_directory', './results'))
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize execution manager (with mock executors for now)
        self.execution_manager = ExecutionManager(
            max_concurrent=4,
            atlas_config={},  # TODO: Load from configuration
            qe_config={}      # TODO: Load from configuration
        )

        # Initialize task manager
        self.task_manager = TaskManager(
            workspace_dir=self.results_dir,
            database_path=self.results_dir / 'workflow_tasks.db'
        )

    def generate_all_calculation_tasks(self) -> List[CalculationTask]:
        """Generate all calculation tasks for the workflow."""
        logger.info("Generating calculation tasks...")

        tasks = []
        parameter_space = self.param_enumerator.discover_parameter_space()

        for structure, combination in parameter_space:
            logger.info(f"Processing {structure.name} with {combination.name}")

            # Create output directory
            output_base = self.results_dir / structure.name / combination.name
            output_base.mkdir(parents=True, exist_ok=True)

            # Generate volume series
            volume_series = self.config.generate_volume_series(structure)

            for vol_idx, volume_scale in enumerate(volume_series):
                # Create volume-specific directory
                vol_dir = output_base / f"{volume_scale:.5f}"
                vol_dir.mkdir(parents=True, exist_ok=True)

                # Generate input files
                input_files = self.input_generator.generate_input_files(
                    structure, combination, volume_scale, vol_dir
                )

                # Create task definition in task manager
                task_def = self.task_manager.create_task(
                    structure_name=structure.name,
                    combination_name=combination.name,
                    software=combination.software,
                    volume_point=volume_scale,
                    input_file=str(input_files.get('input', '')),
                    working_directory=str(vol_dir),
                    priority=TaskPriority.NORMAL
                )

                # Create workflow task
                task = CalculationTask(
                    structure_name=structure.name,
                    combination_name=combination.name,
                    volume_point=volume_scale,
                    volume_index=vol_idx,
                    input_files=input_files,
                    output_dir=vol_dir,
                    software=combination.software
                )
                tasks.append(task)

        logger.info(f"Generated {len(tasks)} calculation tasks")
        return tasks

    def execute_dry_run(self) -> WorkflowResults:
        """Execute dry run to generate input files without calculations."""
        logger.info("Starting dry run execution...")
        start_time = datetime.now()

        # Generate all tasks
        tasks = self.generate_all_calculation_tasks()

        # Count results by structure
        results_by_structure = {}
        for task in tasks:
            if task.structure_name not in results_by_structure:
                results_by_structure[task.structure_name] = {
                    'combinations': {},
                    'total_points': 0
                }

            if task.combination_name not in results_by_structure[task.structure_name]['combinations']:
                results_by_structure[task.structure_name]['combinations'][task.combination_name] = 0

            results_by_structure[task.structure_name]['combinations'][task.combination_name] += 1
            results_by_structure[task.structure_name]['total_points'] += 1

        execution_time = (datetime.now() - start_time).total_seconds()

        results = WorkflowResults(
            total_tasks=len(tasks),
            completed_tasks=len(tasks),  # All tasks completed in dry run
            failed_tasks=0,
            execution_time=execution_time,
            results_by_structure=results_by_structure
        )

        logger.info(f"Dry run completed in {execution_time:.2f} seconds")
        return results

    def execute_calculation(self, task: CalculationTask) -> Dict[str, Any]:
        """
        Execute actual calculation using execution engine.

        Args:
            task: Calculation task to execute

        Returns:
            Dictionary with execution results
        """
        # Find input file based on software
        if task.software == 'atlas':
            input_file = task.output_dir / 'atlas.in'
        elif task.software == 'qe':
            input_file = task.output_dir / 'job.in'
        else:
            input_file = task.output_dir / f'{task.software}.in'

        if not input_file.exists():
            return {
                'task': task,
                'status': 'failed',
                'error': f'Input file not found: {input_file}',
                'volume': task.volume_point,
                'execution_time': 0.0
            }

        # Generate task ID
        task_id = f"{task.structure_name}_{task.combination_name}_{task.volume_point:.5f}"

        logger.info(f"Executing {task.software} calculation: {task_id}")

        try:
            # Update task status to running
            self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)

            # Execute calculation
            result = self.execution_manager.execute_calculation(
                software=task.software,
                input_file=input_file,
                working_dir=task.output_dir,
                task_id=task_id
            )

            # Update task status based on result
            if result.status == ExecutionStatus.COMPLETED:
                task_status = TaskStatus.COMPLETED
            elif result.status == ExecutionStatus.TIMEOUT:
                task_status = TaskStatus.TIMEOUT
            else:
                task_status = TaskStatus.FAILED

            self.task_manager.update_task_status(
                task_id=task_id,
                status=task_status,
                energy=result.energy,
                execution_time=result.execution_time,
                return_code=result.return_code,
                error_message=result.error_message,
                converged=result.convergence_achieved
            )

            # Convert to workflow result format
            return {
                'task': task,
                'status': result.status.value,
                'energy': result.energy,
                'volume': task.volume_point,
                'execution_time': result.execution_time,
                'converged': result.convergence_achieved,
                'return_code': result.return_code,
                'error': result.error_message
            }

        except Exception as e:
            logger.error(f"Calculation execution failed for {task_id}: {e}")

            # Update task status to failed
            self.task_manager.update_task_status(
                task_id=task_id,
                status=TaskStatus.FAILED,
                error_message=str(e)
            )

            return {
                'task': task,
                'status': 'failed',
                'error': str(e),
                'volume': task.volume_point,
                'execution_time': 0.0
            }

    def execute_full_workflow(self) -> WorkflowResults:
        """Execute full workflow with actual calculations."""
        logger.info("Starting full workflow execution...")
        start_time = datetime.now()

        # Generate all tasks
        tasks = self.generate_all_calculation_tasks()

        # Execute calculations
        results = []
        completed_count = 0
        failed_count = 0

        for i, task in enumerate(tasks):
            logger.info(f"Processing task {i+1}/{len(tasks)}: {task.structure_name}/{task.combination_name}/{task.volume_point:.5f}")

            result = self.execute_calculation(task)
            results.append(result)

            if result['status'] in ['completed', 'placeholder_success']:
                completed_count += 1
            else:
                failed_count += 1
                logger.warning(f"Task failed: {result.get('error', 'Unknown error')}")

            # Log progress
            progress = (i + 1) / len(tasks) * 100
            logger.info(f"Progress: {progress:.1f}% ({completed_count} completed, {failed_count} failed)")

        # Count results by structure
        results_by_structure = {}
        for result in results:
            task = result['task']
            struct_name = task.structure_name
            combo_name = task.combination_name

            if struct_name not in results_by_structure:
                results_by_structure[struct_name] = {
                    'combinations': {},
                    'total_points': 0
                }

            if combo_name not in results_by_structure[struct_name]['combinations']:
                results_by_structure[struct_name]['combinations'][combo_name] = 0

            results_by_structure[struct_name]['combinations'][combo_name] += 1
            results_by_structure[struct_name]['total_points'] += 1

        execution_time = (datetime.now() - start_time).total_seconds()

        workflow_results = WorkflowResults(
            total_tasks=len(tasks),
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            execution_time=execution_time,
            results_by_structure=results_by_structure
        )

        logger.info(f"Full workflow completed in {execution_time:.2f} seconds")
        return workflow_results

    def save_workflow_summary(self, results: WorkflowResults) -> None:
        """Save workflow execution summary."""
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system': self.config.system,
            'description': self.config.description,
            'total_tasks': results.total_tasks,
            'completed_tasks': results.completed_tasks,
            'failed_tasks': results.failed_tasks,
            'execution_time': results.execution_time,
            'results_by_structure': results.results_by_structure
        }

        summary_file = self.results_dir / 'workflow_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Workflow summary saved to {summary_file}")

    def print_summary(self, results: WorkflowResults) -> None:
        """Print workflow execution summary."""
        print(f"\n{'='*60}")
        print(f"ATLAS-QE EOS Workflow Summary")
        print(f"{'='*60}")
        print(f"System: {self.config.system}")
        print(f"Description: {self.config.description}")
        print(f"")
        print(f"Execution Results:")
        print(f"  Total tasks: {results.total_tasks}")
        print(f"  Completed: {results.completed_tasks}")
        print(f"  Failed: {results.failed_tasks}")
        print(f"  Execution time: {results.execution_time:.2f} seconds")
        print(f"")
        print(f"Results by Structure:")
        for struct_name, struct_data in results.results_by_structure.items():
            print(f"  {struct_name}:")
            print(f"    Total points: {struct_data['total_points']}")
            print(f"    Combinations:")
            for combo_name, point_count in struct_data['combinations'].items():
                print(f"      {combo_name}: {point_count} points")
        print(f"{'='*60}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS-QE EOS Workflow Runner (Modular Version)"
    )
    parser.add_argument(
        "config_file",
        type=Path,
        help="YAML configuration file"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate input files without executing calculations"
    )
    parser.add_argument(
        "--log-level",
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help="Logging level"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(level=args.log_level)
    logger = logging.getLogger("eos-workflow")

    try:
        # Initialize workflow runner
        runner = ModularEOSWorkflowRunner(args.config_file)

        if args.dry_run:
            # Execute dry run
            results = runner.execute_dry_run()
            logger.info("Dry run mode: input files generated, no calculations executed")
        else:
            # Execute full workflow
            results = runner.execute_full_workflow()
            logger.info("Full workflow execution completed")

        # Save and display results
        runner.save_workflow_summary(results)
        runner.print_summary(results)

        logger.info("Workflow execution completed successfully")

    except Exception as e:
        logger.error(f"Workflow execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()