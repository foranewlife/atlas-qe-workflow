"""
EOS (Equation of State) workflow orchestration engine.

This module provides the main workflow orchestration for multi-structure,
multi-method EOS studies with parallel execution and results management.
"""

import json
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from .configuration import (
    WorkflowConfiguration, ConfigurationLoader, ParameterSpaceEnumerator,
    StructureConfig, ParameterCombination
)
from .template_engine import InputFileGenerator

logger = logging.getLogger(__name__)


@dataclass
class CalculationTask:
    """Individual calculation task definition."""
    structure: StructureConfig
    combination: ParameterCombination
    volume_point: float
    volume_index: int
    output_dir: Path
    input_files: Dict[str, Path]

    @property
    def task_id(self) -> str:
        """Generate unique task identifier."""
        return f"{self.structure.name}_{self.combination.name}_{self.volume_index:02d}"


@dataclass
class CalculationResult:
    """Result from a single calculation."""
    task_id: str
    structure_name: str
    combination_name: str
    volume_scale: float
    volume_index: int
    status: str
    energy: Optional[float] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    output_files: Optional[Dict[str, Path]] = None


@dataclass
class WorkflowResults:
    """Complete workflow execution results."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time: float
    results_by_structure: Dict[str, Dict[str, List[CalculationResult]]]
    summary_file: Path


class CalculationExecutor:
    """Executes individual calculation tasks."""

    def __init__(self, dry_run: bool = False):
        """Initialize calculation executor."""
        self.dry_run = dry_run

    def execute_task(self, task: CalculationTask) -> CalculationResult:
        """
        Execute a single calculation task.

        Args:
            task: Calculation task to execute

        Returns:
            Calculation result
        """
        start_time = datetime.now()

        logger.info(
            f"Executing {task.structure.name}/{task.combination.name} "
            f"volume={task.volume_point:.5f}"
        )

        try:
            if self.dry_run:
                # Dry run - just validate inputs exist
                result = self._dry_run_execution(task)
            else:
                # Real execution
                result = self._real_execution(task)

            execution_time = (datetime.now() - start_time).total_seconds()
            result.execution_time = execution_time

            return result

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(f"Task {task.task_id} failed: {e}")

            return CalculationResult(
                task_id=task.task_id,
                structure_name=task.structure.name,
                combination_name=task.combination.name,
                volume_scale=task.volume_point,
                volume_index=task.volume_index,
                status='failed',
                error_message=str(e),
                execution_time=execution_time
            )

    def _dry_run_execution(self, task: CalculationTask) -> CalculationResult:
        """Perform dry run execution (validation only)."""
        # Check that input files exist
        for file_type, file_path in task.input_files.items():
            if not file_path.exists():
                raise FileNotFoundError(f"Input file not found: {file_path}")

        return CalculationResult(
            task_id=task.task_id,
            structure_name=task.structure.name,
            combination_name=task.combination.name,
            volume_scale=task.volume_point,
            volume_index=task.volume_index,
            status='dry_run_success',
            energy=None
        )

    def _real_execution(self, task: CalculationTask) -> CalculationResult:
        """Perform real calculation execution."""
        # Mock implementation - in real version, this would:
        # 1. Change to task.output_dir
        # 2. Execute appropriate software (ATLAS or QE)
        # 3. Parse output for energy and properties
        # 4. Return results

        # Simulate calculation time
        import time
        time.sleep(0.1)

        # Mock energy calculation (parabolic EOS around equilibrium)
        equilibrium_volume = 1.0
        bulk_modulus = 200.0  # GPa
        energy_0 = -10.0

        volume_deviation = task.volume_point - equilibrium_volume
        mock_energy = energy_0 + 0.5 * bulk_modulus * volume_deviation**2

        return CalculationResult(
            task_id=task.task_id,
            structure_name=task.structure.name,
            combination_name=task.combination.name,
            volume_scale=task.volume_point,
            volume_index=task.volume_index,
            status='completed',
            energy=mock_energy
        )


class ResultsManager:
    """Manages workflow results and output files."""

    def __init__(self, results_dir: Path):
        """Initialize results manager."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_results(
        self,
        results: List[CalculationResult],
        config: WorkflowConfiguration
    ) -> Path:
        """
        Save calculation results to files.

        Args:
            results: List of calculation results
            config: Workflow configuration

        Returns:
            Path to workflow summary file
        """
        # Group results by structure and combination
        grouped_results = self._group_results(results)

        # Save individual EOS data files
        for structure_name, combinations in grouped_results.items():
            for combination_name, combination_results in combinations.items():
                self._save_combination_results(
                    structure_name, combination_name, combination_results
                )

        # Save workflow summary
        summary_file = self._save_workflow_summary(results, config)

        return summary_file

    def _group_results(
        self,
        results: List[CalculationResult]
    ) -> Dict[str, Dict[str, List[CalculationResult]]]:
        """Group results by structure and combination."""
        grouped = {}

        for result in results:
            if result.status in ['completed', 'dry_run_success']:
                structure = result.structure_name
                combination = result.combination_name

                if structure not in grouped:
                    grouped[structure] = {}
                if combination not in grouped[structure]:
                    grouped[structure][combination] = []

                grouped[structure][combination].append(result)

        return grouped

    def _save_combination_results(
        self,
        structure_name: str,
        combination_name: str,
        results: List[CalculationResult]
    ) -> None:
        """Save results for a specific structure-combination pair."""
        results_dir = self.results_dir / structure_name / combination_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Sort by volume
        results.sort(key=lambda x: x.volume_scale)

        # Save detailed results
        detailed_results = []
        for result in results:
            result_dict = {
                'task_id': result.task_id,
                'volume_scale': result.volume_scale,
                'volume_index': result.volume_index,
                'status': result.status,
                'execution_time': result.execution_time
            }
            if result.energy is not None:
                result_dict['energy'] = result.energy
            if result.error_message:
                result_dict['error_message'] = result.error_message

            detailed_results.append(result_dict)

        results_file = results_dir / "calculation_results.json"
        with open(results_file, 'w') as f:
            json.dump(detailed_results, f, indent=2)

        # Save EOS data for plotting (only if we have energies)
        energy_results = [r for r in results if r.energy is not None]
        if energy_results:
            eos_data = {
                'volumes': [r.volume_scale for r in energy_results],
                'energies': [r.energy for r in energy_results],
                'structure': structure_name,
                'combination': combination_name,
                'software': results[0].combination_name.split('_')[0] if results else 'unknown'
            }

            eos_file = results_dir / "eos_data.json"
            with open(eos_file, 'w') as f:
                json.dump(eos_data, f, indent=2)

            logger.info(f"Saved EOS data: {eos_file}")

    def _save_workflow_summary(
        self,
        results: List[CalculationResult],
        config: WorkflowConfiguration
    ) -> Path:
        """Save complete workflow summary."""
        successful_results = [r for r in results if r.status in ['completed', 'dry_run_success']]
        failed_results = [r for r in results if r.status == 'failed']

        # Group successful results by structure
        grouped_results = self._group_results(successful_results)

        summary = {
            'config_system': config.system,
            'config_description': config.description,
            'execution_timestamp': datetime.now().isoformat(),
            'total_calculations': len(results),
            'successful_calculations': len(successful_results),
            'failed_calculations': len(failed_results),
            'structures': list(grouped_results.keys()),
            'results_by_structure': {
                structure: list(combinations.keys())
                for structure, combinations in grouped_results.items()
            }
        }

        summary_file = self.results_dir / "workflow_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"Workflow summary saved: {summary_file}")
        return summary_file


class EOSWorkflowRunner:
    """
    Main EOS workflow orchestrator.

    Coordinates configuration loading, task generation, parallel execution,
    and results management for complete EOS studies.
    """

    def __init__(self, config_path: Path, results_dir: Optional[Path] = None):
        """Initialize EOS workflow runner."""
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent

        # Load configuration
        loader = ConfigurationLoader(self.config_path)
        self.config = loader.load_configuration()

        # Setup results directory
        if results_dir is None:
            results_dir = Path(self.config.data_paths.get('base_directory', './results'))
        self.results_dir = Path(results_dir)

        # Initialize components
        self.input_generator = InputFileGenerator(self.config, self.config_dir)
        self.results_manager = ResultsManager(self.results_dir)

    def run_workflow(
        self,
        max_workers: int = 4,
        dry_run: bool = False
    ) -> WorkflowResults:
        """
        Execute complete EOS workflow.

        Args:
            max_workers: Maximum number of parallel workers
            dry_run: If True, only generate inputs without executing

        Returns:
            Workflow execution results
        """
        start_time = datetime.now()
        logger.info("Starting EOS workflow execution")

        # Discover parameter space
        enumerator = ParameterSpaceEnumerator(self.config)
        parameter_space = enumerator.discover_parameter_space()

        # Generate calculation tasks
        tasks = self._generate_calculation_tasks(parameter_space)
        logger.info(f"Generated {len(tasks)} calculation tasks")

        # Execute calculations
        executor = CalculationExecutor(dry_run=dry_run)
        results = self._run_parallel_calculations(tasks, executor, max_workers)

        # Save results
        summary_file = self.results_manager.save_results(results, self.config)

        # Generate workflow results
        execution_time = (datetime.now() - start_time).total_seconds()
        completed_count = len([r for r in results if r.status in ['completed', 'dry_run_success']])
        failed_count = len([r for r in results if r.status == 'failed'])

        grouped_results = self.results_manager._group_results(results)

        workflow_results = WorkflowResults(
            total_tasks=len(tasks),
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            execution_time=execution_time,
            results_by_structure=grouped_results,
            summary_file=summary_file
        )

        logger.info(f"Workflow completed: {completed_count}/{len(tasks)} successful in {execution_time:.1f}s")
        return workflow_results

    def _generate_calculation_tasks(
        self,
        parameter_space: List[tuple]
    ) -> List[CalculationTask]:
        """Generate calculation tasks from parameter space."""
        tasks = []

        for structure, combination in parameter_space:
            volume_scales = self.config.generate_volume_series(structure)

            for i, volume_scale in enumerate(volume_scales):
                # Create output directory
                output_dir = (self.results_dir / structure.name /
                             combination.name / f"{volume_scale:.5f}")

                # Generate input files
                input_files = self.input_generator.generate_input_files(
                    structure, combination, volume_scale, output_dir
                )

                task = CalculationTask(
                    structure=structure,
                    combination=combination,
                    volume_point=volume_scale,
                    volume_index=i,
                    output_dir=output_dir,
                    input_files=input_files
                )

                tasks.append(task)

        return tasks

    def _run_parallel_calculations(
        self,
        tasks: List[CalculationTask],
        executor: CalculationExecutor,
        max_workers: int
    ) -> List[CalculationResult]:
        """Execute calculations in parallel."""
        logger.info(f"Starting {len(tasks)} calculations with {max_workers} workers")

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as thread_executor:
            future_to_task = {
                thread_executor.submit(executor.execute_task, task): task
                for task in tasks
            }

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed: {result.task_id}")
                except Exception as exc:
                    logger.error(f"Task execution failed: {exc}")
                    # Create failed result
                    failed_result = CalculationResult(
                        task_id=task.task_id,
                        structure_name=task.structure.name,
                        combination_name=task.combination.name,
                        volume_scale=task.volume_point,
                        volume_index=task.volume_index,
                        status='failed',
                        error_message=str(exc)
                    )
                    results.append(failed_result)

        return results

    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of calculations to be performed."""
        enumerator = ParameterSpaceEnumerator(self.config)
        return enumerator.get_calculation_summary()