#!/usr/bin/env python3
"""
ATLAS-QE EOS Workflow Runner

Multi-structure, multi-method equation of state calculation workflow.
Orchestrates parameter combinations, template processing, and parallel execution.

Usage:
    python run_eos_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml
"""

import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import concurrent.futures
from datetime import datetime
import json
import re

# Simple logging setup
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger("eos-workflow")

logger = logging.getLogger(__name__)


class SimpleExampleEngine:
    """Simple template processing engine."""

    def scale_structure_volume(self, poscar_content: str, volume_scale_factor: float) -> str:
        """Scale structure volume in POSCAR content."""
        lines = poscar_content.strip().split('\n')
        if len(lines) < 5:
            raise ValueError("Invalid POSCAR format")

        # Volume scales as cube root of linear scaling
        linear_scale = volume_scale_factor ** (1.0/3.0)

        # Scale lattice constant (line 1, 0-indexed)
        scale_line = lines[1].strip()
        try:
            current_scale = float(scale_line)
            new_scale = current_scale * linear_scale
            lines[1] = f"   {new_scale:.8f}"
        except ValueError:
            logger.warning("Could not parse lattice scale factor")

        return '\n'.join(lines)


@dataclass
class CalculationTask:
    """Individual calculation task definition."""
    structure_name: str
    combination_name: str
    volume_point: float
    volume_index: int
    input_file_path: Path
    output_dir: Path
    software: str
    template_substitutions: Dict[str, Any]


@dataclass
class WorkflowResults:
    """Complete workflow execution results."""
    total_tasks: int
    completed_tasks: int
    failed_tasks: int
    execution_time: float
    results_by_structure: Dict[str, Dict[str, Any]]


class EOSWorkflowRunner:
    """
    Main EOS workflow orchestrator.

    Handles configuration parsing, parameter enumeration,
    template processing, and execution coordination.
    """

    def __init__(self, config_path: Path):
        """Initialize workflow runner with configuration."""
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent
        self.config = self._load_configuration()
        self.engine = SimpleExampleEngine()
        self.results_dir = Path(self.config.get('data_paths', {}).get('base_directory', './results'))

    def _load_configuration(self) -> Dict[str, Any]:
        """Load and validate YAML configuration."""
        logger.info(f"Loading configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required sections
        required_sections = ['pseudopotential_sets', 'structures', 'parameter_combinations']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        logger.info(f"Configuration loaded: {config['system']} - {config['description']}")
        return config

    def discover_parameter_space(self) -> List[Tuple[Dict[str, Any], Dict[str, Any]]]:
        """
        Discover all structure-combination pairs based on applies_to_structures.

        Returns:
            List of (structure, combination) tuples for execution
        """
        parameter_space = []

        for structure in self.config['structures']:
            structure_name = structure['name']

            # Find applicable combinations for this structure
            applicable_combinations = []
            for combination in self.config['parameter_combinations']:
                if structure_name in combination.get('applies_to_structures', []):
                    applicable_combinations.append(combination)

            logger.info(f"Structure '{structure_name}': {len(applicable_combinations)} applicable combinations")

            for combination in applicable_combinations:
                parameter_space.append((structure, combination))

        logger.info(f"Total parameter space: {len(parameter_space)} structure-combination pairs")
        return parameter_space

    def generate_volume_series(self, structure: Dict[str, Any]) -> List[float]:
        """Generate volume scale factors for a structure."""
        volume_range = structure.get('volume_range', [0.8, 1.2])
        volume_points = structure.get('volume_points', 11)

        return np.linspace(volume_range[0], volume_range[1], volume_points).tolist()

    def setup_calculation_directories(self, structure: Dict[str, Any], combination: Dict[str, Any]) -> Path:
        """Create directory structure for calculations."""
        structure_name = structure['name']
        combination_name = combination['name']

        calc_dir = self.results_dir / structure_name / combination_name
        calc_dir.mkdir(parents=True, exist_ok=True)

        return calc_dir

    def process_template_file(self, template_path: Path, substitutions: Dict[str, Any]) -> str:
        """Process template file with substitutions."""
        if not template_path.exists():
            # Try relative to config directory
            template_path = self.config_dir / template_path

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, 'r') as f:
            template_content = f.read()

        # Perform substitutions
        for key, value in substitutions.items():
            placeholder = f"{{{key}}}"
            template_content = template_content.replace(placeholder, str(value))

        return template_content

    def generate_structure_content(self, structure: Dict[str, Any], volume_scale: float) -> str:
        """Generate or scale structure content (POSCAR)."""
        if 'file' in structure:
            # Load structure from file
            structure_file = self.config_dir / structure['file']
            if not structure_file.exists():
                raise FileNotFoundError(f"Structure file not found: {structure_file}")

            with open(structure_file, 'r') as f:
                poscar_content = f.read()

            # Scale volume
            scaled_poscar = self.engine.scale_structure_volume(poscar_content, volume_scale)
            return scaled_poscar

        elif structure.get('structure_type') == 'fcc':
            # Generate FCC structure
            lattice_param = structure.get('lattice_parameter', 4.0)
            elements = structure['elements']

            # Simple FCC POSCAR generation
            scaled_lattice = lattice_param * (volume_scale ** (1.0/3.0))

            poscar_lines = [
                f"FCC {elements[0]} - volume scale {volume_scale:.5f}",
                f"   {scaled_lattice:.8f}",
                "   0.000000   0.500000   0.500000",
                "   0.500000   0.000000   0.500000",
                "   0.500000   0.500000   0.000000",
                " ".join(elements),
                "   1",
                "Direct",
                "   0.000000   0.000000   0.000000"
            ]
            return "\n".join(poscar_lines)

        else:
            raise ValueError(f"Unknown structure specification: {structure}")

    def generate_pseudopotential_lines(self, structure: Dict[str, Any], combination: Dict[str, Any]) -> Dict[str, str]:
        """Generate software-specific pseudopotential lines."""
        elements = structure['elements']
        pp_set_name = combination['pseudopotential_set']
        pp_set = self.config['pseudopotential_sets'][pp_set_name]

        software = combination['software']

        if software == 'atlas':
            # Generate ATLAS format
            elements_line = f"ELEMENTS = {' '.join(elements)}"

            pp_files = [pp_set[elem] for elem in elements]
            ppfile_line = f"ppfile = {' '.join(pp_files)}"

            return {
                'ELEMENTS': elements_line,
                'PPFILE': ppfile_line
            }

        elif software == 'qe':
            # Generate QE ATOMIC_SPECIES
            species_lines = []
            for elem in elements:
                pp_file = pp_set[elem]
                # Approximate atomic masses (should be looked up properly)
                masses = {'Ga': 69.723, 'As': 74.922, 'Mg': 24.305}
                mass = masses.get(elem, 1.0)
                species_lines.append(f"   {elem}  {mass:.3f}  {pp_file}")

            atomic_species = "ATOMIC_SPECIES\n" + "\n".join(species_lines)

            return {
                'ATOMIC_SPECIES': atomic_species
            }

        return {}

    def create_calculation_task(
        self,
        structure: Dict[str, Any],
        combination: Dict[str, Any],
        volume_scale: float,
        volume_index: int
    ) -> CalculationTask:
        """Create a single calculation task."""

        # Setup directories
        calc_base_dir = self.setup_calculation_directories(structure, combination)
        volume_dir = calc_base_dir / f"{volume_scale:.5f}"
        volume_dir.mkdir(exist_ok=True)

        # Generate structure file
        poscar_content = self.generate_structure_content(structure, volume_scale)
        poscar_file = volume_dir / "POSCAR"
        with open(poscar_file, 'w') as f:
            f.write(poscar_content)

        # Process template
        template_file = combination['template_file']
        template_substitutions = combination.get('template_substitutions', {})

        processed_template = self.process_template_file(
            self.config_dir / template_file,
            template_substitutions
        )

        # Add pseudopotential lines for ATLAS
        if combination['software'] == 'atlas':
            pp_lines = self.generate_pseudopotential_lines(structure, combination)

            # Insert after first line (comment)
            template_lines = processed_template.split('\n')
            insert_pos = 1
            for key, line in pp_lines.items():
                template_lines.insert(insert_pos, line)
                insert_pos += 1

            processed_template = '\n'.join(template_lines)

        # Write input file
        software = combination['software']
        input_filename = 'atlas.in' if software == 'atlas' else 'job.in'
        input_file = volume_dir / input_filename

        with open(input_file, 'w') as f:
            f.write(processed_template)

        return CalculationTask(
            structure_name=structure['name'],
            combination_name=combination['name'],
            volume_point=volume_scale,
            volume_index=volume_index,
            input_file_path=input_file,
            output_dir=volume_dir,
            software=software,
            template_substitutions=template_substitutions
        )

    def execute_calculation(self, task: CalculationTask) -> Dict[str, Any]:
        """Execute a single calculation task (placeholder)."""
        logger.info(f"Executing {task.structure_name}/{task.combination_name} volume={task.volume_point:.5f}")

        # Placeholder execution - in real implementation, this would:
        # 1. Change to task.output_dir
        # 2. Execute appropriate software (ATLAS or QE)
        # 3. Parse output for energy/properties
        # 4. Return results

        # Simulate some calculation time
        import time
        time.sleep(0.1)

        # Mock energy calculation (parabolic EOS)
        equilibrium_volume = 1.0
        bulk_modulus = 200.0  # GPa
        energy_0 = -10.0

        volume_deviation = task.volume_point - equilibrium_volume
        mock_energy = energy_0 + 0.5 * bulk_modulus * volume_deviation**2

        return {
            'task_id': f"{task.structure_name}_{task.combination_name}_{task.volume_index:02d}",
            'structure': task.structure_name,
            'combination': task.combination_name,
            'volume_scale': task.volume_point,
            'energy': mock_energy,
            'status': 'completed',
            'software': task.software,
            'input_file': str(task.input_file_path),
            'output_dir': str(task.output_dir)
        }

    def run_parallel_calculations(self, tasks: List[CalculationTask], max_workers: int = 4) -> List[Dict[str, Any]]:
        """Execute calculations in parallel."""
        logger.info(f"Starting {len(tasks)} calculations with {max_workers} workers")

        results = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {executor.submit(self.execute_calculation, task): task for task in tasks}

            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                    logger.debug(f"Completed: {result['task_id']}")
                except Exception as exc:
                    logger.error(f"Task {task.structure_name}/{task.combination_name} failed: {exc}")
                    results.append({
                        'task_id': f"{task.structure_name}_{task.combination_name}_{task.volume_index:02d}",
                        'status': 'failed',
                        'error': str(exc)
                    })

        return results

    def save_results(self, results: List[Dict[str, Any]]) -> None:
        """Save calculation results to files."""
        # Group results by structure and combination
        grouped_results = {}

        for result in results:
            if result.get('status') == 'completed':
                structure = result['structure']
                combination = result['combination']

                if structure not in grouped_results:
                    grouped_results[structure] = {}
                if combination not in grouped_results[structure]:
                    grouped_results[structure][combination] = []

                grouped_results[structure][combination].append(result)

        # Save individual EOS data
        for structure_name, combinations in grouped_results.items():
            for combination_name, combination_results in combinations.items():
                results_dir = self.results_dir / structure_name / combination_name

                # Sort by volume
                combination_results.sort(key=lambda x: x['volume_scale'])

                # Save detailed results
                results_file = results_dir / "calculation_results.json"
                with open(results_file, 'w') as f:
                    json.dump(combination_results, f, indent=2)

                # Save EOS data for plotting
                eos_data = {
                    'volumes': [r['volume_scale'] for r in combination_results],
                    'energies': [r['energy'] for r in combination_results],
                    'structure': structure_name,
                    'combination': combination_name,
                    'software': combination_results[0]['software']
                }

                eos_file = results_dir / "eos_data.json"
                with open(eos_file, 'w') as f:
                    json.dump(eos_data, f, indent=2)

                logger.info(f"Saved EOS data: {eos_file}")

        # Save complete workflow summary
        summary_file = self.results_dir / "workflow_summary.json"
        workflow_summary = {
            'config_file': str(self.config_path),
            'execution_timestamp': datetime.now().isoformat(),
            'total_calculations': len(results),
            'successful_calculations': len([r for r in results if r.get('status') == 'completed']),
            'failed_calculations': len([r for r in results if r.get('status') == 'failed']),
            'structures': list(grouped_results.keys()),
            'results_by_structure': {
                structure: list(combinations.keys())
                for structure, combinations in grouped_results.items()
            }
        }

        with open(summary_file, 'w') as f:
            json.dump(workflow_summary, f, indent=2)

        logger.info(f"Workflow summary saved: {summary_file}")

    def run_workflow(self, max_workers: int = 4) -> WorkflowResults:
        """Execute complete EOS workflow."""
        start_time = datetime.now()
        logger.info("Starting EOS workflow execution")

        # Discover parameter space
        parameter_space = self.discover_parameter_space()

        # Generate all calculation tasks
        all_tasks = []

        for structure, combination in parameter_space:
            volume_scales = self.generate_volume_series(structure)

            for i, volume_scale in enumerate(volume_scales):
                task = self.create_calculation_task(structure, combination, volume_scale, i)
                all_tasks.append(task)

        logger.info(f"Generated {len(all_tasks)} calculation tasks")

        # Execute calculations
        results = self.run_parallel_calculations(all_tasks, max_workers)

        # Save results
        self.save_results(results)

        # Generate summary
        execution_time = (datetime.now() - start_time).total_seconds()
        completed_count = len([r for r in results if r.get('status') == 'completed'])
        failed_count = len([r for r in results if r.get('status') == 'failed'])

        workflow_results = WorkflowResults(
            total_tasks=len(all_tasks),
            completed_tasks=completed_count,
            failed_tasks=failed_count,
            execution_time=execution_time,
            results_by_structure={}
        )

        logger.info(f"Workflow completed: {completed_count}/{len(all_tasks)} successful in {execution_time:.1f}s")
        return workflow_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="ATLAS-QE EOS Workflow Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python run_eos_workflow.py examples/gaas_eos_study/gaas_eos_study.yaml
    python run_eos_workflow.py config.yaml --max-workers 8 --verbose
        """
    )

    parser.add_argument(
        'config_file',
        type=Path,
        help='YAML configuration file for EOS study'
    )

    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Maximum number of parallel workers (default: 4)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Generate inputs but do not execute calculations'
    )

    args = parser.parse_args()

    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level=log_level)

    try:
        # Initialize and run workflow
        runner = EOSWorkflowRunner(args.config_file)

        if args.dry_run:
            logger.info("Dry run mode: generating inputs only")
            parameter_space = runner.discover_parameter_space()

            total_tasks = 0
            for structure, combination in parameter_space:
                volume_scales = runner.generate_volume_series(structure)
                total_tasks += len(volume_scales)

            logger.info(f"Would generate {total_tasks} calculation tasks")

        else:
            results = runner.run_workflow(max_workers=args.max_workers)

            print(f"\n🎉 EOS Workflow Completed!")
            print(f"Total calculations: {results.total_tasks}")
            print(f"Successful: {results.completed_tasks}")
            print(f"Failed: {results.failed_tasks}")
            print(f"Execution time: {results.execution_time:.1f} seconds")
            print(f"Results saved in: {runner.results_dir}")

    except Exception as e:
        logger.error(f"Workflow failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()