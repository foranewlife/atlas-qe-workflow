"""
Workflow orchestration engine for ATLAS-QE materials calculations.

This module coordinates the complete materials calculation workflow:
- Parameter space enumeration and discovery
- Intelligent scheduling and resource management
- Cache-aware computation execution
- Results aggregation and analysis
"""

import yaml
import time
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .parameter_set import ParameterSet, ParameterSetManager, ParameterSetStatus
from .cache_system import ParameterSetCache, CacheMatchType
from .pseudopotential import PseudopotentialManager
from ..calculators import ATLASCalculator, QECalculator, CalculationStatus
from ..utils.structure_generation import StructureGenerator

logger = logging.getLogger(__name__)


class WorkflowStage(Enum):
    """Stages of the materials workflow."""
    CONVERGENCE_TESTING = "convergence_testing"
    STRUCTURE_OPTIMIZATION = "structure_optimization"
    EOS_CALCULATION = "eos_calculation"
    ANALYSIS = "analysis"


@dataclass
class WorkflowConfig:
    """Configuration for workflow execution."""
    max_parallel_jobs: int = 4
    timeout_per_calculation: float = 3600  # seconds
    retry_failed_calculations: bool = True
    max_retries: int = 2
    cleanup_workspaces: bool = False
    resource_monitoring: bool = True


class WorkflowEngine:
    """
    Main workflow orchestration engine.

    Coordinates parameter space enumeration, intelligent scheduling,
    cache-aware execution, and results aggregation for complete
    materials calculation workflows.
    """

    def __init__(
        self,
        parameter_manager: Optional[ParameterSetManager] = None,
        cache_system: Optional[ParameterSetCache] = None,
        pseudopotential_manager: Optional[PseudopotentialManager] = None,
        atlas_executable: Optional[str] = None,
        qe_executable: Optional[str] = None,
        base_workspace: Optional[Path] = None
    ):
        """
        Initialize workflow engine.

        Args:
            parameter_manager: Parameter set management system
            cache_system: Intelligent caching system
            pseudopotential_manager: Pseudopotential management
            atlas_executable: Path to ATLAS executable
            qe_executable: Path to QE executable
            base_workspace: Base directory for calculations
        """
        self.parameter_manager = parameter_manager or ParameterSetManager()
        self.cache_system = cache_system or ParameterSetCache(parameter_manager=self.parameter_manager)
        self.pp_manager = pseudopotential_manager or PseudopotentialManager()

        # Calculator setup
        self.calculators = {}
        if atlas_executable:
            self.calculators["atlas"] = ATLASCalculator(executable_path=atlas_executable)
        if qe_executable:
            self.calculators["qe"] = QECalculator(executable_path=qe_executable)

        self.base_workspace = base_workspace or Path("data/calculations")
        self.structure_generator = StructureGenerator()

        logger.info("Initialized WorkflowEngine")

    def discover_parameter_space(self, config: Dict[str, Any]) -> List[ParameterSet]:
        """
        Discover the complete parameter space from configuration.

        Enumerates all possible parameter combinations for the given
        system configuration, including pseudopotential combinations.

        Args:
            config: System configuration dictionary

        Returns:
            List of all parameter sets in the target space
        """
        logger.info(f"Discovering parameter space for system: {config.get('system', 'unknown')}")

        # Validate configuration
        is_valid, errors = self.pp_manager.validate_all_pseudopotential_sets(config)
        if not is_valid:
            raise ValueError(f"Configuration validation failed: {errors}")

        parameter_sets = []
        system = config["system"]
        elements = config["elements"]
        structures = config["structures"]
        volume_range = tuple(config.get("volume_range", [0.8, 1.2]))
        volume_points = config.get("volume_points", 11)

        # Generate parameter sets for each software
        for software in ["atlas", "qe"]:
            if software not in config:
                continue

            software_config = config[software]
            parameter_sets.extend(
                self._enumerate_software_parameter_sets(
                    system, elements, structures, software, software_config,
                    volume_range, volume_points
                )
            )

        logger.info(f"Discovered {len(parameter_sets)} parameter sets")
        return parameter_sets

    def _enumerate_software_parameter_sets(
        self,
        system: str,
        elements: List[str],
        structures: List[str],
        software: str,
        software_config: Dict[str, Any],
        volume_range: Tuple[float, float],
        volume_points: int
    ) -> List[ParameterSet]:
        """Enumerate parameter sets for a specific software."""
        parameter_sets = []

        # Get pseudopotential combinations
        pp_combinations = self.pp_manager.enumerate_pseudopotential_combinations(
            elements, software_config
        )

        # Get software-specific parameter combinations
        software_params = self._get_software_parameter_combinations(software, software_config)

        # Generate all combinations
        for structure in structures:
            for pp_combo in pp_combinations:
                for params in software_params:
                    # Resolve pseudopotential mapping
                    element_pp_mapping = self.pp_manager.resolve_pseudopotential_combination(
                        elements, pp_combo
                    )

                    # Create parameter set
                    param_set = ParameterSet(
                        system=system,
                        software=software,
                        structure=structure,
                        parameters=params,
                        pseudopotential_set=pp_combo["name"],
                        pseudopotential_files=pp_combo["files"],
                        volume_range=volume_range,
                        volume_points=volume_points
                    )

                    # Add element mapping for input generation
                    param_set.parameters["element_pp_mapping"] = element_pp_mapping

                    parameter_sets.append(param_set)

        return parameter_sets

    def _get_software_parameter_combinations(
        self,
        software: str,
        software_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get all parameter combinations for a software."""
        if software == "atlas":
            return self._get_atlas_parameter_combinations(software_config)
        elif software == "qe":
            return self._get_qe_parameter_combinations(software_config)
        else:
            raise ValueError(f"Unknown software: {software}")

    def _get_atlas_parameter_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate ATLAS parameter combinations."""
        combinations = []

        functionals = config.get("functionals", ["kedf701"])
        gaps = config.get("gap", [0.20])

        # Ensure gaps is a list
        if not isinstance(gaps, list):
            gaps = [gaps]

        for functional in functionals:
            for gap in gaps:
                combinations.append({
                    "functional": functional,
                    "gap": gap,
                    "grid_spacing": gap  # Alias for compatibility
                })

        return combinations

    def _get_qe_parameter_combinations(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate QE parameter combinations."""
        combinations = []

        configurations = config.get("configurations", ["default"])
        k_points_list = config.get("k_points", [[8, 8, 8]])

        for configuration in configurations:
            for k_points in k_points_list:
                # Parse configuration (e.g., "ncpp_ecut60", "paw_ecut80")
                params = self._parse_qe_configuration(configuration)
                params["k_points"] = k_points
                params["configuration"] = configuration

                combinations.append(params)

        return combinations

    def _parse_qe_configuration(self, config_name: str) -> Dict[str, Any]:
        """Parse QE configuration string."""
        params = {}

        # Extract pseudopotential type
        if "ncpp" in config_name.lower():
            params["pp_type"] = "ncpp"
        elif "paw" in config_name.lower():
            params["pp_type"] = "paw"
        elif "uspp" in config_name.lower():
            params["pp_type"] = "uspp"

        # Extract energy cutoff
        import re
        ecut_match = re.search(r"ecut(\d+)", config_name.lower())
        if ecut_match:
            params["ecutwfc"] = float(ecut_match.group(1))

        return params

    def execute_incremental_computation(
        self,
        target_parameter_space: List[ParameterSet],
        config: WorkflowConfig = None
    ) -> Dict[str, Any]:
        """
        Execute incremental computation for missing parameter sets.

        Identifies missing calculations from the target space and executes
        them with intelligent caching and parallel processing.

        Args:
            target_parameter_space: Complete target parameter space
            config: Workflow execution configuration

        Returns:
            Dictionary with execution results and statistics
        """
        if config is None:
            config = WorkflowConfig()

        logger.info("Starting incremental computation")

        # Identify missing parameter sets
        missing_param_sets = self.parameter_manager.get_missing_parameter_sets(target_parameter_space)

        if not missing_param_sets:
            logger.info("No missing parameter sets found")
            return {
                "status": "completed",
                "missing_calculations": 0,
                "executed_calculations": 0,
                "cache_hits": 0
            }

        logger.info(f"Found {len(missing_param_sets)} missing parameter sets")

        # Add missing parameter sets to database
        for param_set in missing_param_sets:
            self.parameter_manager.add_parameter_set(param_set)

        # Execute calculations with intelligent scheduling
        results = self._execute_parameter_sets(missing_param_sets, config)

        return results

    def _execute_parameter_sets(
        self,
        parameter_sets: List[ParameterSet],
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Execute a list of parameter sets with parallel processing."""
        results = {
            "total_parameter_sets": len(parameter_sets),
            "completed": 0,
            "failed": 0,
            "cache_hits": 0,
            "execution_times": [],
            "errors": []
        }

        # Check cache for each parameter set first
        cache_hits = []
        calculations_needed = []

        for param_set in parameter_sets:
            cache_match = self.cache_system.check_computation_cache(param_set)

            if cache_match.match_type == CacheMatchType.EXACT:
                results["cache_hits"] += 1
                cache_hits.append(param_set)
                # Update database status
                param_set.status = ParameterSetStatus.COMPLETED
                param_set.results = cache_match.parameter_set.results
                self.parameter_manager.update_parameter_set(param_set)
            else:
                calculations_needed.append(param_set)

        logger.info(f"Cache hits: {len(cache_hits)}, Calculations needed: {len(calculations_needed)}")

        # Execute needed calculations in parallel
        if calculations_needed:
            with concurrent.futures.ThreadPoolExecutor(max_workers=config.max_parallel_jobs) as executor:
                # Submit all calculations
                future_to_param_set = {
                    executor.submit(self._execute_single_calculation, param_set, config): param_set
                    for param_set in calculations_needed
                }

                # Process completed calculations
                for future in concurrent.futures.as_completed(future_to_param_set):
                    param_set = future_to_param_set[future]

                    try:
                        calculation_result = future.result()

                        if calculation_result["status"] == "completed":
                            results["completed"] += 1
                            results["execution_times"].append(calculation_result["execution_time"])
                        else:
                            results["failed"] += 1
                            results["errors"].append({
                                "parameter_set": param_set.fingerprint[:8],
                                "error": calculation_result.get("error", "Unknown error")
                            })

                    except Exception as e:
                        results["failed"] += 1
                        results["errors"].append({
                            "parameter_set": param_set.fingerprint[:8],
                            "error": str(e)
                        })
                        logger.error(f"Error executing parameter set {param_set.fingerprint[:8]}: {e}")

        # Update final statistics
        results["total_cache_hits"] = results["cache_hits"]
        results["total_completed"] = results["completed"] + results["cache_hits"]
        results["success_rate"] = results["total_completed"] / results["total_parameter_sets"]

        logger.info(f"Workflow completed: {results['total_completed']}/{results['total_parameter_sets']} successful")

        return results

    def _execute_single_calculation(
        self,
        param_set: ParameterSet,
        config: WorkflowConfig
    ) -> Dict[str, Any]:
        """Execute a single parameter set calculation."""
        start_time = time.time()

        try:
            # Update status to running
            param_set.status = ParameterSetStatus.RUNNING
            self.parameter_manager.update_parameter_set(param_set)

            # Create workspace
            workspace_name = param_set.get_workspace_name()
            workspace = self.base_workspace / workspace_name
            param_set.workspace_path = str(workspace)

            # Get appropriate calculator
            calculator = self.calculators.get(param_set.software)
            if not calculator:
                raise ValueError(f"Calculator not available for software: {param_set.software}")

            # Generate crystal structure
            structure = self._generate_structure_for_parameter_set(param_set)

            # Generate input files
            calculator.generate_input_files(structure, param_set.parameters, workspace)

            # Execute calculation
            calculation_result = calculator.run_calculation(workspace, config.timeout_per_calculation)

            # Process results
            if calculation_result.status == CalculationStatus.COMPLETED:
                param_set.status = ParameterSetStatus.COMPLETED
                param_set.results = {
                    "energy_data": calculation_result.energy_data,
                    "computational_details": calculation_result.computational_details,
                    "optimized_structure": calculation_result.optimized_structure
                }

                execution_time = time.time() - start_time
                self.parameter_manager.update_parameter_set(param_set)

                logger.info(f"Completed parameter set {param_set.fingerprint[:8]} in {execution_time:.1f}s")

                return {
                    "status": "completed",
                    "execution_time": execution_time,
                    "parameter_set": param_set.fingerprint
                }

            else:
                param_set.status = ParameterSetStatus.FAILED
                self.parameter_manager.update_parameter_set(param_set)

                return {
                    "status": "failed",
                    "error": calculation_result.error_message or "Calculation failed",
                    "parameter_set": param_set.fingerprint
                }

        except Exception as e:
            param_set.status = ParameterSetStatus.FAILED
            self.parameter_manager.update_parameter_set(param_set)

            logger.error(f"Failed to execute parameter set {param_set.fingerprint[:8]}: {e}")

            return {
                "status": "failed",
                "error": str(e),
                "parameter_set": param_set.fingerprint
            }

    def _generate_structure_for_parameter_set(self, param_set: ParameterSet) -> Dict[str, Any]:
        """Generate crystal structure for a parameter set."""
        # Extract elements from parameter set
        if "element_pp_mapping" in param_set.parameters:
            elements = list(param_set.parameters["element_pp_mapping"].keys())
        else:
            # Infer from pseudopotential files
            elements = self._infer_elements_from_pseudopotentials(param_set.pseudopotential_files)

        # Generate structure using the structure generator
        structure = self.structure_generator.generate_structure(
            structure_type=param_set.structure,
            elements=elements,
            lattice_parameter=4.0  # Default, will be varied for EOS
        )

        return structure

    def _infer_elements_from_pseudopotentials(self, pp_files: List[str]) -> List[str]:
        """Infer elements from pseudopotential filenames."""
        elements = []
        for pp_file in pp_files:
            # Simple heuristic: extract element from filename
            # e.g., "Mg_lda.recpot" -> "Mg"
            element = pp_file.split('_')[0].split('.')[0]
            if element not in elements:
                elements.append(element)
        return elements

    def generate_analysis(self, analysis_requests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate analysis based on completed calculations.

        Args:
            analysis_requests: List of analysis configurations

        Returns:
            Dictionary with analysis results
        """
        logger.info(f"Generating {len(analysis_requests)} analysis requests")

        analysis_results = {}

        for i, request in enumerate(analysis_requests):
            try:
                result = self._execute_analysis_request(request)
                analysis_results[f"analysis_{i}"] = result
            except Exception as e:
                logger.error(f"Failed to execute analysis request {i}: {e}")
                analysis_results[f"analysis_{i}"] = {"error": str(e)}

        return analysis_results

    def _execute_analysis_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single analysis request."""
        analysis_type = request.get("type", "comparison")
        filters = request.get("filters", {})

        # Query parameter sets
        parameter_sets = self.parameter_manager.query_parameter_sets(filters)

        if not parameter_sets:
            return {"error": "No parameter sets found matching filters"}

        # Filter to completed calculations only
        completed_sets = [ps for ps in parameter_sets if ps.status == ParameterSetStatus.COMPLETED]

        if not completed_sets:
            return {"error": "No completed calculations found"}

        if analysis_type == "comparison":
            return self._generate_comparison_analysis(completed_sets, request)
        elif analysis_type == "convergence":
            return self._generate_convergence_analysis(completed_sets, request)
        elif analysis_type == "summary":
            return self._generate_summary_analysis(completed_sets, request)
        else:
            return {"error": f"Unknown analysis type: {analysis_type}"}

    def _generate_comparison_analysis(self, parameter_sets: List[ParameterSet], request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comparison analysis between parameter sets."""
        # Placeholder implementation
        return {
            "type": "comparison",
            "parameter_sets": len(parameter_sets),
            "systems": list(set(ps.system for ps in parameter_sets)),
            "software": list(set(ps.software for ps in parameter_sets)),
            "structures": list(set(ps.structure for ps in parameter_sets))
        }

    def _generate_convergence_analysis(self, parameter_sets: List[ParameterSet], request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate convergence analysis."""
        # Placeholder implementation
        return {
            "type": "convergence",
            "parameter_sets_analyzed": len(parameter_sets),
            "convergence_status": "analysis_placeholder"
        }

    def _generate_summary_analysis(self, parameter_sets: List[ParameterSet], request: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary analysis."""
        # Placeholder implementation
        return {
            "type": "summary",
            "total_parameter_sets": len(parameter_sets),
            "completion_rate": sum(1 for ps in parameter_sets if ps.status == ParameterSetStatus.COMPLETED) / len(parameter_sets)
        }

    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status and statistics."""
        stats = self.parameter_manager.get_statistics()
        cache_stats = self.cache_system.get_cache_statistics()

        return {
            "parameter_sets": stats,
            "cache_performance": {
                "hit_rate": cache_stats.hit_rate,
                "exact_hit_rate": cache_stats.exact_hit_rate,
                "total_queries": cache_stats.total_queries,
                "space_used_gb": cache_stats.space_used_gb
            },
            "calculators": list(self.calculators.keys()),
            "base_workspace": str(self.base_workspace)
        }