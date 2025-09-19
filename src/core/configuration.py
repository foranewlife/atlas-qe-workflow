"""
Configuration management for ATLAS-QE workflow system.

This module handles loading, validation, and processing of YAML configuration files
for multi-structure, multi-method EOS studies.
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class StructureConfig:
    """Configuration for a single structure."""
    name: str
    elements: List[str]
    description: str
    volume_range: Tuple[float, float]
    volume_points: int
    file: Optional[str] = None
    structure_type: Optional[str] = None
    lattice_parameter: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StructureConfig':
        """Create StructureConfig from dictionary."""
        return cls(
            name=data['name'],
            elements=data['elements'],
            description=data['description'],
            volume_range=tuple(data['volume_range']),
            volume_points=data['volume_points'],
            file=data.get('file'),
            structure_type=data.get('structure_type'),
            lattice_parameter=data.get('lattice_parameter')
        )


@dataclass
class ParameterCombination:
    """Configuration for a parameter combination."""
    name: str
    software: str
    template_file: str
    applies_to_structures: List[str]
    pseudopotential_set: str
    template_substitutions: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterCombination':
        """Create ParameterCombination from dictionary."""
        return cls(
            name=data['name'],
            software=data['software'],
            template_file=data['template_file'],
            applies_to_structures=data['applies_to_structures'],
            pseudopotential_set=data['pseudopotential_set'],
            template_substitutions=data.get('template_substitutions', {})
        )


@dataclass
class WorkflowConfiguration:
    """Complete workflow configuration."""
    system: str
    description: str
    pseudopotential_sets: Dict[str, Dict[str, str]]
    structures: List[StructureConfig]
    parameter_combinations: List[ParameterCombination]
    data_paths: Dict[str, str]

    def get_applicable_combinations(self, structure_name: str) -> List[ParameterCombination]:
        """Get parameter combinations applicable to a structure."""
        return [
            combo for combo in self.parameter_combinations
            if structure_name in combo.applies_to_structures
        ]

    def generate_volume_series(self, structure: StructureConfig) -> List[float]:
        """Generate volume scale factors for a structure."""
        return np.linspace(
            structure.volume_range[0],
            structure.volume_range[1],
            structure.volume_points
        ).tolist()


class ConfigurationLoader:
    """YAML configuration file loader and validator."""

    def __init__(self, config_path: Path):
        """Initialize configuration loader."""
        self.config_path = Path(config_path)
        self.config_dir = self.config_path.parent

    def load_configuration(self) -> WorkflowConfiguration:
        """Load and validate YAML configuration."""
        logger.info(f"Loading configuration from {self.config_path}")

        with open(self.config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        # Validate required sections
        self._validate_configuration(raw_config)

        # Parse structures
        structures = [
            StructureConfig.from_dict(struct_data)
            for struct_data in raw_config['structures']
        ]

        # Parse parameter combinations
        combinations = [
            ParameterCombination.from_dict(combo_data)
            for combo_data in raw_config['parameter_combinations']
        ]

        # Create configuration object
        config = WorkflowConfiguration(
            system=raw_config['system'],
            description=raw_config['description'],
            pseudopotential_sets=raw_config['pseudopotential_sets'],
            structures=structures,
            parameter_combinations=combinations,
            data_paths=raw_config.get('data_paths', {})
        )

        # Validate cross-references
        self._validate_cross_references(config)

        logger.info(f"Configuration loaded: {config.system} - {config.description}")
        return config

    def _validate_configuration(self, config: Dict[str, Any]) -> None:
        """Validate required configuration sections."""
        required_sections = [
            'system',
            'description',
            'pseudopotential_sets',
            'structures',
            'parameter_combinations'
        ]

        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")

        # Validate structures
        for i, structure in enumerate(config['structures']):
            required_struct_fields = ['name', 'elements', 'description', 'volume_range', 'volume_points']
            for field in required_struct_fields:
                if field not in structure:
                    raise ValueError(f"Structure {i}: missing required field '{field}'")

        # Validate parameter combinations
        for i, combo in enumerate(config['parameter_combinations']):
            required_combo_fields = ['name', 'software', 'template_file', 'applies_to_structures', 'pseudopotential_set']
            for field in required_combo_fields:
                if field not in combo:
                    raise ValueError(f"Parameter combination {i}: missing required field '{field}'")

    def _validate_cross_references(self, config: WorkflowConfiguration) -> None:
        """Validate cross-references between configuration sections."""
        # Collect all structure names
        structure_names = {struct.name for struct in config.structures}

        # Validate applies_to_structures references
        for combo in config.parameter_combinations:
            for struct_name in combo.applies_to_structures:
                if struct_name not in structure_names:
                    raise ValueError(
                        f"Parameter combination '{combo.name}' references unknown structure '{struct_name}'"
                    )

        # Validate pseudopotential set references
        pp_set_names = set(config.pseudopotential_sets.keys())
        for combo in config.parameter_combinations:
            if combo.pseudopotential_set not in pp_set_names:
                raise ValueError(
                    f"Parameter combination '{combo.name}' references unknown pseudopotential set '{combo.pseudopotential_set}'"
                )

        # Validate pseudopotential sets have all required elements
        for combo in config.parameter_combinations:
            pp_set = config.pseudopotential_sets[combo.pseudopotential_set]
            for struct_name in combo.applies_to_structures:
                structure = next(s for s in config.structures if s.name == struct_name)
                for element in structure.elements:
                    if element not in pp_set:
                        raise ValueError(
                            f"Pseudopotential set '{combo.pseudopotential_set}' missing element '{element}' "
                            f"required for structure '{struct_name}'"
                        )

        logger.info("Configuration validation completed successfully")

    def resolve_template_path(self, template_file: str) -> Path:
        """Resolve template file path relative to configuration directory."""
        template_path = Path(template_file)
        if not template_path.is_absolute():
            template_path = self.config_dir / template_path

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        return template_path

    def resolve_structure_path(self, structure_file: str) -> Path:
        """Resolve structure file path relative to configuration directory."""
        structure_path = Path(structure_file)
        if not structure_path.is_absolute():
            structure_path = self.config_dir / structure_path

        if not structure_path.exists():
            raise FileNotFoundError(f"Structure file not found: {structure_path}")

        return structure_path


class ParameterSpaceEnumerator:
    """Enumerate parameter space for workflow execution."""

    def __init__(self, config: WorkflowConfiguration):
        """Initialize parameter space enumerator."""
        self.config = config

    def discover_parameter_space(self) -> List[Tuple[StructureConfig, ParameterCombination]]:
        """
        Discover all structure-combination pairs for execution.

        Returns:
            List of (structure, combination) tuples
        """
        parameter_space = []

        for structure in self.config.structures:
            applicable_combinations = self.config.get_applicable_combinations(structure.name)

            logger.info(f"Structure '{structure.name}': {len(applicable_combinations)} applicable combinations")

            for combination in applicable_combinations:
                parameter_space.append((structure, combination))

        logger.info(f"Total parameter space: {len(parameter_space)} structure-combination pairs")
        return parameter_space

    def count_total_calculations(self) -> int:
        """Count total number of calculations that will be performed."""
        total = 0
        parameter_space = self.discover_parameter_space()

        for structure, combination in parameter_space:
            total += structure.volume_points

        return total

    def get_calculation_summary(self) -> Dict[str, Any]:
        """Get summary of calculations to be performed."""
        parameter_space = self.discover_parameter_space()
        total_calculations = self.count_total_calculations()

        structure_summary = {}
        for structure, combination in parameter_space:
            if structure.name not in structure_summary:
                structure_summary[structure.name] = {
                    'combinations': [],
                    'volume_points': structure.volume_points,
                    'volume_range': structure.volume_range
                }
            structure_summary[structure.name]['combinations'].append(combination.name)

        return {
            'total_calculations': total_calculations,
            'total_combinations': len(parameter_space),
            'total_structures': len(self.config.structures),
            'structures': structure_summary
        }