"""
Configuration management for ATLAS-QE workflow system.

This module handles loading, validation, and processing of YAML configuration files
for multi-structure, multi-method EOS studies, resource management, and distributed execution.
"""

import yaml
import logging
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional, Union
from dataclasses import dataclass, field
import numpy as np

from enum import Enum

# Minimal internal definitions to avoid depending on legacy resource_manager
@dataclass
class ResourceCapability:
    cores: int
    memory_gb: int | None = None
    supports_mpi: bool = True
    max_concurrent_jobs: int = 1
    preferred_software: List[str] = field(default_factory=list)


class ResourceType(Enum):
    LOCAL = "local"
    REMOTE = "remote"

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
    """Configuration for a parameter combination.

    Simplified naming: use 'template' instead of 'template_file'.
    Backward compatible: loader accepts both keys.
    """
    name: str
    software: str
    template: str
    applies_to_structures: List[str]
    pseudopotential_set: str
    template_substitutions: Dict[str, Any]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterCombination':
        """Create ParameterCombination from dictionary (accept legacy keys)."""
        tpl = data.get('template') or data.get('template_file')
        return cls(
            name=data['name'],
            software=data['software'],
            template=tpl,
            applies_to_structures=data['applies_to_structures'],
            pseudopotential_set=data['pseudopotential_set'],
            template_substitutions=data.get('template_substitutions', {})
        )

    @property
    def template_file(self) -> str:
        """Backward compatibility alias of 'template'."""
        return self.template


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

        # Validate parameter combinations (allow 'template' or legacy 'template_file')
        for i, combo in enumerate(config['parameter_combinations']):
            required_combo_fields = ['name', 'software', 'applies_to_structures', 'pseudopotential_set']
            for field in required_combo_fields:
                if field not in combo:
                    raise ValueError(f"Parameter combination {i}: missing required field '{field}'")
            if 'template' not in combo and 'template_file' not in combo:
                raise ValueError(f"Parameter combination {i}: missing required field 'template' (or legacy 'template_file')")

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


@dataclass
class ResourceConfig:
    """Configuration for computational resources."""
    name: str
    resource_type: ResourceType
    capability: ResourceCapability
    software_paths: Dict[str, str]
    hostname: Optional[str] = None
    username: Optional[str] = None
    ssh_key: Optional[str] = None
    remote_work_dir: Optional[str] = None
    monitoring_config: Dict[str, Any] = field(default_factory=dict)
    connection_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, name: str, data: Dict[str, Any]) -> 'ResourceConfig':
        """Create ResourceConfig from dictionary."""
        # Parse capability
        cap_data = data['capability']
        capability = ResourceCapability(
            cores=cap_data['cores'],
            memory_gb=cap_data['memory_gb'],
            supports_mpi=cap_data.get('supports_mpi', True),
            max_concurrent_jobs=cap_data.get('max_concurrent_jobs', 1),
            preferred_software=cap_data.get('preferred_software', [])
        )

        # Determine resource type
        resource_type = ResourceType.REMOTE if 'hostname' in data else ResourceType.LOCAL

        return cls(
            name=name,
            resource_type=resource_type,
            capability=capability,
            software_paths=data['software_paths'],
            hostname=data.get('hostname'),
            username=data.get('username', os.getenv('USER')),
            ssh_key=data.get('connection', {}).get('ssh_key'),
            remote_work_dir=data.get('connection', {}).get('remote_work_dir', '/tmp/atlas_qe_jobs'),
            monitoring_config=data.get('monitoring', {}),
            connection_config=data.get('connection', {})
        )


@dataclass
class DistributedConfig:
    """Configuration for distributed execution."""
    resource_management: Dict[str, Any] = field(default_factory=dict)
    execution_settings: Dict[str, Any] = field(default_factory=dict)
    monitoring_settings: Dict[str, Any] = field(default_factory=dict)
    software_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    logging_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DistributedConfig':
        """Create DistributedConfig from dictionary."""
        return cls(
            resource_management=data.get('resource_management', {}),
            execution_settings=data.get('execution', {}),
            monitoring_settings=data.get('monitoring', {}),
            software_configs=data.get('software_configs', {}),
            logging_config=data.get('logging', {})
        )


class ResourceConfigurationLoader:
    """Loader for resource configuration files."""

    def __init__(self, config_file: Union[str, Path]):
        """Initialize resource configuration loader."""
        self.config_file = Path(config_file)
        self.config_dir = self.config_file.parent

        if not self.config_file.exists():
            raise FileNotFoundError(f"Resource configuration file not found: {self.config_file}")

    def load_resource_configuration(self) -> Tuple[List[ResourceConfig], DistributedConfig]:
        """
        Load resource configuration from YAML file.

        Returns:
            Tuple of (resource_configs, distributed_config)
        """
        try:
            with open(self.config_file, 'r') as f:
                data = yaml.safe_load(f)

            logger.info(f"Loading resource configuration from {self.config_file}")

            # Load resource configurations
            resource_configs = []

            # Load local resources
            for name, config_data in data.get('local_resources', {}).items():
                resource_config = ResourceConfig.from_dict(name, config_data)
                resource_configs.append(resource_config)
                logger.debug(f"Loaded local resource: {name}")

            # Load remote resources
            for name, config_data in data.get('remote_resources', {}).items():
                resource_config = ResourceConfig.from_dict(name, config_data)
                resource_configs.append(resource_config)
                logger.debug(f"Loaded remote resource: {name}")

            # Load distributed configuration
            distributed_config = DistributedConfig.from_dict(data)

            logger.info(f"Loaded {len(resource_configs)} resource configurations")

            return resource_configs, distributed_config

        except Exception as e:
            logger.error(f"Error loading resource configuration: {e}")
            raise


class ConfigurationManager:
    """
    Unified configuration manager for the entire system.

    Manages workflow configurations, resource configurations,
    and provides unified access to all system settings.
    """

    def __init__(
        self,
        workflow_config_file: Optional[Union[str, Path]] = None,
        resource_config_file: Optional[Union[str, Path]] = None
    ):
        """Initialize configuration manager."""
        self.workflow_config: Optional[WorkflowConfiguration] = None
        self.resource_configs: List[ResourceConfig] = []
        self.distributed_config: Optional[DistributedConfig] = None

        # Load configurations if provided
        if workflow_config_file:
            self.load_workflow_configuration(workflow_config_file)

        if resource_config_file:
            self.load_resource_configuration(resource_config_file)

        logger.info("ConfigurationManager initialized")

    def load_workflow_configuration(self, config_file: Union[str, Path]):
        """Load workflow configuration."""
        try:
            loader = ConfigurationLoader(config_file)
            self.workflow_config = loader.load_configuration()
            logger.info("Workflow configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading workflow configuration: {e}")
            raise

    def load_resource_configuration(self, config_file: Union[str, Path]):
        """Load resource configuration."""
        try:
            loader = ResourceConfigurationLoader(config_file)
            self.resource_configs, self.distributed_config = loader.load_resource_configuration()
            logger.info("Resource configuration loaded successfully")
        except Exception as e:
            logger.error(f"Error loading resource configuration: {e}")
            raise

    def get_resource_config(self, resource_name: str) -> Optional[ResourceConfig]:
        """Get configuration for a specific resource."""
        for config in self.resource_configs:
            if config.name == resource_name:
                return config
        return None

    def get_local_resources(self) -> List[ResourceConfig]:
        """Get all local resource configurations."""
        return [config for config in self.resource_configs if config.resource_type == ResourceType.LOCAL]

    def get_remote_resources(self) -> List[ResourceConfig]:
        """Get all remote resource configurations."""
        return [config for config in self.resource_configs if config.resource_type == ResourceType.REMOTE]

    def get_software_config(self, software: str) -> Dict[str, Any]:
        """Get configuration for a specific software."""
        if self.distributed_config:
            return self.distributed_config.software_configs.get(software, {})
        return {}

    def get_execution_settings(self) -> Dict[str, Any]:
        """Get execution settings."""
        if self.distributed_config:
            return self.distributed_config.execution_settings
        return {}

    def get_monitoring_settings(self) -> Dict[str, Any]:
        """Get monitoring settings."""
        if self.distributed_config:
            return self.distributed_config.monitoring_settings
        return {}

    def get_resource_management_settings(self) -> Dict[str, Any]:
        """Get resource management settings."""
        if self.distributed_config:
            return self.distributed_config.resource_management
        return {}

    def validate_configurations(self) -> bool:
        """Validate all loaded configurations."""
        try:
            # Validate workflow configuration
            if self.workflow_config:
                # Basic validation checks
                if not self.workflow_config.system:
                    raise ValueError("System name is required")
                if not self.workflow_config.structures:
                    raise ValueError("At least one structure is required")
                if not self.workflow_config.parameter_combinations:
                    raise ValueError("At least one parameter combination is required")

            # Validate resource configurations
            self._validate_resource_configurations()

            # Validate software paths
            self._validate_software_paths()

            logger.info("All configurations validated successfully")
            return True

        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False

    def _validate_resource_configurations(self):
        """Validate resource configurations."""
        if not self.resource_configs:
            raise ValueError("No resource configurations loaded")

        # Check for duplicate resource names
        names = [config.name for config in self.resource_configs]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate resource names found")

        # Validate each resource configuration
        for config in self.resource_configs:
            if config.capability.cores <= 0:
                raise ValueError(f"Resource {config.name}: cores must be positive")

            if config.capability.memory_gb <= 0:
                raise ValueError(f"Resource {config.name}: memory_gb must be positive")

            if config.capability.max_concurrent_jobs <= 0:
                raise ValueError(f"Resource {config.name}: max_concurrent_jobs must be positive")

    def _validate_software_paths(self):
        """Validate software executable paths."""
        for config in self.resource_configs:
            for software, path in config.software_paths.items():
                if config.resource_type == ResourceType.LOCAL:
                    # For local resources, check if executable exists
                    if not Path(path).exists():
                        logger.warning(f"Local software executable not found: {software} at {path}")
                else:
                    # For remote resources, we can't easily check existence
                    # Just validate that path is not empty
                    if not path.strip():
                        raise ValueError(f"Empty software path for {software} on resource {config.name}")

    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get summary of all loaded configurations."""
        summary = {
            'workflow_config_loaded': self.workflow_config is not None,
            'resource_configs_loaded': len(self.resource_configs),
            'distributed_config_loaded': self.distributed_config is not None
        }

        if self.workflow_config:
            enumerator = ParameterSpaceEnumerator(self.workflow_config)
            workflow_summary = enumerator.get_calculation_summary()
            summary['workflow_summary'] = workflow_summary

        if self.resource_configs:
            total_cores = sum(config.capability.cores for config in self.resource_configs)
            local_count = len(self.get_local_resources())
            remote_count = len(self.get_remote_resources())

            summary['resource_summary'] = {
                'total_resources': len(self.resource_configs),
                'local_resources': local_count,
                'remote_resources': remote_count,
                'total_cores': total_cores,
                'resources': [
                    {
                        'name': config.name,
                        'type': config.resource_type.value,
                        'cores': config.capability.cores,
                        'memory_gb': config.capability.memory_gb,
                        'software': list(config.software_paths.keys())
                    }
                    for config in self.resource_configs
                ]
            }

        return summary

    def export_merged_configuration(self, output_file: Union[str, Path]):
        """Export merged configuration to YAML file."""
        try:
            merged_config = {}

            # Add workflow configuration
            if self.workflow_config:
                merged_config['workflow'] = {
                    'system': self.workflow_config.system,
                    'description': self.workflow_config.description,
                    'structures': [
                        {
                            'name': s.name,
                            'elements': s.elements,
                            'description': s.description,
                            'volume_range': list(s.volume_range),
                            'volume_points': s.volume_points,
                            'file': s.file,
                            'structure_type': s.structure_type,
                            'lattice_parameter': s.lattice_parameter
                        }
                        for s in self.workflow_config.structures
                    ],
                    'parameter_combinations': [
                        {
                            'name': c.name,
                            'software': c.software,
                            'template': c.template,
                            'applies_to_structures': c.applies_to_structures,
                            'pseudopotential_set': c.pseudopotential_set,
                            'template_substitutions': c.template_substitutions
                        }
                        for c in self.workflow_config.parameter_combinations
                    ],
                    'pseudopotential_sets': self.workflow_config.pseudopotential_sets,
                    'data_paths': self.workflow_config.data_paths
                }

            # Add resource configurations
            if self.resource_configs:
                local_resources = {}
                remote_resources = {}

                for config in self.resource_configs:
                    resource_data = {
                        'capability': {
                            'cores': config.capability.cores,
                            'memory_gb': config.capability.memory_gb,
                            'supports_mpi': config.capability.supports_mpi,
                            'max_concurrent_jobs': config.capability.max_concurrent_jobs,
                            'preferred_software': config.capability.preferred_software
                        },
                        'software_paths': config.software_paths,
                        'monitoring': config.monitoring_config
                    }

                    if config.resource_type == ResourceType.LOCAL:
                        local_resources[config.name] = resource_data
                    else:
                        resource_data.update({
                            'hostname': config.hostname,
                            'username': config.username,
                            'connection': config.connection_config
                        })
                        remote_resources[config.name] = resource_data

                merged_config['local_resources'] = local_resources
                merged_config['remote_resources'] = remote_resources

            # Add distributed configuration
            if self.distributed_config:
                merged_config['resource_management'] = self.distributed_config.resource_management
                merged_config['execution'] = self.distributed_config.execution_settings
                merged_config['monitoring'] = self.distributed_config.monitoring_settings
                merged_config['software_configs'] = self.distributed_config.software_configs
                merged_config['logging'] = self.distributed_config.logging_config

            # Write to file
            with open(output_file, 'w') as f:
                yaml.dump(merged_config, f, default_flow_style=False, indent=2)

            logger.info(f"Merged configuration exported to {output_file}")

        except Exception as e:
            logger.error(f"Error exporting merged configuration: {e}")
            raise
