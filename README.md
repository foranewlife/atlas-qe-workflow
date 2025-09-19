# ATLAS-QE Materials Workflow

A comprehensive automated workflow system for materials calculations using ATLAS (OFDFT) and Quantum ESPRESSO (DFT) with intelligent parameter set management and caching.

## Features

- **Complete Workflow Automation**: From convergence testing to EOS analysis
- **Parameter Set Management**: Unique fingerprinting and intelligent caching
- **Multi-element Support**: Proper pseudopotential combination handling
- **Flexible Configuration**: YAML-based system definitions
- **Resource Management**: Local and cluster computation scheduling
- **Analysis Engine**: Comprehensive querying and visualization tools

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd atlas-qe-workflow

# Install with uv (recommended)
uv pip install -e .

# Or with pip
pip install -e .
```

### Basic Usage

```bash
# Run complete workflow for Mg study
atlas-qe-workflow config/systems/Mg_study.yaml

# Convergence testing only
atlas-qe-convergence --system Mg --software qe

# Structure optimization
atlas-qe-optimize --system Mg

# Parameter set workflow
atlas-qe-paramset config/systems/MgAl_study.yaml

# Query and analysis
atlas-qe-query --query "software=atlas AND pseudopotential_set=lda_combo"
```

## Workflow Architecture

```
Convergence Tests → Structure Optimization → EOS Analysis
                 ↓
         Parameter Set Database + Intelligent Cache
```

### Core Components

1. **Parameter Set Engine**: Enumerates and manages calculation parameters
2. **Intelligent Cache**: Avoids duplicate calculations with fingerprinting
3. **Calculator Interfaces**: ATLAS and QE integration
4. **Analysis Engine**: EOS fitting, visualization, and comparison tools
5. **Workflow Orchestration**: End-to-end automation with resource management

## Configuration

### System Configuration Example

```yaml
# Single element system (Mg)
system: "Mg"
elements: ["Mg"]
structures: ["fcc", "bcc", "diamond"]
volume_range: [0.8, 1.2]
volume_points: 11

atlas:
  functionals: ["kedf701", "kedf801"]
  gap: [0.20, 0.25]
  pseudopotentials: ["Mg_lda.recpot", "Mg_pbe.recpot"]

qe:
  configurations: ["ncpp_ecut60", "paw_ecut80"]
  k_points: [[10,10,10], [12,12,12]]
  pseudopotentials: ["Mg.ncpp.upf", "Mg.paw.upf"]
```

### Multi-element System Support

For systems like MgAl, the workflow handles pseudopotential combinations:

```yaml
system: "MgAl"
elements: ["Mg", "Al"]
atlas:
  pseudopotential_sets:
    - name: "lda_combo"
      files: ["Mg_lda.recpot", "Al_lda.recpot"]
    - name: "pbe_combo"
      files: ["Mg_pbe.recpot", "Al_pbe.recpot"]
```

## Project Structure

```
atlas-qe-workflow/
├── src/
│   ├── core/              # Core parameter and cache management
│   ├── calculators/       # ATLAS/QE interfaces
│   ├── analysis/          # EOS analysis and visualization
│   └── utils/             # Utilities and helpers
├── scripts/               # Command-line entry points
├── config/                # Configuration templates
├── tests/                 # Unit and integration tests
├── docs/                  # Documentation
└── examples/              # Example workflows
```

## Development

### Setting up Development Environment

```bash
# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Format code
black src/ tests/

# Type checking
mypy src/
```

### Key Design Principles

1. **Modular Architecture**: Independent components for flexibility
2. **Parameter Fingerprinting**: Unique identification for caching
3. **Incremental Computation**: Only calculate missing parameter sets
4. **Scientific Workflow**: Research-oriented analysis and comparison tools

## License

MIT License

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## Acknowledgments

Built as part of the TB-OFDFT project for large-scale materials simulation using semi-empirical methods.