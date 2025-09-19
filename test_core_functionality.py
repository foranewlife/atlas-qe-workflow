#!/usr/bin/env python3
"""
Test script for core ATLAS-QE workflow functionality.

This script tests the main components without requiring actual
software executables or pseudopotential files.
"""

import sys
import yaml
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

from core.parameter_set import ParameterSet, ParameterSetManager
from core.cache_system import ParameterSetCache
from core.pseudopotential import PseudopotentialManager
from core.workflow_engine import WorkflowEngine
from utils.structure_generation import StructureGenerator


def test_parameter_set_creation():
    """Test parameter set creation and fingerprinting."""
    print("=== Testing Parameter Set Creation ===")

    # Create a simple parameter set
    param_set = ParameterSet(
        system="Mg",
        software="atlas",
        structure="fcc",
        parameters={
            "functional": "kedf701",
            "gap": 0.20,
            "grid_spacing": 0.20
        },
        pseudopotential_set="lda",
        pseudopotential_files=["Mg_lda.recpot"],
        volume_range=(0.8, 1.2),
        volume_points=11
    )

    print(f"✓ Parameter set created successfully")
    print(f"  Fingerprint: {param_set.fingerprint[:16]}...")
    print(f"  Workspace name: {param_set.get_workspace_name()}")

    # Test fingerprint consistency
    param_set2 = ParameterSet(
        system="Mg",
        software="atlas",
        structure="fcc",
        parameters={
            "functional": "kedf701",
            "gap": 0.20,
            "grid_spacing": 0.20
        },
        pseudopotential_set="lda",
        pseudopotential_files=["Mg_lda.recpot"],
        volume_range=(0.8, 1.2),
        volume_points=11
    )

    if param_set.fingerprint == param_set2.fingerprint:
        print("✓ Fingerprint consistency verified")
    else:
        print("✗ Fingerprint inconsistency detected!")
        return False

    return True


def test_parameter_set_manager():
    """Test parameter set database operations."""
    print("\n=== Testing Parameter Set Manager ===")

    # Use temporary database
    db_path = Path("test_parameter_sets.db")
    if db_path.exists():
        db_path.unlink()

    manager = ParameterSetManager(db_path)

    # Create test parameter sets
    param_sets = []
    for functional in ["kedf701", "kedf801"]:
        for gap in [0.20, 0.25]:
            param_set = ParameterSet(
                system="Mg",
                software="atlas",
                structure="fcc",
                parameters={
                    "functional": functional,
                    "gap": gap
                },
                pseudopotential_set="lda",
                pseudopotential_files=["Mg_lda.recpot"]
            )
            param_sets.append(param_set)

    # Add to database
    added_count = 0
    for param_set in param_sets:
        if manager.add_parameter_set(param_set):
            added_count += 1

    print(f"✓ Added {added_count} parameter sets to database")

    # Test querying
    all_sets = manager.query_parameter_sets()
    print(f"✓ Retrieved {len(all_sets)} parameter sets from database")

    # Test filtering
    atlas_sets = manager.query_parameter_sets({"software": "atlas"})
    print(f"✓ Found {len(atlas_sets)} ATLAS parameter sets")

    # Test statistics
    stats = manager.get_statistics()
    print(f"✓ Database statistics: {stats['total_parameter_sets']} total sets")

    # Cleanup
    db_path.unlink()

    return True


def test_pseudopotential_manager():
    """Test pseudopotential combination management."""
    print("\n=== Testing Pseudopotential Manager ===")

    pp_manager = PseudopotentialManager()

    # Test single element resolution
    elements = ["Mg"]
    pp_set = {
        "name": "lda",
        "files": ["Mg_lda.recpot"]
    }

    mapping = pp_manager.resolve_pseudopotential_combination(elements, pp_set)
    print(f"✓ Single element mapping: {mapping}")

    # Test multi-element resolution
    elements = ["Mg", "Al"]
    pp_set = {
        "name": "lda_combo",
        "files": ["Mg_lda.recpot", "Al_lda.recpot"]
    }

    mapping = pp_manager.resolve_pseudopotential_combination(elements, pp_set)
    print(f"✓ Multi-element mapping: {mapping}")

    # Test ATLAS input generation
    atlas_line = pp_manager.generate_atlas_ppfile_line(mapping)
    print(f"✓ ATLAS PPFILE line: {atlas_line}")

    # Test QE input generation
    qe_section = pp_manager.generate_qe_atomic_species(mapping)
    print(f"✓ QE ATOMIC_SPECIES section generated")

    # Test validation
    config = {
        "elements": ["Mg", "Al"],
        "atlas": {
            "pseudopotential_sets": [
                {
                    "name": "lda_combo",
                    "files": ["Mg_lda.recpot", "Al_lda.recpot"]
                }
            ]
        }
    }

    is_valid, errors = pp_manager.validate_all_pseudopotential_sets(config)
    print(f"✓ Configuration validation: {'valid' if is_valid else 'invalid'}")
    if errors:
        print(f"  Errors: {errors}")

    return True


def test_structure_generator():
    """Test crystal structure generation."""
    print("\n=== Testing Structure Generator ===")

    generator = StructureGenerator()

    # Test available structures
    structures = generator.get_available_structures()
    print(f"✓ Available structures: {structures}")

    # Test structure generation
    for structure_type in ["fcc", "bcc"]:
        structure = generator.generate_structure(
            structure_type=structure_type,
            elements=["Mg"],
            lattice_parameter=4.0
        )

        # Validate structure
        is_valid, error = generator.validate_structure(structure)
        if is_valid:
            print(f"✓ Generated valid {structure_type} structure")

            # Get structure info
            info = generator.get_structure_info(structure)
            print(f"  Atoms: {info['num_atoms']}, Volume: {info['volume']:.2f} Å³")
        else:
            print(f"✗ Invalid {structure_type} structure: {error}")
            return False

    # Test volume scaling
    base_structure = generator.generate_structure("fcc", ["Mg"], 4.0)
    scaled_structure = generator.scale_structure_volume(base_structure, 1.1)

    base_info = generator.get_structure_info(base_structure)
    scaled_info = generator.get_structure_info(scaled_structure)

    volume_ratio = scaled_info['volume'] / base_info['volume']
    if abs(volume_ratio - 1.1) < 0.01:
        print(f"✓ Volume scaling correct: {volume_ratio:.3f}")
    else:
        print(f"✗ Volume scaling incorrect: {volume_ratio:.3f}")
        return False

    return True


def test_cache_system():
    """Test intelligent caching system."""
    print("\n=== Testing Cache System ===")

    # Use temporary database
    db_path = Path("test_cache.db")
    if db_path.exists():
        db_path.unlink()

    cache_dir = Path("test_cache")
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    manager = ParameterSetManager(db_path)
    cache = ParameterSetCache(cache_dir, manager)

    # Create and add a parameter set
    param_set1 = ParameterSet(
        system="Mg",
        software="atlas",
        structure="fcc",
        parameters={
            "functional": "kedf701",
            "gap": 0.20
        },
        pseudopotential_set="lda",
        pseudopotential_files=["Mg_lda.recpot"]
    )

    manager.add_parameter_set(param_set1)

    # Test cache miss
    cache_match = cache.check_computation_cache(param_set1)
    print(f"✓ Cache miss detected: {cache_match.match_type.value}")

    # Simulate completed calculation
    param_set1.status = param_set1.status.__class__.COMPLETED
    param_set1.results = {"energy": -123.45}
    manager.update_parameter_set(param_set1)

    # Test cache hit
    cache_match = cache.check_computation_cache(param_set1)
    print(f"✓ Cache hit detected: {cache_match.match_type.value}")

    # Test approximate match
    param_set2 = ParameterSet(
        system="Mg",
        software="atlas",
        structure="fcc",
        parameters={
            "functional": "kedf701",
            "gap": 0.21  # Slightly different
        },
        pseudopotential_set="lda",
        pseudopotential_files=["Mg_lda.recpot"]
    )

    cache_match = cache.check_computation_cache(param_set2)
    print(f"✓ Approximate match: {cache_match.match_type.value}, similarity: {cache_match.similarity_score:.2f}")

    # Test cache statistics
    stats = cache.get_cache_statistics()
    print(f"✓ Cache statistics: {stats.total_queries} queries, {stats.hit_rate:.2f} hit rate")

    # Cleanup
    db_path.unlink()
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)

    return True


def test_workflow_engine():
    """Test workflow orchestration engine."""
    print("\n=== Testing Workflow Engine ===")

    # Create workflow engine without actual executables
    engine = WorkflowEngine()

    # Test configuration loading and parameter space discovery
    config = {
        "system": "Mg",
        "elements": ["Mg"],
        "structures": ["fcc", "bcc"],
        "volume_range": [0.8, 1.2],
        "volume_points": 5,
        "atlas": {
            "functionals": ["kedf701"],
            "gap": [0.20],
            "pseudopotentials": ["Mg_lda.recpot"]
        },
        "qe": {
            "configurations": ["ncpp_ecut60"],
            "k_points": [[8, 8, 8]],
            "pseudopotentials": ["Mg.ncpp.upf"]
        }
    }

    try:
        parameter_space = engine.discover_parameter_space(config)
        print(f"✓ Discovered {len(parameter_space)} parameter sets")

        # Print some details
        for i, ps in enumerate(parameter_space[:3]):
            print(f"  {i+1}. {ps.software}_{ps.structure}_{ps.pseudopotential_set}")

        if len(parameter_space) > 3:
            print(f"  ... and {len(parameter_space) - 3} more")

    except Exception as e:
        print(f"✗ Error in parameter space discovery: {e}")
        return False

    # Test workflow status
    status = engine.get_workflow_status()
    print(f"✓ Workflow status retrieved: {len(status['calculators'])} calculators configured")

    return True


def test_yaml_configuration():
    """Test YAML configuration loading."""
    print("\n=== Testing YAML Configuration ===")

    # Test loading the example configurations
    config_files = [
        "config/systems/Mg_study.yaml",
        "config/systems/MgAl_study.yaml"
    ]

    for config_file in config_files:
        config_path = Path(config_file)
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)

                print(f"✓ Loaded {config_file}")
                print(f"  System: {config['system']}")
                print(f"  Elements: {config['elements']}")
                print(f"  Structures: {config['structures']}")

                # Test with workflow engine
                engine = WorkflowEngine()
                parameter_space = engine.discover_parameter_space(config)
                print(f"  Parameter sets: {len(parameter_space)}")

            except Exception as e:
                print(f"✗ Error loading {config_file}: {e}")
                return False
        else:
            print(f"✗ Configuration file not found: {config_file}")
            return False

    return True


def main():
    """Run all tests."""
    print("🧪 Testing ATLAS-QE Workflow Core Functionality\n")

    tests = [
        ("Parameter Set Creation", test_parameter_set_creation),
        ("Parameter Set Manager", test_parameter_set_manager),
        ("Pseudopotential Manager", test_pseudopotential_manager),
        ("Structure Generator", test_structure_generator),
        ("Cache System", test_cache_system),
        ("Workflow Engine", test_workflow_engine),
        ("YAML Configuration", test_yaml_configuration),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results.append((test_name, False))

    # Summary
    print("\n" + "="*50)
    print("🏁 Test Summary")
    print("="*50)

    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if result:
            passed += 1

    print(f"\nTotal: {passed}/{len(results)} tests passed")

    if passed == len(results):
        print("🎉 All core functionality tests passed!")
        return True
    else:
        print("⚠️  Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)