"""
Unit tests for parameter set management.
"""

import unittest
import tempfile
from pathlib import Path

from src.core.parameter_set import ParameterSet, ParameterSetManager, ParameterSetStatus


class TestParameterSet(unittest.TestCase):
    """Test ParameterSet class functionality."""

    def test_parameter_set_creation(self):
        """Test basic parameter set creation."""
        param_set = ParameterSet(
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

        self.assertEqual(param_set.system, "Mg")
        self.assertEqual(param_set.software, "atlas")
        self.assertEqual(param_set.structure, "fcc")
        self.assertIsNotNone(param_set.fingerprint)
        self.assertEqual(param_set.status, ParameterSetStatus.PENDING)

    def test_fingerprint_consistency(self):
        """Test that identical parameter sets have identical fingerprints."""
        param_set1 = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        param_set2 = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        self.assertEqual(param_set1.fingerprint, param_set2.fingerprint)

    def test_fingerprint_uniqueness(self):
        """Test that different parameter sets have different fingerprints."""
        param_set1 = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        param_set2 = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf801", "gap": 0.20},  # Different functional
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        self.assertNotEqual(param_set1.fingerprint, param_set2.fingerprint)

    def test_workspace_name_generation(self):
        """Test workspace name generation."""
        param_set = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda_combo",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        workspace_name = param_set.get_workspace_name()
        expected_parts = ["atlas", "fcc", "kedf701", "gap020", "lda_combo"]

        for part in expected_parts:
            self.assertIn(part, workspace_name)

    def test_serialization(self):
        """Test parameter set serialization/deserialization."""
        param_set = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        # Test to_dict
        data = param_set.to_dict()
        self.assertIsInstance(data, dict)
        self.assertEqual(data["system"], "Mg")
        self.assertEqual(data["status"], "pending")

        # Test from_dict
        restored_param_set = ParameterSet.from_dict(data)
        self.assertEqual(restored_param_set.fingerprint, param_set.fingerprint)
        self.assertEqual(restored_param_set.system, param_set.system)


class TestParameterSetManager(unittest.TestCase):
    """Test ParameterSetManager class functionality."""

    def setUp(self):
        """Set up test database."""
        self.temp_dir = tempfile.mkdtemp()
        self.db_path = Path(self.temp_dir) / "test.db"
        self.manager = ParameterSetManager(self.db_path)

    def tearDown(self):
        """Clean up test database."""
        if self.db_path.exists():
            self.db_path.unlink()

    def test_database_initialization(self):
        """Test database is properly initialized."""
        self.assertTrue(self.db_path.exists())

    def test_add_parameter_set(self):
        """Test adding parameter sets to database."""
        param_set = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        # First addition should succeed
        result = self.manager.add_parameter_set(param_set)
        self.assertTrue(result)

        # Duplicate addition should fail
        result = self.manager.add_parameter_set(param_set)
        self.assertFalse(result)

    def test_get_parameter_set(self):
        """Test retrieving parameter sets by fingerprint."""
        param_set = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        self.manager.add_parameter_set(param_set)

        # Test retrieval
        retrieved = self.manager.get_parameter_set(param_set.fingerprint)
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.fingerprint, param_set.fingerprint)
        self.assertEqual(retrieved.system, param_set.system)

        # Test non-existent fingerprint
        fake_fingerprint = "0" * 64
        retrieved = self.manager.get_parameter_set(fake_fingerprint)
        self.assertIsNone(retrieved)

    def test_update_parameter_set(self):
        """Test updating parameter sets."""
        param_set = ParameterSet(
            system="Mg",
            software="atlas",
            structure="fcc",
            parameters={"functional": "kedf701", "gap": 0.20},
            pseudopotential_set="lda",
            pseudopotential_files=["Mg_lda.recpot"]
        )

        self.manager.add_parameter_set(param_set)

        # Update status
        param_set.status = ParameterSetStatus.COMPLETED
        param_set.results = {"energy": -123.45}
        self.manager.update_parameter_set(param_set)

        # Verify update
        retrieved = self.manager.get_parameter_set(param_set.fingerprint)
        self.assertEqual(retrieved.status, ParameterSetStatus.COMPLETED)
        self.assertIsNotNone(retrieved.results)
        self.assertEqual(retrieved.results["energy"], -123.45)

    def test_query_parameter_sets(self):
        """Test querying parameter sets with filters."""
        # Add multiple parameter sets
        param_sets = []
        for software in ["atlas", "qe"]:
            for structure in ["fcc", "bcc"]:
                param_set = ParameterSet(
                    system="Mg",
                    software=software,
                    structure=structure,
                    parameters={"test": "value"},
                    pseudopotential_set="lda",
                    pseudopotential_files=["Mg_lda.recpot"]
                )
                param_sets.append(param_set)
                self.manager.add_parameter_set(param_set)

        # Test query all
        all_sets = self.manager.query_parameter_sets()
        self.assertEqual(len(all_sets), 4)

        # Test filtered query
        atlas_sets = self.manager.query_parameter_sets({"software": "atlas"})
        self.assertEqual(len(atlas_sets), 2)

        fcc_sets = self.manager.query_parameter_sets({"structure": "fcc"})
        self.assertEqual(len(fcc_sets), 2)

        # Test multiple filters
        atlas_fcc = self.manager.query_parameter_sets({
            "software": "atlas",
            "structure": "fcc"
        })
        self.assertEqual(len(atlas_fcc), 1)

    def test_get_missing_parameter_sets(self):
        """Test detection of missing parameter sets."""
        # Create target parameter space
        target_space = []
        for i in range(3):
            param_set = ParameterSet(
                system="Mg",
                software="atlas",
                structure="fcc",
                parameters={"functional": f"kedf{i}"},
                pseudopotential_set="lda",
                pseudopotential_files=["Mg_lda.recpot"]
            )
            target_space.append(param_set)

        # Add only one to database
        self.manager.add_parameter_set(target_space[0])

        # Check missing
        missing = self.manager.get_missing_parameter_sets(target_space)
        self.assertEqual(len(missing), 2)

        # Add all and check again
        for param_set in target_space[1:]:
            self.manager.add_parameter_set(param_set)

        missing = self.manager.get_missing_parameter_sets(target_space)
        self.assertEqual(len(missing), 0)

    def test_get_statistics(self):
        """Test database statistics."""
        # Add some parameter sets
        for i in range(3):
            param_set = ParameterSet(
                system="Mg",
                software="atlas",
                structure="fcc",
                parameters={"functional": f"kedf{i}"},
                pseudopotential_set="lda",
                pseudopotential_files=["Mg_lda.recpot"]
            )
            self.manager.add_parameter_set(param_set)

        stats = self.manager.get_statistics()
        self.assertEqual(stats["total_parameter_sets"], 3)
        self.assertIn("status_breakdown", stats)
        self.assertIn("database_path", stats)


if __name__ == "__main__":
    unittest.main()