"""
Unit tests for pseudopotential management.
"""

import unittest
import tempfile
from pathlib import Path

from src.core.pseudopotential import PseudopotentialManager


class TestPseudopotentialManager(unittest.TestCase):
    """Test PseudopotentialManager functionality."""

    def setUp(self):
        """Set up test pseudopotential manager."""
        self.temp_dir = tempfile.mkdtemp()
        self.pp_manager = PseudopotentialManager(Path(self.temp_dir))

    def test_single_element_resolution(self):
        """Test pseudopotential resolution for single element systems."""
        elements = ["Mg"]
        pp_set = {
            "name": "lda",
            "files": ["Mg_lda.recpot"]
        }

        mapping = self.pp_manager.resolve_pseudopotential_combination(elements, pp_set)

        self.assertEqual(len(mapping), 1)
        self.assertIn("Mg", mapping)
        self.assertEqual(mapping["Mg"], "Mg_lda.recpot")

    def test_multi_element_resolution(self):
        """Test pseudopotential resolution for multi-element systems."""
        elements = ["Mg", "Al"]
        pp_set = {
            "name": "lda_combo",
            "files": ["Mg_lda.recpot", "Al_lda.recpot"]
        }

        mapping = self.pp_manager.resolve_pseudopotential_combination(elements, pp_set)

        self.assertEqual(len(mapping), 2)
        self.assertIn("Mg", mapping)
        self.assertIn("Al", mapping)
        self.assertEqual(mapping["Mg"], "Mg_lda.recpot")
        self.assertEqual(mapping["Al"], "Al_lda.recpot")

    def test_resolution_validation(self):
        """Test validation of pseudopotential combinations."""
        elements = ["Mg", "Al"]

        # Valid combination
        valid_pp_set = {
            "name": "lda_combo",
            "files": ["Mg_lda.recpot", "Al_lda.recpot"]
        }

        is_valid, error = self.pp_manager.validate_pseudopotential_set(elements, valid_pp_set)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Invalid combination - wrong number of files
        invalid_pp_set = {
            "name": "incomplete",
            "files": ["Mg_lda.recpot"]  # Missing Al pseudopotential
        }

        is_valid, error = self.pp_manager.validate_pseudopotential_set(elements, invalid_pp_set)
        self.assertFalse(is_valid)
        self.assertIn("does not match", error)

        # Invalid combination - missing name
        nameless_pp_set = {
            "files": ["Mg_lda.recpot", "Al_lda.recpot"]
        }

        is_valid, error = self.pp_manager.validate_pseudopotential_set(elements, nameless_pp_set)
        self.assertFalse(is_valid)
        self.assertIn("name", error)

    def test_atlas_input_generation(self):
        """Test ATLAS PPFILE line generation."""
        mapping = {
            "Mg": "Mg_lda.recpot",
            "Al": "Al_lda.recpot"
        }

        ppfile_line = self.pp_manager.generate_atlas_ppfile_line(mapping)

        self.assertIn("PPFILE =", ppfile_line)
        self.assertIn("Mg_lda.recpot", ppfile_line)
        self.assertIn("Al_lda.recpot", ppfile_line)

    def test_qe_input_generation(self):
        """Test QE ATOMIC_SPECIES section generation."""
        mapping = {
            "Mg": "Mg_lda.recpot",
            "Al": "Al_lda.recpot"
        }

        atomic_species = self.pp_manager.generate_qe_atomic_species(mapping)

        self.assertIn("ATOMIC_SPECIES", atomic_species)
        self.assertIn("Mg", atomic_species)
        self.assertIn("Al", atomic_species)
        self.assertIn("24.305", atomic_species)  # Mg mass
        self.assertIn("26.982", atomic_species)  # Al mass

    def test_configuration_validation(self):
        """Test complete configuration validation."""
        # Valid single element configuration
        single_element_config = {
            "elements": ["Mg"],
            "atlas": {
                "pseudopotentials": ["Mg_lda.recpot", "Mg_pbe.recpot"]
            },
            "qe": {
                "pseudopotentials": ["Mg.ncpp.upf", "Mg.paw.upf"]
            }
        }

        is_valid, errors = self.pp_manager.validate_all_pseudopotential_sets(single_element_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Valid multi-element configuration
        multi_element_config = {
            "elements": ["Mg", "Al"],
            "atlas": {
                "pseudopotential_sets": [
                    {
                        "name": "lda_combo",
                        "files": ["Mg_lda.recpot", "Al_lda.recpot"]
                    }
                ]
            },
            "qe": {
                "pseudopotential_sets": [
                    {
                        "name": "ncpp_lda",
                        "files": ["Mg.ncpp.lda.upf", "Al.ncpp.lda.upf"]
                    }
                ]
            }
        }

        is_valid, errors = self.pp_manager.validate_all_pseudopotential_sets(multi_element_config)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)

        # Invalid configuration - incomplete pseudopotential set
        invalid_config = {
            "elements": ["Mg", "Al"],
            "atlas": {
                "pseudopotential_sets": [
                    {
                        "name": "incomplete",
                        "files": ["Mg_lda.recpot"]  # Missing Al
                    }
                ]
            }
        }

        is_valid, errors = self.pp_manager.validate_all_pseudopotential_sets(invalid_config)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)

    def test_pseudopotential_enumeration(self):
        """Test enumeration of pseudopotential combinations."""
        # Single element system
        elements = ["Mg"]
        config = {
            "pseudopotentials": ["Mg_lda.recpot", "Mg_pbe.recpot"]
        }

        combinations = self.pp_manager.enumerate_pseudopotential_combinations(elements, config)
        self.assertEqual(len(combinations), 2)

        # Multi-element system
        elements = ["Mg", "Al"]
        config = {
            "pseudopotential_sets": [
                {
                    "name": "lda_combo",
                    "files": ["Mg_lda.recpot", "Al_lda.recpot"]
                },
                {
                    "name": "pbe_combo",
                    "files": ["Mg_pbe.recpot", "Al_pbe.recpot"]
                }
            ]
        }

        combinations = self.pp_manager.enumerate_pseudopotential_combinations(elements, config)
        self.assertEqual(len(combinations), 2)
        self.assertEqual(combinations[0]["name"], "lda_combo")
        self.assertEqual(combinations[1]["name"], "pbe_combo")

    def test_pseudopotential_type_inference(self):
        """Test pseudopotential type inference from filenames."""
        test_cases = [
            ("Mg_lda.recpot", "lda"),
            ("Al_pbe.recpot", "pbe"),
            ("Si_pbesol.recpot", "pbesol"),
            ("Mg.paw.upf", "paw"),
            ("Al.ncpp.upf", "ncpp"),
            ("Si.uspp.upf", "uspp"),
            ("unknown.txt", "unknown")
        ]

        for filename, expected_type in test_cases:
            with self.subTest(filename=filename):
                inferred_type = self.pp_manager._infer_pseudopotential_type(filename)
                self.assertEqual(inferred_type, expected_type)

    def test_pseudopotential_info(self):
        """Test pseudopotential file information extraction."""
        # Test with non-existent file
        info = self.pp_manager.get_pseudopotential_info("Mg_lda.recpot")

        self.assertEqual(info["filename"], "Mg_lda.recpot")
        self.assertEqual(info["type"], "lda")
        self.assertIsNotNone(info["exists"])  # Should be False or None

        # Create a test file
        test_file = Path(self.temp_dir) / "Mg_pbe.recpot"
        test_file.write_text("test pseudopotential data")

        info = self.pp_manager.get_pseudopotential_info("Mg_pbe.recpot")
        self.assertEqual(info["filename"], "Mg_pbe.recpot")
        self.assertEqual(info["type"], "pbe")
        self.assertTrue(info["exists"])
        self.assertGreater(info["size"], 0)


if __name__ == "__main__":
    unittest.main()