"""
Unit tests for crystal structure generation.
"""

import unittest
import numpy as np

from src.utils.structure_generation import StructureGenerator


class TestStructureGenerator(unittest.TestCase):
    """Test StructureGenerator functionality."""

    def setUp(self):
        """Set up structure generator."""
        self.generator = StructureGenerator()

    def test_available_structures(self):
        """Test that basic crystal structures are available."""
        structures = self.generator.get_available_structures()

        expected_structures = ["fcc", "bcc", "diamond", "sc"]
        for structure in expected_structures:
            self.assertIn(structure, structures)

    def test_fcc_structure_generation(self):
        """Test FCC structure generation."""
        structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg"],
            lattice_parameter=4.0
        )

        # Validate structure
        is_valid, error = self.generator.validate_structure(structure)
        self.assertTrue(is_valid, f"FCC structure invalid: {error}")

        # Check basic properties
        self.assertEqual(structure["structure_type"], "fcc")
        self.assertEqual(len(structure["species"]), 4)  # FCC has 4 atoms per unit cell
        self.assertEqual(len(structure["coords"]), 4)

        # Check lattice parameter
        lattice = np.array(structure["lattice"])
        self.assertAlmostEqual(np.linalg.norm(lattice[0]), 4.0, places=5)

    def test_bcc_structure_generation(self):
        """Test BCC structure generation."""
        structure = self.generator.generate_structure(
            structure_type="bcc",
            elements=["Fe"],
            lattice_parameter=2.87
        )

        # Validate structure
        is_valid, error = self.generator.validate_structure(structure)
        self.assertTrue(is_valid, f"BCC structure invalid: {error}")

        # Check basic properties
        self.assertEqual(structure["structure_type"], "bcc")
        self.assertEqual(len(structure["species"]), 2)  # BCC has 2 atoms per unit cell
        self.assertEqual(len(structure["coords"]), 2)

    def test_diamond_structure_generation(self):
        """Test diamond structure generation."""
        structure = self.generator.generate_structure(
            structure_type="diamond",
            elements=["C"],
            lattice_parameter=3.57
        )

        # Validate structure
        is_valid, error = self.generator.validate_structure(structure)
        self.assertTrue(is_valid, f"Diamond structure invalid: {error}")

        # Check basic properties
        self.assertEqual(structure["structure_type"], "diamond")
        self.assertEqual(len(structure["species"]), 8)  # Diamond has 8 atoms per unit cell

    def test_multi_element_structure(self):
        """Test structure generation with multiple elements."""
        structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg", "Al"],
            lattice_parameter=4.0
        )

        # Check that elements are distributed cyclically
        species = structure["species"]
        self.assertIn("Mg", species)
        self.assertIn("Al", species)

        # For FCC with 2 elements, should have 2 Mg and 2 Al atoms
        mg_count = species.count("Mg")
        al_count = species.count("Al")
        self.assertEqual(mg_count, 2)
        self.assertEqual(al_count, 2)

    def test_supercell_generation(self):
        """Test supercell generation."""
        base_structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg"],
            lattice_parameter=4.0,
            supercell=[1, 1, 1]
        )

        supercell_structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg"],
            lattice_parameter=4.0,
            supercell=[2, 2, 2]
        )

        # Supercell should have 8x more atoms
        base_atoms = len(base_structure["species"])
        supercell_atoms = len(supercell_structure["species"])
        self.assertEqual(supercell_atoms, base_atoms * 8)

        # Lattice vectors should be scaled
        base_lattice = np.array(base_structure["lattice"])
        supercell_lattice = np.array(supercell_structure["lattice"])

        expected_lattice = base_lattice * 2
        np.testing.assert_array_almost_equal(supercell_lattice, expected_lattice)

    def test_volume_scaling(self):
        """Test volume scaling functionality."""
        base_structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg"],
            lattice_parameter=4.0
        )

        # Scale volume by factor of 1.21 (20% increase)
        scaled_structure = self.generator.scale_structure_volume(base_structure, 1.21)

        # Check volume change
        base_info = self.generator.get_structure_info(base_structure)
        scaled_info = self.generator.get_structure_info(scaled_structure)

        volume_ratio = scaled_info["volume"] / base_info["volume"]
        self.assertAlmostEqual(volume_ratio, 1.21, places=2)

        # Check that lattice parameter is correctly scaled
        # Volume scales as cube of linear dimension, so linear scale = volume_scale^(1/3)
        expected_linear_scale = 1.21**(1/3)
        actual_linear_scale = scaled_structure["lattice_parameter"] / base_structure["lattice_parameter"]
        self.assertAlmostEqual(actual_linear_scale, expected_linear_scale, places=3)

    def test_volume_series_generation(self):
        """Test generation of volume series for EOS calculations."""
        base_structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg"],
            lattice_parameter=4.0
        )

        volume_series = self.generator.generate_volume_series(
            base_structure,
            volume_range=(0.9, 1.1),
            num_points=5
        )

        # Check series length
        self.assertEqual(len(volume_series), 5)

        # Check volume scaling
        base_info = self.generator.get_structure_info(base_structure)
        base_volume = base_info["volume"]

        expected_scales = [0.9, 0.95, 1.0, 1.05, 1.1]
        for i, structure in enumerate(volume_series):
            info = self.generator.get_structure_info(structure)
            volume_ratio = info["volume"] / base_volume
            self.assertAlmostEqual(volume_ratio, expected_scales[i], places=2)
            self.assertAlmostEqual(structure["volume_scale"], expected_scales[i], places=5)

    def test_structure_validation(self):
        """Test structure validation functionality."""
        # Valid structure
        valid_structure = {
            "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "species": ["Mg", "Mg"],
            "coords": [[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]]
        }

        is_valid, error = self.generator.validate_structure(valid_structure)
        self.assertTrue(is_valid)
        self.assertEqual(error, "")

        # Invalid structure - missing lattice
        invalid_structure = {
            "species": ["Mg"],
            "coords": [[0.0, 0.0, 0.0]]
        }

        is_valid, error = self.generator.validate_structure(invalid_structure)
        self.assertFalse(is_valid)
        self.assertIn("lattice", error)

        # Invalid structure - mismatched species and coords
        mismatched_structure = {
            "lattice": [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
            "species": ["Mg", "Al"],
            "coords": [[0.0, 0.0, 0.0]]  # Only one coordinate for two species
        }

        is_valid, error = self.generator.validate_structure(mismatched_structure)
        self.assertFalse(is_valid)
        self.assertIn("match", error)

        # Invalid structure - zero volume lattice
        zero_volume_structure = {
            "lattice": [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],  # Linearly dependent
            "species": ["Mg"],
            "coords": [[0.0, 0.0, 0.0]]
        }

        is_valid, error = self.generator.validate_structure(zero_volume_structure)
        self.assertFalse(is_valid)
        self.assertIn("zero volume", error)

    def test_structure_info_extraction(self):
        """Test extraction of structure information."""
        structure = self.generator.generate_structure(
            structure_type="fcc",
            elements=["Mg", "Al"],
            lattice_parameter=4.0
        )

        info = self.generator.get_structure_info(structure)

        # Check required fields
        required_fields = ["volume", "num_atoms", "unique_species", "composition", "density"]
        for field in required_fields:
            self.assertIn(field, info)

        # Check values
        self.assertGreater(info["volume"], 0)
        self.assertEqual(info["num_atoms"], 4)  # FCC unit cell
        self.assertEqual(set(info["unique_species"]), {"Mg", "Al"})
        self.assertEqual(info["composition"]["Mg"], 2)
        self.assertEqual(info["composition"]["Al"], 2)
        self.assertGreater(info["density"], 0)

    def test_unknown_structure_type(self):
        """Test handling of unknown structure types."""
        with self.assertRaises(ValueError):
            self.generator.generate_structure(
                structure_type="unknown_structure",
                elements=["Mg"],
                lattice_parameter=4.0
            )


if __name__ == "__main__":
    unittest.main()