#!/usr/bin/env python3
"""
Unit tests for customer segmentation module.
"""

import os
import sys
import unittest
import tempfile
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.customer_segmentation import (
    currency_to_float,
    validate_dataframe,
    ensure_output_dir,
    calculate_adaptive_sample_size,
    REQUIRED_COLUMNS,
    CLUSTERING_FEATURES
)


class TestCurrencyConversion(unittest.TestCase):
    """Test currency conversion function."""

    def test_simple_currency(self):
        """Test simple currency strings."""
        self.assertEqual(currency_to_float("£100.50"), 100.50)
        self.assertEqual(currency_to_float("€200.75"), 200.75)
        self.assertEqual(currency_to_float("$50.25"), 50.25)

    def test_currency_with_thousands_separator(self):
        """Test currency with thousands separators."""
        self.assertEqual(currency_to_float("£1,234.56"), 1234.56)
        self.assertEqual(currency_to_float("€10,000.00"), 10000.00)

    def test_european_format(self):
        """Test European number format (comma as decimal)."""
        self.assertEqual(currency_to_float("€100,50"), 100.50)

    def test_whitespace(self):
        """Test handling of whitespace."""
        self.assertEqual(currency_to_float("  £100.50  "), 100.50)
        self.assertEqual(currency_to_float("£ 100.50"), 100.50)

    def test_invalid_input(self):
        """Test invalid input handling."""
        self.assertTrue(np.isnan(currency_to_float("invalid")))
        self.assertTrue(np.isnan(currency_to_float("")))

    def test_negative_values(self):
        """Test negative currency values."""
        self.assertEqual(currency_to_float("-£50.00"), -50.00)


class TestDataValidation(unittest.TestCase):
    """Test data validation functions."""

    def test_validate_empty_dataframe(self):
        """Test validation of empty dataframe."""
        df = pd.DataFrame()
        with self.assertRaises(ValueError) as context:
            validate_dataframe(df, REQUIRED_COLUMNS)
        self.assertIn("empty", str(context.exception).lower())

    def test_validate_missing_columns(self):
        """Test validation with missing columns."""
        df = pd.DataFrame({'Customer ID': [1, 2, 3]})
        with self.assertRaises(ValueError) as context:
            validate_dataframe(df, REQUIRED_COLUMNS)
        self.assertIn("missing", str(context.exception).lower())

    def test_validate_correct_dataframe(self):
        """Test validation with correct dataframe."""
        data = {col: [1, 2, 3] for col in REQUIRED_COLUMNS}
        df = pd.DataFrame(data)
        # Should not raise an exception
        try:
            validate_dataframe(df, REQUIRED_COLUMNS)
        except ValueError:
            self.fail("validate_dataframe raised ValueError unexpectedly")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def test_ensure_output_dir(self):
        """Test output directory creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "test_output")
            ensure_output_dir(output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertTrue(os.path.isdir(output_path))

    def test_calculate_adaptive_sample_size(self):
        """Test adaptive sample size calculation."""
        # Small dataset - should use all
        self.assertEqual(calculate_adaptive_sample_size(100, 1000, 0.5), 100)

        # Medium dataset - should use fraction
        self.assertEqual(calculate_adaptive_sample_size(2000, 1000, 0.5), 1000)

        # Large dataset - should use max
        self.assertEqual(calculate_adaptive_sample_size(10000, 1000, 0.5), 1000)

        # Very small dataset
        result = calculate_adaptive_sample_size(50, 1000, 0.5)
        self.assertGreaterEqual(result, 50)


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""

    def create_sample_data(self):
        """Create sample data for testing."""
        data = {
            'Customer ID': ['C1', 'C1', 'C2', 'C2', 'C3'],
            'Order ID': ['O1', 'O2', 'O3', 'O4', 'O5'],
            'Order_Date': ['01JAN2023', '15FEB2023', '10JAN2023', '20FEB2023', '05MAR2023'],
            'Delivery_Date': ['05JAN2023', '20FEB2023', '15JAN2023', '25FEB2023', '10MAR2023'],
            'Total Revenue': ['£100.00', '£200.00', '£150.00', '£75.00', '£300.00'],
            'Unit Cost': ['£10.00', '£20.00', '£15.00', '£7.50', '£30.00'],
            'Customer_BirthDate': ['15MAR1985', '15MAR1985', '22JUL1990', '22JUL1990', '10DEC1988']
        }
        return pd.DataFrame(data)

    def test_sample_data_structure(self):
        """Test that sample data has correct structure."""
        df = self.create_sample_data()
        self.assertEqual(len(df), 5)
        self.assertTrue(all(col in df.columns for col in REQUIRED_COLUMNS))


class TestClusteringValidation(unittest.TestCase):
    """Test clustering-related functions."""

    def test_clustering_features_defined(self):
        """Test that clustering features are properly defined."""
        self.assertIsInstance(CLUSTERING_FEATURES, list)
        self.assertGreater(len(CLUSTERING_FEATURES), 0)
        expected_features = ['Frequency', 'Recency', 'CLV', 'Average Unit Cost', 'Customer Age']
        self.assertEqual(CLUSTERING_FEATURES, expected_features)


if __name__ == '__main__':
    unittest.main()
