"""
Unit Tests for Membership Functions

Tests for the MembershipFunctions class.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import numpy as np
from membership_functions import MembershipFunctions as MF


class TestMembershipFunctions(unittest.TestCase):
    """Test cases for membership functions."""
    
    def test_universe_generation(self):
        """Test universe of discourse generation."""
        age_universe = MF.get_universe('age')
        self.assertEqual(age_universe[0], 29)
        self.assertGreaterEqual(age_universe[-1], 77)
        
        risk_universe = MF.get_universe('risk')
        self.assertEqual(risk_universe[0], 0)
        self.assertLessEqual(risk_universe[-1], 1.01)
    
    def test_membership_values_at_peaks(self):
        """Test that membership is 1.0 at peak values."""
        # Age: young peaks at 29
        fuzz = MF.fuzzify('age', 29)
        self.assertEqual(fuzz['young'], 1.0)
        
        # Age: middle peaks at 50
        fuzz = MF.fuzzify('age', 50)
        self.assertEqual(fuzz['middle'], 1.0)
        
        # Age: old peaks at 77
        fuzz = MF.fuzzify('age', 77)
        self.assertEqual(fuzz['old'], 1.0)
    
    def test_membership_values_at_boundaries(self):
        """Test membership values at term boundaries."""
        # At age 45, young should be 0 and middle should be > 0
        fuzz = MF.fuzzify('age', 45)
        self.assertAlmostEqual(fuzz['young'], 0.0, places=2)
        self.assertGreater(fuzz['middle'], 0)
    
    def test_fuzzify_returns_all_terms(self):
        """Test that fuzzify returns all terms for a variable."""
        fuzz = MF.fuzzify('age', 50)
        self.assertIn('young', fuzz)
        self.assertIn('middle', fuzz)
        self.assertIn('old', fuzz)
        
        fuzz = MF.fuzzify('risk', 0.5)
        self.assertIn('low', fuzz)
        self.assertIn('medium', fuzz)
        self.assertIn('high', fuzz)
    
    def test_membership_sum_reasonable(self):
        """Test that membership values are reasonable (can overlap)."""
        fuzz = MF.fuzzify('age', 50)
        total = sum(fuzz.values())
        # Sum can be > 1 due to overlapping MFs, but should be reasonable
        self.assertGreater(total, 0)
        self.assertLess(total, 3)  # At most 3 terms fully activated
    
    def test_get_all_terms(self):
        """Test getting all terms for a variable."""
        age_terms = MF.get_all_terms('age')
        self.assertEqual(set(age_terms), {'young', 'middle', 'old'})
        
        risk_terms = MF.get_all_terms('risk')
        self.assertEqual(set(risk_terms), {'low', 'medium', 'high'})
    
    def test_edge_values(self):
        """Test membership at edge values."""
        # Minimum age
        fuzz = MF.fuzzify('age', 29)
        self.assertEqual(fuzz['young'], 1.0)
        self.assertEqual(fuzz['old'], 0.0)
        
        # Maximum age
        fuzz = MF.fuzzify('age', 77)
        self.assertEqual(fuzz['old'], 1.0)
        self.assertEqual(fuzz['young'], 0.0)
    
    def test_oldpeak_membership(self):
        """Test ST depression (oldpeak) membership."""
        # Low oldpeak = 0
        fuzz = MF.fuzzify('oldpeak', 0)
        self.assertEqual(fuzz['low'], 1.0)
        
        # High oldpeak = 6.2
        fuzz = MF.fuzzify('oldpeak', 6.2)
        self.assertEqual(fuzz['high'], 1.0)
    
    def test_ranges_defined(self):
        """Test all ranges are properly defined."""
        expected_vars = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'risk']
        for var in expected_vars:
            self.assertIn(var, MF.RANGES)
            self.assertIn(var, MF.MF_PARAMS)


class TestMembershipShapes(unittest.TestCase):
    """Test the shapes of membership functions."""
    
    def test_triangular_shape(self):
        """Test that MFs have proper triangular shape."""
        universe = MF.get_universe('age')
        young_mf = MF.get_membership('age', 'young', universe)
        
        # Should be all non-negative
        self.assertTrue(np.all(young_mf >= 0))
        
        # Should have max of 1
        self.assertAlmostEqual(np.max(young_mf), 1.0, places=5)
        
        # Should be 0 at some point (not all 1s)
        self.assertTrue(np.any(young_mf == 0))
    
    def test_mf_continuity(self):
        """Test membership function continuity."""
        universe = MF.get_universe('chol')
        mf = MF.get_membership('chol', 'normal', universe)
        
        # Check no sudden jumps (gradient should be reasonable)
        gradient = np.abs(np.diff(mf))
        max_gradient = np.max(gradient)
        self.assertLess(max_gradient, 0.1)  # No jumps > 0.1


if __name__ == '__main__':
    unittest.main()

