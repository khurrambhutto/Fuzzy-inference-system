"""
Unit Tests for Fuzzy Inference

Tests for the FuzzyInference and HeartDiseaseFIS classes.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import unittest
import numpy as np
from fuzzy_system import HeartDiseaseFIS
from inference import FuzzyInference


class TestHeartDiseaseFIS(unittest.TestCase):
    """Test cases for the main FIS."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fis = HeartDiseaseFIS()
    
    def test_predict_returns_valid_range(self):
        """Test that predictions are in valid range [0, 1]."""
        risk = self.fis.predict(
            age=50, trestbps=120, chol=200, thalach=150, oldpeak=1.0
        )
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
    
    def test_high_risk_case(self):
        """Test that high-risk inputs produce high risk output."""
        risk = self.fis.predict(
            age=70,        # Old
            trestbps=180,  # High BP
            chol=400,      # High cholesterol
            thalach=80,    # Low max HR
            oldpeak=5.0    # High ST depression
        )
        self.assertGreater(risk, 0.5)
    
    def test_low_risk_case(self):
        """Test that low-risk inputs produce low risk output."""
        risk = self.fis.predict(
            age=35,        # Young
            trestbps=110,  # Normal BP
            chol=180,      # Normal cholesterol
            thalach=180,   # High max HR
            oldpeak=0.2    # Low ST depression
        )
        self.assertLess(risk, 0.5)
    
    def test_predict_class(self):
        """Test binary classification."""
        # High risk case
        pred = self.fis.predict_class(
            age=70, trestbps=180, chol=400, thalach=80, oldpeak=5.0
        )
        self.assertEqual(pred, 1)
        
        # Low risk case
        pred = self.fis.predict_class(
            age=35, trestbps=110, chol=180, thalach=180, oldpeak=0.2
        )
        self.assertEqual(pred, 0)
    
    def test_risk_label(self):
        """Test risk label generation."""
        self.assertEqual(self.fis.get_risk_label(0.2), "Low")
        self.assertEqual(self.fis.get_risk_label(0.5), "Medium")
        self.assertEqual(self.fis.get_risk_label(0.8), "High")
    
    def test_input_clipping(self):
        """Test that out-of-range inputs are clipped."""
        # Should not raise error with extreme values
        risk = self.fis.predict(
            age=100,       # Beyond max (77)
            trestbps=250,  # Beyond max (200)
            chol=600,      # Beyond max (564)
            thalach=250,   # Beyond max (202)
            oldpeak=10.0   # Beyond max (6.2)
        )
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
    
    def test_boundary_values(self):
        """Test with boundary values."""
        # Minimum values
        risk_min = self.fis.predict(
            age=29, trestbps=90, chol=120, thalach=70, oldpeak=0
        )
        self.assertGreaterEqual(risk_min, 0)
        
        # Maximum values
        risk_max = self.fis.predict(
            age=77, trestbps=200, chol=564, thalach=202, oldpeak=6.2
        )
        self.assertLessEqual(risk_max, 1)


class TestFuzzyInference(unittest.TestCase):
    """Test cases for the step-by-step inference engine."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = FuzzyInference()
        self.sample_inputs = {
            'age': 50,
            'trestbps': 130,
            'chol': 220,
            'thalach': 150,
            'oldpeak': 1.5
        }
    
    def test_fuzzify_all(self):
        """Test fuzzification step."""
        fuzzified = self.engine.fuzzify_all(self.sample_inputs)
        
        # Check all variables are fuzzified
        for var in self.sample_inputs.keys():
            self.assertIn(var, fuzzified)
            # Each variable should have membership values
            self.assertTrue(len(fuzzified[var]) > 0)
    
    def test_evaluate_rules(self):
        """Test rule evaluation step."""
        fuzzified = self.engine.fuzzify_all(self.sample_inputs)
        rule_activations = self.engine.evaluate_rules(fuzzified)
        
        # Should have 15 rules
        self.assertEqual(len(rule_activations), 15)
        
        # Each rule should have activation and consequent
        for rule, (activation, consequent) in rule_activations.items():
            self.assertGreaterEqual(activation, 0)
            self.assertLessEqual(activation, 1)
            self.assertIn(consequent, ['low', 'medium', 'high'])
    
    def test_aggregate(self):
        """Test aggregation step."""
        fuzzified = self.engine.fuzzify_all(self.sample_inputs)
        rule_activations = self.engine.evaluate_rules(fuzzified)
        universe, aggregated = self.engine.aggregate(rule_activations)
        
        # Universe should be valid
        self.assertTrue(len(universe) > 0)
        
        # Aggregated should have same length as universe
        self.assertEqual(len(aggregated), len(universe))
        
        # Values should be in [0, 1]
        self.assertTrue(np.all(aggregated >= 0))
        self.assertTrue(np.all(aggregated <= 1))
    
    def test_defuzzify(self):
        """Test defuzzification step."""
        fuzzified = self.engine.fuzzify_all(self.sample_inputs)
        rule_activations = self.engine.evaluate_rules(fuzzified)
        universe, aggregated = self.engine.aggregate(rule_activations)
        risk = self.engine.defuzzify(universe, aggregated)
        
        self.assertGreaterEqual(risk, 0)
        self.assertLessEqual(risk, 1)
    
    def test_full_inference(self):
        """Test complete inference pipeline."""
        result = self.engine.infer(self.sample_inputs)
        
        # Check all steps are in result
        self.assertIn('fuzzified', result)
        self.assertIn('rule_activations', result)
        self.assertIn('aggregated', result)
        self.assertIn('risk', result)
        self.assertIn('risk_label', result)
        
        # Check risk is valid
        self.assertGreaterEqual(result['risk'], 0)
        self.assertLessEqual(result['risk'], 1)
        
        # Check label is valid
        self.assertIn(result['risk_label'], ['Low', 'Medium', 'High'])
    
    def test_different_defuzz_methods(self):
        """Test different defuzzification methods."""
        methods = ['centroid', 'bisector', 'mom', 'som', 'lom']
        
        for method in methods:
            result = self.engine.infer(self.sample_inputs, method=method)
            self.assertGreaterEqual(result['risk'], 0)
            self.assertLessEqual(result['risk'], 1)


class TestConsistency(unittest.TestCase):
    """Test consistency between FIS and Inference engine."""
    
    def test_fis_inference_consistency(self):
        """Test that FIS and Inference engine give similar results."""
        fis = HeartDiseaseFIS()
        engine = FuzzyInference()
        
        test_cases = [
            {'age': 35, 'trestbps': 110, 'chol': 180, 'thalach': 180, 'oldpeak': 0.2},
            {'age': 50, 'trestbps': 130, 'chol': 220, 'thalach': 150, 'oldpeak': 1.5},
            {'age': 70, 'trestbps': 180, 'chol': 350, 'thalach': 90, 'oldpeak': 4.0},
        ]
        
        for inputs in test_cases:
            fis_risk = fis.predict(**inputs)
            engine_result = engine.infer(inputs)
            engine_risk = engine_result['risk']
            
            # Results should be close (within 0.1)
            self.assertAlmostEqual(fis_risk, engine_risk, delta=0.15)


if __name__ == '__main__':
    unittest.main()

