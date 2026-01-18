"""
Fuzzy Inference Module

Implements step-by-step Mamdani inference for educational purposes.
Shows fuzzification, rule evaluation, aggregation, and defuzzification.
"""

import numpy as np
import skfuzzy as fuzz

from membership_functions import MembershipFunctions as MF


class FuzzyInference:
    """
    Step-by-step Mamdani Fuzzy Inference Engine.
    
    This class provides detailed inference steps for educational purposes,
    showing exactly how fuzzy logic processes inputs.
    """
    
    def __init__(self):
        """Initialize inference engine."""
        self.last_fuzzified = {}
        self.last_rule_activations = {}
        self.last_aggregated = None
        self.last_defuzzified = None
    
    def fuzzify_all(self, inputs: dict) -> dict:
        """
        Step 1: Fuzzification
        
        Convert crisp inputs to fuzzy membership degrees.
        
        Args:
            inputs: Dict with keys (age, trestbps, chol, thalach, oldpeak)
            
        Returns:
            Dict of {variable: {term: degree}}
        """
        self.last_fuzzified = {}
        
        for var, value in inputs.items():
            self.last_fuzzified[var] = MF.fuzzify(var, value)
        
        return self.last_fuzzified
    
    def evaluate_rules(self, fuzzified: dict) -> dict:
        """
        Step 2: Rule Evaluation
        
        Apply fuzzy rules using MIN for AND operations.
        
        Args:
            fuzzified: Output from fuzzify_all()
            
        Returns:
            Dict of {rule_name: (activation_strength, consequent_term)}
        """
        self.last_rule_activations = {}
        
        # Rule 1: Old AND HighBP AND HighChol -> High Risk
        activation = min(
            fuzzified['age']['old'],
            fuzzified['trestbps']['high'],
            fuzzified['chol']['high']
        )
        self.last_rule_activations['R1'] = (activation, 'high')
        
        # Rule 2: HighOldpeak -> High Risk
        activation = fuzzified['oldpeak']['high']
        self.last_rule_activations['R2'] = (activation, 'high')
        
        # Rule 3: HighBP AND HighChol -> High Risk
        activation = min(
            fuzzified['trestbps']['high'],
            fuzzified['chol']['high']
        )
        self.last_rule_activations['R3'] = (activation, 'high')
        
        # Rule 4: LowHR AND HighOldpeak -> High Risk
        activation = min(
            fuzzified['thalach']['low'],
            fuzzified['oldpeak']['high']
        )
        self.last_rule_activations['R4'] = (activation, 'high')
        
        # Rule 5: Old AND LowHR -> High Risk
        activation = min(
            fuzzified['age']['old'],
            fuzzified['thalach']['low']
        )
        self.last_rule_activations['R5'] = (activation, 'high')
        
        # Rule 6: HighBP AND LowHR -> High Risk
        activation = min(
            fuzzified['trestbps']['high'],
            fuzzified['thalach']['low']
        )
        self.last_rule_activations['R6'] = (activation, 'high')
        
        # Rule 7: Old AND MedOldpeak -> Medium Risk
        activation = min(
            fuzzified['age']['old'],
            fuzzified['oldpeak']['medium']
        )
        self.last_rule_activations['R7'] = (activation, 'medium')
        
        # Rule 8: HighChol AND MedOldpeak -> Medium Risk
        activation = min(
            fuzzified['chol']['high'],
            fuzzified['oldpeak']['medium']
        )
        self.last_rule_activations['R8'] = (activation, 'medium')
        
        # Rule 9: Middle AND HighBP -> Medium Risk
        activation = min(
            fuzzified['age']['middle'],
            fuzzified['trestbps']['high']
        )
        self.last_rule_activations['R9'] = (activation, 'medium')
        
        # Rule 10: Young AND NormalBP AND HighHR -> Low Risk
        activation = min(
            fuzzified['age']['young'],
            fuzzified['trestbps']['normal'],
            fuzzified['thalach']['high']
        )
        self.last_rule_activations['R10'] = (activation, 'low')
        
        # Rule 11: Young AND NormalChol -> Low Risk
        activation = min(
            fuzzified['age']['young'],
            fuzzified['chol']['normal']
        )
        self.last_rule_activations['R11'] = (activation, 'low')
        
        # Rule 12: Middle AND NormalBP AND NormalChol -> Low Risk
        activation = min(
            fuzzified['age']['middle'],
            fuzzified['trestbps']['normal'],
            fuzzified['chol']['normal']
        )
        self.last_rule_activations['R12'] = (activation, 'low')
        
        # Rule 13: LowBP AND HighHR -> Low Risk
        activation = min(
            fuzzified['trestbps']['low'],
            fuzzified['thalach']['high']
        )
        self.last_rule_activations['R13'] = (activation, 'low')
        
        # Rule 14: Middle AND LowOldpeak -> Low Risk
        activation = min(
            fuzzified['age']['middle'],
            fuzzified['oldpeak']['low']
        )
        self.last_rule_activations['R14'] = (activation, 'low')
        
        # Rule 15: NormalChol AND NormalBP AND LowOldpeak -> Low Risk
        activation = min(
            fuzzified['chol']['normal'],
            fuzzified['trestbps']['normal'],
            fuzzified['oldpeak']['low']
        )
        self.last_rule_activations['R15'] = (activation, 'low')
        
        return self.last_rule_activations
    
    def aggregate(self, rule_activations: dict) -> tuple:
        """
        Step 3: Aggregation
        
        Combine rule outputs using MAX aggregation.
        
        Args:
            rule_activations: Output from evaluate_rules()
            
        Returns:
            Tuple of (universe, aggregated_output)
        """
        universe = MF.get_universe('risk', 0.01)
        
        # Get membership functions for output
        low_mf = MF.get_membership('risk', 'low', universe)
        medium_mf = MF.get_membership('risk', 'medium', universe)
        high_mf = MF.get_membership('risk', 'high', universe)
        
        # Find max activation for each output term
        low_activation = max(
            [act for act, term in rule_activations.values() if term == 'low'],
            default=0
        )
        medium_activation = max(
            [act for act, term in rule_activations.values() if term == 'medium'],
            default=0
        )
        high_activation = max(
            [act for act, term in rule_activations.values() if term == 'high'],
            default=0
        )
        
        # Clip membership functions by activation level
        low_clipped = np.fmin(low_activation, low_mf)
        medium_clipped = np.fmin(medium_activation, medium_mf)
        high_clipped = np.fmin(high_activation, high_mf)
        
        # Aggregate using MAX
        aggregated = np.fmax(low_clipped, np.fmax(medium_clipped, high_clipped))
        
        self.last_aggregated = (universe, aggregated)
        return self.last_aggregated
    
    def defuzzify(self, universe: np.ndarray, aggregated: np.ndarray, 
                  method: str = 'centroid') -> float:
        """
        Step 4: Defuzzification
        
        Convert fuzzy output to crisp value.
        
        Args:
            universe: Output universe
            aggregated: Aggregated fuzzy output
            method: Defuzzification method ('centroid', 'bisector', 'mom', 'som', 'lom')
            
        Returns:
            Crisp output value
        """
        if np.sum(aggregated) == 0:
            # No rules fired
            self.last_defuzzified = 0.5
            return 0.5
        
        self.last_defuzzified = fuzz.defuzz(universe, aggregated, method)
        return self.last_defuzzified
    
    def infer(self, inputs: dict, method: str = 'centroid') -> dict:
        """
        Complete inference pipeline.
        
        Args:
            inputs: Dict with (age, trestbps, chol, thalach, oldpeak)
            method: Defuzzification method
            
        Returns:
            Dict with all inference steps and final result
        """
        # Step 1: Fuzzification
        fuzzified = self.fuzzify_all(inputs)
        
        # Step 2: Rule Evaluation
        rule_activations = self.evaluate_rules(fuzzified)
        
        # Step 3: Aggregation
        universe, aggregated = self.aggregate(rule_activations)
        
        # Step 4: Defuzzification
        risk = self.defuzzify(universe, aggregated, method)
        
        return {
            'fuzzified': fuzzified,
            'rule_activations': rule_activations,
            'aggregated': (universe, aggregated),
            'risk': risk,
            'risk_label': self._get_label(risk)
        }
    
    def _get_label(self, risk: float) -> str:
        """Convert risk score to label."""
        if risk < 0.35:
            return "Low"
        elif risk < 0.65:
            return "Medium"
        else:
            return "High"

