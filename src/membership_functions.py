"""
Membership Functions Module

Defines triangular membership functions for all fuzzy variables.
Based on medical literature and dataset analysis.
"""

import numpy as np
import skfuzzy as fuzz


class MembershipFunctions:
    """
    Defines membership functions for heart disease risk assessment.
    
    Input Variables:
        - age: Patient age (29-77 years)
        - trestbps: Resting blood pressure (90-200 mm Hg)
        - chol: Serum cholesterol (120-564 mg/dl)
        - thalach: Maximum heart rate (70-202 bpm)
        - oldpeak: ST depression (0-6.2)
    
    Output Variable:
        - risk: Heart disease risk (0-1)
    """
    
    # Universe ranges based on dataset analysis
    RANGES = {
        'age': (29, 77),
        'trestbps': (90, 200),
        'chol': (120, 564),
        'thalach': (70, 202),
        'oldpeak': (0.0, 6.2),
        'risk': (0.0, 1.0)
    }
    
    # Triangular MF parameters [left, peak, right]
    MF_PARAMS = {
        'age': {
            'young': [29, 29, 45],
            'middle': [35, 50, 65],
            'old': [55, 77, 77]
        },
        'trestbps': {
            'low': [90, 90, 110],
            'normal': [100, 120, 140],
            'high': [130, 200, 200]
        },
        'chol': {
            'low': [120, 120, 180],
            'normal': [160, 200, 240],
            'high': [220, 564, 564]
        },
        'thalach': {
            'low': [70, 70, 120],
            'normal': [100, 150, 180],
            'high': [160, 202, 202]
        },
        'oldpeak': {
            'low': [0, 0, 1.5],
            'medium': [1, 2.5, 4],
            'high': [3, 6.2, 6.2]
        },
        'risk': {
            'low': [0, 0, 0.4],
            'medium': [0.25, 0.5, 0.75],
            'high': [0.6, 1.0, 1.0]
        }
    }
    
    @classmethod
    def get_universe(cls, variable: str, step: float = 1.0) -> np.ndarray:
        """Generate universe of discourse for a variable."""
        low, high = cls.RANGES[variable]
        if variable in ['oldpeak', 'risk']:
            step = 0.01
        return np.arange(low, high + step, step)
    
    @classmethod
    def get_membership(cls, variable: str, term: str, universe: np.ndarray) -> np.ndarray:
        """Calculate membership values for a term."""
        params = cls.MF_PARAMS[variable][term]
        return fuzz.trimf(universe, params)
    
    @classmethod
    def fuzzify(cls, variable: str, value: float) -> dict:
        """
        Fuzzify a crisp input value.
        
        Args:
            variable: Variable name (age, trestbps, etc.)
            value: Crisp input value
            
        Returns:
            Dictionary of {term: membership_degree}
        """
        universe = cls.get_universe(variable)
        result = {}
        
        for term in cls.MF_PARAMS[variable].keys():
            mf = cls.get_membership(variable, term, universe)
            # Interpolate to find membership at exact value
            result[term] = float(fuzz.interp_membership(universe, mf, value))
        
        return result
    
    @classmethod
    def get_all_terms(cls, variable: str) -> list:
        """Get all linguistic terms for a variable."""
        return list(cls.MF_PARAMS[variable].keys())

