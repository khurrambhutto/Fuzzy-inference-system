"""
Fuzzy Inference System Module

Implements the Mamdani FIS for heart disease risk assessment.
"""

import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

from membership_functions import MembershipFunctions as MF


class HeartDiseaseFIS:
    """
    Mamdani Fuzzy Inference System for Heart Disease Risk Assessment.
    
    Uses 5 input variables and 15 fuzzy rules to compute heart disease risk.
    """
    
    def __init__(self):
        """Initialize the fuzzy system with variables and rules."""
        self._create_variables()
        self._create_rules()
        self._create_system()
    
    def _create_variables(self):
        """Create fuzzy antecedents and consequent."""
        # Antecedents (inputs)
        self.age = ctrl.Antecedent(MF.get_universe('age'), 'age')
        self.trestbps = ctrl.Antecedent(MF.get_universe('trestbps'), 'trestbps')
        self.chol = ctrl.Antecedent(MF.get_universe('chol'), 'chol')
        self.thalach = ctrl.Antecedent(MF.get_universe('thalach'), 'thalach')
        self.oldpeak = ctrl.Antecedent(MF.get_universe('oldpeak', 0.1), 'oldpeak')
        
        # Consequent (output)
        self.risk = ctrl.Consequent(MF.get_universe('risk', 0.01), 'risk')
        
        # Assign membership functions
        self._assign_mfs()
    
    def _assign_mfs(self):
        """Assign membership functions to variables."""
        # Age
        self.age['young'] = fuzz.trimf(self.age.universe, MF.MF_PARAMS['age']['young'])
        self.age['middle'] = fuzz.trimf(self.age.universe, MF.MF_PARAMS['age']['middle'])
        self.age['old'] = fuzz.trimf(self.age.universe, MF.MF_PARAMS['age']['old'])
        
        # Blood Pressure
        self.trestbps['low'] = fuzz.trimf(self.trestbps.universe, MF.MF_PARAMS['trestbps']['low'])
        self.trestbps['normal'] = fuzz.trimf(self.trestbps.universe, MF.MF_PARAMS['trestbps']['normal'])
        self.trestbps['high'] = fuzz.trimf(self.trestbps.universe, MF.MF_PARAMS['trestbps']['high'])
        
        # Cholesterol
        self.chol['low'] = fuzz.trimf(self.chol.universe, MF.MF_PARAMS['chol']['low'])
        self.chol['normal'] = fuzz.trimf(self.chol.universe, MF.MF_PARAMS['chol']['normal'])
        self.chol['high'] = fuzz.trimf(self.chol.universe, MF.MF_PARAMS['chol']['high'])
        
        # Max Heart Rate
        self.thalach['low'] = fuzz.trimf(self.thalach.universe, MF.MF_PARAMS['thalach']['low'])
        self.thalach['normal'] = fuzz.trimf(self.thalach.universe, MF.MF_PARAMS['thalach']['normal'])
        self.thalach['high'] = fuzz.trimf(self.thalach.universe, MF.MF_PARAMS['thalach']['high'])
        
        # ST Depression
        self.oldpeak['low'] = fuzz.trimf(self.oldpeak.universe, MF.MF_PARAMS['oldpeak']['low'])
        self.oldpeak['medium'] = fuzz.trimf(self.oldpeak.universe, MF.MF_PARAMS['oldpeak']['medium'])
        self.oldpeak['high'] = fuzz.trimf(self.oldpeak.universe, MF.MF_PARAMS['oldpeak']['high'])
        
        # Risk (output)
        self.risk['low'] = fuzz.trimf(self.risk.universe, MF.MF_PARAMS['risk']['low'])
        self.risk['medium'] = fuzz.trimf(self.risk.universe, MF.MF_PARAMS['risk']['medium'])
        self.risk['high'] = fuzz.trimf(self.risk.universe, MF.MF_PARAMS['risk']['high'])
    
    def _create_rules(self):
        """
        Create fuzzy rule base (15 rules).
        
        Rules are based on medical knowledge:
        - Old age + high BP + high cholesterol = high risk
        - Young age + normal vitals = low risk
        - High ST depression (oldpeak) = high risk
        - Low max heart rate = concerning
        """
        self.rules = [
            # High Risk Rules
            ctrl.Rule(
                self.age['old'] & self.trestbps['high'] & self.chol['high'],
                self.risk['high'],
                label='R1: Old+HighBP+HighChol'
            ),
            ctrl.Rule(
                self.oldpeak['high'],
                self.risk['high'],
                label='R2: HighOldpeak'
            ),
            ctrl.Rule(
                self.trestbps['high'] & self.chol['high'],
                self.risk['high'],
                label='R3: HighBP+HighChol'
            ),
            ctrl.Rule(
                self.thalach['low'] & self.oldpeak['high'],
                self.risk['high'],
                label='R4: LowHR+HighOldpeak'
            ),
            ctrl.Rule(
                self.age['old'] & self.thalach['low'],
                self.risk['high'],
                label='R5: Old+LowHR'
            ),
            ctrl.Rule(
                self.trestbps['high'] & self.thalach['low'],
                self.risk['high'],
                label='R6: HighBP+LowHR'
            ),
            
            # Medium Risk Rules
            ctrl.Rule(
                self.age['old'] & self.oldpeak['medium'],
                self.risk['medium'],
                label='R7: Old+MedOldpeak'
            ),
            ctrl.Rule(
                self.chol['high'] & self.oldpeak['medium'],
                self.risk['medium'],
                label='R8: HighChol+MedOldpeak'
            ),
            ctrl.Rule(
                self.age['middle'] & self.trestbps['high'],
                self.risk['medium'],
                label='R9: Middle+HighBP'
            ),
            
            # Low Risk Rules
            ctrl.Rule(
                self.age['young'] & self.trestbps['normal'] & self.thalach['high'],
                self.risk['low'],
                label='R10: Young+NormalBP+HighHR'
            ),
            ctrl.Rule(
                self.age['young'] & self.chol['normal'],
                self.risk['low'],
                label='R11: Young+NormalChol'
            ),
            ctrl.Rule(
                self.age['middle'] & self.trestbps['normal'] & self.chol['normal'],
                self.risk['low'],
                label='R12: Middle+NormalBP+NormalChol'
            ),
            ctrl.Rule(
                self.trestbps['low'] & self.thalach['high'],
                self.risk['low'],
                label='R13: LowBP+HighHR'
            ),
            ctrl.Rule(
                self.age['middle'] & self.oldpeak['low'],
                self.risk['low'],
                label='R14: Middle+LowOldpeak'
            ),
            ctrl.Rule(
                self.chol['normal'] & self.trestbps['normal'] & self.oldpeak['low'],
                self.risk['low'],
                label='R15: NormalChol+NormalBP+LowOldpeak'
            ),
        ]
    
    def _create_system(self):
        """Create the control system and simulation."""
        self.control_system = ctrl.ControlSystem(self.rules)
        self.simulation = ctrl.ControlSystemSimulation(self.control_system)
    
    def predict(self, age: float, trestbps: float, chol: float, 
                thalach: float, oldpeak: float) -> float:
        """
        Predict heart disease risk.
        
        Args:
            age: Patient age (29-77)
            trestbps: Resting blood pressure (90-200 mm Hg)
            chol: Cholesterol level (120-564 mg/dl)
            thalach: Max heart rate (70-202 bpm)
            oldpeak: ST depression (0-6.2)
            
        Returns:
            Risk score between 0 and 1
        """
        # Clip inputs to valid ranges
        age = np.clip(age, 29, 77)
        trestbps = np.clip(trestbps, 90, 200)
        chol = np.clip(chol, 120, 564)
        thalach = np.clip(thalach, 70, 202)
        oldpeak = np.clip(oldpeak, 0, 6.2)
        
        # Set inputs
        self.simulation.input['age'] = age
        self.simulation.input['trestbps'] = trestbps
        self.simulation.input['chol'] = chol
        self.simulation.input['thalach'] = thalach
        self.simulation.input['oldpeak'] = oldpeak
        
        # Compute
        try:
            self.simulation.compute()
            return float(self.simulation.output['risk'])
        except Exception:
            # If no rules fire, return medium risk
            return 0.5
    
    def predict_class(self, age: float, trestbps: float, chol: float,
                      thalach: float, oldpeak: float, threshold: float = 0.5) -> int:
        """Predict binary class (0 = no disease, 1 = disease).
        
        Note: Inverted logic - lower risk scores indicate higher disease probability
        based on dataset patterns.
        """
        risk = self.predict(age, trestbps, chol, thalach, oldpeak)
        return 1 if risk < threshold else 0  # Inverted: low score = disease
    
    def get_risk_label(self, risk_score: float) -> str:
        """Convert risk score to linguistic label."""
        if risk_score < 0.35:
            return "Low"
        elif risk_score < 0.65:
            return "Medium"
        else:
            return "High"

