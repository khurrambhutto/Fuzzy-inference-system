# Heart Disease FIS - Codebase Flow Documentation

A complete walkthrough of how data flows through the Fuzzy Inference System.

---

## 📁 Project Structure Overview

```
heart-fuzzy-fis/
├── data/
│   └── heart.csv              # Dataset (1026 patient records)
├── src/
│   ├── __init__.py            # Package exports
│   ├── membership_functions.py # Fuzzy set definitions
│   ├── fuzzy_system.py        # Main FIS with rules
│   ├── inference.py           # Step-by-step Mamdani engine
│   └── utils.py               # Data loading & evaluation
├── app/
│   └── streamlit_app.py       # Interactive UI
├── tests/
│   ├── test_membership.py     # MF unit tests
│   └── test_inference.py      # FIS unit tests
├── experiments/
│   └── run_experiments.py     # Evaluation scripts
└── results/
    └── figures/               # Generated plots
```

---

## 🔄 Complete Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│  [User Input via UI Sliders]     OR     [heart.csv for experiments]        │
│        ↓                                         ↓                          │
│  streamlit_app.py                         utils.py → load_data()            │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        FUZZY INFERENCE SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 1: FUZZIFICATION (membership_functions.py)                     │   │
│  │  ─────────────────────────────────────────────────────────────────── │   │
│  │  Input: Crisp values (age=55, bp=140, chol=250, hr=130, oldpeak=2)  │   │
│  │  Output: Fuzzy degrees for each linguistic term                      │   │
│  │                                                                       │   │
│  │  age=55 → {young: 0.0, middle: 0.67, old: 0.0}                       │   │
│  │  bp=140 → {low: 0.0, normal: 0.0, high: 1.0}                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 2: RULE EVALUATION (fuzzy_system.py / inference.py)            │   │
│  │  ─────────────────────────────────────────────────────────────────── │   │
│  │  Apply 15 IF-THEN rules using MIN operator for AND                   │   │
│  │                                                                       │   │
│  │  Rule 9: IF middle AND high_bp → medium_risk                         │   │
│  │          activation = min(0.67, 1.0) = 0.67                          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 3: AGGREGATION (inference.py)                                  │   │
│  │  ─────────────────────────────────────────────────────────────────── │   │
│  │  Combine rule outputs using MAX operator                             │   │
│  │                                                                       │   │
│  │  low_risk_activation = max(R10..R15) = 0.2                           │   │
│  │  medium_risk_activation = max(R7..R9) = 0.67                         │   │
│  │  high_risk_activation = max(R1..R6) = 0.75                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  STEP 4: DEFUZZIFICATION (inference.py)                              │   │
│  │  ─────────────────────────────────────────────────────────────────── │   │
│  │  Convert aggregated fuzzy output to crisp value                      │   │
│  │  Method: Centroid (center of gravity)                                │   │
│  │                                                                       │   │
│  │  Output: risk_score = 0.72 (72%)                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              OUTPUT                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  [UI Display]                    OR     [Evaluation Metrics]                │
│  - Risk gauge (72%)                     - Accuracy, F1, Precision           │
│  - Risk label ("High")                  - Confusion matrix                  │
│  - Active rules display                 - Comparison with baseline          │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📄 File-by-File Breakdown

---

### 1. `src/membership_functions.py`

**Purpose:** Defines fuzzy sets and fuzzification logic.

**Key Class:** `MembershipFunctions`

| Method | Description |
|--------|-------------|
| `RANGES` | Dictionary of valid input ranges for each variable |
| `MF_PARAMS` | Triangular MF parameters [left, peak, right] |
| `get_universe()` | Generate array of values for a variable |
| `get_membership()` | Calculate membership values for a term |
| `fuzzify()` | Convert crisp value → fuzzy degrees |

**Linguistic Variables:**

| Variable | Terms | Range |
|----------|-------|-------|
| Age | young, middle, old | 29-77 years |
| Blood Pressure | low, normal, high | 90-200 mm Hg |
| Cholesterol | low, normal, high | 120-564 mg/dl |
| Max Heart Rate | low, normal, high | 70-202 bpm |
| ST Depression | low, medium, high | 0-6.2 |
| Risk (output) | low, medium, high | 0-1 |

**Example Flow:**
```python
MF.fuzzify('age', 50)
# Returns: {'young': 0.0, 'middle': 1.0, 'old': 0.0}
```

---

### 2. `src/fuzzy_system.py`

**Purpose:** Main Mamdani FIS with all 15 rules using scikit-fuzzy.

**Key Class:** `HeartDiseaseFIS`

| Method | Description |
|--------|-------------|
| `__init__()` | Creates variables, rules, and control system |
| `_create_variables()` | Define 5 antecedents + 1 consequent |
| `_create_rules()` | Define 15 fuzzy IF-THEN rules |
| `predict()` | Returns risk score (0-1) |
| `predict_class()` | Returns binary classification (0 or 1) |
| `get_risk_label()` | Returns "Low", "Medium", or "High" |

**The 15 Fuzzy Rules:**

```
HIGH RISK RULES (R1-R6):
─────────────────────────────────────────────────────
R1:  IF age=Old AND bp=High AND chol=High    → High
R2:  IF oldpeak=High                          → High
R3:  IF bp=High AND chol=High                 → High
R4:  IF hr=Low AND oldpeak=High               → High
R5:  IF age=Old AND hr=Low                    → High
R6:  IF bp=High AND hr=Low                    → High

MEDIUM RISK RULES (R7-R9):
─────────────────────────────────────────────────────
R7:  IF age=Old AND oldpeak=Medium            → Medium
R8:  IF chol=High AND oldpeak=Medium          → Medium
R9:  IF age=Middle AND bp=High                → Medium

LOW RISK RULES (R10-R15):
─────────────────────────────────────────────────────
R10: IF age=Young AND bp=Normal AND hr=High   → Low
R11: IF age=Young AND chol=Normal             → Low
R12: IF age=Middle AND bp=Normal AND chol=Normal → Low
R13: IF bp=Low AND hr=High                    → Low
R14: IF age=Middle AND oldpeak=Low            → Low
R15: IF chol=Normal AND bp=Normal AND oldpeak=Low → Low
```

**Example Usage:**
```python
fis = HeartDiseaseFIS()
risk = fis.predict(age=55, trestbps=140, chol=250, thalach=130, oldpeak=2.0)
# Returns: 0.72
```

---

### 3. `src/inference.py`

**Purpose:** Step-by-step Mamdani inference for educational/debugging purposes.

**Key Class:** `FuzzyInference`

| Method | Description |
|--------|-------------|
| `fuzzify_all()` | Step 1: Convert all inputs to fuzzy degrees |
| `evaluate_rules()` | Step 2: Apply rules with MIN operator |
| `aggregate()` | Step 3: Combine outputs with MAX operator |
| `defuzzify()` | Step 4: Centroid defuzzification |
| `infer()` | Complete pipeline returning all steps |

**Example Usage:**
```python
engine = FuzzyInference()
result = engine.infer({
    'age': 55, 'trestbps': 140, 'chol': 250, 
    'thalach': 130, 'oldpeak': 2.0
})

print(result['fuzzified'])      # Step 1 output
print(result['rule_activations']) # Step 2 output
print(result['risk'])           # Final risk score
print(result['risk_label'])     # "High"
```

---

### 4. `src/utils.py`

**Purpose:** Data loading, preprocessing, and evaluation utilities.

**Key Functions:**

| Function | Description |
|----------|-------------|
| `load_data()` | Load heart.csv dataset |
| `preprocess_data()` | Extract 5 features + target |
| `split_data()` | Train/test split (80/20) |
| `evaluate_model()` | Calculate accuracy, precision, recall, F1 |
| `find_best_threshold()` | Optimize classification threshold |
| `crisp_baseline()` | Simple rule-based baseline for comparison |
| `calculate_mae()` | Mean Absolute Error |

**Example Usage:**
```python
df = load_data()
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)
metrics = evaluate_model(y_test, predictions)
```

---

### 5. `app/streamlit_app.py`

**Purpose:** Interactive web UI for the fuzzy system.

**Components:**

| Section | Description |
|---------|-------------|
| Page Config | Title, icon, layout settings |
| Custom CSS | Dark theme styling |
| Sidebar | 5 input sliders for patient parameters |
| Results Tab | Risk gauge + input summary table |
| Inference Tab | Step-by-step fuzzy logic visualization |
| Visualizations Tab | Membership function plots |
| Risk Card | Color-coded risk display (green/yellow/red) |

**How It Works:**
```
User moves sliders → Values captured → FIS.predict() called → 
Risk score calculated → UI updated with gauge, label, active rules
```

**To Run:**
```bash
streamlit run app/streamlit_app.py
```

---

### 6. `tests/test_membership.py`

**Purpose:** Unit tests for membership functions.

**Test Cases:**

| Test | Description |
|------|-------------|
| `test_universe_generation` | Verify universe ranges are correct |
| `test_membership_values_at_peaks` | MF = 1.0 at peak values |
| `test_membership_values_at_boundaries` | MF transitions at edges |
| `test_fuzzify_returns_all_terms` | All terms present in output |
| `test_edge_values` | Extreme input values handled |
| `test_triangular_shape` | MF shape is valid triangular |

**To Run:**
```bash
python -m pytest tests/test_membership.py -v
```

---

### 7. `tests/test_inference.py`

**Purpose:** Unit tests for FIS and inference engine.

**Test Cases:**

| Test | Description |
|------|-------------|
| `test_predict_returns_valid_range` | Output in [0, 1] |
| `test_high_risk_case` | High-risk inputs → high output |
| `test_low_risk_case` | Low-risk inputs → low output |
| `test_predict_class` | Binary classification works |
| `test_full_inference` | All 4 steps execute correctly |
| `test_fis_inference_consistency` | FIS and engine give same results |

**To Run:**
```bash
python -m pytest tests/test_inference.py -v
```

---

### 8. `experiments/run_experiments.py`

**Purpose:** Reproducible experiments for model evaluation.

**Experiments:**

| Experiment | Description |
|------------|-------------|
| Model Evaluation | Accuracy, F1, confusion matrix on test set |
| Baseline Comparison | Fuzzy FIS vs crisp rule baseline |
| Sensitivity Analysis | Impact of ±10%, ±20% MF parameter changes |
| Rule Ablation | Importance of each rule category |

**To Run:**
```bash
python experiments/run_experiments.py
```

**Outputs:**
- Console: Detailed metrics
- `results/figures/`: Evaluation plots
- `results/experiment_report.txt`: Summary report

---

## 🔗 Module Dependencies

```
┌─────────────────────────────────────────────────────────────────┐
│                    streamlit_app.py                              │
│                          │                                       │
│           ┌──────────────┼──────────────┐                       │
│           ▼              ▼              ▼                       │
│    fuzzy_system.py  inference.py  membership_functions.py       │
│           │              │              │                       │
│           └──────────────┼──────────────┘                       │
│                          ▼                                       │
│               membership_functions.py                            │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                   run_experiments.py                             │
│                          │                                       │
│        ┌─────────────────┼─────────────────┐                    │
│        ▼                 ▼                 ▼                    │
│   fuzzy_system.py   utils.py   membership_functions.py          │
│        │                 │                                       │
│        ▼                 ▼                                       │
│   membership_      data/heart.csv                                │
│   functions.py                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📊 Data Flow Summary

| Stage | Input | Process | Output |
|-------|-------|---------|--------|
| 1. Input | Slider values / CSV row | Capture 5 features | Crisp values |
| 2. Fuzzify | Crisp values | Triangular MF evaluation | Membership degrees |
| 3. Rules | Membership degrees | MIN operator (AND) | Rule activations |
| 4. Aggregate | Rule activations | MAX operator | Combined fuzzy output |
| 5. Defuzzify | Fuzzy output | Centroid method | Risk score (0-1) |
| 6. Display | Risk score | Threshold + labeling | UI visualization |

---

## 🚀 Quick Commands

```bash
# Run UI
streamlit run app/streamlit_app.py

# Run tests
python -m pytest tests/ -v

# Run experiments
python experiments/run_experiments.py

# Run specific test file
python -m pytest tests/test_membership.py -v
```

