"""
Experiment Runner

Runs all experiments for the fuzzy inference system:
1. Model evaluation on test set
2. Comparison with crisp baseline
3. Sensitivity analysis on MF parameters
4. Rule ablation study

Usage:
    python experiments/run_experiments.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from fuzzy_system import HeartDiseaseFIS
from membership_functions import MembershipFunctions as MF
from utils import (
    load_data, preprocess_data, split_data, 
    evaluate_model, evaluate_with_threshold,
    find_best_threshold, crisp_baseline, calculate_mae
)


def run_model_evaluation(fis, X_test, y_test):
    """Evaluate model on test set."""
    print("\n" + "="*60)
    print("EXPERIMENT 1: Model Evaluation")
    print("="*60)
    
    # Get predictions
    risk_scores = []
    for _, row in X_test.iterrows():
        risk = fis.predict(
            age=row['age'],
            trestbps=row['trestbps'],
            chol=row['chol'],
            thalach=row['thalach'],
            oldpeak=row['oldpeak']
        )
        risk_scores.append(risk)
    
    # Evaluate with default threshold (0.5)
    results_05 = evaluate_with_threshold(y_test.values, risk_scores, 0.5)
    print("\nWith threshold = 0.5:")
    print(f"  Accuracy:  {results_05['accuracy']:.4f}")
    print(f"  Precision: {results_05['precision']:.4f}")
    print(f"  Recall:    {results_05['recall']:.4f}")
    print(f"  F1 Score:  {results_05['f1']:.4f}")
    
    # Find best threshold
    best_threshold, best_f1 = find_best_threshold(y_test.values, risk_scores)
    results_best = evaluate_with_threshold(y_test.values, risk_scores, best_threshold)
    print(f"\nWith optimal threshold = {best_threshold:.2f}:")
    print(f"  Accuracy:  {results_best['accuracy']:.4f}")
    print(f"  Precision: {results_best['precision']:.4f}")
    print(f"  Recall:    {results_best['recall']:.4f}")
    print(f"  F1 Score:  {results_best['f1']:.4f}")
    
    # Calculate MAE
    mae = calculate_mae(y_test.values, risk_scores)
    print(f"\nMean Absolute Error: {mae:.4f}")
    
    return {
        'threshold_05': results_05,
        'threshold_best': results_best,
        'best_threshold': best_threshold,
        'mae': mae,
        'risk_scores': risk_scores
    }


def run_baseline_comparison(X_test, y_test, fis_results):
    """Compare with crisp rule baseline."""
    print("\n" + "="*60)
    print("EXPERIMENT 2: Baseline Comparison")
    print("="*60)
    
    # Get baseline predictions
    baseline_preds = crisp_baseline(X_test)
    baseline_results = evaluate_model(y_test.values, baseline_preds)
    
    print("\nCrisp Rule Baseline:")
    print(f"  Accuracy:  {baseline_results['accuracy']:.4f}")
    print(f"  Precision: {baseline_results['precision']:.4f}")
    print(f"  Recall:    {baseline_results['recall']:.4f}")
    print(f"  F1 Score:  {baseline_results['f1']:.4f}")
    
    # Compare
    print("\nComparison (Fuzzy FIS vs Baseline):")
    fis_acc = fis_results['threshold_best']['accuracy']
    base_acc = baseline_results['accuracy']
    diff = fis_acc - base_acc
    print(f"  Accuracy Difference: {diff:+.4f} ({'Better' if diff > 0 else 'Worse'})")
    
    fis_f1 = fis_results['threshold_best']['f1']
    base_f1 = baseline_results['f1']
    diff_f1 = fis_f1 - base_f1
    print(f"  F1 Difference: {diff_f1:+.4f} ({'Better' if diff_f1 > 0 else 'Worse'})")
    
    return baseline_results


def run_sensitivity_analysis(X_test, y_test, original_f1):
    """Sensitivity analysis on MF parameters."""
    print("\n" + "="*60)
    print("EXPERIMENT 3: Sensitivity Analysis")
    print("="*60)
    
    # Test variations in age MF parameters
    variations = [-0.2, -0.1, 0, 0.1, 0.2]  # ±10%, ±20%
    results = []
    
    print("\nVarying age MF boundaries by ±10%, ±20%:")
    print("-" * 40)
    
    for var in variations:
        # Modify MF parameters temporarily
        original_params = MF.MF_PARAMS['age'].copy()
        
        # Adjust boundaries
        for term in MF.MF_PARAMS['age']:
            params = MF.MF_PARAMS['age'][term]
            # Scale the spread of the MF
            center = params[1]
            new_params = [
                max(29, params[0] + (params[0] - center) * var),
                center,
                min(77, params[2] + (params[2] - center) * var)
            ]
            MF.MF_PARAMS['age'][term] = new_params
        
        # Create new FIS with modified parameters
        fis = HeartDiseaseFIS()
        
        # Evaluate
        risk_scores = []
        for _, row in X_test.iterrows():
            risk = fis.predict(
                age=row['age'],
                trestbps=row['trestbps'],
                chol=row['chol'],
                thalach=row['thalach'],
                oldpeak=row['oldpeak']
            )
            risk_scores.append(risk)
        
        best_threshold, _ = find_best_threshold(y_test.values, risk_scores)
        eval_results = evaluate_with_threshold(y_test.values, risk_scores, best_threshold)
        
        print(f"  Variation {var*100:+.0f}%: F1 = {eval_results['f1']:.4f} "
              f"(Δ = {eval_results['f1'] - original_f1:+.4f})")
        
        results.append({
            'variation': var,
            'f1': eval_results['f1'],
            'accuracy': eval_results['accuracy']
        })
        
        # Restore original parameters
        MF.MF_PARAMS['age'] = original_params
    
    return results


def run_rule_ablation(X_test, y_test):
    """Rule ablation study - remove rules one by one."""
    print("\n" + "="*60)
    print("EXPERIMENT 4: Rule Ablation Study")
    print("="*60)
    
    # Get baseline with all rules
    fis = HeartDiseaseFIS()
    
    risk_scores = []
    for _, row in X_test.iterrows():
        risk = fis.predict(
            age=row['age'],
            trestbps=row['trestbps'],
            chol=row['chol'],
            thalach=row['thalach'],
            oldpeak=row['oldpeak']
        )
        risk_scores.append(risk)
    
    best_threshold, _ = find_best_threshold(y_test.values, risk_scores)
    baseline = evaluate_with_threshold(y_test.values, risk_scores, best_threshold)
    baseline_f1 = baseline['f1']
    
    print(f"\nBaseline F1 (all 15 rules): {baseline_f1:.4f}")
    print("\nRule importance (F1 drop when removed):")
    print("-" * 40)
    
    # Test removing each rule type
    rule_categories = {
        'High Risk Rules (R1-R6)': [0, 1, 2, 3, 4, 5],
        'Medium Risk Rules (R7-R9)': [6, 7, 8],
        'Low Risk Rules (R10-R15)': [9, 10, 11, 12, 13, 14]
    }
    
    for category, indices in rule_categories.items():
        # Create FIS without these rules
        fis_ablated = HeartDiseaseFIS()
        remaining_rules = [r for i, r in enumerate(fis_ablated.rules) if i not in indices]
        
        if len(remaining_rules) > 0:
            from skfuzzy import control as ctrl
            fis_ablated.control_system = ctrl.ControlSystem(remaining_rules)
            fis_ablated.simulation = ctrl.ControlSystemSimulation(fis_ablated.control_system)
            
            risk_scores_ablated = []
            for _, row in X_test.iterrows():
                try:
                    risk = fis_ablated.predict(
                        age=row['age'],
                        trestbps=row['trestbps'],
                        chol=row['chol'],
                        thalach=row['thalach'],
                        oldpeak=row['oldpeak']
                    )
                except:
                    risk = 0.5
                risk_scores_ablated.append(risk)
            
            eval_ablated = evaluate_with_threshold(y_test.values, risk_scores_ablated, best_threshold)
            f1_drop = baseline_f1 - eval_ablated['f1']
            print(f"  Without {category}: F1 = {eval_ablated['f1']:.4f} (Δ = {-f1_drop:+.4f})")


def generate_report(results_dir):
    """Generate summary report."""
    print("\n" + "="*60)
    print("GENERATING REPORT")
    print("="*60)
    
    report_path = results_dir / 'experiment_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("Heart Disease FIS - Experiment Results\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*50 + "\n\n")
        f.write("See console output for detailed results.\n")
        f.write("Figures saved in results/figures/\n")
    
    print(f"Report saved to: {report_path}")


def plot_results(fis_results, y_test, results_dir):
    """Generate visualization plots."""
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)
    
    # 1. Risk score distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    risk_scores = np.array(fis_results['risk_scores'])
    y_test_arr = y_test.values
    
    # Distribution by class
    axes[0].hist(risk_scores[y_test_arr == 0], bins=20, alpha=0.7, label='No Disease', color='green')
    axes[0].hist(risk_scores[y_test_arr == 1], bins=20, alpha=0.7, label='Disease', color='red')
    axes[0].axvline(x=fis_results['best_threshold'], color='black', linestyle='--', label=f'Threshold ({fis_results["best_threshold"]:.2f})')
    axes[0].set_xlabel('Risk Score')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Risk Score Distribution by Class')
    axes[0].legend()
    
    # Confusion matrix
    cm = fis_results['threshold_best']['confusion_matrix']
    im = axes[1].imshow(cm, cmap='Blues')
    axes[1].set_xticks([0, 1])
    axes[1].set_yticks([0, 1])
    axes[1].set_xticklabels(['No Disease', 'Disease'])
    axes[1].set_yticklabels(['No Disease', 'Disease'])
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    axes[1].set_title('Confusion Matrix')
    
    for i in range(2):
        for j in range(2):
            axes[1].text(j, i, cm[i, j], ha='center', va='center', fontsize=16)
    
    plt.colorbar(im, ax=axes[1])
    plt.tight_layout()
    plt.savefig(figures_dir / 'evaluation_results.png', dpi=150)
    plt.close()
    
    print(f"Figures saved to: {figures_dir}")


def main():
    """Run all experiments."""
    print("\n" + "="*60)
    print("HEART DISEASE FIS - EXPERIMENT SUITE")
    print("="*60)
    
    # Setup
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    print(f"Dataset: {len(df)} samples")
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Positive class ratio: {y.mean():.2%}")
    
    # Create FIS
    fis = HeartDiseaseFIS()
    
    # Run experiments
    fis_results = run_model_evaluation(fis, X_test, y_test)
    baseline_results = run_baseline_comparison(X_test, y_test, fis_results)
    sensitivity_results = run_sensitivity_analysis(X_test, y_test, fis_results['threshold_best']['f1'])
    run_rule_ablation(X_test, y_test)
    
    # Generate outputs
    plot_results(fis_results, y_test, results_dir)
    generate_report(results_dir)
    
    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*60)


if __name__ == '__main__':
    main()

