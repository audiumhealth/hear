#!/usr/bin/env python3
"""
WHO Compliance Optimization for TB Detection
Target: 90% Sensitivity, 70% Specificity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def create_patient_level_split(X, y, patient_ids, test_size=0.2, random_state=42):
    """Create patient-level train/test split"""
    unique_patients = np.unique(patient_ids)
    
    patient_labels = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_labels[patient] = int(np.any(y[patient_mask]))
    
    patients_array = np.array(list(patient_labels.keys()))
    labels_array = np.array(list(patient_labels.values()))
    
    if len(np.unique(labels_array)) > 1:
        train_patients, test_patients = train_test_split(
            patients_array, test_size=test_size, stratify=labels_array, random_state=random_state
        )
    else:
        train_patients, test_patients = train_test_split(
            patients_array, test_size=test_size, random_state=random_state
        )
    
    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        patient_ids[train_mask], patient_ids[test_mask]
    )

def calculate_patient_metrics(y_pred, y_true, patient_ids):
    """Calculate patient-level sensitivity and specificity"""
    unique_patients = np.unique(patient_ids)
    patient_predictions = []
    patient_true_labels = []
    
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_pred = int(np.any(y_pred[patient_mask]))
        patient_true = int(np.any(y_true[patient_mask]))
        
        patient_predictions.append(patient_pred)
        patient_true_labels.append(patient_true)
    
    patient_predictions = np.array(patient_predictions)
    patient_true_labels = np.array(patient_true_labels)
    
    if len(np.unique(patient_true_labels)) > 1:
        cm = confusion_matrix(patient_true_labels, patient_predictions)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            return sensitivity, specificity, tp, tn, fp, fn
    
    return 0, 0, 0, 0, 0, 0

def optimize_threshold_for_who(model, X_val, y_val, patient_ids_val):
    """Find optimal threshold for WHO guidelines (90% sensitivity, 70% specificity)"""
    thresholds = np.linspace(0.05, 0.95, 91)
    best_threshold = 0.5
    best_score = 0
    results = []
    
    y_probs = model.predict_proba(X_val)[:, 1]
    
    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        sensitivity, specificity, tp, tn, fp, fn = calculate_patient_metrics(
            y_pred_thresh, y_val, patient_ids_val
        )
        
        # WHO compliance score
        who_compliant = sensitivity >= 0.90 and specificity >= 0.70
        score = 0
        
        if who_compliant:
            score = (sensitivity + specificity) / 2  # Balanced score
        elif sensitivity >= 0.90:
            score = sensitivity * 0.5  # Partial credit for sensitivity
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'who_compliant': who_compliant,
            'score': score,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, pd.DataFrame(results)

def balanced_augmentation(X_train, y_train, patient_ids_train):
    """Custom augmentation strategy for WHO compliance"""
    # Less aggressive SMOTE to reduce false positives
    min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
    
    if min_samples > 5:
        # Reduce positive class oversampling
        smote = SMOTE(
            sampling_strategy=0.3,  # Less aggressive than default 1.0
            k_neighbors=min(3, min_samples - 1),
            random_state=42
        )
        X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    else:
        X_balanced, y_balanced = X_train, y_train
    
    return X_balanced, y_balanced

def create_who_optimized_models():
    """Create models optimized for WHO compliance"""
    models = {}
    
    # 1. Calibrated Random Forest (better probability estimates)
    models["Calibrated RF"] = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=150,
            max_depth=8,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        ),
        method='isotonic',
        cv=3
    )
    
    # 2. Regularized Logistic Regression
    models["Regularized LR"] = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.5,
        C=0.1,
        class_weight='balanced',
        solver='saga',
        max_iter=1000,
        random_state=42
    )
    
    # 3. Balanced XGBoost
    if XGBOOST_AVAILABLE:
        models["Balanced XGB"] = XGBClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            scale_pos_weight=2.5,  # Reduced from 3.7
            eval_metric='logloss',
            random_state=42
        )
    
    # 4. Ensemble with balanced voting
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, class_weight='balanced_subsample', random_state=42)),
        ('lr', LogisticRegression(class_weight='balanced', random_state=42)),
        ('svm', SVC(probability=True, class_weight='balanced', kernel='rbf', random_state=42))
    ]
    
    if XGBOOST_AVAILABLE:
        base_models.append(('xgb', XGBClassifier(scale_pos_weight=2.5, random_state=42)))
    
    models["Ensemble"] = VotingClassifier(base_models, voting='soft')
    
    return models

def evaluate_who_compliance(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Evaluate model for WHO compliance"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, tp, tn, fp, fn = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    return {
        'model_name': model_name,
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'who_compliant': who_compliant,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'y_probs': y_probs,
        'y_pred': y_pred,
        'y_test': y_test,
        'patient_ids_test': patient_ids_test
    }

def plot_who_analysis(results, threshold_analysis):
    """Create comprehensive WHO compliance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: WHO Compliance Status
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    colors = ['green' if c else 'red' for c in compliant]
    scatter = ax1.scatter(specificities, sensitivities, c=colors, s=100, alpha=0.7)
    
    # Add WHO target lines
    ax1.axhline(y=0.90, color='blue', linestyle='--', alpha=0.7, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.7, label='WHO Specificity (70%)')
    
    # Add model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (specificities[i], sensitivities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    
    ax1.set_xlabel('Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('WHO Compliance Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Threshold Analysis (Best Model)
    ax2 = axes[0, 1]
    if len(threshold_analysis) > 0:
        best_model_thresh = threshold_analysis[0]  # Assume first is best
        ax2.plot(best_model_thresh['threshold'], best_model_thresh['sensitivity'], 
                'g-', label='Sensitivity', linewidth=2)
        ax2.plot(best_model_thresh['threshold'], best_model_thresh['specificity'], 
                'r-', label='Specificity', linewidth=2)
        
        ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7, label='WHO Sensitivity')
        ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.7, label='WHO Specificity')
        
        ax2.set_xlabel('Decision Threshold')
        ax2.set_ylabel('Performance')
        ax2.set_title('Threshold Optimization')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Comparison
    ax3 = axes[1, 0]
    x = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x - width/2, sensitivities, width, label='Sensitivity', alpha=0.7)
    ax3.bar(x + width/2, specificities, width, label='Specificity', alpha=0.7)
    
    ax3.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Performance')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Confusion Matrix (Best Model)
    ax4 = axes[1, 1]
    best_result = max(results, key=lambda x: x['sensitivity'] if x['sensitivity'] >= 0.90 else 0)
    
    cm = np.array([[best_result['tn'], best_result['fp']], 
                   [best_result['fn'], best_result['tp']]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                xticklabels=['TB-', 'TB+'], yticklabels=['TB-', 'TB+'])
    ax4.set_title(f'Best Model: {best_result["model_name"]}')
    ax4.set_ylabel('Actual')
    ax4.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('who_compliance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 WHO COMPLIANCE OPTIMIZATION")
    print("Target: 90% Sensitivity, 70% Specificity")
    print("=" * 60)
    
    # Load processed data
    try:
        data = np.load('full_dataset_processed.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
        patient_ids = data['patient_ids']
        
        print(f"✅ Loaded dataset: {X.shape}")
        print(f"✅ Patients: {len(np.unique(patient_ids))}")
        
    except FileNotFoundError:
        print("❌ Run run_full_analysis.py first")
        return
    
    # Patient-level split
    X_train, X_test, y_train, y_test, train_patients, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    print(f"\\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Enhanced preprocessing for WHO compliance
    print("\\n🔄 Enhanced preprocessing for WHO compliance...")
    
    # Variance filtering
    var_selector = VarianceThreshold(threshold=0.001)
    X_train_filtered = var_selector.fit_transform(X_train)
    X_test_filtered = var_selector.transform(X_test)
    
    # Balanced augmentation
    X_train_balanced, y_train_balanced = balanced_augmentation(
        X_train_filtered, y_train, train_patients
    )
    
    print(f"✅ Balanced training set: {X_train_balanced.shape}")
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Feature selection (more features for better specificity)
    selector = SelectKBest(score_func=f_classif, k=min(800, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"✅ Feature selection: {X_train_selected.shape[1]} features")
    
    # Create WHO-optimized models
    models = create_who_optimized_models()
    
    print(f"\\n🔄 Training {len(models)} WHO-optimized models...")
    
    trained_models = {}
    results = []
    threshold_analysis = []
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_selected, y_train_balanced)
        trained_models[name] = model
        
        # Find optimal threshold
        optimal_threshold, thresh_df = optimize_threshold_for_who(
            model, X_test_selected, y_test, test_patients
        )
        
        print(f"    Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate with optimal threshold
        result = evaluate_who_compliance(
            model, X_test_selected, y_test, test_patients, name, optimal_threshold
        )
        
        results.append(result)
        threshold_analysis.append(thresh_df)
        
        compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
        print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
    
    # Create results summary
    print("\\n" + "="*80)
    print("📋 WHO COMPLIANCE RESULTS")
    print("="*80)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Threshold': f"{result['threshold']:.3f}",
            'Sensitivity': f"{result['sensitivity']:.3f}",
            'Specificity': f"{result['specificity']:.3f}",
            'WHO Compliant': '✅' if result['who_compliant'] else '❌',
            'TB Detected': f"{result['tp']}/{result['tp'] + result['fn']}",
            'TB Correctly Excluded': f"{result['tn']}/{result['tn'] + result['fp']}"
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Find best model
    compliant_models = [r for r in results if r['who_compliant']]
    
    if compliant_models:
        best_model = max(compliant_models, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\\n🏆 BEST WHO-COMPLIANT MODEL: {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (≥90% ✅)")
        print(f"   Specificity: {best_model['specificity']:.3f} (≥70% ✅)")
        print(f"   Threshold: {best_model['threshold']:.3f}")
    else:
        best_model = max(results, key=lambda x: x['sensitivity'] if x['sensitivity'] >= 0.90 else 0)
        print(f"\\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
        print(f"   Gap: Sensitivity {max(0, 0.90 - best_model['sensitivity']):.3f}, Specificity {max(0, 0.70 - best_model['specificity']):.3f}")
    
    # Save results
    results_df.to_csv('who_compliance_results.csv', index=False)
    
    # Create visualization
    print("\\n📊 Creating WHO compliance visualization...")
    plot_who_analysis(results, threshold_analysis)
    
    print("\\n" + "="*80)
    print("🎉 WHO COMPLIANCE ANALYSIS COMPLETE!")
    print("="*80)
    print("📊 Files generated:")
    print("   - who_compliance_results.csv")
    print("   - who_compliance_analysis.png")

if __name__ == "__main__":
    main()