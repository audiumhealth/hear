#!/usr/bin/env python3
"""
WHO Compliance Optimization - FINAL VERSION
Target: 90% Sensitivity, 70% Specificity
Robust implementation with proven techniques
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc, roc_auc_score, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.neural_network import MLPClassifier
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
    """Create patient-level train/test split with stratification"""
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
    """Calculate patient-level metrics"""
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
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            return sensitivity, specificity, precision, npv, tp, tn, fp, fn, patient_true_labels, patient_predictions
    
    return 0, 0, 0, 0, 0, 0, 0, 0, patient_true_labels, patient_predictions

def optimized_preprocessing(X_train, y_train, X_test):
    """Optimized preprocessing for WHO compliance"""
    print("  🔄 Optimized preprocessing...")
    
    # 1. Variance filtering
    var_selector = VarianceThreshold(threshold=0.01)
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    print(f"    After variance filtering: {X_train_var.shape[1]} features")
    
    # 2. Balanced sampling with conservative approach
    if len(np.unique(y_train)) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        if min_samples > 10:
            # Conservative SMOTE to avoid overfitting
            smote = SMOTE(
                sampling_strategy=0.3,  # Conservative oversampling
                k_neighbors=min(5, min_samples - 1),
                random_state=42
            )
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_var, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_var, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_var, y_train
    
    print(f"    After balanced sampling: {X_train_balanced.shape[0]} samples")
    
    # 3. Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_var)
    
    # 4. Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(800, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"    Final features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train_balanced, scaler, var_selector, selector

def create_who_targeted_models():
    """Create models specifically targeted for WHO compliance"""
    models = {}
    
    # Model 1: Proven WHO compliant model (from V1)
    models["WHO Compliant LR"] = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.7,
        C=0.01,
        class_weight={0: 1, 1: 3},
        solver='saga',
        max_iter=2000,
        random_state=42
    )
    
    # Model 2: Highly regularized alternatives
    models["Ridge LR"] = LogisticRegression(
        penalty='l2',
        C=0.01,
        class_weight={0: 1, 1: 2.5},
        solver='liblinear',
        random_state=42
    )
    
    # Model 3: Conservative Random Forest
    models["Conservative RF"] = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features=0.3,
        class_weight={0: 1, 1: 2},
        random_state=42,
        n_jobs=-1
    )
    
    # Model 4: Regularized XGBoost
    if XGBOOST_AVAILABLE:
        models["Conservative XGB"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=0.5,
            scale_pos_weight=2.0,
            eval_metric='logloss',
            random_state=42
        )
    
    # Model 5: Calibrated ensemble
    base_models = [
        ('lr1', LogisticRegression(C=0.01, class_weight={0: 1, 1: 3}, random_state=42)),
        ('lr2', LogisticRegression(C=0.1, class_weight={0: 1, 1: 2}, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, class_weight={0: 1, 1: 2}, random_state=42))
    ]
    
    models["Calibrated Ensemble"] = CalibratedClassifierCV(
        VotingClassifier(base_models, voting='soft'),
        method='isotonic',
        cv=3
    )
    
    # Model 6: SVM with specific parameters
    models["Tuned SVM"] = SVC(
        kernel='rbf',
        C=0.1,
        gamma='scale',
        class_weight={0: 1, 1: 2},
        probability=True,
        random_state=42
    )
    
    return models

def comprehensive_threshold_optimization(model, X_val, y_val, patient_ids_val):
    """Comprehensive threshold optimization focusing on WHO compliance"""
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    y_probs = model.predict_proba(X_val)[:, 1]
    
    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        
        sensitivity, specificity, precision, npv, tp, tn, fp, fn, patient_true, patient_pred = calculate_patient_metrics(
            y_pred_thresh, y_val, patient_ids_val
        )
        
        # WHO compliance
        who_compliant = sensitivity >= 0.90 and specificity >= 0.70
        
        # Scoring with WHO prioritization
        if who_compliant:
            score = 1000 + (sensitivity + specificity)  # High bonus for compliance
        elif sensitivity >= 0.90:
            score = 500 + specificity * 100  # Reward specificity if sensitivity met
        elif specificity >= 0.70:
            score = 100 + sensitivity * 100  # Reward sensitivity if specificity met
        else:
            score = sensitivity * 50 + specificity * 50  # Base score
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'who_compliant': who_compliant,
            'score': score,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['score'])
    
    return best_result['threshold'], pd.DataFrame(results)

def evaluate_final_model(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Final evaluation with comprehensive metrics"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, precision, npv, tp, tn, fp, fn, patient_true, patient_pred = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # Additional metrics
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # ROC-AUC and PR-AUC
    unique_patients = np.unique(patient_ids_test)
    patient_probs = []
    
    for patient in unique_patients:
        patient_mask = patient_ids_test == patient
        patient_prob = np.max(y_probs[patient_mask])
        patient_probs.append(patient_prob)
    
    patient_probs = np.array(patient_probs)
    
    if len(np.unique(patient_true)) > 1:
        roc_auc = roc_auc_score(patient_true, patient_probs)
        pr_auc = average_precision_score(patient_true, patient_probs)
    else:
        roc_auc = pr_auc = 0
    
    return {
        'model_name': model_name,
        'threshold': threshold,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'npv': npv,
        'f1_score': f1_score,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'who_compliant': who_compliant,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'patient_true': patient_true,
        'patient_probs': patient_probs
    }

def plot_final_who_analysis(results, threshold_analyses):
    """Create final WHO compliance visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: WHO Compliance Status
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    colors = ['green' if c else 'red' for c in compliant]
    scatter = ax1.scatter(specificities, sensitivities, c=colors, s=150, alpha=0.7)
    
    # WHO compliance zone
    ax1.axhline(y=0.90, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity (70%)')
    ax1.fill_between([0.70, 1.0], 0.90, 1.0, alpha=0.1, color='green', label='WHO Compliant Zone')
    
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
    
    # Plot 2: Threshold optimization for best model
    ax2 = axes[0, 1]
    best_idx = np.argmax([r['sensitivity'] * r['specificity'] for r in results])
    best_thresh_df = threshold_analyses[best_idx]
    
    ax2.plot(best_thresh_df['threshold'], best_thresh_df['sensitivity'], 
             'g-', label='Sensitivity', linewidth=2)
    ax2.plot(best_thresh_df['threshold'], best_thresh_df['specificity'], 
             'r-', label='Specificity', linewidth=2)
    
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    # Mark optimal threshold
    optimal_thresh = results[best_idx]['threshold']
    ax2.axvline(x=optimal_thresh, color='black', linestyle=':', alpha=0.8, 
                label=f'Optimal ({optimal_thresh:.3f})')
    
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Performance')
    ax2.set_title(f'Threshold Analysis: {results[best_idx]["model_name"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance comparison
    ax3 = axes[1, 0]
    x = np.arange(len(models))
    width = 0.35
    
    ax3.bar(x - width/2, sensitivities, width, label='Sensitivity', alpha=0.8)
    ax3.bar(x + width/2, specificities, width, label='Specificity', alpha=0.8)
    
    ax3.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Performance')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ROC curves
    ax4 = axes[1, 1]
    for result in results:
        if result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true'], result['patient_probs'])
            ax4.plot(fpr, tpr, label=f'{result["model_name"]} (AUC={result["roc_auc"]:.3f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('who_compliance_final.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 WHO COMPLIANCE OPTIMIZATION - FINAL VERSION")
    print("Target: 90% Sensitivity, 70% Specificity")
    print("Robust implementation with proven techniques")
    print("=" * 70)
    
    # Load processed data
    try:
        data = np.load('full_dataset_processed.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
        patient_ids = data['patient_ids']
        
        print(f"✅ Loaded dataset: {X.shape}")
        print(f"✅ Patients: {len(np.unique(patient_ids))}")
        print(f"✅ TB positive rate: {sum(y)/len(y)*100:.1f}%")
        
    except FileNotFoundError:
        print("❌ Run run_full_analysis.py first")
        return
    
    # Create train/validation/test splits
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42
    )
    
    print(f"\\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Validation: {len(X_val)} files, {len(np.unique(val_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Optimized preprocessing
    print("\\n🔄 Optimized preprocessing...")
    X_train_processed, X_val_processed, y_train_processed, scaler, var_selector, selector = optimized_preprocessing(
        X_train, y_train, X_val
    )
    
    # Apply preprocessing to test set
    X_test_var = var_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_var)
    X_test_processed = selector.transform(X_test_scaled)
    
    # Create targeted models
    models = create_who_targeted_models()
    
    print(f"\\n🔄 Training {len(models)} WHO-targeted models...")
    
    results = []
    threshold_analyses = []
    
    for name, model in models.items():
        print(f"\\n  Training {name}...")
        model.fit(X_train_processed, y_train_processed)
        
        # Comprehensive threshold optimization
        optimal_threshold, thresh_df = comprehensive_threshold_optimization(
            model, X_val_processed, y_val, val_patients
        )
        
        print(f"    Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate on test set
        result = evaluate_final_model(
            model, X_test_processed, y_test, test_patients, name, optimal_threshold
        )
        
        results.append(result)
        threshold_analyses.append(thresh_df)
        
        compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
        print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
    
    # Results summary
    print("\\n" + "="*80)
    print("📋 FINAL WHO COMPLIANCE RESULTS")
    print("="*80)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Threshold': f"{result['threshold']:.3f}",
            'Sensitivity': f"{result['sensitivity']:.3f}",
            'Specificity': f"{result['specificity']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'NPV': f"{result['npv']:.3f}",
            'F1-Score': f"{result['f1_score']:.3f}",
            'ROC-AUC': f"{result['roc_auc']:.3f}",
            'WHO Compliant': '✅' if result['who_compliant'] else '❌',
            'TB Detected': f"{result['tp']}/{result['tp'] + result['fn']}",
            'TB Excluded': f"{result['tn']}/{result['tn'] + result['fp']}"
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Find best models
    compliant_models = [r for r in results if r['who_compliant']]
    
    if compliant_models:
        best_model = max(compliant_models, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\\n🏆 BEST WHO-COMPLIANT MODEL: {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (≥90% ✅)")
        print(f"   Specificity: {best_model['specificity']:.3f} (≥70% ✅)")
        print(f"   Precision: {best_model['precision']:.3f}")
        print(f"   F1-Score: {best_model['f1_score']:.3f}")
        print(f"   Threshold: {best_model['threshold']:.3f}")
        
        print(f"\\n🎉 WHO-COMPLIANT MODELS: {len(compliant_models)}/{len(results)}")
        
        # List all compliant models
        print("\\n✅ All WHO-compliant models:")
        for model in compliant_models:
            print(f"   - {model['model_name']}: {model['sensitivity']:.3f} sensitivity, {model['specificity']:.3f} specificity")
            
    else:
        best_model = max(results, key=lambda x: x['sensitivity'] + x['specificity'])
        print(f"\\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
        
        sens_gap = max(0, 0.90 - best_model['sensitivity'])
        spec_gap = max(0, 0.70 - best_model['specificity'])
        print(f"   Gap: Sensitivity -{sens_gap:.3f}, Specificity -{spec_gap:.3f}")
    
    # Save results
    results_df.to_csv('who_compliance_final_results.csv', index=False)
    
    # Create visualization
    print("\\n📊 Creating final WHO compliance visualization...")
    plot_final_who_analysis(results, threshold_analyses)
    
    print("\\n" + "="*80)
    print("🎉 FINAL WHO COMPLIANCE ANALYSIS COMPLETE!")
    print("="*80)
    print("📊 Files generated:")
    print("   - who_compliance_final_results.csv")
    print("   - who_compliance_final.png")

if __name__ == "__main__":
    main()