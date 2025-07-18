#!/usr/bin/env python3
"""
WHO Compliance - AGGRESSIVE OPTIMIZATION
Target: 90% Sensitivity, 70% Specificity
Final attempt with aggressive regularization and cost-sensitive learning
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
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
        patient_ids[train_mask], patient_ids[test_patients]
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
            return sensitivity, specificity, tp, tn, fp, fn, patient_true_labels, patient_predictions
    
    return 0, 0, 0, 0, 0, 0, patient_true_labels, patient_predictions

def aggressive_preprocessing(X_train, y_train, X_test):
    """Aggressive preprocessing to maximize specificity"""
    print("  🔄 Aggressive preprocessing for WHO compliance...")
    
    # 1. Strict variance filtering
    var_selector = VarianceThreshold(threshold=0.02)  # Stricter
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    print(f"    After strict variance filtering: {X_train_var.shape[1]} features")
    
    # 2. Minimal oversampling + undersampling
    pos_count = np.sum(y_train)
    neg_count = len(y_train) - pos_count
    
    if pos_count > 0 and neg_count > 0:
        # Conservative SMOTE
        smote = SMOTE(
            sampling_strategy=0.2,  # Very conservative
            k_neighbors=3,
            random_state=42
        )
        X_smote, y_smote = smote.fit_resample(X_train_var, y_train)
        
        # Undersample majority class
        undersampler = RandomUnderSampler(
            sampling_strategy=0.6,  # Reduce majority class
            random_state=42
        )
        X_train_balanced, y_train_balanced = undersampler.fit_resample(X_smote, y_smote)
    else:
        X_train_balanced, y_train_balanced = X_train_var, y_train
    
    print(f"    After balanced sampling: {X_train_balanced.shape[0]} samples")
    
    # 3. Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_var)
    
    # 4. Aggressive feature selection
    selector = SelectKBest(score_func=f_classif, k=min(400, X_train_scaled.shape[1]))  # Fewer features
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"    Final features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train_balanced, scaler, var_selector, selector

def create_cost_sensitive_models():
    """Create cost-sensitive models for WHO compliance"""
    models = {}
    
    # Model 1: Extremely regularized LR with high false positive cost
    models["Ultra Regularized LR"] = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.9,  # Heavy L1 regularization
        C=0.001,       # Very strong regularization
        class_weight={0: 1, 1: 1.5},  # Moderate class weight
        solver='saga',
        max_iter=3000,
        random_state=42
    )
    
    # Model 2: Random Forest with extreme regularization
    models["Ultra Conservative RF"] = RandomForestClassifier(
        n_estimators=300,
        max_depth=3,           # Very shallow
        min_samples_split=50,  # Very conservative
        min_samples_leaf=25,   # Large leaves
        max_features=0.1,      # Very few features
        class_weight={0: 1, 1: 1.2},
        random_state=42,
        n_jobs=-1
    )
    
    # Model 3: SVM with tight margin
    models["Tight Margin SVM"] = SVC(
        kernel='rbf',
        C=0.01,              # Very tight margin
        gamma='scale',
        class_weight={0: 1, 1: 1.5},
        probability=True,
        random_state=42
    )
    
    # Model 4: XGBoost with heavy regularization
    if XGBOOST_AVAILABLE:
        models["Heavy Reg XGB"] = XGBClassifier(
            n_estimators=300,
            max_depth=2,         # Very shallow
            learning_rate=0.01,  # Very slow
            subsample=0.5,       # Strong bagging
            colsample_bytree=0.5,
            reg_alpha=1.0,       # Heavy L1
            reg_lambda=1.0,      # Heavy L2
            scale_pos_weight=1.2,
            eval_metric='logloss',
            random_state=42
        )
    
    return models

class WHOThresholdOptimizer:
    """Specialized threshold optimizer for WHO compliance"""
    
    def __init__(self, sensitivity_target=0.90, specificity_target=0.70):
        self.sensitivity_target = sensitivity_target
        self.specificity_target = specificity_target
    
    def optimize(self, model, X_val, y_val, patient_ids_val):
        """Find threshold that maximizes WHO compliance"""
        
        # Get prediction probabilities
        y_probs = model.predict_proba(X_val)[:, 1]
        
        # Test thresholds with focus on WHO targets
        thresholds = np.linspace(0.01, 0.99, 199)  # More granular
        results = []
        
        for threshold in thresholds:
            y_pred = (y_probs >= threshold).astype(int)
            
            sensitivity, specificity, tp, tn, fp, fn, _, _ = calculate_patient_metrics(
                y_pred, y_val, patient_ids_val
            )
            
            # WHO compliance check
            sens_meets = sensitivity >= self.sensitivity_target
            spec_meets = specificity >= self.specificity_target
            who_compliant = sens_meets and spec_meets
            
            # Multi-objective scoring
            if who_compliant:
                # Bonus for WHO compliance
                score = 10000 + sensitivity + specificity
            elif sens_meets:
                # Prioritize specificity if sensitivity is met
                score = 5000 + specificity * 1000
            else:
                # Prioritize sensitivity
                score = sensitivity * 1000
            
            results.append({
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'who_compliant': who_compliant,
                'score': score,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            })
        
        # Find best threshold
        best_result = max(results, key=lambda x: x['score'])
        
        return best_result['threshold'], pd.DataFrame(results)

def evaluate_with_who_focus(model, X_test, y_test, patient_ids_test, model_name, threshold):
    """Evaluate model with WHO focus"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, tp, tn, fp, fn, patient_true, patient_pred = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # Calculate additional metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # Patient-level probabilities
    unique_patients = np.unique(patient_ids_test)
    patient_probs = []
    
    for patient in unique_patients:
        patient_mask = patient_ids_test == patient
        patient_prob = np.max(y_probs[patient_mask])
        patient_probs.append(patient_prob)
    
    patient_probs = np.array(patient_probs)
    
    # AUC metrics
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

def create_who_dashboard(results, threshold_analyses):
    """Create WHO compliance dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: WHO Compliance Map
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    # Create compliance map
    colors = ['green' if c else 'red' for c in compliant]
    sizes = [200 if c else 100 for c in compliant]
    
    scatter = ax1.scatter(specificities, sensitivities, c=colors, s=sizes, alpha=0.7)
    
    # WHO targets
    ax1.axhline(y=0.90, color='blue', linestyle='--', linewidth=2, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', linewidth=2, label='WHO Specificity (70%)')
    ax1.fill_between([0.70, 1.0], 0.90, 1.0, alpha=0.2, color='green', label='WHO Compliant Zone')
    
    # Model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (specificities[i], sensitivities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
    
    ax1.set_xlabel('Specificity', fontsize=12)
    ax1.set_ylabel('Sensitivity', fontsize=12)
    ax1.set_title('WHO Compliance Map', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Performance Metrics
    ax2 = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    
    sens_bars = ax2.bar(x - width/2, sensitivities, width, label='Sensitivity', alpha=0.8, color='green')
    spec_bars = ax2.bar(x + width/2, specificities, width, label='Specificity', alpha=0.8, color='red')
    
    # WHO target lines
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('Performance', fontsize=12)
    ax2.set_title('Performance vs WHO Targets', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # Plot 3: Best Model Threshold Analysis
    ax3 = axes[1, 0]
    best_idx = np.argmax([r['sensitivity'] * r['specificity'] for r in results])
    best_thresh_df = threshold_analyses[best_idx]
    
    ax3.plot(best_thresh_df['threshold'], best_thresh_df['sensitivity'], 
             'g-', linewidth=3, label='Sensitivity')
    ax3.plot(best_thresh_df['threshold'], best_thresh_df['specificity'], 
             'r-', linewidth=3, label='Specificity')
    
    # WHO targets
    ax3.axhline(y=0.90, color='green', linestyle='--', alpha=0.8, linewidth=2)
    ax3.axhline(y=0.70, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Optimal threshold
    optimal_thresh = results[best_idx]['threshold']
    ax3.axvline(x=optimal_thresh, color='black', linestyle=':', alpha=0.8, linewidth=2,
                label=f'Optimal ({optimal_thresh:.3f})')
    
    ax3.set_xlabel('Decision Threshold', fontsize=12)
    ax3.set_ylabel('Performance', fontsize=12)
    ax3.set_title(f'Threshold Analysis: {results[best_idx]["model_name"]}', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ROC Curves
    ax4 = axes[1, 1]
    for result in results:
        if result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true'], result['patient_probs'])
            ax4.plot(fpr, tpr, linewidth=2, label=f'{result["model_name"]} (AUC={result["roc_auc"]:.3f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curves', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('who_aggressive_final.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 WHO COMPLIANCE - AGGRESSIVE OPTIMIZATION")
    print("Target: 90% Sensitivity, 70% Specificity")
    print("Cost-sensitive learning with aggressive regularization")
    print("=" * 70)
    
    # Load data
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
    
    # Create splits
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42
    )
    
    print(f"\\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Validation: {len(X_val)} files, {len(np.unique(val_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Aggressive preprocessing
    X_train_processed, X_val_processed, y_train_processed, scaler, var_selector, selector = aggressive_preprocessing(
        X_train, y_train, X_val
    )
    
    # Apply to test set
    X_test_var = var_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_var)
    X_test_processed = selector.transform(X_test_scaled)
    
    # Create cost-sensitive models
    models = create_cost_sensitive_models()
    
    print(f"\\n🔄 Training {len(models)} cost-sensitive models...")
    
    # Initialize WHO optimizer
    optimizer = WHOThresholdOptimizer()
    
    results = []
    threshold_analyses = []
    
    for name, model in models.items():
        print(f"\\n  Training {name}...")
        model.fit(X_train_processed, y_train_processed)
        
        # WHO-focused threshold optimization
        optimal_threshold, thresh_df = optimizer.optimize(
            model, X_val_processed, y_val, val_patients
        )
        
        print(f"    Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate on test set
        result = evaluate_with_who_focus(
            model, X_test_processed, y_test, test_patients, name, optimal_threshold
        )
        
        results.append(result)
        threshold_analyses.append(thresh_df)
        
        compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
        print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
    
    # Results summary
    print("\\n" + "="*90)
    print("📋 AGGRESSIVE WHO COMPLIANCE RESULTS")
    print("="*90)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Threshold': f"{result['threshold']:.3f}",
            'Sensitivity': f"{result['sensitivity']:.3f}",
            'Specificity': f"{result['specificity']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'F1-Score': f"{result['f1_score']:.3f}",
            'ROC-AUC': f"{result['roc_auc']:.3f}",
            'WHO Compliant': '✅' if result['who_compliant'] else '❌',
            'TB Detected': f"{result['tp']}/{result['tp'] + result['fn']}",
            'TB Excluded': f"{result['tn']}/{result['tn'] + result['fp']}"
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Find WHO compliant models
    compliant_models = [r for r in results if r['who_compliant']]
    
    if compliant_models:
        best_model = max(compliant_models, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\\n🏆 BEST WHO-COMPLIANT MODEL: {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (≥90% ✅)")
        print(f"   Specificity: {best_model['specificity']:.3f} (≥70% ✅)")
        print(f"   Threshold: {best_model['threshold']:.3f}")
        
        print(f"\\n🎉 SUCCESS! WHO-COMPLIANT MODELS: {len(compliant_models)}/{len(results)}")
        
    else:
        best_model = max(results, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
        
        sens_gap = max(0, 0.90 - best_model['sensitivity'])
        spec_gap = max(0, 0.70 - best_model['specificity'])
        print(f"   Remaining gap: Sensitivity -{sens_gap:.3f}, Specificity -{spec_gap:.3f}")
        
        print(f"\\n⚠️  WHO compliance not achieved with current approach")
    
    # Save results
    results_df.to_csv('who_aggressive_results.csv', index=False)
    
    # Create dashboard
    print("\\n📊 Creating WHO compliance dashboard...")
    create_who_dashboard(results, threshold_analyses)
    
    print("\\n" + "="*90)
    print("🎉 AGGRESSIVE WHO OPTIMIZATION COMPLETE!")
    print("="*90)
    print("📊 Files generated:")
    print("   - who_aggressive_results.csv")
    print("   - who_aggressive_final.png")
    
    # Final recommendations
    print("\\n💡 FINAL RECOMMENDATIONS:")
    if compliant_models:
        print("✅ WHO compliance achieved!")
        print("   Deploy the best compliant model for clinical use.")
    else:
        print("❌ WHO compliance not fully achieved.")
        print("   Consider:")
        print("   - Two-stage classification (sensitivity screening + specificity refinement)")
        print("   - Ensemble with rejection option")
        print("   - Additional feature engineering")
        print("   - Collect more balanced training data")

if __name__ == "__main__":
    main()