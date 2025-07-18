#!/usr/bin/env python3
"""
WHO Compliance Optimization V2 - Enhanced Specificity Focus
Target: 90% Sensitivity, 70% Specificity
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
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
    
    # Stratified split
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
    """Calculate patient-level metrics with detailed breakdown"""
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
            return sensitivity, specificity, precision, npv, tp, tn, fp, fn
    
    return 0, 0, 0, 0, 0, 0, 0, 0

def advanced_preprocessing(X_train, y_train, X_test, patient_ids_train):
    """Advanced preprocessing pipeline for WHO compliance"""
    print("  🔄 Advanced preprocessing pipeline...")
    
    # 1. Remove low-variance features
    var_selector = VarianceThreshold(threshold=0.005)  # Stricter threshold
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    print(f"    After variance filtering: {X_train_var.shape[1]} features")
    
    # 2. Advanced sampling strategy
    # Use SMOTETomek for better boundary handling
    if len(np.unique(y_train)) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        if min_samples > 10:
            # Borderline SMOTE + Tomek links cleanup
            smote_tomek = SMOTETomek(
                smote=BorderlineSMOTE(
                    sampling_strategy=0.4,  # Conservative oversampling
                    k_neighbors=5,
                    random_state=42
                ),
                random_state=42
            )
            X_train_balanced, y_train_balanced = smote_tomek.fit_resample(X_train_var, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_var, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_var, y_train
    
    print(f"    After advanced sampling: {X_train_balanced.shape[0]} samples")
    
    # 3. Robust scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_var)
    
    # 4. Advanced feature selection
    # Combine multiple selection methods
    
    # Step 4a: Statistical selection
    selector_stats = SelectKBest(score_func=f_classif, k=min(1200, X_train_scaled.shape[1]))
    X_train_stats = selector_stats.fit_transform(X_train_scaled, y_train_balanced)
    X_test_stats = selector_stats.transform(X_test_scaled)
    
    # Step 4b: Recursive feature elimination with Random Forest
    rf_selector = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rfe_selector = RFE(rf_selector, n_features_to_select=min(600, X_train_stats.shape[1]), step=0.1)
    X_train_selected = rfe_selector.fit_transform(X_train_stats, y_train_balanced)
    X_test_selected = rfe_selector.transform(X_test_stats)
    
    print(f"    Final features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train_balanced, scaler, var_selector, selector_stats, rfe_selector

def create_advanced_who_models():
    """Create advanced models optimized for WHO compliance"""
    models = {}
    
    # 1. Highly regularized Logistic Regression (proven to work)
    models["Elastic Net LR"] = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.7,  # More L1 regularization
        C=0.01,        # Stronger regularization
        class_weight={0: 1, 1: 3},  # Balanced but not extreme
        solver='saga',
        max_iter=2000,
        random_state=42
    )
    
    # 2. Calibrated Random Forest with balanced parameters
    models["Tuned RF"] = CalibratedClassifierCV(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=6,           # Shallower trees
            min_samples_split=15,  # More conservative splits
            min_samples_leaf=8,    # Larger leaf sizes
            max_features=0.3,      # Reduced feature sampling
            class_weight={0: 1, 1: 2.5},  # Moderate class weighting
            random_state=42,
            n_jobs=-1
        ),
        method='isotonic',
        cv=3
    )
    
    # 3. Gradient Boosting with regularization
    models["Regularized GBM"] = GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.03,    # Slower learning
        max_depth=4,           # Shallow trees
        min_samples_split=20,  # Conservative splits
        min_samples_leaf=10,   # Large leaves
        subsample=0.7,         # Bagging
        max_features=0.3,      # Feature sampling
        random_state=42
    )
    
    # 4. Advanced XGBoost with regularization
    if XGBOOST_AVAILABLE:
        models["Tuned XGB"] = XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.3,         # L1 regularization
            reg_lambda=0.3,        # L2 regularization
            scale_pos_weight=2.0,  # Moderate class weighting
            eval_metric='logloss',
            random_state=42,
            n_jobs=-1
        )
    
    # 5. Neural Network with dropout
    models["Regularized NN"] = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        alpha=0.01,            # L2 regularization
        learning_rate='adaptive',
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.2,
        random_state=42
    )
    
    # 6. SVM with balanced parameters
    models["Balanced SVM"] = SVC(
        kernel='rbf',
        C=0.1,                 # Regularization
        gamma='scale',
        class_weight={0: 1, 1: 2},
        probability=True,
        random_state=42
    )
    
    # 7. Ensemble with diversity
    base_models = [
        ('lr', LogisticRegression(C=0.01, class_weight={0: 1, 1: 3}, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=6, class_weight={0: 1, 1: 2}, random_state=42)),
        ('svm', SVC(probability=True, C=0.1, class_weight={0: 1, 1: 2}, random_state=42))
    ]
    
    if XGBOOST_AVAILABLE:
        base_models.append(('xgb', XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.05, 
            scale_pos_weight=2.0, reg_alpha=0.1, random_state=42
        )))
    
    models["Diverse Ensemble"] = VotingClassifier(base_models, voting='soft')
    
    return models

def optimize_threshold_grid_search(model, X_val, y_val, patient_ids_val):
    """Comprehensive threshold optimization with grid search"""
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    
    y_probs = model.predict_proba(X_val)[:, 1]
    
    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        
        sensitivity, specificity, precision, npv, tp, tn, fp, fn = calculate_patient_metrics(
            y_pred_thresh, y_val, patient_ids_val
        )
        
        # Multiple scoring strategies
        who_compliant = sensitivity >= 0.90 and specificity >= 0.70
        
        # Scoring function that prioritizes WHO compliance
        if who_compliant:
            score = 100 + (sensitivity + specificity) / 2  # Bonus for compliance
        elif sensitivity >= 0.90:
            score = 50 + specificity  # Reward specificity if sensitivity is met
        else:
            score = sensitivity * 10  # Minimal score if sensitivity not met
        
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

def evaluate_who_compliance_v2(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Enhanced evaluation with additional metrics"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, precision, npv, tp, tn, fp, fn = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # Additional metrics
    ppv = precision
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    
    # ROC-AUC and PR-AUC
    unique_patients = np.unique(patient_ids_test)
    patient_true_labels = []
    patient_probs = []
    
    for patient in unique_patients:
        patient_mask = patient_ids_test == patient
        patient_true = int(np.any(y_test[patient_mask]))
        patient_prob = np.max(y_probs[patient_mask])
        
        patient_true_labels.append(patient_true)
        patient_probs.append(patient_prob)
    
    patient_true_labels = np.array(patient_true_labels)
    patient_probs = np.array(patient_probs)
    
    if len(np.unique(patient_true_labels)) > 1:
        from sklearn.metrics import roc_auc_score, average_precision_score
        roc_auc = roc_auc_score(patient_true_labels, patient_probs)
        pr_auc = average_precision_score(patient_true_labels, patient_probs)
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
        'y_probs': y_probs,
        'y_pred': y_pred,
        'patient_true_labels': patient_true_labels,
        'patient_probs': patient_probs
    }

def plot_comprehensive_who_analysis(results, threshold_analyses):
    """Create comprehensive WHO compliance visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Plot 1: WHO Compliance Scatter
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    colors = ['green' if c else 'red' for c in compliant]
    scatter = ax1.scatter(specificities, sensitivities, c=colors, s=120, alpha=0.7)
    
    # WHO target box
    ax1.axhline(y=0.90, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity (70%)')
    ax1.fill_between([0.70, 1.0], 0.90, 1.0, alpha=0.1, color='green', label='WHO Compliant Zone')
    
    # Add model labels
    for i, model in enumerate(models):
        ax1.annotate(model, (specificities[i], sensitivities[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    ax1.set_xlabel('Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('WHO Compliance Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: Best Model Threshold Analysis
    ax2 = axes[0, 1]
    # Find best performing model
    best_idx = np.argmax([r['sensitivity'] + r['specificity'] for r in results])
    best_thresh_df = threshold_analyses[best_idx]
    
    ax2.plot(best_thresh_df['threshold'], best_thresh_df['sensitivity'], 
             'g-', label='Sensitivity', linewidth=2)
    ax2.plot(best_thresh_df['threshold'], best_thresh_df['specificity'], 
             'r-', label='Specificity', linewidth=2)
    ax2.plot(best_thresh_df['threshold'], best_thresh_df['precision'], 
             'b-', label='Precision', linewidth=2)
    
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    # Mark optimal threshold
    optimal_thresh = results[best_idx]['threshold']
    ax2.axvline(x=optimal_thresh, color='black', linestyle=':', alpha=0.8, label=f'Optimal ({optimal_thresh:.3f})')
    
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Performance')
    ax2.set_title(f'Threshold Analysis: {results[best_idx]["model_name"]}')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Performance Comparison
    ax3 = axes[0, 2]
    x = np.arange(len(models))
    width = 0.25
    
    ax3.bar(x - width, sensitivities, width, label='Sensitivity', alpha=0.8)
    ax3.bar(x, specificities, width, label='Specificity', alpha=0.8)
    ax3.bar(x + width, [r['f1_score'] for r in results], width, label='F1-Score', alpha=0.8)
    
    ax3.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax3.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Performance')
    ax3.set_title('Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: ROC Curves
    ax4 = axes[1, 0]
    for result in results:
        if result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true_labels'], result['patient_probs'])
            ax4.plot(fpr, tpr, label=f'{result["model_name"]} (AUC={result["roc_auc"]:.3f})')
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.set_title('ROC Curves')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Precision-Recall Curves
    ax5 = axes[1, 1]
    for result in results:
        if result['pr_auc'] > 0:
            precision, recall, _ = precision_recall_curve(result['patient_true_labels'], result['patient_probs'])
            ax5.plot(recall, precision, label=f'{result["model_name"]} (AUC={result["pr_auc"]:.3f})')
    
    ax5.set_xlabel('Recall (Sensitivity)')
    ax5.set_ylabel('Precision')
    ax5.set_title('Precision-Recall Curves')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Confusion Matrix Heatmap (Best Model)
    ax6 = axes[1, 2]
    best_result = results[best_idx]
    
    cm = np.array([[best_result['tn'], best_result['fp']], 
                   [best_result['fn'], best_result['tp']]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax6,
                xticklabels=['TB-', 'TB+'], yticklabels=['TB-', 'TB+'])
    ax6.set_title(f'Best Model: {best_result["model_name"]}\\n'
                  f'Sens: {best_result["sensitivity"]:.3f}, Spec: {best_result["specificity"]:.3f}')
    ax6.set_ylabel('Actual')
    ax6.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig('who_compliance_v2_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 WHO COMPLIANCE OPTIMIZATION V2")
    print("Target: 90% Sensitivity, 70% Specificity")
    print("Advanced preprocessing & threshold optimization")
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
    
    # Create validation split for threshold optimization
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42  # 0.25 * 0.8 = 0.2 of total
    )
    
    print(f"\\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Validation: {len(X_val)} files, {len(np.unique(val_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Advanced preprocessing
    print("\\n🔄 Advanced preprocessing pipeline...")
    X_train_processed, X_val_processed, y_train_processed, scaler, var_selector, stats_selector, rfe_selector = advanced_preprocessing(
        X_train, y_train, X_val, train_patients
    )
    
    # Apply same preprocessing to test set
    X_test_var = var_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_var)
    X_test_stats = stats_selector.transform(X_test_scaled)
    X_test_processed = rfe_selector.transform(X_test_stats)
    
    print(f"✅ Final processed shapes: Train {X_train_processed.shape}, Val {X_val_processed.shape}, Test {X_test_processed.shape}")
    
    # Create advanced WHO-optimized models
    models = create_advanced_who_models()
    
    print(f"\\n🔄 Training {len(models)} advanced WHO-optimized models...")
    
    trained_models = {}
    results = []
    threshold_analyses = []
    
    for name, model in models.items():
        print(f"\\n  Training {name}...")
        model.fit(X_train_processed, y_train_processed)
        trained_models[name] = model
        
        # Comprehensive threshold optimization
        optimal_threshold, thresh_df = optimize_threshold_grid_search(
            model, X_val_processed, y_val, val_patients
        )
        
        print(f"    Optimal threshold: {optimal_threshold:.3f}")
        
        # Evaluate on test set with optimal threshold
        result = evaluate_who_compliance_v2(
            model, X_test_processed, y_test, test_patients, name, optimal_threshold
        )
        
        results.append(result)
        threshold_analyses.append(thresh_df)
        
        compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
        print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
        print(f"    F1-Score: {result['f1_score']:.3f}, ROC-AUC: {result['roc_auc']:.3f}")
    
    # Create comprehensive results summary
    print("\\n" + "="*90)
    print("📋 COMPREHENSIVE WHO COMPLIANCE RESULTS")
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
            'PR-AUC': f"{result['pr_auc']:.3f}",
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
        print(f"   F1-Score: {best_model['f1_score']:.3f}")
        print(f"   Threshold: {best_model['threshold']:.3f}")
        
        # Count compliant models
        compliant_count = len(compliant_models)
        print(f"\\n📊 WHO-COMPLIANT MODELS: {compliant_count}/{len(results)}")
        
    else:
        # Find closest to WHO compliance
        best_model = max(results, key=lambda x: (
            x['sensitivity'] if x['sensitivity'] >= 0.90 else x['sensitivity'] * 0.5,
            x['specificity']
        ))
        print(f"\\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
        print(f"   Gap: Sensitivity {max(0, 0.90 - best_model['sensitivity']):.3f}, Specificity {max(0, 0.70 - best_model['specificity']):.3f}")
        
        print(f"\\n📊 WHO-COMPLIANT MODELS: 0/{len(results)}")
    
    # Save results
    results_df.to_csv('who_compliance_v2_results.csv', index=False)
    
    # Create comprehensive visualization
    print("\\n📊 Creating comprehensive WHO compliance visualization...")
    plot_comprehensive_who_analysis(results, threshold_analyses)
    
    print("\\n" + "="*90)
    print("🎉 WHO COMPLIANCE OPTIMIZATION V2 COMPLETE!")
    print("="*90)
    print("📊 Files generated:")
    print("   - who_compliance_v2_results.csv")
    print("   - who_compliance_v2_analysis.png")
    
    # Provide recommendations
    print("\\n💡 RECOMMENDATIONS:")
    if compliant_models:
        print("✅ WHO compliance achieved! Deploy recommended model.")
    else:
        print("❌ WHO compliance not fully achieved. Consider:")
        print("   - Ensemble of best models")
        print("   - Cost-sensitive learning")
        print("   - Multi-stage classification")
        print("   - Additional feature engineering")

if __name__ == "__main__":
    main()