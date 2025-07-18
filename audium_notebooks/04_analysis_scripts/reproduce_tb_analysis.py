#!/usr/bin/env python3
"""
Reproduction Script for TB Detection Analysis
Simple script to reproduce the WHO-compliant TB detection results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score, average_precision_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

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
    
    return (X[train_mask], X[test_mask], y[train_mask], y[test_mask], 
            patient_ids[train_mask], patient_ids[test_mask])

def calculate_patient_metrics(y_pred, y_true, patient_ids):
    """Calculate patient-level metrics"""
    unique_patients = np.unique(patient_ids)
    patient_predictions = []
    patient_true_labels = []
    
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_pred = int(np.mean(y_pred[patient_mask]) >= 0.5)
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
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
            return sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn
    
    return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

def preprocess_data(X_train, y_train, X_test):
    """Preprocess data for optimal performance"""
    print("  🔄 Preprocessing data...")
    
    # Variance filtering
    var_selector = VarianceThreshold(threshold=0.01)
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    # Balanced sampling
    if len(np.unique(y_train)) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        if min_samples > 10:
            smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_var, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_var, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_var, y_train
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_var)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(800, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    return X_train_selected, X_test_selected, y_train_balanced, scaler, var_selector, selector

def optimize_threshold(model, X_val, y_val, patient_ids_val):
    """Optimize threshold for WHO compliance"""
    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_score = 0
    
    y_probs = model.predict_proba(X_val)[:, 1]
    
    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        
        sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn = calculate_patient_metrics(
            y_pred_thresh, y_val, patient_ids_val
        )
        
        # WHO compliance scoring
        who_compliant = sensitivity >= 0.90 and specificity >= 0.70
        
        if who_compliant:
            score = 1000 + (sensitivity + specificity)
        elif sensitivity >= 0.90:
            score = 500 + specificity * 100
        elif specificity >= 0.70:
            score = 100 + sensitivity * 100
        else:
            score = sensitivity * 50 + specificity * 50
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold

def evaluate_model(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Evaluate model performance"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # Calculate patient-level ROC AUC
    unique_patients = np.unique(patient_ids_test)
    patient_probs = []
    patient_true = []
    
    for patient in unique_patients:
        patient_mask = patient_ids_test == patient
        patient_prob = np.max(y_probs[patient_mask])
        patient_label = int(np.any(y_test[patient_mask]))
        patient_probs.append(patient_prob)
        patient_true.append(patient_label)
    
    patient_probs = np.array(patient_probs)
    patient_true = np.array(patient_true)
    
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
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'who_compliant': who_compliant,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'patient_true': patient_true,
        'patient_probs': patient_probs
    }

def create_summary_plot(results):
    """Create summary visualization"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: WHO Compliance
    ax1 = axes[0, 0]
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    colors = ['green' if c else 'red' for c in compliant]
    ax1.scatter(specificities, sensitivities, c=colors, s=150, alpha=0.7)
    
    ax1.axhline(y=0.90, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity (70%)')
    ax1.fill_between([0.70, 1.0], 0.90, 1.0, alpha=0.1, color='green', label='WHO Compliant Zone')
    
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
    
    # Plot 2: Performance Comparison
    ax2 = axes[0, 1]
    x = np.arange(len(models))
    width = 0.35
    
    ax2.bar(x - width/2, sensitivities, width, label='Sensitivity', alpha=0.8)
    ax2.bar(x + width/2, specificities, width, label='Specificity', alpha=0.8)
    
    ax2.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax2.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_xlabel('Models')
    ax2.set_ylabel('Performance')
    ax2.set_title('Performance Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(models, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: ROC Curves
    ax3 = axes[1, 0]
    for result in results:
        if result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true'], result['patient_probs'])
            ax3.plot(fpr, tpr, label=f'{result["model_name"]} (AUC={result["roc_auc"]:.3f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curves')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary Metrics
    ax4 = axes[1, 1]
    f1_scores = [r['f1_score'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    
    ax4.bar(x - width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    ax4.bar(x + width/2, roc_aucs, width, label='ROC AUC', alpha=0.8)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Score')
    ax4.set_title('F1 Score and ROC AUC')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tb_detection_reproduction_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 TB DETECTION ANALYSIS - REPRODUCTION SCRIPT")
    print("Target: 90% Sensitivity, 70% Specificity (WHO Guidelines)")
    print("=" * 60)
    
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
        print("❌ File 'full_dataset_processed.npz' not found")
        return
    
    # Create splits
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42
    )
    
    print(f"\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Validation: {len(X_val)} files, {len(np.unique(val_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Preprocess data
    X_train_processed, X_val_processed, y_train_processed, scaler, var_selector, selector = preprocess_data(
        X_train, y_train, X_val
    )
    
    X_test_var = var_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_var)
    X_test_processed = selector.transform(X_test_scaled)
    
    # Define models
    models = {
        "WHO_Optimized_LR": LogisticRegression(
            penalty='elasticnet', l1_ratio=0.7, C=0.01, 
            class_weight={0: 1, 1: 3}, solver='saga', 
            max_iter=2000, random_state=42
        ),
        "Conservative_RF": RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_split=20,
            min_samples_leaf=10, max_features=0.3, 
            class_weight={0: 1, 1: 2}, random_state=42, n_jobs=-1
        ),
        "Regularized_GB": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_split=20, min_samples_leaf=10,
            random_state=42
        ),
        "Tuned_SVM": SVC(
            kernel='rbf', C=0.1, gamma='scale', 
            class_weight={0: 1, 1: 2}, probability=True, random_state=42
        )
    }
    
    print(f"\n🔄 Training {len(models)} optimized models...")
    
    results = []
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        try:
            model.fit(X_train_processed, y_train_processed)
            
            # Optimize threshold
            optimal_threshold = optimize_threshold(model, X_val_processed, y_val, val_patients)
            print(f"    Optimal threshold: {optimal_threshold:.3f}")
            
            # Evaluate
            result = evaluate_model(model, X_test_processed, y_test, test_patients, name, optimal_threshold)
            results.append(result)
            
            compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
            print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
            
        except Exception as e:
            print(f"    ❌ Failed to train {name}: {str(e)}")
    
    # Summary
    print("\n" + "="*80)
    print("📋 REPRODUCTION RESULTS")
    print("="*80)
    
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
            'WHO Compliant': '✅' if result['who_compliant'] else '❌'
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Find best model
    compliant_models = [r for r in results if r['who_compliant']]
    
    if compliant_models:
        best_model = max(compliant_models, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\n🏆 BEST WHO-COMPLIANT MODEL: {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (≥90% ✅)")
        print(f"   Specificity: {best_model['specificity']:.3f} (≥70% ✅)")
        print(f"   Threshold: {best_model['threshold']:.3f}")
        print(f"\n🎉 WHO-COMPLIANT MODELS: {len(compliant_models)}/{len(results)}")
    else:
        best_model = max(results, key=lambda x: x['sensitivity'] + x['specificity'])
        print(f"\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
    
    # Save and visualize
    results_df.to_csv('tb_detection_reproduction_results.csv', index=False)
    create_summary_plot(results)
    
    print("\n" + "="*80)
    print("🎉 REPRODUCTION ANALYSIS COMPLETE!")
    print("="*80)
    print("📊 Files generated:")
    print("   - tb_detection_reproduction_results.csv")
    print("   - tb_detection_reproduction_results.png")

if __name__ == "__main__":
    main()