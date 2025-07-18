#!/usr/bin/env python3
"""
Quick TB Detection Analysis for Full Dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, fbeta_score, roc_auc_score, accuracy_score
)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

def create_patient_level_split(X, y, patient_ids, test_size=0.2, random_state=42):
    """Create patient-level train/test split"""
    unique_patients = np.unique(patient_ids)
    
    # Calculate patient-level labels
    patient_labels = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_labels[patient] = int(np.any(y[patient_mask]))
    
    # Split patients
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

def evaluate_model(model, X_test, y_test, test_patients, model_name):
    """Evaluate model with patient-level aggregation"""
    # File-level predictions
    y_pred_file = model.predict(X_test)
    y_prob_file = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred_file
    
    # Patient-level aggregation
    unique_patients = np.unique(test_patients)
    patient_predictions = []
    patient_true_labels = []
    patient_probs = []
    
    for patient in unique_patients:
        patient_mask = test_patients == patient
        patient_files_pred = y_pred_file[patient_mask]
        patient_files_true = y_test[patient_mask]
        patient_files_prob = y_prob_file[patient_mask]
        
        # Any positive file makes patient positive
        patient_pred_any = int(np.any(patient_files_pred))
        patient_true_any = int(np.any(patient_files_true))
        patient_prob_max = np.max(patient_files_prob)
        
        patient_predictions.append(patient_pred_any)
        patient_true_labels.append(patient_true_any)
        patient_probs.append(patient_prob_max)
    
    patient_predictions = np.array(patient_predictions)
    patient_true_labels = np.array(patient_true_labels)
    patient_probs = np.array(patient_probs)
    
    # Calculate metrics
    cm_patient = confusion_matrix(patient_true_labels, patient_predictions)
    
    if cm_patient.shape == (2, 2):
        tn_p, fp_p, fn_p, tp_p = cm_patient.ravel()
        sensitivity = tp_p / (tp_p + fn_p) if (tp_p + fn_p) > 0 else 0
        specificity = tn_p / (tn_p + fp_p) if (tn_p + fp_p) > 0 else 0
        precision = tp_p / (tp_p + fp_p) if (tp_p + fp_p) > 0 else 0
        f2_score = fbeta_score(patient_true_labels, patient_predictions, beta=2, average='weighted')
        roc_auc = roc_auc_score(patient_true_labels, patient_probs) if len(np.unique(patient_true_labels)) > 1 else 0
        
        precision_vals, recall_vals, _ = precision_recall_curve(patient_true_labels, patient_probs)
        pr_auc = auc(recall_vals, precision_vals) if len(np.unique(patient_true_labels)) > 1 else 0
    else:
        tp_p = fn_p = tn_p = fp_p = 0
        sensitivity = specificity = precision = f2_score = roc_auc = pr_auc = 0
    
    return {
        'model_name': model_name,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f2_score': f2_score,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'tp': tp_p,
        'fn': fn_p,
        'tn': tn_p,
        'fp': fp_p,
        'cm': cm_patient,
        'patient_true_labels': patient_true_labels,
        'patient_probs': patient_probs
    }

def main():
    print("🚀 QUICK TB DETECTION ANALYSIS")
    print("=" * 50)
    
    # Load processed data
    try:
        data = np.load('full_dataset_processed.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
        patient_ids = data['patient_ids']
        
        print(f"✅ Loaded dataset: {X.shape}")
        print(f"✅ Patients: {len(np.unique(patient_ids))}")
        print(f"✅ TB positive: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
        
    except FileNotFoundError:
        print("❌ Run run_full_analysis.py first")
        return
    
    # Patient-level split
    X_train, X_test, y_train, y_test, train_patients, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    print(f"\\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Quick preprocessing
    print("\\n🔄 Preprocessing...")
    
    # Variance filtering
    var_selector = VarianceThreshold(threshold=0.001)
    X_train_filtered = var_selector.fit_transform(X_train)
    X_test_filtered = var_selector.transform(X_test)
    
    # SMOTE
    min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
    if min_samples > 5:
        smote = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train)
    else:
        X_train_balanced, y_train_balanced = X_train_filtered, y_train
    
    # Scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Feature selection
    k_features = min(500, X_train_scaled.shape[1])  # Reduced for speed
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"✅ Final training shape: {X_train_selected.shape}")
    
    # Train models (reduced set for speed)
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=50,  # Reduced for speed
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=50,  # Reduced for speed
            learning_rate=0.1,
            max_depth=6,
            random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            class_weight='balanced',
            random_state=42
        )
    }
    
    print("\\n🔄 Training models...")
    results = {}
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train_selected, y_train_balanced)
        
        result = evaluate_model(model, X_test_selected, y_test, test_patients, name)
        results[name] = result
        
        print(f"    Sensitivity: {result['sensitivity']:.3f}")
        print(f"    Specificity: {result['specificity']:.3f}")
        print(f"    F2-Score: {result['f2_score']:.3f}")
    
    # Generate summary
    print("\\n" + "="*60)
    print("📋 RESULTS SUMMARY")
    print("="*60)
    
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Sensitivity': f"{result['sensitivity']:.3f}",
            'Specificity': f"{result['specificity']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'F2-Score': f"{result['f2_score']:.3f}",
            'ROC-AUC': f"{result['roc_auc']:.3f}",
            'PR-AUC': f"{result['pr_auc']:.3f}",
            'Clinical Target': '✅' if result['sensitivity'] >= 0.8 else '❌',
            'TB Detected': f"{result['tp']}/{result['tp'] + result['fn']}"
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('quick_tb_results.csv', index=False)
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['sensitivity'])
    print(f"\\n🏆 BEST MODEL: {best_model[0]}")
    print(f"   Sensitivity: {best_model[1]['sensitivity']:.3f}")
    print(f"   Specificity: {best_model[1]['specificity']:.3f}")
    print(f"   F2-Score: {best_model[1]['f2_score']:.3f}")
    
    # Create basic visualization
    print("\\n📊 Creating visualization...")
    
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Confusion matrices
    plt.subplot(2, 2, 1)
    cm = best_model[1]['cm']
    if cm.size > 0:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['TB-', 'TB+'], yticklabels=['TB-', 'TB+'])
        plt.title(f'{best_model[0]}\\nConfusion Matrix')
    
    # Plot 2: ROC curves
    plt.subplot(2, 2, 2)
    for name, result in results.items():
        if result['roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true_labels'], result['patient_probs'])
            plt.plot(fpr, tpr, label=f'{name} (AUC={result["roc_auc"]:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall curves
    plt.subplot(2, 2, 3)
    for name, result in results.items():
        if result['pr_auc'] > 0:
            precision, recall, _ = precision_recall_curve(result['patient_true_labels'], result['patient_probs'])
            plt.plot(recall, precision, label=f'{name} (AUC={result["pr_auc"]:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Performance metrics
    plt.subplot(2, 2, 4)
    metrics = ['Sensitivity', 'Specificity', 'Precision', 'F2-Score']
    model_names = list(results.keys())
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        values = [results[model][metric.lower().replace('-', '_')] for model in model_names]
        plt.bar(x + i * width, values, width, label=metric)
    
    plt.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Target')
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Metrics')
    plt.xticks(x + width * 1.5, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tb_analysis_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\\n" + "="*60)
    print("🎉 ANALYSIS COMPLETE!")
    print("="*60)
    print("📊 Files generated:")
    print("   - quick_tb_results.csv")
    print("   - tb_analysis_results.png")

if __name__ == "__main__":
    main()