#!/usr/bin/env python3
"""
Complete TB Detection Analysis with Visualizations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, fbeta_score, roc_auc_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# Data augmentation
from imblearn.over_sampling import SMOTE

# Feature engineering
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from scipy import stats

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

def create_patient_level_split(X, y, patient_ids, test_size=0.2, random_state=42):
    """
    Create train/test split ensuring patients don't appear in both sets
    """
    unique_patients = np.unique(patient_ids)
    
    # Calculate patient-level labels (any TB positive file makes patient positive)
    patient_labels = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_labels[patient] = int(np.any(y[patient_mask]))
    
    # Split patients
    patients_array = np.array(list(patient_labels.keys()))
    labels_array = np.array(list(patient_labels.values()))
    
    # Only do stratified split if we have both classes
    if len(np.unique(labels_array)) > 1:
        train_patients, test_patients = train_test_split(
            patients_array, test_size=test_size, stratify=labels_array, random_state=random_state
        )
    else:
        train_patients, test_patients = train_test_split(
            patients_array, test_size=test_size, random_state=random_state
        )
    
    # Create file-level splits
    train_mask = np.isin(patient_ids, train_patients)
    test_mask = np.isin(patient_ids, test_patients)
    
    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        patient_ids[train_mask], patient_ids[test_mask]
    )

def evaluate_advanced_model(model, X_test, y_test, test_patients, model_name):
    """
    Advanced evaluation with both file-level and patient-level metrics
    """
    # File-level predictions
    y_pred_file = model.predict(X_test)
    
    if hasattr(model, "predict_proba"):
        y_prob_file = model.predict_proba(X_test)[:, 1]
    else:
        y_prob_file = y_pred_file
    
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
        
        # Patient-level aggregation strategies
        # 1. Any positive file makes patient positive (sensitive)
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
    # File-level metrics
    cm_file = confusion_matrix(y_test, y_pred_file)
    if cm_file.shape == (2, 2):
        tn_f, fp_f, fn_f, tp_f = cm_file.ravel()
    else:
        tn_f, fp_f, fn_f, tp_f = 0, 0, 0, 0
    
    # Patient-level metrics
    cm_patient = confusion_matrix(patient_true_labels, patient_predictions)
    if cm_patient.shape == (2, 2):
        tn_p, fp_p, fn_p, tp_p = cm_patient.ravel()
    else:
        tn_p, fp_p, fn_p, tp_p = 0, 0, 0, 0
    
    # Calculate clinical metrics
    def safe_divide(a, b):
        return a / b if b > 0 else 0
    
    # File-level metrics
    file_metrics = {
        'sensitivity': safe_divide(tp_f, tp_f + fn_f),
        'specificity': safe_divide(tn_f, tn_f + fp_f),
        'precision': safe_divide(tp_f, tp_f + fp_f),
        'npv': safe_divide(tn_f, tn_f + fn_f),
        'f1': f1_score(y_test, y_pred_file, average='weighted') if len(np.unique(y_test)) > 1 else 0,
        'f2': fbeta_score(y_test, y_pred_file, beta=2, average='weighted') if len(np.unique(y_test)) > 1 else 0,
        'accuracy': accuracy_score(y_test, y_pred_file)
    }
    
    # Patient-level metrics
    patient_metrics = {
        'sensitivity': safe_divide(tp_p, tp_p + fn_p),
        'specificity': safe_divide(tn_p, tn_p + fp_p),
        'precision': safe_divide(tp_p, tp_p + fp_p),
        'npv': safe_divide(tn_p, tn_p + fn_p),
        'f1': f1_score(patient_true_labels, patient_predictions, average='weighted') if len(np.unique(patient_true_labels)) > 1 else 0,
        'f2': fbeta_score(patient_true_labels, patient_predictions, beta=2, average='weighted') if len(np.unique(patient_true_labels)) > 1 else 0,
        'accuracy': accuracy_score(patient_true_labels, patient_predictions)
    }
    
    # AUC metrics
    try:
        if len(np.unique(y_test)) > 1:
            file_roc_auc = roc_auc_score(y_test, y_prob_file)
            precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob_file)
            file_pr_auc = auc(recall_vals, precision_vals)
        else:
            file_roc_auc = file_pr_auc = 0.0
            
        if len(np.unique(patient_true_labels)) > 1:
            patient_roc_auc = roc_auc_score(patient_true_labels, patient_probs)
            precision_vals_p, recall_vals_p, _ = precision_recall_curve(patient_true_labels, patient_probs)
            patient_pr_auc = auc(recall_vals_p, precision_vals_p)
        else:
            patient_roc_auc = patient_pr_auc = 0.0
    except:
        file_roc_auc = patient_roc_auc = file_pr_auc = patient_pr_auc = 0.0
    
    return {
        'model_name': model_name,
        'file_metrics': file_metrics,
        'patient_metrics': patient_metrics,
        'file_roc_auc': file_roc_auc,
        'patient_roc_auc': patient_roc_auc,
        'file_pr_auc': file_pr_auc,
        'patient_pr_auc': patient_pr_auc,
        'file_cm': cm_file,
        'patient_cm': cm_patient,
        'file_predictions': y_pred_file,
        'patient_predictions': patient_predictions,
        'file_probs': y_prob_file,
        'patient_probs': patient_probs,
        'y_test': y_test,
        'patient_true_labels': patient_true_labels,
        'n_patients': len(unique_patients),
        'n_files': len(y_test),
        'tp_p': tp_p, 'fn_p': fn_p, 'tn_p': tn_p, 'fp_p': fp_p
    }

def plot_confusion_matrices(advanced_results, figsize=(20, 12)):
    """Plot confusion matrices for all models"""
    n_models = len(advanced_results)
    fig, axes = plt.subplots(2, n_models, figsize=figsize)
    
    if n_models == 1:
        axes = axes.reshape(2, 1)
    
    for i, (name, result) in enumerate(advanced_results.items()):
        # File-level confusion matrix
        ax1 = axes[0, i]
        cm_file = result['file_cm']
        if cm_file.size > 0:
            sns.heatmap(cm_file, annot=True, fmt='d', cmap='Blues', ax=ax1,
                       xticklabels=['TB-', 'TB+'], yticklabels=['TB-', 'TB+'])
            ax1.set_title(f'{name}\\nFile-level Confusion Matrix')
            ax1.set_xlabel('Predicted')
            ax1.set_ylabel('Actual')
        
        # Patient-level confusion matrix
        ax2 = axes[1, i]
        cm_patient = result['patient_cm']
        if cm_patient.size > 0:
            sns.heatmap(cm_patient, annot=True, fmt='d', cmap='Oranges', ax=ax2,
                       xticklabels=['TB-', 'TB+'], yticklabels=['TB-', 'TB+'])
            ax2.set_title(f'{name}\\nPatient-level Confusion Matrix')
            ax2.set_xlabel('Predicted')
            ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_roc_curves(advanced_results, figsize=(15, 6)):
    """Plot ROC curves for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # File-level ROC curves
    for name, result in advanced_results.items():
        if result['file_roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['y_test'], result['file_probs'])
            ax1.plot(fpr, tpr, label=f'{name} (AUC={result["file_roc_auc"]:.3f})')
    
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('File-level ROC Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Patient-level ROC curves
    for name, result in advanced_results.items():
        if result['patient_roc_auc'] > 0:
            fpr, tpr, _ = roc_curve(result['patient_true_labels'], result['patient_probs'])
            ax2.plot(fpr, tpr, label=f'{name} (AUC={result["patient_roc_auc"]:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('Patient-level ROC Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curves(advanced_results, figsize=(15, 6)):
    """Plot Precision-Recall curves for all models"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # File-level PR curves
    for name, result in advanced_results.items():
        if result['file_pr_auc'] > 0:
            precision, recall, _ = precision_recall_curve(result['y_test'], result['file_probs'])
            ax1.plot(recall, precision, label=f'{name} (AUC={result["file_pr_auc"]:.3f})')
    
    ax1.set_xlabel('Recall')
    ax1.set_ylabel('Precision')
    ax1.set_title('File-level Precision-Recall Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Patient-level PR curves
    for name, result in advanced_results.items():
        if result['patient_pr_auc'] > 0:
            precision, recall, _ = precision_recall_curve(result['patient_true_labels'], result['patient_probs'])
            ax2.plot(recall, precision, label=f'{name} (AUC={result["patient_pr_auc"]:.3f})')
    
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.set_title('Patient-level Precision-Recall Curves')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('precision_recall_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_metrics(advanced_results, figsize=(15, 10)):
    """Plot performance metrics comparison"""
    models = list(advanced_results.keys())
    
    # Extract metrics
    file_metrics = {
        'Sensitivity': [advanced_results[m]['file_metrics']['sensitivity'] for m in models],
        'Specificity': [advanced_results[m]['file_metrics']['specificity'] for m in models],
        'Precision': [advanced_results[m]['file_metrics']['precision'] for m in models],
        'F2-Score': [advanced_results[m]['file_metrics']['f2'] for m in models],
        'Accuracy': [advanced_results[m]['file_metrics']['accuracy'] for m in models]
    }
    
    patient_metrics = {
        'Sensitivity': [advanced_results[m]['patient_metrics']['sensitivity'] for m in models],
        'Specificity': [advanced_results[m]['patient_metrics']['specificity'] for m in models],
        'Precision': [advanced_results[m]['patient_metrics']['precision'] for m in models],
        'F2-Score': [advanced_results[m]['patient_metrics']['f2'] for m in models],
        'Accuracy': [advanced_results[m]['patient_metrics']['accuracy'] for m in models]
    }
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    # File-level metrics
    file_df = pd.DataFrame(file_metrics, index=models)
    file_df.plot(kind='bar', ax=ax1, width=0.8)
    ax1.set_title('File-level Performance Metrics')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Target (80%)')
    
    # Patient-level metrics
    patient_df = pd.DataFrame(patient_metrics, index=models)
    patient_df.plot(kind='bar', ax=ax2, width=0.8)
    ax2.set_title('Patient-level Performance Metrics')
    ax2.set_ylabel('Score')
    ax2.set_ylim(0, 1)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Clinical Target (80%)')
    
    plt.tight_layout()
    plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🚀 STARTING COMPLETE TB DETECTION ANALYSIS")
    print("=" * 60)
    
    # Load processed data
    try:
        data = np.load('full_dataset_processed.npz', allow_pickle=True)
        X = data['X']
        y = data['y']
        file_keys = data['file_keys']
        patient_ids = data['patient_ids']
        
        print(f"✅ Loaded processed dataset: {X.shape}")
        print(f"✅ Patients: {len(np.unique(patient_ids))}")
        print(f"✅ TB positive rate: {sum(y)/len(y)*100:.1f}%")
        
    except FileNotFoundError:
        print("❌ Processed dataset not found. Run run_full_analysis.py first.")
        return
    
    # Patient-level split
    X_train, X_test, y_train, y_test, train_patients, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    print(f"\\n🔄 Patient-level split completed")
    print(f"📊 Train: {len(X_train)} files from {len(np.unique(train_patients))} patients")
    print(f"📊 Test: {len(X_test)} files from {len(np.unique(test_patients))} patients")
    print(f"📈 Train TB rate: {sum(y_train)/len(y_train)*100:.1f}%")
    print(f"📈 Test TB rate: {sum(y_test)/len(y_test)*100:.1f}%")
    
    # Apply preprocessing
    print("\\n🔄 Applying preprocessing...")
    
    # Remove features with zero variance
    var_selector = VarianceThreshold(threshold=0.001)
    X_train_filtered = var_selector.fit_transform(X_train)
    X_test_filtered = var_selector.transform(X_test)
    
    print(f"📊 Features after variance filtering: {X_train_filtered.shape[1]} (was {X_train.shape[1]})")
    
    # Apply SMOTE for class balancing
    if len(np.unique(y_train)) > 1 and np.sum(y_train) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        k_neighbors = min(5, min_samples - 1)
        
        if k_neighbors > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            X_train_balanced, y_train_balanced = smote.fit_resample(X_train_filtered, y_train)
            print(f"✅ SMOTE applied: {len(X_train_balanced)} samples")
        else:
            X_train_balanced, y_train_balanced = X_train_filtered, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_filtered, y_train
    
    # Feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_filtered)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(1000, X_train_scaled.shape[1]))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"✅ Feature selection: {X_train_selected.shape[1]} features selected")
    
    # Define models
    pos_weight = len(y_train_balanced[y_train_balanced == 0]) / len(y_train_balanced[y_train_balanced == 1]) if len(y_train_balanced[y_train_balanced == 1]) > 0 else 1.0
    
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            min_samples_split=5,
            subsample=0.8,
            random_state=42
        ),
        
        "Logistic Regression": LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.1,
            class_weight='balanced',
            random_state=42
        ),
        
        "SVM": SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            probability=True,
            class_weight='balanced',
            random_state=42
        )
    }
    
    # Add XGBoost if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_weight,
            random_state=42,
            eval_metric='logloss'
        )
    
    # Train and evaluate models
    print("\\n🚀 Training and evaluating models...")
    results = {}
    
    for name, model in models.items():
        print(f"🔄 Training: {name}")
        model.fit(X_train_selected, y_train_balanced)
        
        result = evaluate_advanced_model(
            model, X_test_selected, y_test, test_patients, name
        )
        results[name] = result
        
        print(f"  ✅ Patient Sensitivity: {result['patient_metrics']['sensitivity']:.3f}")
        print(f"  ✅ Patient Specificity: {result['patient_metrics']['specificity']:.3f}")
        print(f"  ✅ Patient F2-Score: {result['patient_metrics']['f2']:.3f}")
    
    # Generate visualizations
    print("\\n📊 Generating visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Confusion Matrices
    print("📊 Creating confusion matrices...")
    plot_confusion_matrices(results)
    
    # 2. ROC Curves
    print("📊 Creating ROC curves...")
    plot_roc_curves(results)
    
    # 3. Precision-Recall Curves
    print("📊 Creating Precision-Recall curves...")
    plot_precision_recall_curves(results)
    
    # 4. Performance Metrics
    print("📊 Creating performance metrics plot...")
    plot_performance_metrics(results)
    
    # Generate detailed report
    print("\\n📋 Generating detailed report...")
    
    # Create results summary
    summary_data = []
    for name, result in results.items():
        summary_data.append({
            'Model': name,
            'Patient Sensitivity': f"{result['patient_metrics']['sensitivity']:.3f}",
            'Patient Specificity': f"{result['patient_metrics']['specificity']:.3f}",
            'Patient Precision': f"{result['patient_metrics']['precision']:.3f}",
            'Patient F2-Score': f"{result['patient_metrics']['f2']:.3f}",
            'Patient ROC-AUC': f"{result['patient_roc_auc']:.3f}",
            'Patient PR-AUC': f"{result['patient_pr_auc']:.3f}",
            'Clinical Target (≥80%)': '✅' if result['patient_metrics']['sensitivity'] >= 0.8 else '❌',
            'TB Patients Detected': f"{result['tp_p']}/{result['tp_p'] + result['fn_p']}"
        })
    
    results_df = pd.DataFrame(summary_data)
    print("\\n" + "="*80)
    print("📋 FINAL RESULTS SUMMARY")
    print("="*80)
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv('tb_detection_results.csv', index=False)
    print("\\n✅ Results saved to 'tb_detection_results.csv'")
    
    # Best model
    best_model = max(results.items(), key=lambda x: x[1]['patient_metrics']['sensitivity'])
    print(f"\\n🏆 BEST MODEL: {best_model[0]}")
    print(f"   Patient Sensitivity: {best_model[1]['patient_metrics']['sensitivity']:.3f}")
    print(f"   Patient Specificity: {best_model[1]['patient_metrics']['specificity']:.3f}")
    print(f"   Patient F2-Score: {best_model[1]['patient_metrics']['f2']:.3f}")
    
    print("\\n" + "="*80)
    print("🎉 COMPLETE TB DETECTION ANALYSIS FINISHED!")
    print("="*80)
    print("📊 Generated files:")
    print("   - confusion_matrices.png")
    print("   - roc_curves.png") 
    print("   - precision_recall_curves.png")
    print("   - performance_metrics.png")
    print("   - tb_detection_results.csv")

if __name__ == "__main__":
    main()