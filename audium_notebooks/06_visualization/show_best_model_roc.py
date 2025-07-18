#!/usr/bin/env python3
"""
Display ROC curve for the best WHO-compliant model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
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
    
    return (
        X[train_mask], X[test_mask],
        y[train_mask], y[test_mask],
        patient_ids[train_mask], patient_ids[test_mask]
    )

def preprocess_data(X_train, y_train, X_test):
    """Preprocessing pipeline"""
    # Variance filtering
    var_selector = VarianceThreshold(threshold=0.01)
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    # Balanced sampling
    if len(np.unique(y_train)) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        if min_samples > 10:
            smote = SMOTE(sampling_strategy=0.3, k_neighbors=min(5, min_samples - 1), random_state=42)
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
    
    return X_train_selected, X_test_selected, y_train_balanced

def get_patient_level_predictions(y_probs, y_test, patient_ids_test):
    """Convert file-level predictions to patient-level"""
    unique_patients = np.unique(patient_ids_test)
    patient_true_labels = []
    patient_probs = []
    
    for patient in unique_patients:
        patient_mask = patient_ids_test == patient
        patient_true = int(np.any(y_test[patient_mask]))
        patient_prob = np.max(y_probs[patient_mask])  # Use max probability
        
        patient_true_labels.append(patient_true)
        patient_probs.append(patient_prob)
    
    return np.array(patient_true_labels), np.array(patient_probs)

def plot_best_model_roc():
    """Plot ROC curve for the best WHO-compliant model"""
    
    # Load data
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
    
    # Create splits
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42
    )
    
    # Preprocess data
    X_train_processed, X_test_processed, y_train_processed = preprocess_data(
        X_train, y_train, X_test
    )
    
    # Train the best model (Regularized Logistic Regression)
    best_model = LogisticRegression(
        penalty='elasticnet',
        l1_ratio=0.7,
        C=0.01,
        class_weight={0: 1, 1: 3},
        solver='saga',
        max_iter=2000,
        random_state=42
    )
    
    print("🔄 Training best model (Regularized Logistic Regression)...")
    best_model.fit(X_train_processed, y_train_processed)
    
    # Get predictions
    y_probs_test = best_model.predict_proba(X_test_processed)[:, 1]
    
    # Convert to patient-level
    patient_true_labels, patient_probs = get_patient_level_predictions(
        y_probs_test, y_test, test_patients
    )
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(patient_true_labels, patient_probs)
    roc_auc = auc(fpr, tpr)
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    
    # Plot ROC curve
    plt.plot(fpr, tpr, color='blue', lw=3, 
             label=f'WHO-Compliant Model (AUC = {roc_auc:.3f})')
    
    # Plot diagonal reference line
    plt.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.7,
             label='Random Classifier (AUC = 0.500)')
    
    # Mark the optimal operating point (WHO compliance threshold = 0.810)
    optimal_threshold = 0.810
    
    # Find the point on ROC curve closest to this threshold
    threshold_idx = np.argmin(np.abs(thresholds - optimal_threshold))
    optimal_fpr = fpr[threshold_idx]
    optimal_tpr = tpr[threshold_idx]
    
    plt.plot(optimal_fpr, optimal_tpr, 'ro', markersize=10, 
             label=f'WHO Operating Point (Threshold = {optimal_threshold:.3f})')
    
    # Add annotations
    plt.annotate(f'WHO Compliance Point\\nSensitivity: {optimal_tpr:.3f}\\nSpecificity: {1-optimal_fpr:.3f}',
                xy=(optimal_fpr, optimal_tpr), xytext=(optimal_fpr + 0.2, optimal_tpr - 0.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))
    
    # Formatting
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
    plt.title('ROC Curve: Best WHO-Compliant TB Detection Model\\n(Regularized Logistic Regression)', 
              fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add performance text
    plt.text(0.02, 0.98, f'Patient-Level Performance:\\n' +
                         f'• Total Patients: {len(patient_true_labels)}\\n' +
                         f'• TB Patients: {np.sum(patient_true_labels)}\\n' +
                         f'• Non-TB Patients: {len(patient_true_labels) - np.sum(patient_true_labels)}\\n' +
                         f'• ROC-AUC: {roc_auc:.3f}',
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # Add WHO targets
    plt.axhline(y=0.9, color='green', linestyle=':', alpha=0.7, label='WHO Sensitivity Target (90%)')
    plt.axvline(x=0.3, color='green', linestyle=':', alpha=0.7, label='WHO Specificity Target (70%)')
    
    plt.tight_layout()
    plt.savefig('best_model_roc_curve.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed performance
    print(f"\\n📊 BEST MODEL ROC ANALYSIS")
    print(f"=" * 50)
    print(f"Model: Regularized Logistic Regression")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"WHO Threshold: {optimal_threshold:.3f}")
    print(f"Sensitivity at WHO threshold: {optimal_tpr:.3f}")
    print(f"Specificity at WHO threshold: {1-optimal_fpr:.3f}")
    print(f"WHO Compliance: {'✅ YES' if optimal_tpr >= 0.9 and (1-optimal_fpr) >= 0.7 else '❌ NO'}")
    
    # Save ROC data
    roc_data = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })
    roc_data.to_csv('best_model_roc_data.csv', index=False)
    
    print(f"\\n📁 Files generated:")
    print(f"   - best_model_roc_curve.png")
    print(f"   - best_model_roc_data.csv")

if __name__ == "__main__":
    plot_best_model_roc()