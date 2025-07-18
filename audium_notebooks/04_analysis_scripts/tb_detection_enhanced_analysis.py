#!/usr/bin/env python3
"""
Enhanced TB Detection Analysis with WHO Compliance Optimization
Target: 90% Sensitivity, 70% Specificity
Comprehensive algorithm comparison with robust evaluation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (confusion_matrix, roc_curve, precision_recall_curve, 
                           auc, roc_auc_score, average_precision_score, 
                           classification_report, accuracy_score, f1_score)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Optional XGBoost and LightGBM
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

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
    """Calculate patient-level metrics using majority voting"""
    unique_patients = np.unique(patient_ids)
    patient_predictions = []
    patient_true_labels = []
    
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        # Majority vote for predictions
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

def advanced_preprocessing(X_train, y_train, X_test, method='robust'):
    """Advanced preprocessing with multiple options"""
    print(f"  🔄 Advanced preprocessing ({method})...")
    
    # Remove low variance features
    var_selector = VarianceThreshold(threshold=0.01)
    X_train_var = var_selector.fit_transform(X_train)
    X_test_var = var_selector.transform(X_test)
    
    print(f"    After variance filtering: {X_train_var.shape[1]} features")
    
    # Handle class imbalance
    if len(np.unique(y_train)) > 1:
        min_samples = min(np.sum(y_train), len(y_train) - np.sum(y_train))
        if min_samples > 10:
            # Try different sampling strategies
            if method == 'smote':
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))
            elif method == 'borderline':
                sampler = BorderlineSMOTE(random_state=42, k_neighbors=min(5, min_samples - 1))
            elif method == 'adasyn':
                sampler = ADASYN(random_state=42, n_neighbors=min(5, min_samples - 1))
            elif method == 'smotetomek':
                sampler = SMOTETomek(random_state=42)
            else:
                sampler = SMOTE(random_state=42, k_neighbors=min(5, min_samples - 1), sampling_strategy=0.5)
            
            X_train_balanced, y_train_balanced = sampler.fit_resample(X_train_var, y_train)
        else:
            X_train_balanced, y_train_balanced = X_train_var, y_train
    else:
        X_train_balanced, y_train_balanced = X_train_var, y_train
    
    print(f"    After balanced sampling: {X_train_balanced.shape[0]} samples")
    
    # Scaling
    if method == 'robust':
        scaler = RobustScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    elif method == 'power':
        scaler = PowerTransformer(method='yeo-johnson')
    else:
        scaler = RobustScaler()
    
    X_train_scaled = scaler.fit_transform(X_train_balanced)
    X_test_scaled = scaler.transform(X_test_var)
    
    # Feature selection
    n_features = min(1000, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"    Final features: {X_train_selected.shape[1]}")
    
    return X_train_selected, X_test_selected, y_train_balanced, scaler, var_selector, selector

def create_comprehensive_models():
    """Create comprehensive set of models for comparison"""
    models = {}
    
    # Logistic Regression variants
    models["LR_L1"] = LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42)
    models["LR_L2"] = LogisticRegression(penalty='l2', C=0.1, solver='liblinear', random_state=42)
    models["LR_Elastic"] = LogisticRegression(penalty='elasticnet', l1_ratio=0.5, C=0.1, solver='saga', max_iter=2000, random_state=42)
    
    # Tree-based models
    models["Random_Forest"] = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    models["Gradient_Boosting"] = GradientBoostingClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
    models["AdaBoost"] = AdaBoostClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
    models["Decision_Tree"] = DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
    
    # SVM variants
    models["SVM_RBF"] = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
    models["SVM_Linear"] = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
    models["SVM_Poly"] = SVC(kernel='poly', degree=3, C=1.0, probability=True, random_state=42)
    
    # Neural networks
    models["MLP_Small"] = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    models["MLP_Large"] = MLPClassifier(hidden_layer_sizes=(200, 100), max_iter=1000, random_state=42)
    models["MLP_Deep"] = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1000, random_state=42)
    
    # Other classifiers
    models["Naive_Bayes"] = GaussianNB()
    models["KNN"] = KNeighborsClassifier(n_neighbors=5)
    models["Ridge"] = RidgeClassifier(alpha=1.0, random_state=42)
    
    # Ensemble methods
    base_models = [
        ('lr', LogisticRegression(C=0.1, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)),
        ('svm', SVC(probability=True, random_state=42))
    ]
    models["Voting_Soft"] = VotingClassifier(base_models, voting='soft')
    models["Voting_Hard"] = VotingClassifier(base_models, voting='hard')
    
    # Calibrated classifiers
    models["Calibrated_RF"] = CalibratedClassifierCV(RandomForestClassifier(n_estimators=100, random_state=42), cv=3)
    models["Calibrated_SVM"] = CalibratedClassifierCV(SVC(random_state=42), cv=3)
    
    # XGBoost and LightGBM if available
    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42)
        models["XGBoost_Regularized"] = XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05, 
            reg_alpha=0.5, reg_lambda=0.5, random_state=42
        )
    
    if LIGHTGBM_AVAILABLE:
        models["LightGBM"] = LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, verbose=-1)
    
    return models

def who_threshold_optimization(model, X_val, y_val, patient_ids_val, target_sensitivity=0.90, target_specificity=0.70):
    """Optimize threshold for WHO compliance with comprehensive search"""
    thresholds = np.linspace(0.01, 0.99, 199)
    results = []
    
    y_probs = model.predict_proba(X_val)[:, 1]
    
    for threshold in thresholds:
        y_pred_thresh = (y_probs >= threshold).astype(int)
        
        sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn = calculate_patient_metrics(
            y_pred_thresh, y_val, patient_ids_val
        )
        
        # WHO compliance check
        who_compliant = sensitivity >= target_sensitivity and specificity >= target_specificity
        
        # Multi-objective scoring
        if who_compliant:
            score = 1000 + (sensitivity + specificity) * 100  # Bonus for compliance
        else:
            # Penalty for missing targets
            sens_penalty = max(0, target_sensitivity - sensitivity) * 500
            spec_penalty = max(0, target_specificity - specificity) * 300
            score = (sensitivity + specificity) * 100 - sens_penalty - spec_penalty
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'accuracy': accuracy,
            'f1_score': f1,
            'who_compliant': who_compliant,
            'score': score,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
    
    # Find best threshold
    best_result = max(results, key=lambda x: x['score'])
    
    return best_result['threshold'], pd.DataFrame(results)

def comprehensive_evaluation(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Comprehensive model evaluation with all metrics"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    # Patient-level metrics
    sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    # WHO compliance
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # ROC and PR curves
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
        
        # Calculate curves
        fpr, tpr, _ = roc_curve(patient_true, patient_probs)
        precision_curve, recall_curve, _ = precision_recall_curve(patient_true, patient_probs)
    else:
        roc_auc = pr_auc = 0
        fpr = tpr = precision_curve = recall_curve = np.array([])
    
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
        'patient_probs': patient_probs,
        'fpr': fpr, 'tpr': tpr,
        'precision_curve': precision_curve, 'recall_curve': recall_curve
    }

def create_comprehensive_plots(results, threshold_analyses):
    """Create comprehensive visualization plots"""
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: WHO Compliance Scatter
    ax1 = plt.subplot(3, 3, 1)
    models = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    compliant = [r['who_compliant'] for r in results]
    
    colors = ['green' if c else 'red' for c in compliant]
    scatter = ax1.scatter(specificities, sensitivities, c=colors, s=100, alpha=0.7)
    
    # WHO compliance zone
    ax1.axhline(y=0.90, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity (90%)')
    ax1.axvline(x=0.70, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity (70%)')
    ax1.fill_between([0.70, 1.0], 0.90, 1.0, alpha=0.2, color='green', label='WHO Compliant Zone')
    
    ax1.set_xlabel('Specificity')
    ax1.set_ylabel('Sensitivity')
    ax1.set_title('WHO Compliance Status')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Plot 2: ROC Curves
    ax2 = plt.subplot(3, 3, 2)
    for result in results:
        if len(result['fpr']) > 0 and result['roc_auc'] > 0:
            ax2.plot(result['fpr'], result['tpr'], 
                    label=f"{result['model_name']} (AUC={result['roc_auc']:.3f})", 
                    linewidth=2 if result['who_compliant'] else 1)
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Precision-Recall Curves
    ax3 = plt.subplot(3, 3, 3)
    for result in results:
        if len(result['precision_curve']) > 0 and result['pr_auc'] > 0:
            ax3.plot(result['recall_curve'], result['precision_curve'], 
                    label=f"{result['model_name']} (AUC={result['pr_auc']:.3f})", 
                    linewidth=2 if result['who_compliant'] else 1)
    
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curves')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Performance Metrics Comparison
    ax4 = plt.subplot(3, 3, 4)
    x = np.arange(len(models))
    width = 0.2
    
    ax4.bar(x - width, sensitivities, width, label='Sensitivity', alpha=0.8)
    ax4.bar(x, specificities, width, label='Specificity', alpha=0.8)
    ax4.bar(x + width, [r['accuracy'] for r in results], width, label='Accuracy', alpha=0.8)
    
    ax4.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax4.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    ax4.set_xlabel('Models')
    ax4.set_ylabel('Performance')
    ax4.set_title('Performance Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: F1 Score and ROC AUC
    ax5 = plt.subplot(3, 3, 5)
    f1_scores = [r['f1_score'] for r in results]
    roc_aucs = [r['roc_auc'] for r in results]
    
    ax5.bar(x - width/2, f1_scores, width, label='F1 Score', alpha=0.8)
    ax5.bar(x + width/2, roc_aucs, width, label='ROC AUC', alpha=0.8)
    
    ax5.set_xlabel('Models')
    ax5.set_ylabel('Score')
    ax5.set_title('F1 Score and ROC AUC')
    ax5.set_xticks(x)
    ax5.set_xticklabels(models, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Threshold Analysis for Best Model
    ax6 = plt.subplot(3, 3, 6)
    best_idx = np.argmax([r['sensitivity'] * r['specificity'] for r in results])
    best_thresh_df = threshold_analyses[best_idx]
    
    ax6.plot(best_thresh_df['threshold'], best_thresh_df['sensitivity'], 
             'g-', label='Sensitivity', linewidth=2)
    ax6.plot(best_thresh_df['threshold'], best_thresh_df['specificity'], 
             'r-', label='Specificity', linewidth=2)
    ax6.plot(best_thresh_df['threshold'], best_thresh_df['f1_score'], 
             'b-', label='F1 Score', linewidth=2)
    
    ax6.axhline(y=0.90, color='green', linestyle='--', alpha=0.7)
    ax6.axhline(y=0.70, color='red', linestyle='--', alpha=0.7)
    
    optimal_thresh = results[best_idx]['threshold']
    ax6.axvline(x=optimal_thresh, color='black', linestyle=':', alpha=0.8, 
                label=f'Optimal ({optimal_thresh:.3f})')
    
    ax6.set_xlabel('Decision Threshold')
    ax6.set_ylabel('Performance')
    ax6.set_title(f'Threshold Analysis: {results[best_idx]["model_name"]}')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Confusion Matrix Heatmap for Best Model
    ax7 = plt.subplot(3, 3, 7)
    best_result = results[best_idx]
    cm = np.array([[best_result['tn'], best_result['fp']], 
                   [best_result['fn'], best_result['tp']]])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No TB', 'TB'], yticklabels=['No TB', 'TB'], ax=ax7)
    ax7.set_title(f'Confusion Matrix: {best_result["model_name"]}')
    ax7.set_xlabel('Predicted')
    ax7.set_ylabel('Actual')
    
    # Plot 8: Model Ranking
    ax8 = plt.subplot(3, 3, 8)
    ranking_scores = [r['sensitivity'] * r['specificity'] for r in results]
    sorted_indices = np.argsort(ranking_scores)[::-1]
    
    sorted_models = [models[i] for i in sorted_indices]
    sorted_scores = [ranking_scores[i] for i in sorted_indices]
    sorted_compliant = [compliant[i] for i in sorted_indices]
    
    colors = ['green' if c else 'red' for c in sorted_compliant]
    bars = ax8.barh(range(len(sorted_models)), sorted_scores, color=colors, alpha=0.7)
    
    ax8.set_yticks(range(len(sorted_models)))
    ax8.set_yticklabels(sorted_models)
    ax8.set_xlabel('Sensitivity × Specificity')
    ax8.set_title('Model Ranking')
    ax8.grid(True, alpha=0.3)
    
    # Plot 9: WHO Compliance Summary
    ax9 = plt.subplot(3, 3, 9)
    compliant_count = sum(compliant)
    non_compliant_count = len(compliant) - compliant_count
    
    ax9.pie([compliant_count, non_compliant_count], 
            labels=[f'WHO Compliant\n({compliant_count})', f'Non-Compliant\n({non_compliant_count})'],
            colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
    ax9.set_title('WHO Compliance Summary')
    
    plt.tight_layout()
    plt.savefig('tb_detection_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    print("🎯 ENHANCED TB DETECTION ANALYSIS")
    print("Target: 90% Sensitivity, 70% Specificity (WHO Guidelines)")
    print("Comprehensive algorithm comparison with robust evaluation")
    print("=" * 80)
    
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
        print("❌ File 'full_dataset_processed.npz' not found")
        print("Please ensure the embeddings file is available")
        return
    
    # Create train/validation/test splits
    X_temp, X_test, y_temp, y_test, patients_temp, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val, train_patients, val_patients = create_patient_level_split(
        X_temp, y_temp, patients_temp, test_size=0.25, random_state=42
    )
    
    print(f"\n📊 Train: {len(X_train)} files, {len(np.unique(train_patients))} patients")
    print(f"📊 Validation: {len(X_val)} files, {len(np.unique(val_patients))} patients")
    print(f"📊 Test: {len(X_test)} files, {len(np.unique(test_patients))} patients")
    
    # Preprocessing
    X_train_processed, X_val_processed, y_train_processed, scaler, var_selector, selector = advanced_preprocessing(
        X_train, y_train, X_val, method='robust'
    )
    
    # Apply preprocessing to test set
    X_test_var = var_selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_var)
    X_test_processed = selector.transform(X_test_scaled)
    
    # Create comprehensive model set
    models = create_comprehensive_models()
    
    print(f"\n🔄 Training {len(models)} models...")
    
    results = []
    threshold_analyses = []
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        try:
            model.fit(X_train_processed, y_train_processed)
            
            # WHO-optimized threshold
            optimal_threshold, thresh_df = who_threshold_optimization(
                model, X_val_processed, y_val, val_patients
            )
            
            print(f"    Optimal threshold: {optimal_threshold:.3f}")
            
            # Comprehensive evaluation
            result = comprehensive_evaluation(
                model, X_test_processed, y_test, test_patients, name, optimal_threshold
            )
            
            results.append(result)
            threshold_analyses.append(thresh_df)
            
            compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
            print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
            
        except Exception as e:
            print(f"    ❌ Failed to train {name}: {str(e)}")
            continue
    
    # Results summary
    print("\n" + "="*100)
    print("📋 COMPREHENSIVE TB DETECTION RESULTS")
    print("="*100)
    
    summary_data = []
    for result in results:
        summary_data.append({
            'Model': result['model_name'],
            'Threshold': f"{result['threshold']:.3f}",
            'Sensitivity': f"{result['sensitivity']:.3f}",
            'Specificity': f"{result['specificity']:.3f}",
            'Precision': f"{result['precision']:.3f}",
            'NPV': f"{result['npv']:.3f}",
            'Accuracy': f"{result['accuracy']:.3f}",
            'F1-Score': f"{result['f1_score']:.3f}",
            'ROC-AUC': f"{result['roc_auc']:.3f}",
            'PR-AUC': f"{result['pr_auc']:.3f}",
            'WHO Compliant': '✅' if result['who_compliant'] else '❌'
        })
    
    results_df = pd.DataFrame(summary_data)
    print(results_df.to_string(index=False))
    
    # Find WHO compliant models
    compliant_models = [r for r in results if r['who_compliant']]
    
    if compliant_models:
        best_model = max(compliant_models, key=lambda x: (x['sensitivity'] + x['specificity']) / 2)
        print(f"\n🏆 BEST WHO-COMPLIANT MODEL: {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (≥90% ✅)")
        print(f"   Specificity: {best_model['specificity']:.3f} (≥70% ✅)")
        print(f"   Precision: {best_model['precision']:.3f}")
        print(f"   F1-Score: {best_model['f1_score']:.3f}")
        print(f"   ROC-AUC: {best_model['roc_auc']:.3f}")
        print(f"   Threshold: {best_model['threshold']:.3f}")
        
        print(f"\n🎉 WHO-COMPLIANT MODELS: {len(compliant_models)}/{len(results)}")
        
    else:
        best_model = max(results, key=lambda x: x['sensitivity'] + x['specificity'])
        print(f"\n🏆 BEST MODEL (Closest to WHO): {best_model['model_name']}")
        print(f"   Sensitivity: {best_model['sensitivity']:.3f} (Target: ≥90%)")
        print(f"   Specificity: {best_model['specificity']:.3f} (Target: ≥70%)")
        
        sens_gap = max(0, 0.90 - best_model['sensitivity'])
        spec_gap = max(0, 0.70 - best_model['specificity'])
        print(f"   Gap: Sensitivity -{sens_gap:.3f}, Specificity -{spec_gap:.3f}")
    
    # Save results
    results_df.to_csv('tb_detection_comprehensive_results.csv', index=False)
    
    # Create comprehensive visualization
    print("\n📊 Creating comprehensive visualization...")
    create_comprehensive_plots(results, threshold_analyses)
    
    print("\n" + "="*100)
    print("🎉 ENHANCED TB DETECTION ANALYSIS COMPLETE!")
    print("="*100)
    print("📊 Files generated:")
    print("   - tb_detection_comprehensive_results.csv")
    print("   - tb_detection_comprehensive_analysis.png")

if __name__ == "__main__":
    main()