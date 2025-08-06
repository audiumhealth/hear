#!/usr/bin/env python3
"""
Complete TB Detection Analysis for New UCSF R2D2 Training Dataset
Comprehensive analysis with WHO algorithm optimization and full reporting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Metrics imports
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, fbeta_score, roc_auc_score, accuracy_score, 
    classification_report, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold, RFE
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

def load_embeddings_and_labels(embeddings_path, metadata_path, labels_path):
    """Load embeddings and create labels"""
    print("📂 Loading embeddings and labels...")
    
    # Load embeddings
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    print(f"✅ Loaded {len(embeddings_data.files)} embedding files")
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_path)
    print(f"✅ Loaded metadata for {len(metadata_df)} files")
    
    # Load labels
    labels_df = pd.read_csv(labels_path)
    print(f"✅ Loaded labels for {len(labels_df)} patients")
    
    # Create mapping from patient_id to label
    label_map = dict(zip(labels_df['StudyID'], labels_df['Label']))
    
    # Process embeddings
    X_list = []
    y_list = []
    patient_ids_list = []
    
    for file_key in embeddings_data.files:
        patient_id = file_key.split('/')[0]
        
        if patient_id not in label_map:
            print(f"⚠️  No label found for patient {patient_id}")
            continue
        
        embeddings = embeddings_data[file_key]
        label = 1 if label_map[patient_id] == 'TB Positive' else 0
        
        # Each embedding represents one clip
        for embedding in embeddings:
            X_list.append(embedding)
            y_list.append(label)
            patient_ids_list.append(patient_id)
    
    X = np.array(X_list)
    y = np.array(y_list)
    patient_ids = np.array(patient_ids_list)
    
    print(f"✅ Final dataset: {X.shape[0]} clips, {len(np.unique(patient_ids))} patients")
    print(f"✅ TB Positive: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    
    return X, y, patient_ids

def create_patient_level_split(X, y, patient_ids, test_size=0.2, random_state=42):
    """Create patient-level train/test split to prevent data leakage"""
    unique_patients = np.unique(patient_ids)
    
    # Calculate patient-level labels
    patient_labels = {}
    for patient in unique_patients:
        patient_mask = patient_ids == patient
        patient_labels[patient] = int(np.any(y[patient_mask]))
    
    # Split patients
    patients_array = np.array(list(patient_labels.keys()))
    labels_array = np.array(list(patient_labels.values()))
    
    # Check if we have enough patients for each class to do stratified split
    unique_labels, counts = np.unique(labels_array, return_counts=True)
    min_class_size = np.min(counts)
    
    if len(unique_labels) > 1 and min_class_size >= 2:
        # Use stratified split if we have at least 2 patients per class
        train_patients, test_patients = train_test_split(
            patients_array, test_size=test_size, stratify=labels_array, random_state=random_state
        )
    else:
        # Use random split if we don't have enough patients for stratification
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

def evaluate_model_comprehensive(model, X_test, y_test, test_patients, model_name):
    """Comprehensive model evaluation with patient-level aggregation"""
    # File-level predictions
    y_pred_file = model.predict(X_test)
    y_prob_file = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred_file
    
    # Patient-level aggregation strategies
    unique_patients = np.unique(test_patients)
    
    # Strategy 1: Any positive (OR logic)
    patient_predictions_any = []
    patient_true_labels = []
    patient_probs_max = []
    patient_probs_mean = []
    
    # Strategy 2: Majority vote
    patient_predictions_majority = []
    
    # Strategy 3: Threshold-based
    patient_predictions_threshold = []
    
    for patient in unique_patients:
        patient_mask = test_patients == patient
        patient_files_pred = y_pred_file[patient_mask]
        patient_files_true = y_test[patient_mask]
        patient_files_prob = y_prob_file[patient_mask]
        
        # True label (any positive file makes patient positive)
        patient_true_any = int(np.any(patient_files_true))
        patient_true_labels.append(patient_true_any)
        
        # Strategy 1: Any positive
        patient_pred_any = int(np.any(patient_files_pred))
        patient_predictions_any.append(patient_pred_any)
        
        # Strategy 2: Majority vote
        patient_pred_majority = int(np.mean(patient_files_pred) > 0.5)
        patient_predictions_majority.append(patient_pred_majority)
        
        # Strategy 3: Threshold-based (e.g., >30% of clips positive)
        patient_pred_threshold = int(np.mean(patient_files_pred) > 0.3)
        patient_predictions_threshold.append(patient_pred_threshold)
        
        # Probability aggregation
        patient_prob_max = np.max(patient_files_prob)
        patient_prob_mean = np.mean(patient_files_prob)
        patient_probs_max.append(patient_prob_max)
        patient_probs_mean.append(patient_prob_mean)
    
    # Convert to arrays
    patient_predictions_any = np.array(patient_predictions_any)
    patient_predictions_majority = np.array(patient_predictions_majority)
    patient_predictions_threshold = np.array(patient_predictions_threshold)
    patient_true_labels = np.array(patient_true_labels)
    patient_probs_max = np.array(patient_probs_max)
    patient_probs_mean = np.array(patient_probs_mean)
    
    # Calculate metrics for each strategy
    strategies = [
        ("Any Positive", patient_predictions_any, patient_probs_max),
        ("Majority Vote", patient_predictions_majority, patient_probs_mean),
        ("Threshold (30%)", patient_predictions_threshold, patient_probs_mean)
    ]
    
    results = {}
    
    for strategy_name, predictions, probs in strategies:
        # Calculate confusion matrix
        cm = confusion_matrix(patient_true_labels, predictions)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
            f2 = fbeta_score(patient_true_labels, predictions, beta=2, average='weighted')
            
            # ROC AUC and PR AUC
            if len(np.unique(patient_true_labels)) > 1:
                roc_auc = roc_auc_score(patient_true_labels, probs)
                pr_auc = average_precision_score(patient_true_labels, probs)
            else:
                roc_auc = 0
                pr_auc = 0
            
            # Accuracy
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            
        else:
            tp = fn = tn = fp = 0
            sensitivity = specificity = precision = npv = f1 = f2 = roc_auc = pr_auc = accuracy = 0
        
        results[strategy_name] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'npv': npv,
            'f1_score': f1,
            'f2_score': f2,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'accuracy': accuracy,
            'tp': tp,
            'fn': fn,
            'tn': tn,
            'fp': fp,
            'cm': cm,
            'predictions': predictions,
            'probabilities': probs
        }
    
    return {
        'model_name': model_name,
        'strategies': results,
        'patient_true_labels': patient_true_labels,
        'unique_patients': unique_patients
    }

def optimize_who_algorithm(X_train, y_train, X_test, y_test, test_patients, target_sensitivity=0.9, min_specificity=0.7):
    """Optimize algorithm selection for WHO-like performance requirements"""
    print(f"\n🎯 WHO ALGORITHM OPTIMIZATION")
    print(f"Target: ≥{target_sensitivity*100:.0f}% sensitivity, ≥{min_specificity*100:.0f}% specificity")
    print("=" * 60)
    
    # WHO-optimized model configurations
    who_models = {
        "WHO-RF": RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "WHO-GB": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        ),
        "WHO-LR": LogisticRegression(
            penalty='l1',
            solver='liblinear',
            C=0.01,
            class_weight='balanced',
            random_state=42,
            max_iter=2000
        ),
        "WHO-SVM": SVC(
            kernel='rbf',
            C=0.1,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        "WHO-ET": ExtraTreesClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_models = {}
    
    for model_name, model in who_models.items():
        print(f"\n🔍 Optimizing {model_name}...")
        
        try:
            # Train model
            model.fit(X_train, y_train)
            
            # Comprehensive evaluation
            results = evaluate_model_comprehensive(model, X_test, y_test, test_patients, model_name)
            
            # Check each strategy for WHO compliance
            for strategy_name, strategy_results in results['strategies'].items():
                sens = strategy_results['sensitivity']
                spec = strategy_results['specificity']
                
                print(f"  {strategy_name}: Sens={sens:.3f}, Spec={spec:.3f}", end="")
                
                # Check WHO compliance
                if sens >= target_sensitivity and spec >= min_specificity:
                    print(" ✅ WHO COMPLIANT")
                    
                    # Calculate WHO score (prioritize sensitivity, then specificity)
                    who_score = sens + 0.5 * spec
                    
                    key = f"{model_name}_{strategy_name}"
                    best_models[key] = {
                        'model': model,
                        'strategy': strategy_name,
                        'results': strategy_results,
                        'who_score': who_score,
                        'full_results': results
                    }
                elif sens >= target_sensitivity:
                    print(f" ⚠️  Low specificity ({spec:.3f} < {min_specificity:.3f})")
                elif spec >= min_specificity:
                    print(f" ⚠️  Low sensitivity ({sens:.3f} < {target_sensitivity:.3f})")
                else:
                    print(f" ❌ Both metrics below threshold")
                    
        except Exception as e:
            print(f"❌ Error with {model_name}: {str(e)}")
    
    # Select best WHO-compliant model
    if best_models:
        best_key = max(best_models.keys(), key=lambda k: best_models[k]['who_score'])
        best_model_info = best_models[best_key]
        
        print(f"\n🏆 BEST WHO-COMPLIANT MODEL: {best_key}")
        print(f"   WHO Score: {best_model_info['who_score']:.3f}")
        print(f"   Sensitivity: {best_model_info['results']['sensitivity']:.3f}")
        print(f"   Specificity: {best_model_info['results']['specificity']:.3f}")
        print(f"   Precision: {best_model_info['results']['precision']:.3f}")
        print(f"   F1 Score: {best_model_info['results']['f1_score']:.3f}")
        
        return best_model_info, best_models
    else:
        print("\n❌ No WHO-compliant models found!")
        return None, best_models

def create_comprehensive_visualizations(all_results, who_results, output_dir):
    """Create comprehensive visualizations"""
    print("\n📊 Creating comprehensive visualizations...")
    
    # Create results directory
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Performance comparison across all models and strategies
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Collect data for visualization
    model_names = []
    sensitivities = []
    specificities = []
    precisions = []
    f1_scores = []
    roc_aucs = []
    
    for result in all_results:
        model_name = result['model_name']
        for strategy_name, strategy_results in result['strategies'].items():
            full_name = f"{model_name}_{strategy_name}"
            model_names.append(full_name)
            sensitivities.append(strategy_results['sensitivity'])
            specificities.append(strategy_results['specificity'])
            precisions.append(strategy_results['precision'])
            f1_scores.append(strategy_results['f1_score'])
            roc_aucs.append(strategy_results['roc_auc'])
    
    # Plot metrics
    axes[0, 0].bar(range(len(model_names)), sensitivities, color='lightblue', alpha=0.7)
    axes[0, 0].set_title('Sensitivity (Recall)')
    axes[0, 0].set_ylabel('Sensitivity')
    axes[0, 0].axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥90%')
    axes[0, 0].legend()
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    axes[0, 1].bar(range(len(model_names)), specificities, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Specificity')
    axes[0, 1].set_ylabel('Specificity')
    axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥70%')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    axes[0, 2].bar(range(len(model_names)), precisions, color='lightcoral', alpha=0.7)
    axes[0, 2].set_title('Precision')
    axes[0, 2].set_ylabel('Precision')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    axes[1, 0].bar(range(len(model_names)), f1_scores, color='lightyellow', alpha=0.7)
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    axes[1, 1].bar(range(len(model_names)), roc_aucs, color='lightpink', alpha=0.7)
    axes[1, 1].set_title('ROC AUC')
    axes[1, 1].set_ylabel('ROC AUC')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    # WHO compliance chart
    who_compliant = []
    for i, (sens, spec) in enumerate(zip(sensitivities, specificities)):
        compliant = sens >= 0.9 and spec >= 0.7
        who_compliant.append(1 if compliant else 0)
    
    axes[1, 2].bar(range(len(model_names)), who_compliant, color='lightsteelblue', alpha=0.7)
    axes[1, 2].set_title('WHO Compliance')
    axes[1, 2].set_ylabel('WHO Compliant (1=Yes, 0=No)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    
    # Set x-axis labels for all subplots
    for ax in axes.flat:
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_results)))
    
    for i, result in enumerate(all_results):
        model_name = result['model_name']
        patient_true_labels = result['patient_true_labels']
        
        for strategy_name, strategy_results in result['strategies'].items():
            if len(np.unique(patient_true_labels)) > 1:
                probabilities = strategy_results['probabilities']
                fpr, tpr, _ = roc_curve(patient_true_labels, probabilities)
                roc_auc = auc(fpr, tpr)
                
                label = f"{model_name}_{strategy_name} (AUC={roc_auc:.3f})"
                plt.plot(fpr, tpr, label=label, color=colors[i], alpha=0.8)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - Patient-Level Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'roc_curves_comprehensive.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Confusion matrices for WHO-compliant models
    if who_results:
        n_who_models = len(who_results)
        cols = min(3, n_who_models)
        rows = (n_who_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
        if n_who_models == 1:
            axes = [axes]
        elif rows == 1:
            axes = axes.reshape(1, -1)
        
        for i, (model_key, model_info) in enumerate(who_results.items()):
            if i >= len(axes.flat):
                break
                
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            cm = model_info['results']['cm']
            if cm.shape == (2, 2):
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"{model_key}\nSens: {model_info['results']['sensitivity']:.3f}, Spec: {model_info['results']['specificity']:.3f}")
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
            else:
                ax.text(0.5, 0.5, 'No valid confusion matrix', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"{model_key}\n(Insufficient data)")
        
        # Hide unused subplots
        for i in range(n_who_models, rows * cols):
            if i < len(axes.flat):
                row = i // cols
                col = i % cols
                ax = axes[row, col] if rows > 1 else axes[col]
                ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'who_compliant_confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"✅ Visualizations saved to {results_dir}")

def generate_comprehensive_report(all_results, who_results, output_dir):
    """Generate comprehensive analysis report"""
    print("\n📋 Generating comprehensive report...")
    
    # Create results directory
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # 1. Detailed results CSV
    detailed_results = []
    
    for result in all_results:
        model_name = result['model_name']
        for strategy_name, strategy_results in result['strategies'].items():
            detailed_results.append({
                'Model': model_name,
                'Strategy': strategy_name,
                'Sensitivity': strategy_results['sensitivity'],
                'Specificity': strategy_results['specificity'],
                'Precision': strategy_results['precision'],
                'NPV': strategy_results['npv'],
                'F1_Score': strategy_results['f1_score'],
                'F2_Score': strategy_results['f2_score'],
                'ROC_AUC': strategy_results['roc_auc'],
                'PR_AUC': strategy_results['pr_auc'],
                'Accuracy': strategy_results['accuracy'],
                'TP': strategy_results['tp'],
                'FP': strategy_results['fp'],
                'TN': strategy_results['tn'],
                'FN': strategy_results['fn'],
                'WHO_Compliant': (strategy_results['sensitivity'] >= 0.9 and strategy_results['specificity'] >= 0.7)
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(results_dir, 'detailed_analysis_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    # 2. WHO-compliant models summary
    if who_results:
        who_summary = []
        for model_key, model_info in who_results.items():
            results = model_info['results']
            who_summary.append({
                'Model_Strategy': model_key,
                'WHO_Score': model_info['who_score'],
                'Sensitivity': results['sensitivity'],
                'Specificity': results['specificity'],
                'Precision': results['precision'],
                'F1_Score': results['f1_score'],
                'ROC_AUC': results['roc_auc']
            })
        
        who_df = pd.DataFrame(who_summary)
        who_path = os.path.join(results_dir, 'who_compliant_models.csv')
        who_df.to_csv(who_path, index=False)
    
    # 3. Executive summary
    summary_path = os.path.join(results_dir, 'executive_summary.txt')
    
    with open(summary_path, 'w') as f:
        f.write("TB DETECTION ANALYSIS - EXECUTIVE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Dataset summary
        if all_results:
            f.write("DATASET SUMMARY\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total Patients: {len(all_results[0]['unique_patients'])}\n")
            f.write(f"TB Positive: {sum(all_results[0]['patient_true_labels'])}\n")
            f.write(f"TB Negative: {len(all_results[0]['patient_true_labels']) - sum(all_results[0]['patient_true_labels'])}\n\n")
        
        # Models analyzed
        f.write("MODELS ANALYZED\n")
        f.write("-" * 20 + "\n")
        unique_models = set()
        for result in all_results:
            unique_models.add(result['model_name'])
        f.write(f"Number of Models: {len(unique_models)}\n")
        f.write(f"Models: {', '.join(sorted(unique_models))}\n")
        f.write(f"Aggregation Strategies: Any Positive, Majority Vote, Threshold (30%)\n\n")
        
        # WHO compliance summary
        f.write("WHO COMPLIANCE SUMMARY\n")
        f.write("-" * 25 + "\n")
        
        total_combinations = len(detailed_results)
        who_compliant_count = sum(1 for r in detailed_results if r['WHO_Compliant'])
        
        f.write(f"Total Model-Strategy Combinations: {total_combinations}\n")
        f.write(f"WHO Compliant: {who_compliant_count}\n")
        f.write(f"WHO Compliance Rate: {who_compliant_count/total_combinations*100:.1f}%\n\n")
        
        if who_results:
            best_model = max(who_results.items(), key=lambda x: x[1]['who_score'])
            f.write("BEST WHO-COMPLIANT MODEL\n")
            f.write("-" * 25 + "\n")
            f.write(f"Model: {best_model[0]}\n")
            f.write(f"WHO Score: {best_model[1]['who_score']:.3f}\n")
            f.write(f"Sensitivity: {best_model[1]['results']['sensitivity']:.3f}\n")
            f.write(f"Specificity: {best_model[1]['results']['specificity']:.3f}\n")
            f.write(f"Precision: {best_model[1]['results']['precision']:.3f}\n")
            f.write(f"F1 Score: {best_model[1]['results']['f1_score']:.3f}\n\n")
        
        # Performance highlights
        f.write("PERFORMANCE HIGHLIGHTS\n")
        f.write("-" * 22 + "\n")
        
        # Best sensitivity
        best_sens = max(detailed_results, key=lambda x: x['Sensitivity'])
        f.write(f"Highest Sensitivity: {best_sens['Sensitivity']:.3f} ({best_sens['Model']}_{best_sens['Strategy']})\n")
        
        # Best specificity
        best_spec = max(detailed_results, key=lambda x: x['Specificity'])
        f.write(f"Highest Specificity: {best_spec['Specificity']:.3f} ({best_spec['Model']}_{best_spec['Strategy']})\n")
        
        # Best ROC AUC
        best_roc = max(detailed_results, key=lambda x: x['ROC_AUC'])
        f.write(f"Highest ROC AUC: {best_roc['ROC_AUC']:.3f} ({best_roc['Model']}_{best_roc['Strategy']})\n")
    
    print(f"✅ Comprehensive report generated")
    print(f"   Detailed results: {detailed_path}")
    print(f"   WHO summary: {who_path if who_results else 'None'}")
    print(f"   Executive summary: {summary_path}")
    
    return detailed_path, who_path if who_results else None, summary_path

def main():
    parser = argparse.ArgumentParser(description='Complete TB Detection Analysis')
    parser.add_argument('--embeddings', type=str, help='Path to embeddings file')
    parser.add_argument('--metadata', type=str, help='Path to metadata file')
    parser.add_argument('--labels', type=str, help='Path to labels file')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Set up default paths if not provided
    if not args.output:
        args.output = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    if not args.embeddings:
        args.embeddings = os.path.join(args.output, 'data', 'complete_embeddings.npz')
    
    if not args.metadata:
        args.metadata = os.path.join(args.output, 'data', 'complete_embeddings_metadata.csv')
    
    if not args.labels:
        args.labels = os.path.join(args.output, 'data', 'clean_patients_fixed.csv')
    
    print("🎯 COMPLETE TB DETECTION ANALYSIS")
    print("=" * 50)
    print(f"Embeddings: {args.embeddings}")
    print(f"Metadata: {args.metadata}")
    print(f"Labels: {args.labels}")
    print(f"Output: {args.output}")
    
    # Check if files exist
    if not os.path.exists(args.embeddings):
        print(f"❌ Embeddings file not found: {args.embeddings}")
        print("Please generate embeddings first using 02_generate_embeddings_full.py")
        return
    
    # Load data
    X, y, patient_ids = load_embeddings_and_labels(args.embeddings, args.metadata, args.labels)
    
    # Patient-level split
    X_train, X_test, y_train, y_test, train_patients, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    print(f"\n📊 Dataset Split:")
    print(f"Train: {len(X_train)} clips, {len(np.unique(train_patients))} patients")
    print(f"Test: {len(X_test)} clips, {len(np.unique(test_patients))} patients")
    
    # Preprocessing
    print("\n🔄 Preprocessing...")
    
    # Variance filtering
    var_selector = VarianceThreshold(threshold=0.001)
    X_train_filtered = var_selector.fit_transform(X_train)
    X_test_filtered = var_selector.transform(X_test)
    
    # SMOTE for balancing
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
    k_features = min(500, X_train_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=k_features)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_balanced)
    X_test_selected = selector.transform(X_test_scaled)
    
    print(f"✅ Final training shape: {X_train_selected.shape}")
    
    # Train comprehensive model suite
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
        "Logistic Regression": LogisticRegression(penalty='l1', solver='liblinear', C=0.1, class_weight='balanced', random_state=42),
        "SVM": SVC(kernel='rbf', C=1.0, gamma='scale', class_weight='balanced', probability=True, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', alpha=0.01, random_state=42, max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, class_weight='balanced', random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    }
    
    print(f"\n🤖 Training {len(models)} models...")
    all_results = []
    
    for model_name, model in models.items():
        print(f"  🔄 Training {model_name}...")
        
        try:
            # Train model
            model.fit(X_train_selected, y_train_balanced)
            
            # Comprehensive evaluation
            results = evaluate_model_comprehensive(model, X_test_selected, y_test, test_patients, model_name)
            all_results.append(results)
            
            # Print summary
            best_strategy = max(results['strategies'].items(), key=lambda x: x[1]['f1_score'])
            print(f"    ✅ Best strategy: {best_strategy[0]} (F1: {best_strategy[1]['f1_score']:.3f})")
            
        except Exception as e:
            print(f"    ❌ Error training {model_name}: {e}")
    
    # WHO algorithm optimization
    who_best, who_all = optimize_who_algorithm(
        X_train_selected, y_train_balanced, X_test_selected, y_test, test_patients
    )
    
    # Generate visualizations
    create_comprehensive_visualizations(all_results, who_all, args.output)
    
    # Generate comprehensive report
    generate_comprehensive_report(all_results, who_all, args.output)
    
    print(f"\n🎉 COMPLETE ANALYSIS FINISHED!")
    print(f"Check {args.output}/results/ for all outputs")

if __name__ == "__main__":
    main()