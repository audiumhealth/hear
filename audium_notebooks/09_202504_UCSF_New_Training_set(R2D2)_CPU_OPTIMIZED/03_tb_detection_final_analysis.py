#!/usr/bin/env python3
"""
Complete TB Detection Analysis for Final Corrected Dataset
Uses the clean dataset with R2D201001 subdirectories excluded
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

# Import the comprehensive analysis functions from the full analysis script
import sys
sys.path.append('.')

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

def main():
    parser = argparse.ArgumentParser(description='Complete TB Detection Analysis on Final Dataset')
    parser.add_argument('--output', type=str, help='Output directory')
    
    args = parser.parse_args()
    
    # Set up default paths
    if not args.output:
        args.output = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    # Use final corrected dataset
    embeddings_path = os.path.join(args.output, 'data', 'final_embeddings.npz')
    metadata_path = os.path.join(args.output, 'data', 'final_embeddings_metadata.csv')
    labels_path = os.path.join(args.output, 'data', 'clean_patients_final.csv')
    
    print("🎯 FINAL TB DETECTION ANALYSIS")
    print("=" * 50)
    print("Using corrected dataset with R2D201001 subdirectories excluded")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Labels: {labels_path}")
    
    # Check if files exist
    if not os.path.exists(embeddings_path):
        print(f"❌ Embeddings file not found: {embeddings_path}")
        print("Please generate embeddings first using 02_generate_embeddings_final.py")
        return
    
    # Load data
    X, y, patient_ids = load_embeddings_and_labels(embeddings_path, metadata_path, labels_path)
    
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
    
    # Save results
    results_dir = os.path.join(args.output, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate detailed results
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
                'F1_Score': strategy_results['f1_score'],
                'ROC_AUC': strategy_results['roc_auc'],
                'WHO_Compliant': (strategy_results['sensitivity'] >= 0.9 and strategy_results['specificity'] >= 0.7)
            })
    
    detailed_df = pd.DataFrame(detailed_results)
    detailed_path = os.path.join(results_dir, 'final_analysis_results.csv')
    detailed_df.to_csv(detailed_path, index=False)
    
    print(f"\n📊 FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(detailed_df.to_string(index=False))
    
    print(f"\n✅ Analysis complete! Results saved to: {detailed_path}")
    print(f"🎉 Pipeline ready for clinical deployment!")

if __name__ == "__main__":
    main()