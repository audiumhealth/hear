#!/usr/bin/env python3
"""
TB Detection Analysis for New UCSF R2D2 Training Dataset
Based on 04_analysis_scripts/quick_tb_analysis.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, fbeta_score, roc_auc_score, accuracy_score, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

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
        # For very small datasets, use smaller test size
        if len(patients_array) < 5:
            test_size = min(test_size, 1/len(patients_array))
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

def optimize_model_for_targets(X_train, y_train, X_test, y_test, test_patients, target_specificity=0.7):
    """Optimize model to achieve target specificity (≥70%) then maximize sensitivity"""
    print(f"\\n🎯 Optimizing for ≥{target_specificity*100:.0f}% specificity")
    
    # Test different models and parameters
    models_to_test = [
        ('Random Forest', RandomForestClassifier(random_state=42, n_jobs=-1)),
        ('Gradient Boosting', GradientBoostingClassifier(random_state=42)),
        ('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
        ('SVM', SVC(random_state=42, probability=True)),
        ('MLP', MLPClassifier(random_state=42, max_iter=1000))
    ]
    
    best_model = None
    best_results = None
    best_score = 0
    
    for model_name, model in models_to_test:
        print(f"  🔍 Testing {model_name}...")
        
        try:
            # Fit model
            model.fit(X_train, y_train)
            
            # Evaluate
            results = evaluate_model(model, X_test, y_test, test_patients, model_name)
            
            # Check if it meets specificity requirement
            if results['specificity'] >= target_specificity:
                # Score by sensitivity if specificity requirement is met
                score = results['sensitivity']
                print(f"    ✅ Specificity: {results['specificity']:.3f}, Sensitivity: {results['sensitivity']:.3f}")
                
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_results = results
                    
            else:
                print(f"    ❌ Specificity: {results['specificity']:.3f} < {target_specificity:.3f}")
                
        except Exception as e:
            print(f"    ❌ Error: {e}")
    
    return best_model, best_results

def create_visualizations(results_list, output_dir):
    """Create performance visualizations"""
    print("\\n📊 Creating visualizations...")
    
    # Create results directory
    results_dir = os.path.join(output_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Performance comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Extract metrics
    models = [r['model_name'] for r in results_list]
    sensitivities = [r['sensitivity'] for r in results_list]
    specificities = [r['specificity'] for r in results_list]
    roc_aucs = [r['roc_auc'] for r in results_list]
    f2_scores = [r['f2_score'] for r in results_list]
    
    # Sensitivity comparison
    axes[0, 0].bar(models, sensitivities, color='lightblue')
    axes[0, 0].set_title('Sensitivity (Recall)')
    axes[0, 0].set_ylabel('Sensitivity')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Target ≥70%')
    axes[0, 0].legend()
    
    # Specificity comparison
    axes[0, 1].bar(models, specificities, color='lightgreen')
    axes[0, 1].set_title('Specificity')
    axes[0, 1].set_ylabel('Specificity')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='Target ≥70%')
    axes[0, 1].legend()
    
    # ROC AUC comparison
    axes[1, 0].bar(models, roc_aucs, color='lightcoral')
    axes[1, 0].set_title('ROC AUC')
    axes[1, 0].set_ylabel('ROC AUC')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # F2 Score comparison
    axes[1, 1].bar(models, f2_scores, color='lightyellow')
    axes[1, 1].set_title('F2 Score')
    axes[1, 1].set_ylabel('F2 Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC curves
    plt.figure(figsize=(10, 8))
    for result in results_list:
        if len(np.unique(result['patient_true_labels'])) > 1:
            fpr, tpr, _ = roc_curve(result['patient_true_labels'], result['patient_probs'])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {roc_auc:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(results_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confusion matrices
    n_models = len(results_list)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results_list):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        cm = result['cm']
        if cm.shape == (2, 2):
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{result['model_name']}\\nSens: {result['sensitivity']:.3f}, Spec: {result['specificity']:.3f}")
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        else:
            ax.text(0.5, 0.5, 'No valid confusion matrix', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"{result['model_name']}\\n(Insufficient data)")
    
    # Hide unused subplots
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to {results_dir}")

def main():
    parser = argparse.ArgumentParser(description='TB Detection Analysis for New UCSF R2D2 Dataset')
    parser.add_argument('--dataset', choices=['small', 'medium', 'mini', 'all'], default='small',
                       help='Dataset size to analyze')
    parser.add_argument('--optimize', action='store_true',
                       help='Run optimization for target performance')
    
    args = parser.parse_args()
    
    # Set up paths
    output_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    embeddings_path = os.path.join(output_dir, 'data', 'new_ucsf_embeddings.npz')
    metadata_path = os.path.join(output_dir, 'data', 'new_ucsf_embeddings_metadata.csv')
    
    # Load patient labels
    if args.dataset == 'small':
        labels_path = os.path.join(output_dir, 'data', 'clean_small_test_patients.csv')
    elif args.dataset == 'medium':
        labels_path = os.path.join(output_dir, 'data', 'clean_medium_test_patients.csv')
    elif args.dataset == 'mini':
        labels_path = os.path.join(output_dir, 'data', 'mini_test_patients.csv')
    else:  # all
        labels_path = os.path.join(output_dir, 'data', 'all_clean_patients.csv')
    
    print(f"🎯 TB Detection Analysis - {args.dataset.upper()} Dataset")
    print("=" * 60)
    
    # Load data
    X, y, patient_ids = load_embeddings_and_labels(embeddings_path, metadata_path, labels_path)
    
    # Patient-level split
    X_train, X_test, y_train, y_test, train_patients, test_patients = create_patient_level_split(
        X, y, patient_ids, test_size=0.2, random_state=42
    )
    
    print(f"\\n📊 Train: {len(X_train)} clips, {len(np.unique(train_patients))} patients")
    print(f"📊 Test: {len(X_test)} clips, {len(np.unique(test_patients))} patients")
    
    # Quick preprocessing
    print("\\n🔄 Preprocessing...")
    
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
    
    # Train and evaluate models
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
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
        ),
        "SVM": SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=42
        ),
        "MLP": MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            random_state=42,
            max_iter=1000
        )
    }
    
    print(f"\\n🤖 Training {len(models)} models...")
    results_list = []
    
    for model_name, model in models.items():
        print(f"  🔄 Training {model_name}...")
        
        try:
            # Train model
            model.fit(X_train_selected, y_train_balanced)
            
            # Evaluate
            results = evaluate_model(model, X_test_selected, y_test, test_patients, model_name)
            results_list.append(results)
            
            print(f"    ✅ Sensitivity: {results['sensitivity']:.3f}, Specificity: {results['specificity']:.3f}")
            
        except Exception as e:
            print(f"    ❌ Error training {model_name}: {e}")
    
    # Optimization if requested
    if args.optimize:
        print("\\n🎯 Running optimization...")
        best_model, best_results = optimize_model_for_targets(
            X_train_selected, y_train_balanced, X_test_selected, y_test, test_patients
        )
        
        if best_results:
            print(f"\\n🏆 Best optimized model: {best_results['model_name']}")
            print(f"   Sensitivity: {best_results['sensitivity']:.3f}")
            print(f"   Specificity: {best_results['specificity']:.3f}")
            results_list.append(best_results)
    
    # Create comprehensive results summary
    print("\\n📋 FINAL RESULTS SUMMARY")
    print("=" * 60)
    
    results_df = pd.DataFrame([
        {
            'Model': r['model_name'],
            'Sensitivity': f"{r['sensitivity']:.3f}",
            'Specificity': f"{r['specificity']:.3f}",
            'Precision': f"{r['precision']:.3f}",
            'F2 Score': f"{r['f2_score']:.3f}",
            'ROC AUC': f"{r['roc_auc']:.3f}",
            'PR AUC': f"{r['pr_auc']:.3f}"
        }
        for r in results_list
    ])
    
    print(results_df.to_string(index=False))
    
    # Save results
    results_path = os.path.join(output_dir, 'results', f'{args.dataset}_dataset_results.csv')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    results_df.to_csv(results_path, index=False)
    print(f"\\n✅ Results saved to: {results_path}")
    
    # Create visualizations
    create_visualizations(results_list, output_dir)
    
    print(f"\\n🎉 Analysis complete! Check {output_dir}/results/ for outputs.")

if __name__ == "__main__":
    main()