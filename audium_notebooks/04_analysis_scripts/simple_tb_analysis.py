#!/usr/bin/env python3
"""
Simple TB Detection Analysis with Basic Libraries
WHO compliance analysis with minimal dependencies
"""

import numpy as np
import sys
import os

# Try to import required libraries
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("⚠️  pandas not available - results will be printed to console")

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("⚠️  matplotlib not available - no plots will be generated")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("❌ scikit-learn not available - cannot run analysis")
    sys.exit(1)

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

def simple_preprocessing(X_train, y_train, X_test):
    """Simple preprocessing"""
    print("  🔄 Preprocessing data...")
    
    # Remove features with zero variance
    var_mask = np.var(X_train, axis=0) > 0.01
    X_train_var = X_train[:, var_mask]
    X_test_var = X_test[:, var_mask]
    
    print(f"    After variance filtering: {X_train_var.shape[1]} features")
    
    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_var)
    X_test_scaled = scaler.transform(X_test_var)
    
    return X_train_scaled, X_test_scaled, y_train, scaler, var_mask

def optimize_threshold_for_who(model, X_val, y_val, patient_ids_val):
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

def evaluate_model_performance(model, X_test, y_test, patient_ids_test, model_name, threshold=0.5):
    """Evaluate model performance"""
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)
    
    sensitivity, specificity, precision, npv, accuracy, f1, tp, tn, fp, fn = calculate_patient_metrics(
        y_pred, y_test, patient_ids_test
    )
    
    who_compliant = sensitivity >= 0.90 and specificity >= 0.70
    
    # Patient-level ROC AUC
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
    else:
        roc_auc = 0
    
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
        'who_compliant': who_compliant,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def main():
    print("🎯 SIMPLE TB DETECTION ANALYSIS")
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
    
    # Preprocess
    X_train_processed, X_val_processed, y_train_processed, scaler, var_mask = simple_preprocessing(
        X_train, y_train, X_val
    )
    
    X_test_var = X_test[:, var_mask]
    X_test_processed = scaler.transform(X_test_var)
    
    # Define models
    models = {
        "Logistic_Regression": LogisticRegression(
            penalty='l2', C=0.1, class_weight='balanced', 
            max_iter=1000, random_state=42
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=10,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    }
    
    print(f"\n🔄 Training {len(models)} models...")
    
    results = []
    
    for name, model in models.items():
        print(f"\n  Training {name}...")
        
        try:
            model.fit(X_train_processed, y_train_processed)
            
            # Optimize threshold
            optimal_threshold = optimize_threshold_for_who(model, X_val_processed, y_val, val_patients)
            print(f"    Optimal threshold: {optimal_threshold:.3f}")
            
            # Evaluate
            result = evaluate_model_performance(
                model, X_test_processed, y_test, test_patients, name, optimal_threshold
            )
            results.append(result)
            
            compliance_status = "✅ COMPLIANT" if result['who_compliant'] else "❌ NOT COMPLIANT"
            print(f"    Sensitivity: {result['sensitivity']:.3f}, Specificity: {result['specificity']:.3f} {compliance_status}")
            
        except Exception as e:
            print(f"    ❌ Failed to train {name}: {str(e)}")
    
    # Results summary
    print("\n" + "="*80)
    print("📋 SIMPLE TB DETECTION RESULTS")
    print("="*80)
    
    # Print results table
    print(f"{'Model':<20} {'Thresh':<8} {'Sens':<6} {'Spec':<6} {'Prec':<6} {'F1':<6} {'AUC':<6} {'WHO':<5}")
    print("-" * 65)
    
    for result in results:
        print(f"{result['model_name']:<20} {result['threshold']:<8.3f} {result['sensitivity']:<6.3f} "
              f"{result['specificity']:<6.3f} {result['precision']:<6.3f} {result['f1_score']:<6.3f} "
              f"{result['roc_auc']:<6.3f} {'✅' if result['who_compliant'] else '❌':<5}")
    
    # Find best model
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
    
    # Save results if pandas available
    if PANDAS_AVAILABLE:
        try:
            import pandas as pd
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
            results_df.to_csv('simple_tb_analysis_results.csv', index=False)
            print("\n📊 Results saved to: simple_tb_analysis_results.csv")
        except Exception as e:
            print(f"\n⚠️  Could not save CSV file: {str(e)}")
    
    print("\n" + "="*80)
    print("🎉 SIMPLE TB DETECTION ANALYSIS COMPLETE!")
    print("="*80)

if __name__ == "__main__":
    main()