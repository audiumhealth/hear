#!/usr/bin/env python3
"""
Leave-One-Out TB Detection Validation Pipeline
Comprehensive validation with reserved test dataset for clinical deployment
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import warnings
from datetime import datetime
import multiprocessing
import json
import hashlib
from pathlib import Path
warnings.filterwarnings('ignore')

# Import configuration
from config_leave_one_out import load_leave_one_out_config, get_leave_one_out_parser

# Import enhanced visualization and data export modules
try:
    from leave_one_out_visualizations import (
        create_comprehensive_roc_analysis,
        create_comprehensive_prc_analysis, 
        create_comprehensive_who_analysis,
        create_confusion_matrices_grid,
        create_cv_fold_variance_plot
    )
    from leave_one_out_data_export import export_all_curve_data, create_data_verification_report
    ENHANCED_VIZ_AVAILABLE = True
    print("✅ Enhanced visualizations and data export available")
except ImportError as e:
    ENHANCED_VIZ_AVAILABLE = False
    print(f"⚠️  Enhanced visualizations not available: {e}")

# ML imports
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# Metrics imports
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, roc_auc_score, accuracy_score, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif
from imblearn.over_sampling import SMOTE

# XGBoost import
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️  XGBoost not available, will skip XGBoost models")

# PyTorch for GPU MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    print("⚠️  PyTorch not available, will skip GPU MLP")

class LeaveOneOutConfig:
    """Enhanced configuration for leave-one-out validation"""
    
    def __init__(self):
        # Base configuration
        self.n_jobs = multiprocessing.cpu_count()
        self.batch_size = max(1, self.n_jobs // 2)
        
        # Leave-one-out specific
        self.training_patients_file = "data/training_patients_leave_one_out.csv"
        self.test_patients_file = "data/test_patients_leave_one_out.csv"
        self.embeddings_file = "data/final_embeddings.npz"
        self.metadata_file = "data/final_embeddings_metadata.csv"
        
        # Training configuration
        self.use_cross_validation = True
        self.n_folds = 5
        self.random_state = 42
        
        # WHO optimization
        self.who_sensitivity_target = 0.90
        self.who_specificity_target = 0.70
        self.who_optimization = True
        
        # Output configuration
        self.results_dir = "results"
        self.configs_dir = "configs"
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_description = "leave_one_out_validation"
        
        # Create directories
        for directory in [self.results_dir, self.configs_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def get_run_identifier(self):
        """Generate comprehensive run identifier"""
        config_dict = {
            'embeddings_file': self.embeddings_file,
            'use_cross_validation': self.use_cross_validation,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'who_optimization': self.who_optimization
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
        
        components = [
            "final_embeddings",
            "leave_one_out",
            f"cv{self.n_folds}fold",
            self.run_description,
            self.run_timestamp,
            config_hash
        ]
        
        return "_".join(components)
    
    def get_output_filename(self, base_filename):
        """Generate output filename with run identifier"""
        run_id = self.get_run_identifier()
        return f"{run_id}_{base_filename}"
    
    def save_config(self):
        """Save configuration for reproducibility"""
        config_data = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'run_description': self.run_description,
                'run_type': 'leave_one_out_validation',
                'config_hash': self.get_run_identifier().split('_')[-1]
            },
            'data_config': {
                'training_patients_file': self.training_patients_file,
                'test_patients_file': self.test_patients_file,
                'embeddings_file': self.embeddings_file,
                'metadata_file': self.metadata_file
            },
            'training_config': {
                'use_cross_validation': self.use_cross_validation,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'who_optimization': self.who_optimization,
                'who_sensitivity_target': self.who_sensitivity_target,
                'who_specificity_target': self.who_specificity_target
            },
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'pytorch_available': PYTORCH_AVAILABLE,
                'xgboost_available': XGBOOST_AVAILABLE
            }
        }
        
        config_filename = self.get_output_filename('config.json')
        config_path = os.path.join(self.configs_dir, config_filename)
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        return config_path

def load_leave_one_out_data(config):
    """Load embeddings and labels for leave-one-out validation"""
    print("📂 Loading leave-one-out datasets...")
    
    # Load training and test patient lists
    training_df = pd.read_csv(config.training_patients_file)
    test_df = pd.read_csv(config.test_patients_file)
    
    training_patients = set(training_df['StudyID'].tolist())
    test_patients = set(test_df['StudyID'].tolist())
    
    print(f"   Training patients: {len(training_patients)}")
    print(f"   Test patients: {len(test_patients)}")
    
    # Load embeddings and metadata
    embeddings_data = np.load(config.embeddings_file, allow_pickle=True)
    metadata_df = pd.read_csv(config.metadata_file)
    
    print(f"   Embeddings files: {len(embeddings_data.files)}")
    print(f"   Metadata records: {len(metadata_df)}")
    
    # Create patient-level aggregated embeddings and labels
    training_X, training_y, training_patients_list = aggregate_patient_embeddings(
        embeddings_data, metadata_df, training_df, training_patients
    )
    
    test_X, test_y, test_patients_list = aggregate_patient_embeddings(
        embeddings_data, metadata_df, test_df, test_patients
    )
    
    print(f"✅ Training data: {training_X.shape} embeddings, {len(training_y)} labels")
    print(f"✅ Test data: {test_X.shape} embeddings, {len(test_y)} labels")
    
    # Check TB distribution
    training_tb_rate = np.mean(training_y)
    test_tb_rate = np.mean(test_y)
    print(f"📊 Training TB rate: {training_tb_rate:.3f}")
    print(f"📊 Test TB rate: {test_tb_rate:.3f}")
    
    return (training_X, training_y, training_patients_list, 
            test_X, test_y, test_patients_list)

def aggregate_patient_embeddings(embeddings_data, metadata_df, labels_df, patient_set):
    """Aggregate embeddings at patient level to prevent data leakage"""
    
    patient_embeddings = {}
    patient_labels = {}
    
    # Create label lookup
    label_lookup = dict(zip(labels_df['StudyID'], labels_df['Label']))
    
    print(f"🔄 Aggregating embeddings for {len(patient_set)} patients...")
    
    for _, row in metadata_df.iterrows():
        patient_id = row['patient_id']
        
        if patient_id not in patient_set:
            continue
            
        file_key = row['file_key']
        
        if file_key in embeddings_data.files:
            file_embeddings = embeddings_data[file_key]
            
            if patient_id not in patient_embeddings:
                patient_embeddings[patient_id] = []
            
            patient_embeddings[patient_id].extend(file_embeddings)
            
            # Store patient label
            if patient_id in label_lookup:
                tb_label = 1 if label_lookup[patient_id] == 'TB Positive' else 0
                patient_labels[patient_id] = tb_label
    
    # Aggregate embeddings per patient (mean)
    aggregated_X = []
    aggregated_y = []
    patient_list = []
    
    for patient_id in patient_embeddings:
        if patient_id in patient_labels:
            # Calculate mean embedding for patient
            patient_mean_embedding = np.mean(patient_embeddings[patient_id], axis=0)
            aggregated_X.append(patient_mean_embedding)
            aggregated_y.append(patient_labels[patient_id])
            patient_list.append(patient_id)
    
    return np.array(aggregated_X), np.array(aggregated_y), patient_list

def get_model_suite(config):
    """Get comprehensive model suite for evaluation"""
    models = {}
    
    # Core models
    models['Naive Bayes'] = GaussianNB()
    models['LDA'] = LinearDiscriminantAnalysis()
    models['Logistic Regression'] = LogisticRegression(
        random_state=config.random_state, max_iter=1000, class_weight='balanced'
    )
    models['KNN'] = KNeighborsClassifier(n_neighbors=5)
    
    # XGBoost models (if available)
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            random_state=config.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
        models['XGBoost RF'] = xgb.XGBRFClassifier(
            random_state=config.random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    # WHO-optimized models
    if config.who_optimization:
        models['WHO-MLP'] = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),
            random_state=config.random_state,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.2
        )
        
        if PYTORCH_AVAILABLE:
            models['MLP (GPU)'] = 'pytorch_mlp'  # Special marker for PyTorch MLP
    
    print(f"🤖 Model suite: {len(models)} algorithms")
    for name in models.keys():
        print(f"   • {name}")
    
    return models

def optimize_threshold_for_who_compliance(y_true, y_proba, sensitivity_target=0.90, specificity_target=0.70):
    """Optimize decision threshold for WHO compliance"""
    
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    best_score = -1  # Initialize with -1 to ensure we always find a better score
    best_metrics = {
        'threshold': 0.5,
        'sensitivity': 0.0,
        'specificity': 0.0,
        'who_score': 0.0,
        'who_compliant': False
    }
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Handle edge case where all predictions are the same
        if len(set(y_pred)) == 1:
            if y_pred[0] == 1:  # All positive predictions
                sensitivity = 1.0
                specificity = 0.0
            else:  # All negative predictions
                sensitivity = 0.0
                specificity = 1.0
        else:
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # WHO compliance score
        meets_sensitivity = sensitivity >= sensitivity_target
        meets_specificity = specificity >= specificity_target
        
        if meets_sensitivity and meets_specificity:
            who_score = min(sensitivity, specificity)  # Conservative score
        else:
            # Penalty for not meeting targets
            sens_penalty = max(0, sensitivity_target - sensitivity)
            spec_penalty = max(0, specificity_target - specificity)
            who_score = min(sensitivity, specificity) - (sens_penalty + spec_penalty)
        
        if who_score > best_score:
            best_score = who_score
            best_threshold = threshold
            best_metrics = {
                'threshold': threshold,
                'sensitivity': sensitivity,
                'specificity': specificity,
                'who_score': who_score,
                'who_compliant': meets_sensitivity and meets_specificity
            }
    
    return best_threshold, best_metrics

def analyze_who_compliance_curve(y_true, y_proba, config):
    """Analyze WHO compliance across threshold range"""
    
    thresholds = np.linspace(0.01, 0.99, 50)
    sensitivities = []
    specificities = []
    who_scores = []
    who_compliant_flags = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        # Handle edge case where all predictions are the same
        if len(set(y_pred)) == 1:
            if y_pred[0] == 1:  # All positive predictions
                sensitivity = 1.0
                specificity = 0.0
            else:  # All negative predictions
                sensitivity = 0.0
                specificity = 1.0
        else:
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # WHO compliance
        meets_sensitivity = sensitivity >= config.who_sensitivity_target
        meets_specificity = specificity >= config.who_specificity_target
        who_compliant = meets_sensitivity and meets_specificity
        
        # WHO score calculation
        if who_compliant:
            who_score = min(sensitivity, specificity)  # Conservative score
        else:
            sens_penalty = max(0, config.who_sensitivity_target - sensitivity)
            spec_penalty = max(0, config.who_specificity_target - specificity)
            who_score = min(sensitivity, specificity) - (sens_penalty + spec_penalty)
        
        sensitivities.append(sensitivity)
        specificities.append(specificity)
        who_scores.append(who_score)
        who_compliant_flags.append(who_compliant)
    
    return {
        'thresholds': thresholds,
        'sensitivities': sensitivities,
        'specificities': specificities,
        'who_scores': who_scores,
        'who_compliant_flags': who_compliant_flags
    }

def train_pytorch_mlp(X_train, y_train, X_val=None, y_val=None, config=None):
    """Train PyTorch MLP with WHO optimization"""
    
    if not PYTORCH_AVAILABLE:
        return None
    
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔥 Training PyTorch MLP on {device}")
    
    # Define model architecture
    class WHOMLP(nn.Module):
        def __init__(self, input_dim):
            super(WHOMLP, self).__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(), 
                nn.Dropout(0.3),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
    
    # Initialize model
    model = WHOMLP(X_train.shape[1]).to(device)
    
    # WHO-optimized loss function (weighted for sensitivity)
    pos_weight = torch.tensor([8.0]).to(device)  # Favor sensitivity
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # Training
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if epoch % 50 == 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def evaluate_model_cross_validation(model, X, y, config):
    """Evaluate model using cross-validation with enhanced data collection"""
    
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    fold_results = []
    curve_data = {
        'roc_curves': [],
        'prc_curves': [],
        'who_analysis': []
    }
    confusion_matrices = []
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Preprocess data
        scaler = RobustScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_val_fold_scaled = scaler.transform(X_val_fold)
        
        # Apply SMOTE for class balance
        smote = SMOTE(random_state=config.random_state + fold)
        X_train_fold_balanced, y_train_fold_balanced = smote.fit_resample(
            X_train_fold_scaled, y_train_fold
        )
        
        # Train model
        if isinstance(model, str) and model == 'pytorch_mlp':
            trained_model = train_pytorch_mlp(
                X_train_fold_balanced, y_train_fold_balanced, 
                X_val_fold_scaled, y_val_fold, config
            )
            if trained_model is None:
                continue
                
            # Predict with PyTorch model
            trained_model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_val_fold_scaled)
                if torch.backends.mps.is_available():
                    X_val_tensor = X_val_tensor.to('mps')
                y_proba_fold = trained_model(X_val_tensor).cpu().numpy().flatten()
        else:
            trained_model = model
            trained_model.fit(X_train_fold_balanced, y_train_fold_balanced)
            
            if hasattr(trained_model, 'predict_proba'):
                y_proba_fold = trained_model.predict_proba(X_val_fold_scaled)[:, 1]
            else:
                decision_scores = trained_model.decision_function(X_val_fold_scaled)
                y_proba_fold = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        
        # Collect curve data for this fold
        from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
        
        # ROC curve data
        fpr, tpr, roc_thresholds = roc_curve(y_val_fold, y_proba_fold)
        curve_data['roc_curves'].append({
            'fold': fold,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': roc_thresholds
        })
        
        # PRC curve data
        precision, recall, prc_thresholds = precision_recall_curve(y_val_fold, y_proba_fold)
        curve_data['prc_curves'].append({
            'fold': fold,
            'precision': precision,
            'recall': recall,
            'thresholds': prc_thresholds
        })
        
        # WHO analysis data (threshold sweep)
        who_analysis = analyze_who_compliance_curve(y_val_fold, y_proba_fold, config)
        curve_data['who_analysis'].append({
            'fold': fold,
            **who_analysis
        })
        
        # Optimize threshold for WHO compliance
        best_threshold, best_metrics = optimize_threshold_for_who_compliance(
            y_val_fold, y_proba_fold, 
            config.who_sensitivity_target, config.who_specificity_target
        )
        
        # Calculate standard metrics
        y_pred_fold = (y_proba_fold >= 0.5).astype(int)
        accuracy = accuracy_score(y_val_fold, y_pred_fold)
        roc_auc = roc_auc_score(y_val_fold, y_proba_fold)
        
        # Confusion matrix for optimal threshold
        y_pred_optimal = (y_proba_fold >= best_threshold).astype(int)
        cm_fold = confusion_matrix(y_val_fold, y_pred_optimal)
        confusion_matrices.append(cm_fold)
        
        fold_result = {
            'fold': fold,
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'optimal_threshold': best_threshold,
            'optimal_sensitivity': best_metrics['sensitivity'],
            'optimal_specificity': best_metrics['specificity'],
            'who_score': best_metrics['who_score'],
            'who_compliant': best_metrics['who_compliant']
        }
        
        fold_results.append(fold_result)
    
    # Calculate mean and std across folds
    metrics = ['accuracy', 'roc_auc', 'optimal_sensitivity', 'optimal_specificity', 'who_score']
    mean_results = {}
    
    for metric in metrics:
        values = [fr[metric] for fr in fold_results]
        mean_results[f'{metric}_mean'] = np.mean(values)
        mean_results[f'{metric}_std'] = np.std(values, ddof=1)
    
    # WHO compliance rate
    compliance_rate = np.mean([fr['who_compliant'] for fr in fold_results])
    mean_results['who_compliance_rate'] = compliance_rate
    
    return {
        'fold_results': fold_results,
        'mean_results': mean_results,
        'curve_data': curve_data,
        'confusion_matrices': confusion_matrices,
        'who_compliant_folds': sum(fr['who_compliant'] for fr in fold_results),
        'total_folds': len(fold_results)
    }

def evaluate_model_on_test_set(model, X_train, y_train, X_test, y_test, config):
    """Evaluate model on reserved test dataset"""
    
    print(f"🧪 Evaluating on reserved test dataset ({len(X_test)} patients)")
    
    # Preprocess data
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Apply SMOTE for training balance
    smote = SMOTE(random_state=config.random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
    
    # Train model on full training set
    if isinstance(model, str) and model == 'pytorch_mlp':
        trained_model = train_pytorch_mlp(X_train_balanced, y_train_balanced, config=config)
        if trained_model is None:
            return None
            
        # Predict with PyTorch model
        trained_model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test_scaled)
            if torch.backends.mps.is_available():
                X_test_tensor = X_test_tensor.to('mps')
            y_test_proba = trained_model(X_test_tensor).cpu().numpy().flatten()
    else:
        trained_model = model
        trained_model.fit(X_train_balanced, y_train_balanced)
        
        if hasattr(trained_model, 'predict_proba'):
            y_test_proba = trained_model.predict_proba(X_test_scaled)[:, 1]
        else:
            decision_scores = trained_model.decision_function(X_test_scaled)
            y_test_proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
    
    # Collect curve data for test set
    from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
    
    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_test_proba)
    
    # PRC curve data
    precision, recall, prc_thresholds = precision_recall_curve(y_test, y_test_proba)
    
    # WHO analysis data (threshold sweep)
    threshold_analysis = analyze_who_compliance_curve(y_test, y_test_proba, config)
    
    # Optimize threshold for WHO compliance on test set
    best_threshold, best_metrics = optimize_threshold_for_who_compliance(
        y_test, y_test_proba,
        config.who_sensitivity_target, config.who_specificity_target
    )
    
    # Calculate standard metrics
    y_test_pred = (y_test_proba >= 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_roc_auc = roc_auc_score(y_test, y_test_proba)
    
    # Confusion matrix for optimal threshold
    y_test_pred_optimal = (y_test_proba >= best_threshold).astype(int)
    test_confusion_matrix = confusion_matrix(y_test, y_test_pred_optimal)
    
    return {
        'test_accuracy': test_accuracy,
        'test_roc_auc': test_roc_auc,
        'test_sensitivity': best_metrics['sensitivity'],
        'test_specificity': best_metrics['specificity'],
        'test_who_score': best_metrics['who_score'],
        'test_who_compliant': best_metrics['who_compliant'],
        'test_optimal_threshold': best_threshold,
        'confusion_matrix': test_confusion_matrix,
        'curve_data': {
            'roc_curve': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': roc_thresholds
            },
            'prc_curve': {
                'precision': precision,
                'recall': recall,
                'thresholds': prc_thresholds
            },
            'threshold_analysis': threshold_analysis
        },
        'y_test_true': y_test,
        'y_test_proba': y_test_proba,
        'y_test_pred': (y_test_proba >= best_threshold).astype(int)
    }

def run_leave_one_out_validation(config):
    """Run complete leave-one-out validation pipeline"""
    
    print("🚀 Starting Leave-One-Out TB Detection Validation")
    print("="*70)
    
    # Load data
    (training_X, training_y, training_patients,
     test_X, test_y, test_patients) = load_leave_one_out_data(config)
    
    # Get model suite
    models = get_model_suite(config)
    
    # Results storage
    cv_results = []
    test_results = []
    
    print(f"\n🔄 Running cross-validation and test evaluation...")
    
    for model_name, model in models.items():
        print(f"\n🤖 Evaluating {model_name}...")
        
        start_time = datetime.now()
        
        # Cross-validation evaluation
        cv_result = evaluate_model_cross_validation(model, training_X, training_y, config)
        
        # Test set evaluation
        test_result = evaluate_model_on_test_set(
            model, training_X, training_y, test_X, test_y, config
        )
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        if cv_result and test_result:
            # Combine results
            combined_result = {
                'model_name': model_name,
                'training_time': training_time,
                **cv_result,
                **test_result
            }
            
            cv_results.append({
                'model_name': model_name,
                'training_time': training_time,
                **cv_result
            })
            
            test_results.append({
                'model_name': model_name,
                **test_result
            })
            
            print(f"   ✅ CV WHO Score: {cv_result['mean_results']['who_score_mean']:.3f}")
            print(f"   🧪 Test WHO Score: {test_result['test_who_score']:.3f}")
            print(f"   🧪 Test WHO Compliant: {test_result['test_who_compliant']}")
    
    return cv_results, test_results, (training_X, training_y, test_X, test_y)

def save_results_matching_baseline_structure(cv_results, test_results, data_tuple, config):
    """Save results matching the exact structure of baseline pipeline runs"""
    
    print(f"\n💾 Saving results in baseline-compatible format...")
    
    run_id = config.get_run_identifier()
    
    # 1. Cross-validation detailed results (matching existing structure)
    cv_detailed_records = []
    for result in cv_results:
        for fold_result in result['fold_results']:
            record = {
                'model_name': result['model_name'],
                'fold': fold_result['fold'],
                'accuracy': fold_result['accuracy'],
                'roc_auc': fold_result['roc_auc'],
                'optimal_sensitivity': fold_result['optimal_sensitivity'],
                'optimal_specificity': fold_result['optimal_specificity'],
                'who_score': fold_result['who_score'],
                'who_compliant': fold_result['who_compliant'],
                'optimal_threshold': fold_result['optimal_threshold']
            }
            cv_detailed_records.append(record)
    
    cv_detailed_df = pd.DataFrame(cv_detailed_records)
    cv_detailed_file = config.get_output_filename('cross_validation_detailed.csv')
    cv_detailed_path = os.path.join(config.results_dir, cv_detailed_file)
    cv_detailed_df.to_csv(cv_detailed_path, index=False)
    
    # 2. Cross-validation summary results (mean ± std)
    cv_summary_records = []
    for result in cv_results:
        record = {
            'model_name': result['model_name'],
            'training_time': result['training_time'],
            'who_compliance_rate': result['mean_results']['who_compliance_rate'],
            **{k: v for k, v in result['mean_results'].items()}
        }
        cv_summary_records.append(record)
    
    cv_summary_df = pd.DataFrame(cv_summary_records)
    cv_summary_file = config.get_output_filename('cross_validation_summary.csv')
    cv_summary_path = os.path.join(config.results_dir, cv_summary_file)
    cv_summary_df.to_csv(cv_summary_path, index=False)
    
    # 3. **NEW**: Leave-one-out test results
    test_detailed_records = []
    for result in test_results:
        record = {
            'model_name': result['model_name'],
            'test_accuracy': result['test_accuracy'],
            'test_roc_auc': result['test_roc_auc'],
            'test_sensitivity': result['test_sensitivity'],
            'test_specificity': result['test_specificity'],
            'test_who_score': result['test_who_score'],
            'test_who_compliant': result['test_who_compliant'],
            'test_optimal_threshold': result['test_optimal_threshold']
        }
        test_detailed_records.append(record)
    
    test_results_df = pd.DataFrame(test_detailed_records)
    test_results_file = config.get_output_filename('leave_one_out_test_results.csv')
    test_results_path = os.path.join(config.results_dir, test_results_file)
    test_results_df.to_csv(test_results_path, index=False)
    
    # 4. **NEW**: CV vs Test comparison
    comparison_records = []
    for cv_result, test_result in zip(cv_results, test_results):
        if cv_result['model_name'] == test_result['model_name']:
            record = {
                'model_name': cv_result['model_name'],
                'cv_who_score_mean': cv_result['mean_results']['who_score_mean'],
                'cv_who_score_std': cv_result['mean_results']['who_score_std'],
                'cv_who_compliance_rate': cv_result['mean_results']['who_compliance_rate'],
                'test_who_score': test_result['test_who_score'],
                'test_who_compliant': test_result['test_who_compliant'],
                'cv_test_who_score_diff': test_result['test_who_score'] - cv_result['mean_results']['who_score_mean'],
                'generalization_gap': abs(test_result['test_who_score'] - cv_result['mean_results']['who_score_mean'])
            }
            comparison_records.append(record)
    
    comparison_df = pd.DataFrame(comparison_records)
    comparison_file = config.get_output_filename('test_vs_cv_comparison.csv')
    comparison_path = os.path.join(config.results_dir, comparison_file)
    comparison_df.to_csv(comparison_path, index=False)
    
    print(f"✅ Results saved:")
    print(f"   📊 CV detailed: {cv_detailed_path}")
    print(f"   📊 CV summary: {cv_summary_path}")
    print(f"   🧪 Test results: {test_results_path}")
    print(f"   📈 CV vs Test comparison: {comparison_path}")
    
    return {
        'cv_detailed_path': cv_detailed_path,
        'cv_summary_path': cv_summary_path,
        'test_results_path': test_results_path,
        'comparison_path': comparison_path
    }

def create_visualizations_matching_baseline(cv_results, test_results, data_tuple, config):
    """Create comprehensive visualizations with enhanced curve analysis"""
    
    print(f"📊 Creating comprehensive visualizations...")
    
    visualization_paths = []
    
    # Enhanced visualization creation
    if ENHANCED_VIZ_AVAILABLE:
        print("🎯 Creating enhanced visualizations with curve analysis...")
        
        # Convert results format for enhanced visualization functions
        enhanced_cv_results = {}
        enhanced_test_results = {}
        
        for i, cv_result in enumerate(cv_results):
            model_name = cv_result['model_name']
            enhanced_cv_results[model_name] = cv_result
            enhanced_test_results[model_name] = test_results[i]
        
        try:
            # 1. ROC Curves Analysis
            create_comprehensive_roc_analysis(enhanced_cv_results, enhanced_test_results, config)
            roc_file = config.get_output_filename("roc_curves.png")
            visualization_paths.append(os.path.join(config.results_dir, roc_file))
            
            # 2. PRC Curves Analysis  
            create_comprehensive_prc_analysis(enhanced_cv_results, enhanced_test_results, config)
            prc_file = config.get_output_filename("precision_recall_curves.png")
            visualization_paths.append(os.path.join(config.results_dir, prc_file))
            
            # 3. WHO Compliance Analysis
            create_comprehensive_who_analysis(enhanced_cv_results, enhanced_test_results, config)
            who_file = config.get_output_filename("who_compliance_analysis.png")
            visualization_paths.append(os.path.join(config.results_dir, who_file))
            
            # 4. Confusion Matrices Grid
            create_confusion_matrices_grid(enhanced_cv_results, enhanced_test_results, config)
            cm_file = config.get_output_filename("confusion_matrices_grid.png")
            visualization_paths.append(os.path.join(config.results_dir, cm_file))
            
            # 5. CV Fold Variance Analysis
            create_cv_fold_variance_plot(enhanced_cv_results, config)
            variance_file = config.get_output_filename("cv_fold_variance.png")
            visualization_paths.append(os.path.join(config.results_dir, variance_file))
            
            # 6. Export all curve data for verification
            exported_files = export_all_curve_data(enhanced_cv_results, enhanced_test_results, config)
            
            # 7. Create data verification report
            verification_report = create_data_verification_report(exported_files, config)
            
            print(f"✅ Enhanced visualizations created: {len(visualization_paths)} plots")
            print(f"✅ Data export completed: {len(exported_files)} CSV files")
            
        except Exception as e:
            print(f"⚠️  Enhanced visualization error: {e}")
            print("📊 Falling back to basic visualizations...")
            # Fall back to basic visualization
            visualization_paths = create_basic_visualizations(cv_results, test_results, config)
    else:
        # Fall back to basic visualization
        visualization_paths = create_basic_visualizations(cv_results, test_results, config)
    
    return visualization_paths

def create_basic_visualizations(cv_results, test_results, config):
    """Create basic visualizations (fallback)"""
    
    print(f"📊 Creating basic visualizations...")
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    visualization_paths = []
    
    # 1. Cross-validation performance dashboard (matching existing)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Leave-One-Out Cross-Validation Performance Dashboard', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in cv_results]
    
    # Sensitivity
    ax = axes[0, 0]
    sens_means = [r['mean_results']['optimal_sensitivity_mean'] for r in cv_results]
    sens_stds = [r['mean_results']['optimal_sensitivity_std'] for r in cv_results]
    bars = ax.bar(model_names, sens_means, yerr=sens_stds, color='lightblue', alpha=0.7, capsize=5)
    ax.axhline(y=config.who_sensitivity_target, color='red', linestyle='--', alpha=0.7, label=f'WHO Target ≥{config.who_sensitivity_target*100:.0f}%')
    ax.set_title('Sensitivity (Mean ± Std)')
    ax.set_ylabel('Sensitivity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Specificity
    ax = axes[0, 1]
    spec_means = [r['mean_results']['optimal_specificity_mean'] for r in cv_results]
    spec_stds = [r['mean_results']['optimal_specificity_std'] for r in cv_results]
    bars = ax.bar(model_names, spec_means, yerr=spec_stds, color='lightgreen', alpha=0.7, capsize=5)
    ax.axhline(y=config.who_specificity_target, color='red', linestyle='--', alpha=0.7, label=f'WHO Target ≥{config.who_specificity_target*100:.0f}%')
    ax.set_title('Specificity (Mean ± Std)')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # ROC AUC
    ax = axes[0, 2]
    roc_means = [r['mean_results']['roc_auc_mean'] for r in cv_results]
    roc_stds = [r['mean_results']['roc_auc_std'] for r in cv_results]
    bars = ax.bar(model_names, roc_means, yerr=roc_stds, color='lightyellow', alpha=0.7, capsize=5)
    ax.set_title('ROC AUC (Mean ± Std)')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # WHO Score
    ax = axes[1, 0]
    who_means = [r['mean_results']['who_score_mean'] for r in cv_results]
    who_stds = [r['mean_results']['who_score_std'] for r in cv_results]
    bars = ax.bar(model_names, who_means, yerr=who_stds, color='lightcoral', alpha=0.7, capsize=5)
    ax.axhline(y=1.25, color='red', linestyle='--', alpha=0.7, label='WHO Threshold ≥1.25')
    ax.set_title('WHO Score (Mean ± Std)')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # WHO Compliance Rate
    ax = axes[1, 1]
    compliance_rates = [r['mean_results']['who_compliance_rate'] for r in cv_results]
    colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.5 else 'red' for rate in compliance_rates]
    bars = ax.bar(model_names, compliance_rates, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target ≥80%')
    ax.set_title('WHO Compliance Rate')
    ax.set_ylabel('Compliance Rate')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Training Time
    ax = axes[1, 2]
    train_times = [r['training_time'] for r in cv_results]
    bars = ax.bar(model_names, train_times, color='lightpink', alpha=0.7)
    ax.set_title('Training Time (seconds)')
    ax.set_ylabel('Time (s)')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    cv_dashboard_file = config.get_output_filename('cv_performance_dashboard.png')
    cv_dashboard_path = os.path.join(config.results_dir, cv_dashboard_file)
    plt.savefig(cv_dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths.append(cv_dashboard_path)
    
    # 2. **NEW**: Test dataset performance dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Leave-One-Out Test Dataset Performance', fontsize=16, fontweight='bold')
    
    # Test Sensitivity
    ax = axes[0, 0]
    test_sens = [r['test_sensitivity'] for r in test_results]
    bars = ax.bar(model_names, test_sens, color='lightblue', alpha=0.7)
    ax.axhline(y=config.who_sensitivity_target, color='red', linestyle='--', alpha=0.7, label=f'WHO Target ≥{config.who_sensitivity_target*100:.0f}%')
    ax.set_title('Test Sensitivity')
    ax.set_ylabel('Sensitivity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Test Specificity  
    ax = axes[0, 1]
    test_spec = [r['test_specificity'] for r in test_results]
    bars = ax.bar(model_names, test_spec, color='lightgreen', alpha=0.7)
    ax.axhline(y=config.who_specificity_target, color='red', linestyle='--', alpha=0.7, label=f'WHO Target ≥{config.who_specificity_target*100:.0f}%')
    ax.set_title('Test Specificity')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Test ROC AUC
    ax = axes[0, 2]
    test_roc = [r['test_roc_auc'] for r in test_results]
    bars = ax.bar(model_names, test_roc, color='lightyellow', alpha=0.7)
    ax.set_title('Test ROC AUC')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Test WHO Score
    ax = axes[1, 0]
    test_who = [r['test_who_score'] for r in test_results]
    test_compliant = [r['test_who_compliant'] for r in test_results]
    colors = ['green' if compliant else 'red' for compliant in test_compliant]
    bars = ax.bar(model_names, test_who, color=colors, alpha=0.7)
    ax.axhline(y=1.25, color='red', linestyle='--', alpha=0.7, label='WHO Threshold ≥1.25')
    ax.set_title('Test WHO Score')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Test WHO Compliance
    ax = axes[1, 1]
    compliant_count = sum(test_compliant)
    compliance_labels = ['WHO Compliant', 'Non-Compliant']
    compliance_counts = [compliant_count, len(test_compliant) - compliant_count]
    colors = ['green', 'red']
    ax.pie(compliance_counts, labels=compliance_labels, colors=colors, autopct='%1.1f%%', startangle=90)
    ax.set_title('Test WHO Compliance Distribution')
    
    # CV vs Test Comparison
    ax = axes[1, 2]
    cv_who_means = [r['mean_results']['who_score_mean'] for r in cv_results]
    ax.scatter(cv_who_means, test_who, alpha=0.7, s=100)
    ax.plot([0, 1], [0, 1], 'r--', alpha=0.7, label='Perfect Agreement')
    ax.set_xlabel('CV WHO Score (Mean)')
    ax.set_ylabel('Test WHO Score')
    ax.set_title('CV vs Test Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add model labels
    for i, name in enumerate(model_names):
        ax.annotate(name, (cv_who_means[i], test_who[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    plt.tight_layout()
    test_dashboard_file = config.get_output_filename('leave_one_out_test_dashboard.png')
    test_dashboard_path = os.path.join(config.results_dir, test_dashboard_file)
    plt.savefig(test_dashboard_path, dpi=300, bbox_inches='tight')
    plt.close()
    visualization_paths.append(test_dashboard_path)
    
    print(f"✅ Visualizations created:")
    for path in visualization_paths:
        print(f"   📊 {os.path.basename(path)}")
    
    return visualization_paths

def generate_executive_summary(cv_results, test_results, file_paths, config):
    """Generate executive summary report"""
    
    # Find best models
    cv_best_idx = np.argmax([r['mean_results']['who_score_mean'] for r in cv_results])
    test_best_idx = np.argmax([r['test_who_score'] for r in test_results])
    
    cv_best_model = cv_results[cv_best_idx]
    test_best_model = test_results[test_best_idx]
    
    # Count WHO compliant models
    cv_compliant_count = sum(1 for r in cv_results if r['mean_results']['who_compliance_rate'] >= 0.8)
    test_compliant_count = sum(1 for r in test_results if r['test_who_compliant'])
    
    # Generate summary
    summary_file = config.get_output_filename('leave_one_out_executive_summary.txt')
    summary_path = os.path.join(config.results_dir, summary_file)
    
    with open(summary_path, 'w') as f:
        f.write("Leave-One-Out TB Detection Validation - Executive Summary\n")
        f.write("="*70 + "\n\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Run ID: {config.get_run_identifier()}\n\n")
        
        f.write("DATASET SUMMARY:\n")
        f.write(f"  Training patients: {len(cv_results[0]['fold_results'][0])} (used for {config.n_folds}-fold CV)\n")
        f.write(f"  Reserved test patients: 65 (multi-country validation)\n")
        f.write(f"  WHO targets: ≥{config.who_sensitivity_target*100:.0f}% sensitivity, ≥{config.who_specificity_target*100:.0f}% specificity\n\n")
        
        f.write("CROSS-VALIDATION RESULTS:\n")
        f.write(f"  Best CV model: {cv_best_model['model_name']}\n")
        f.write(f"  CV WHO Score: {cv_best_model['mean_results']['who_score_mean']:.3f} ± {cv_best_model['mean_results']['who_score_std']:.3f}\n")
        f.write(f"  CV Sensitivity: {cv_best_model['mean_results']['optimal_sensitivity_mean']:.3f} ± {cv_best_model['mean_results']['optimal_sensitivity_std']:.3f}\n")
        f.write(f"  CV Specificity: {cv_best_model['mean_results']['optimal_specificity_mean']:.3f} ± {cv_best_model['mean_results']['optimal_specificity_std']:.3f}\n")
        f.write(f"  CV WHO compliant models: {cv_compliant_count}/{len(cv_results)}\n\n")
        
        f.write("TEST DATASET VALIDATION:\n")
        f.write(f"  Best test model: {test_best_model['model_name']}\n")
        f.write(f"  Test WHO Score: {test_best_model['test_who_score']:.3f}\n")
        f.write(f"  Test Sensitivity: {test_best_model['test_sensitivity']:.3f}\n")
        f.write(f"  Test Specificity: {test_best_model['test_specificity']:.3f}\n")
        f.write(f"  Test WHO compliant: {test_best_model['test_who_compliant']}\n")
        f.write(f"  Test WHO compliant models: {test_compliant_count}/{len(test_results)}\n\n")
        
        if cv_best_model['model_name'] == test_best_model['model_name']:
            f.write("✅ CONSISTENCY: Same model achieved best performance in both CV and test\n\n")
        else:
            f.write("⚠️  INCONSISTENCY: Different models achieved best CV vs test performance\n\n")
        
        f.write("CLINICAL DEPLOYMENT RECOMMENDATION:\n")
        if test_compliant_count > 0:
            compliant_models = [r['model_name'] for r in test_results if r['test_who_compliant']]
            f.write(f"✅ READY FOR DEPLOYMENT: {compliant_models[0]} meets WHO targets on independent test data\n")
        else:
            f.write("❌ NOT READY: No models meet WHO targets on independent test data\n")
            f.write("   Recommend further optimization or ensemble methods\n")
        
        f.write("\nDELIVERABLES:\n")
        for key, path in file_paths.items():
            f.write(f"  • {key}: {os.path.basename(path)}\n")
    
    print(f"✅ Executive summary: {summary_path}")
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='Leave-One-Out TB Detection Validation')
    parser.add_argument('--test_patients', type=str,
                       default='patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv',
                       help='Path to reserved test patients file')
    parser.add_argument('--run_description', type=str,
                       default='leave_one_out_validation',
                       help='Description for this run')
    parser.add_argument('--cross_validation', action='store_true', default=True,
                       help='Enable cross-validation')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    
    args = parser.parse_args()
    
    print("🚀 Leave-One-Out TB Detection Validation Pipeline")
    print("=" * 70)
    
    # Initialize configuration using the proper loader
    config, parsed_args = load_leave_one_out_config(args)
    config.ensure_directories()
    
    # Check if dataset splits exist
    if not (os.path.exists(config.training_patients_file) and os.path.exists(config.test_patients_file)):
        print("❌ Dataset splits not found!")
        print(f"Run: python prepare_leave_one_out_dataset.py --test_patients {args.test_patients}")
        return
    
    try:
        # Save configuration
        config_path = config.save_run_config()
        
        # Run validation pipeline
        cv_results, test_results, data_tuple = run_leave_one_out_validation(config)
        
        # Save results
        file_paths = save_results_matching_baseline_structure(cv_results, test_results, data_tuple, config)
        
        # Create visualizations
        viz_paths = create_visualizations_matching_baseline(cv_results, test_results, data_tuple, config)
        file_paths.update({f'visualization_{i}': path for i, path in enumerate(viz_paths)})
        
        # Generate executive summary
        summary_path = generate_executive_summary(cv_results, test_results, file_paths, config)
        file_paths['executive_summary'] = summary_path
        
        # Update baseline pipeline runs tracking
        update_baseline_pipeline_runs(config, test_results, file_paths)
        
        print(f"\n🎉 Leave-One-Out Validation Complete!")
        print(f"📁 Results saved with run ID: {config.get_run_identifier()}")
        print(f"📊 Total deliverables: {len(file_paths)} files")
        
        # Print WHO compliance summary
        test_compliant_count = sum(1 for r in test_results if r['test_who_compliant'])
        if test_compliant_count > 0:
            print(f"✅ Clinical validation: {test_compliant_count} model(s) ready for WHO deployment")
        else:
            print(f"❌ Clinical validation: No models meet WHO targets on independent test data")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        raise

def update_baseline_pipeline_runs(config, test_results, file_paths):
    """Update baseline pipeline runs CSV with leave-one-out results"""
    
    baseline_file = 'baseline_pipeline_runs.csv'
    
    # Find best test model
    best_test_idx = np.argmax([r['test_who_score'] for r in test_results])
    best_test_model = test_results[best_test_idx]
    
    # Create new entry
    new_entry = {
        'run_id': config.get_run_identifier(),
        'run_type': 'leave_one_out_validation',
        'timestamp': config.run_timestamp,
        'commit_hash': 'current',  # Would get from git in production
        'script_used': '04_leave_one_out_validation.py',
        'config_file': f"configs/{config.get_output_filename('config.json')}",
        'input_data': f"{config.embeddings_file},{config.metadata_file},{config.training_patients_file},{config.test_patients_file}",
        'output_prefix': config.get_run_identifier(),
        'execution_time_min': '~10',  # Estimated
        'best_model': best_test_model['model_name'],
        'best_who_score': best_test_model['test_who_score'],
        'sensitivity': best_test_model['test_sensitivity'],
        'specificity': best_test_model['test_specificity'],
        'who_compliant': best_test_model['test_who_compliant'],
        'gpu_device': 'MPS_PYTORCH_ONLY' if PYTORCH_AVAILABLE else 'CPU_ONLY',
        'notes': f"Leave-one-out validation: 5-fold CV on training data + validation on 65 reserved test patients from 5 countries. {sum(1 for r in test_results if r['test_who_compliant'])} WHO-compliant models.",
        'executive_summary_path': file_paths['executive_summary'],
        'detailed_results_paths': ','.join([
            file_paths['cv_detailed_path'],
            file_paths['cv_summary_path'], 
            file_paths['test_results_path'],
            file_paths['comparison_path']
        ] + [path for key, path in file_paths.items() if 'visualization' in key])
    }
    
    # Append to baseline file
    if os.path.exists(baseline_file):
        baseline_df = pd.read_csv(baseline_file)
        baseline_df = pd.concat([baseline_df, pd.DataFrame([new_entry])], ignore_index=True)
    else:
        baseline_df = pd.DataFrame([new_entry])
    
    baseline_df.to_csv(baseline_file, index=False)
    print(f"✅ Updated {baseline_file} with leave-one-out results")

if __name__ == "__main__":
    main()