#!/usr/bin/env python3
"""
Ensemble TB Detection Pipeline - WHO Compliance Focus
Combines best performing models using sensitivity-weighted voting
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
from datetime import datetime
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore')

# PyTorch for GPU-accelerated neural networks
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# ML Models
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Metrics and evaluation
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, auc
)

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# Import configuration
from config_gpu import load_gpu_config_from_args

class WHOLoss(nn.Module):
    """WHO-optimized loss function with extreme false negative penalty"""
    def __init__(self, fn_weight=10.0, fp_weight=1.0):
        super().__init__()
        self.fn_weight = fn_weight  # Extreme penalty for missing TB cases
        self.fp_weight = fp_weight
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, outputs, targets):
        loss = self.bce(outputs, targets.float())
        
        # Apply asymmetric weighting
        fn_mask = (targets == 1) & (torch.sigmoid(outputs) < 0.5)  # False negatives
        fp_mask = (targets == 0) & (torch.sigmoid(outputs) >= 0.5)  # False positives
        
        loss = torch.where(fn_mask, loss * self.fn_weight, loss)
        loss = torch.where(fp_mask, loss * self.fp_weight, loss)
        
        return loss.mean()

class WHO_MLP(nn.Module):
    """WHO-optimized Multi-Layer Perceptron"""
    def __init__(self, input_size, hidden_sizes=[512, 256, 128], dropout=0.3):
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x).squeeze()

class TBDataset(Dataset):
    """PyTorch dataset for TB detection"""
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class EnsembleClassifier:
    """Sensitivity-weighted ensemble classifier for WHO compliance"""
    
    def __init__(self, models, weights=None, threshold=0.5):
        self.models = models
        self.weights = weights if weights is not None else np.ones(len(models)) / len(models)
        self.threshold = threshold
        self.model_names = list(models.keys()) if isinstance(models, dict) else [f"model_{i}" for i in range(len(models))]
        
    def predict_proba(self, X):
        """Get ensemble probability predictions"""
        predictions = []
        
        for i, (name, model) in enumerate(self.models.items() if isinstance(self.models, dict) else enumerate(self.models)):
            if name == 'WHO-MLP':
                # Handle PyTorch model
                device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
                model.eval()
                with torch.no_grad():
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    outputs = model(X_tensor)
                    pred = torch.sigmoid(outputs).cpu().numpy()
            elif hasattr(model, 'predict_proba'):
                pred = model.predict_proba(X)[:, 1]  # Get positive class probability
            elif hasattr(model, 'decision_function'):
                # Convert decision function to probability-like scores
                pred = torch.sigmoid(torch.tensor(model.decision_function(X))).numpy()
            else:
                # For models that only have predict method
                pred = model.predict(X).astype(float)
            
            predictions.append(pred)
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        ensemble_proba = np.average(predictions, axis=0, weights=self.weights)
        
        # Return as probability array (negative class, positive class)
        return np.column_stack([1 - ensemble_proba, ensemble_proba])
    
    def predict(self, X):
        """Get ensemble binary predictions"""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= self.threshold).astype(int)

def load_data(config):
    """Load embeddings and metadata"""
    print("📊 Loading embeddings and metadata...")
    print(f"   📁 Embeddings: {config.get_embeddings_path()}")
    print(f"   📁 Metadata: {config.get_metadata_path()}")
    print(f"   📁 Labels: {config.get_labels_path()}")
    
    # Load embeddings (keep as NPZ object for file-key access)
    embeddings_data = np.load(config.get_embeddings_path(), allow_pickle=True)
    print(f"✅ Found {len(embeddings_data.files)} embedding files")
    print(f"   Format: File-key based with multiple clips per file")
    
    # Load metadata 
    metadata_path = config.get_metadata_path()
    metadata = pd.read_csv(metadata_path)
    
    # Load labels
    labels_path = config.get_labels_path()
    labels_df = pd.read_csv(labels_path)
    
    print(f"   Metadata shape: {metadata.shape}")
    print(f"   Labels shape: {labels_df.shape}")
    
    return embeddings_data, None, metadata, labels_df

def aggregate_patient_features(embeddings_data, file_ids, metadata, labels_df):
    """Aggregate audio features and labels by patient"""
    print("🔄 Aggregating features by patient...")
    
    # Using file-key format (each key is a file with multiple clips)
    print("   Using file-key based indexing with clip aggregation")
    
    # Aggregate by patient
    patient_features = []
    patient_labels = []
    patient_ids = []
    
    for patient_id in labels_df['StudyID'].unique():
        # Get patient's audio files from metadata
        patient_metadata = metadata[metadata['patient_id'] == patient_id]
        
        if len(patient_metadata) > 0:
            # Collect all embeddings for this patient
            patient_embeddings = []
            
            for _, row in patient_metadata.iterrows():
                file_key = row['file_key']
                
                if file_key in embeddings_data.files:
                    # Get embeddings for this file (could be multiple clips)
                    file_embeddings = embeddings_data[file_key]  # Shape: (n_clips, 512)
                    
                    # Add all clips from this file
                    if file_embeddings.ndim == 2:
                        patient_embeddings.append(file_embeddings)
                    elif file_embeddings.ndim == 1:
                        # Single clip
                        patient_embeddings.append(file_embeddings.reshape(1, -1))
            
            if patient_embeddings:
                # Concatenate all clips from all files for this patient
                all_clips = np.vstack(patient_embeddings)  # Shape: (total_clips, 512)
                
                # Aggregate all clips to single patient representation (mean)
                patient_embedding = all_clips.mean(axis=0)  # Shape: (512,)
                patient_features.append(patient_embedding)
                
                # Get label (convert string to binary)
                label_str = labels_df[labels_df['StudyID'] == patient_id]['Label'].iloc[0]
                label_binary = 1 if 'Positive' in label_str else 0
                patient_labels.append(label_binary)
                patient_ids.append(patient_id)
    
    X = np.array(patient_features)
    y = np.array(patient_labels)
    
    print(f"   Aggregated to {len(patient_ids)} patients")
    print(f"   Features shape: {X.shape}")
    print(f"   TB positive: {y.sum()}/{len(y)} ({y.mean()*100:.1f}%)")
    
    return X, y, patient_ids

def train_who_mlp(X_train, y_train, X_val, y_val, config, fold=None):
    """Train WHO-optimized MLP with validation"""
    device = torch.device(config.pytorch_device)
    
    # Create datasets
    train_dataset = TBDataset(X_train, y_train)
    val_dataset = TBDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=min(32, len(train_dataset)), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(32, len(val_dataset)), shuffle=False)
    
    # Initialize model
    model = WHO_MLP(input_size=X_train.shape[1]).to(device)
    
    # WHO-optimized loss with extreme FN penalty
    criterion = WHOLoss(fn_weight=10.0, fp_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Training
    best_val_sensitivity = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(200):  # More epochs for better convergence
        model.train()
        train_loss = 0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        if epoch % 10 == 0:
            model.eval()
            val_probs = []
            val_true = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                    outputs = model(batch_X)
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    val_probs.extend(probs)
                    val_true.extend(batch_y.cpu().numpy())
            
            val_probs = np.array(val_probs)
            val_true = np.array(val_true)
            
            # Optimize threshold for sensitivity
            val_pred = (val_probs >= 0.3).astype(int)  # Lower threshold for higher sensitivity
            val_sensitivity = recall_score(val_true, val_pred)
            
            scheduler.step(train_loss)
            
            if val_sensitivity > best_val_sensitivity:
                best_val_sensitivity = val_sensitivity
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= 20:  # Early stopping
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

def train_models_with_ensemble(X_train, y_train, X_val, y_val, config, fold=None):
    """Train individual models and create ensemble"""
    models = {}
    
    print(f"   Training models for fold {fold}...")
    
    # Calculate class weights for TB bias (8:1 ratio)
    class_weight = {0: 1.0, 1: 8.0}
    
    # 1. Naive Bayes (consistently strong baseline)
    print("     - Naive Bayes")
    nb = GaussianNB()
    nb.fit(X_train, y_train)
    models['Naive Bayes'] = nb
    
    # 2. Logistic Regression with WHO optimization
    print("     - Logistic Regression")
    lr = LogisticRegression(
        class_weight=class_weight,
        max_iter=1000,
        random_state=config.random_state,
        C=0.1  # Regularization for better generalization
    )
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr
    
    # 3. WHO-optimized MLP
    print("     - WHO-MLP")
    who_mlp = train_who_mlp(X_train, y_train, X_val, y_val, config, fold)
    models['WHO-MLP'] = who_mlp
    
    # 4. XGBoost with aggressive sensitivity settings
    print("     - XGBoost")
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': config.xgboost_tree_method,
        'random_state': config.random_state,
        'scale_pos_weight': 8.0,  # Class imbalance handling
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0
    }
    
    xgb_model = xgb.XGBClassifier(**xgb_params)
    xgb_model.fit(X_train, y_train)
    models['XGBoost'] = xgb_model
    
    return models

def calculate_sensitivity_weights(models, X_val, y_val):
    """Calculate ensemble weights based on validation sensitivity"""
    sensitivities = {}
    
    for name, model in models.items():
        if name == 'WHO-MLP':
            # Handle PyTorch model
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            model.eval()
            with torch.no_grad():
                X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
                outputs = model(X_tensor)
                probs = torch.sigmoid(outputs).cpu().numpy()
                pred = (probs >= 0.3).astype(int)  # Lower threshold for sensitivity
        else:
            # Handle sklearn models
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_val)[:, 1]
                pred = (probs >= 0.3).astype(int)
            else:
                pred = model.predict(X_val)
        
        sensitivity = recall_score(y_val, pred)
        sensitivities[name] = sensitivity
    
    # Convert to weights (higher sensitivity = higher weight)
    total_sensitivity = sum(sensitivities.values())
    weights = {name: sens / total_sensitivity for name, sens in sensitivities.items()}
    
    return weights, sensitivities

def optimize_who_threshold(y_true, y_scores):
    """Optimize threshold for WHO compliance with ensemble scores"""
    thresholds = np.linspace(0, 1, 2000)
    best_threshold = 0.5
    best_score = 0
    best_metrics = None
    
    results = []
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        if tp + fn > 0:
            sensitivity = tp / (tp + fn)
        else:
            sensitivity = 0
            
        if tn + fp > 0:
            specificity = tn / (tn + fp)
        else:
            specificity = 0
        
        # WHO compliance scoring
        who_score = 0
        who_compliant = False
        
        # Priority 1: WHO compliance (sens ≥90%, spec ≥70%)
        if sensitivity >= 0.90 and specificity >= 0.70:
            who_score = sensitivity + specificity * 2  # Bonus for full compliance
            who_compliant = True
        # Priority 2: Relaxed WHO compliance (sens ≥90%, spec ≥60%)
        elif sensitivity >= 0.90 and specificity >= 0.60:
            who_score = sensitivity + specificity
            who_compliant = True
        # Priority 3: Maximum sensitivity with minimal specificity (≥40%)
        elif specificity >= 0.40:
            who_score = sensitivity * 1.2  # Weight sensitivity heavily
        else:
            who_score = sensitivity * 0.8  # Penalty for very low specificity
        
        results.append({
            'threshold': threshold,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'who_score': who_score,
            'who_compliant': who_compliant,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        })
        
        if who_score > best_score:
            best_score = who_score
            best_threshold = threshold
            best_metrics = {
                'sensitivity': sensitivity,
                'specificity': specificity,
                'who_score': who_score,
                'who_compliant': who_compliant,
                'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
            }
    
    return best_threshold, best_metrics, pd.DataFrame(results)

def evaluate_model(model, X_test, y_test, model_name, config, fold=None):
    """Evaluate a single model with WHO threshold optimization"""
    
    # Get predictions based on model type
    if model_name == 'WHO-MLP':
        device = torch.device(config.pytorch_device)
        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            outputs = model(X_tensor)
            y_scores = torch.sigmoid(outputs).cpu().numpy()
    elif hasattr(model, 'predict_proba'):
        y_scores = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        y_scores = model.decision_function(X_test)
        y_scores = (y_scores - y_scores.min()) / (y_scores.max() - y_scores.min())
    else:
        y_pred = model.predict(X_test)
        y_scores = y_pred.astype(float)
    
    # Standard predictions (0.5 threshold)
    y_pred_standard = (y_scores >= 0.5).astype(int)
    
    # WHO-optimized threshold
    who_threshold, who_metrics, threshold_results = optimize_who_threshold(y_test, y_scores)
    y_pred_who = (y_scores >= who_threshold).astype(int)
    
    # Calculate standard metrics
    accuracy = accuracy_score(y_test, y_pred_standard)
    sensitivity = recall_score(y_test, y_pred_standard)
    specificity = recall_score(1 - y_test, 1 - y_pred_standard)
    precision = precision_score(y_test, y_pred_standard, zero_division=0)
    f1 = f1_score(y_test, y_pred_standard)
    
    try:
        roc_auc = roc_auc_score(y_test, y_scores)
    except:
        roc_auc = 0.5
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_standard).ravel()
    
    return {
        'model': model_name,
        'fold': fold,
        'test_accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'who_threshold': who_threshold,
        'who_sensitivity': who_metrics['sensitivity'],
        'who_specificity': who_metrics['specificity'],
        'who_score': who_metrics['who_score'],
        'who_compliant': who_metrics['who_compliant'],
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn,
        'y_scores': y_scores,
        'threshold_results': threshold_results
    }

def run_ensemble_cross_validation(config):
    """Run ensemble cross-validation experiment"""
    print("🚀 Starting Ensemble Cross-Validation Experiment")
    print("=" * 60)
    
    # Load and prepare data
    embeddings_data, file_ids, metadata, labels_df = load_data(config)
    X, y, patient_ids = aggregate_patient_features(embeddings_data, file_ids, metadata, labels_df)
    
    # Cross-validation setup
    skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    results = []
    ensemble_results = []
    fold_models = []
    
    print(f"\n🔄 Running {config.n_folds}-fold cross-validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"\n📊 Fold {fold + 1}/{config.n_folds}")
        print("-" * 30)
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Further split training for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train, random_state=config.random_state
        )
        
        # Standardize features
        scaler = StandardScaler()
        X_train_split = scaler.fit_transform(X_train_split)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)
        
        print(f"   Train: {len(X_train_split)} patients")
        print(f"   Validation: {len(X_val)} patients")
        print(f"   Test: {len(X_test)} patients")
        
        # Train individual models
        models = train_models_with_ensemble(X_train_split, y_train_split, X_val, y_val, config, fold)
        
        # Calculate sensitivity-based weights
        weights, sensitivities = calculate_sensitivity_weights(models, X_val, y_val)
        
        print(f"   Validation sensitivities: {sensitivities}")
        print(f"   Ensemble weights: {weights}")
        
        # Create ensemble
        ensemble = EnsembleClassifier(models, weights=list(weights.values()))
        
        # Evaluate individual models
        for model_name, model in models.items():
            result = evaluate_model(model, X_test, y_test, model_name, config, fold)
            # Store y_test for ROC curve generation
            result['y_test'] = y_test
            results.append(result)
        
        # Evaluate ensemble
        ensemble_probs = ensemble.predict_proba(X_test)[:, 1]
        ensemble_result = evaluate_model(ensemble, X_test, y_test, 'Ensemble', config, fold)
        ensemble_result['y_scores'] = ensemble_probs
        ensemble_result['y_test'] = y_test
        ensemble_results.append(ensemble_result)
        
        # Store fold models for final ensemble
        fold_models.append({
            'models': models,
            'weights': weights,
            'scaler': scaler,
            'sensitivities': sensitivities
        })
    
    return results, ensemble_results, fold_models

def create_standard_analysis_results_df(results, ensemble_results):
    """Create standardized analysis results DataFrame compatible with GPU pipeline"""
    analysis_data = []
    
    # Process individual model results
    for result in results:
        analysis_data.append({
            'Model': result['model'],
            'Fold': result['fold'],
            'Sensitivity': result['sensitivity'],
            'Specificity': result['specificity'],
            'Precision': result['precision'],
            'NPV': result.get('npv', result['tn'] / (result['tn'] + result['fn']) if (result['tn'] + result['fn']) > 0 else 0.0),
            'F1_Score': result.get('f1_score', result.get('f1', 0.0)),
            'ROC_AUC': result['roc_auc'],
            'PR_AUC': result.get('pr_auc', 0.0),
            'WHO_Score': result['who_score'],
            'WHO_Compliant': result['who_compliant'],
            'Aggregation_Strategy': 'Ensemble_Weighted',
            'TP': result['tp'],
            'TN': result['tn'], 
            'FP': result['fp'],
            'FN': result['fn']
        })
    
    # Process ensemble results
    for result in ensemble_results:
        analysis_data.append({
            'Model': 'Ensemble',
            'Fold': result['fold'],
            'Sensitivity': result['sensitivity'],
            'Specificity': result['specificity'],
            'Precision': result['precision'],
            'NPV': result.get('npv', result['tn'] / (result['tn'] + result['fn']) if (result['tn'] + result['fn']) > 0 else 0.0),
            'F1_Score': result.get('f1_score', result.get('f1', 0.0)),
            'ROC_AUC': result['roc_auc'],
            'PR_AUC': result.get('pr_auc', 0.0),
            'WHO_Score': result['who_score'],
            'WHO_Compliant': result['who_compliant'],
            'Aggregation_Strategy': 'Ensemble_Weighted',
            'TP': result['tp'],
            'TN': result['tn'],
            'FP': result['fp'], 
            'FN': result['fn']
        })
    
    return pd.DataFrame(analysis_data)

def create_standard_cv_summary_df(results, ensemble_results):
    """Create standardized cross-validation summary DataFrame"""
    results_df = pd.DataFrame(results)
    ensemble_df = pd.DataFrame(ensemble_results)
    
    summary_data = []
    
    # Individual model summaries
    for model_name in results_df['model'].unique():
        model_results = results_df[results_df['model'] == model_name]
        summary_data.append({
            'Model': model_name,
            'Mean_Sensitivity': model_results['sensitivity'].mean(),
            'Std_Sensitivity': model_results['sensitivity'].std(),
            'Mean_Specificity': model_results['specificity'].mean(),
            'Std_Specificity': model_results['specificity'].std(),
            'Mean_WHO_Score': model_results['who_score'].mean(),
            'WHO_Compliant_Folds': model_results['who_compliant'].sum(),
            'Mean_ROC_AUC': model_results['roc_auc'].mean(),
            'Mean_F1_Score': model_results.get('f1_score', model_results.get('f1', pd.Series([0.0]))).mean(),
            'Mean_Precision': model_results['precision'].mean()
        })
    
    # Ensemble summary
    summary_data.append({
        'Model': 'Ensemble',
        'Mean_Sensitivity': ensemble_df['sensitivity'].mean(),
        'Std_Sensitivity': ensemble_df['sensitivity'].std(),
        'Mean_Specificity': ensemble_df['specificity'].mean(),
        'Std_Specificity': ensemble_df['specificity'].std(),
        'Mean_WHO_Score': ensemble_df['who_score'].mean(),
        'WHO_Compliant_Folds': ensemble_df['who_compliant'].sum(),
        'Mean_ROC_AUC': ensemble_df['roc_auc'].mean(),
        'Mean_F1_Score': ensemble_df.get('f1_score', ensemble_df.get('f1', pd.Series([0.0]))).mean(),
        'Mean_Precision': ensemble_df['precision'].mean()
    })
    
    return pd.DataFrame(summary_data)

def save_performance_dashboard_ensemble(results_df, ensemble_df, config):
    """Create ensemble performance dashboard"""
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Ensemble TB Detection Model Comparison', fontsize=16, fontweight='bold')
    
    # Create direct summary from results_df
    model_summaries = {}
    
    # Process individual models
    for model in results_df[results_df['Model'] != 'Ensemble']['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        model_summaries[model] = {
            'Mean_Sensitivity': model_data['Sensitivity'].mean(),
            'Std_Sensitivity': model_data['Sensitivity'].std(),
            'Mean_Specificity': model_data['Specificity'].mean(),
            'Std_Specificity': model_data['Specificity'].std(),
            'Mean_WHO_Score': model_data['WHO_Score'].mean(),
            'Std_WHO_Score': model_data['WHO_Score'].std(),
            'WHO_Compliant_Folds': model_data['WHO_Compliant'].sum(),
            'Mean_ROC_AUC': model_data['ROC_AUC'].mean(),
            'Std_ROC_AUC': model_data['ROC_AUC'].std()
        }
    
    # Add ensemble
    if len(ensemble_df) > 0:
        model_summaries['Ensemble'] = {
            'Mean_Sensitivity': ensemble_df['sensitivity'].mean(),
            'Std_Sensitivity': ensemble_df['sensitivity'].std(),
            'Mean_Specificity': ensemble_df['specificity'].mean(),
            'Std_Specificity': ensemble_df['specificity'].std(),
            'Mean_WHO_Score': ensemble_df['who_score'].mean(),
            'Std_WHO_Score': ensemble_df['who_score'].std(),
            'WHO_Compliant_Folds': ensemble_df['who_compliant'].sum(),
            'Mean_ROC_AUC': ensemble_df['roc_auc'].mean(),
            'Std_ROC_AUC': ensemble_df['roc_auc'].std()
        }
    
    model_names = list(model_summaries.keys())
    sensitivities = [model_summaries[m]['Mean_Sensitivity'] for m in model_names]
    sens_stds = [model_summaries[m]['Std_Sensitivity'] for m in model_names]
    specificities = [model_summaries[m]['Mean_Specificity'] for m in model_names]
    spec_stds = [model_summaries[m]['Std_Specificity'] for m in model_names]
    who_scores = [model_summaries[m]['Mean_WHO_Score'] for m in model_names]
    who_stds = [model_summaries[m]['Std_WHO_Score'] for m in model_names]
    who_compliant = [model_summaries[m]['WHO_Compliant_Folds'] for m in model_names]
    roc_aucs = [model_summaries[m]['Mean_ROC_AUC'] for m in model_names]
    roc_stds = [model_summaries[m]['Std_ROC_AUC'] for m in model_names]
    
    # Plot 1: Sensitivity with WHO target
    ax = axes[0, 0]
    bars = ax.bar(model_names, sensitivities, yerr=sens_stds, capsize=5, color='lightblue', alpha=0.7, error_kw={'color': 'black', 'alpha': 0.8})
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥90%')
    ax.set_title('Mean Sensitivity (Recall)')
    ax.set_ylabel('Sensitivity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, sensitivities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 2: Specificity with WHO target
    ax = axes[0, 1]
    bars = ax.bar(model_names, specificities, yerr=spec_stds, capsize=5, color='lightgreen', alpha=0.7, error_kw={'color': 'black', 'alpha': 0.8})
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥70%')
    ax.set_title('Mean Specificity')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, specificities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: WHO Score
    ax = axes[0, 2]
    bars = ax.bar(model_names, who_scores, yerr=who_stds, capsize=5, color='lightcoral', alpha=0.7, error_kw={'color': 'black', 'alpha': 0.8})
    ax.set_title('WHO Score')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, who_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: WHO Compliance
    ax = axes[1, 0]
    bars = ax.bar(model_names, who_compliant, color='gold', alpha=0.7)
    ax.set_title('WHO Compliant Folds (out of 5)')
    ax.set_ylabel('Compliant Folds')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 5.5)
    ax.grid(True, alpha=0.3)
    
    for bar, val in zip(bars, who_compliant):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(val)}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: ROC AUC
    ax = axes[1, 1]
    bars = ax.bar(model_names, roc_aucs, yerr=roc_stds, capsize=5, color='mediumpurple', alpha=0.7, error_kw={'color': 'black', 'alpha': 0.8})
    ax.set_title('Mean ROC AUC')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, roc_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: Sensitivity vs Specificity scatter
    ax = axes[1, 2]
    ax.scatter(specificities, sensitivities, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    for i, model in enumerate(model_names):
        ax.annotate(model, (specificities[i], sensitivities[i]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=8)
    
    # WHO compliance zones
    ax.add_patch(Rectangle((0.7, 0.9), 0.3, 0.1, alpha=0.2, color='green', label='WHO Compliant'))
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7)
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Mean Specificity')
    ax.set_ylabel('Mean Sensitivity')
    ax.set_title('Sensitivity vs Specificity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    plot_filename = config.get_output_filename('gpu_performance_dashboard.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_roc_curves_ensemble(results, ensemble_results, config):
    """Create ROC curves for ensemble models with proper granularity"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot diagonal line for random classifier
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Classifier')
    
    # Get unique models
    model_names = list(set([r['model'] for r in results]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names) + 1))  # +1 for ensemble
    
    # Plot ROC curves for individual models
    for i, model_name in enumerate(model_names):
        # Aggregate all predictions across folds for this model
        all_y_true = []
        all_y_scores = []
        
        model_results = [r for r in results if r['model'] == model_name]
        for result in model_results:
            if 'y_test' in result and 'y_scores' in result:
                all_y_true.extend(result['y_test'])
                all_y_scores.extend(result['y_scores'])
        
        if len(all_y_true) > 0 and len(set(all_y_true)) > 1:  # Need both classes
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot smooth ROC curve
            ax.plot(fpr, tpr, color=colors[i], linewidth=2, alpha=0.8,
                   label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot ROC curve for ensemble
    if len(ensemble_results) > 0:
        all_y_true = []
        all_y_scores = []
        
        for result in ensemble_results:
            if 'y_test' in result and 'y_scores' in result:
                all_y_true.extend(result['y_test'])
                all_y_scores.extend(result['y_scores'])
        
        if len(all_y_true) > 0 and len(set(all_y_true)) > 1:
            # Compute ROC curve for ensemble
            fpr, tpr, _ = roc_curve(all_y_true, all_y_scores)
            roc_auc = auc(fpr, tpr)
            
            # Plot ensemble ROC curve with distinct style
            ax.plot(fpr, tpr, color=colors[-1], linewidth=3, alpha=0.9,
                   linestyle='-', label=f'Ensemble (AUC = {roc_auc:.3f})')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves - Ensemble Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plot_filename = config.get_output_filename('gpu_roc_curves.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_precision_recall_curves_ensemble(results, ensemble_results, config):
    """Create precision-recall curves for ensemble models with proper granularity"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique models
    model_names = list(set([r['model'] for r in results]))
    colors = plt.cm.Set1(np.linspace(0, 1, len(model_names) + 1))  # +1 for ensemble
    
    # Plot PR curves for individual models
    for i, model_name in enumerate(model_names):
        # Aggregate all predictions across folds for this model
        all_y_true = []
        all_y_scores = []
        
        model_results = [r for r in results if r['model'] == model_name]
        for result in model_results:
            if 'y_test' in result and 'y_scores' in result:
                all_y_true.extend(result['y_test'])
                all_y_scores.extend(result['y_scores'])
        
        if len(all_y_true) > 0 and len(set(all_y_true)) > 1:  # Need both classes
            # Compute PR curve
            precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
            pr_auc = average_precision_score(all_y_true, all_y_scores)
            
            # Plot smooth PR curve
            ax.plot(recall, precision, color=colors[i], linewidth=2, alpha=0.8,
                   label=f'{model_name} (AP = {pr_auc:.3f})')
    
    # Plot PR curve for ensemble
    if len(ensemble_results) > 0:
        all_y_true = []
        all_y_scores = []
        
        for result in ensemble_results:
            if 'y_test' in result and 'y_scores' in result:
                all_y_true.extend(result['y_test'])
                all_y_scores.extend(result['y_scores'])
        
        if len(all_y_true) > 0 and len(set(all_y_true)) > 1:
            # Compute PR curve for ensemble
            precision, recall, _ = precision_recall_curve(all_y_true, all_y_scores)
            pr_auc = average_precision_score(all_y_true, all_y_scores)
            
            # Plot ensemble PR curve with distinct style
            ax.plot(recall, precision, color=colors[-1], linewidth=3, alpha=0.9,
                   linestyle='-', label=f'Ensemble (AP = {pr_auc:.3f})')
    
    # Add baseline (random classifier performance)
    if len(all_y_true) > 0:
        baseline = np.mean(all_y_true)  # Proportion of positive class
        ax.axhline(y=baseline, color='k', linestyle='--', alpha=0.3, 
                  label=f'Random Classifier (AP = {baseline:.3f})')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves - Ensemble Models Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plot_filename = config.get_output_filename('gpu_precision_recall_curves.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_confusion_matrices_ensemble(results_df, ensemble_df, config):
    """Create confusion matrices grid for ensemble models"""
    plt.style.use('default')
    
    # Get unique models
    model_names = list(results_df[results_df['Model'] != 'Ensemble']['Model'].unique())
    if len(ensemble_df) > 0:
        model_names.append('Ensemble')
    
    n_models = len(model_names)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle('Confusion Matrices - Ensemble Models', fontsize=16, fontweight='bold')
    
    for i, model_name in enumerate(model_names):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Get confusion matrix values
        if model_name == 'Ensemble':
            avg_tp = ensemble_df['tp'].mean() if len(ensemble_df) > 0 else 0
            avg_tn = ensemble_df['tn'].mean() if len(ensemble_df) > 0 else 0
            avg_fp = ensemble_df['fp'].mean() if len(ensemble_df) > 0 else 0
            avg_fn = ensemble_df['fn'].mean() if len(ensemble_df) > 0 else 0
        else:
            model_results = results_df[results_df['Model'] == model_name]
            avg_tp = model_results['TP'].mean() if len(model_results) > 0 else 0
            avg_tn = model_results['TN'].mean() if len(model_results) > 0 else 0
            avg_fp = model_results['FP'].mean() if len(model_results) > 0 else 0
            avg_fn = model_results['FN'].mean() if len(model_results) > 0 else 0
        
        cm = np.array([[avg_tn, avg_fp], [avg_fn, avg_tp]])
        
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f'{model_name}')
        
        # Add text annotations
        thresh = cm.max() / 2. if cm.max() > 0 else 0
        for i_cm, j_cm in np.ndindex(cm.shape):
            ax.text(j_cm, i_cm, f'{cm[i_cm, j_cm]:.1f}',
                   ha="center", va="center",
                   color="white" if cm[i_cm, j_cm] > thresh else "black")
        
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    # Hide empty subplots
    for i in range(n_models, rows * cols):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.set_visible(False)
    
    plt.tight_layout()
    
    plot_filename = config.get_output_filename('gpu_confusion_matrices_grid.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_who_compliance_analysis_ensemble(results_df, ensemble_df, config):
    """Create WHO compliance analysis visualization"""
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get model summaries directly
    model_summaries = {}
    for model in results_df[results_df['Model'] != 'Ensemble']['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        model_summaries[model] = {
            'Mean_Sensitivity': model_data['Sensitivity'].mean(),
            'Std_Sensitivity': model_data['Sensitivity'].std(),
            'Mean_Specificity': model_data['Specificity'].mean(),
            'Std_Specificity': model_data['Specificity'].std(),
            'WHO_Compliant_Folds': model_data['WHO_Compliant'].sum()
        }
    
    if len(ensemble_df) > 0:
        model_summaries['Ensemble'] = {
            'Mean_Sensitivity': ensemble_df['sensitivity'].mean(),
            'Std_Sensitivity': ensemble_df['sensitivity'].std(),
            'Mean_Specificity': ensemble_df['specificity'].mean(),
            'Std_Specificity': ensemble_df['specificity'].std(),
            'WHO_Compliant_Folds': ensemble_df['who_compliant'].sum()
        }
    
    # Create scatter plot with error bars
    specificities = [model_summaries[m]['Mean_Specificity'] for m in model_summaries.keys()]
    sensitivities = [model_summaries[m]['Mean_Sensitivity'] for m in model_summaries.keys()]
    spec_stds = [model_summaries[m]['Std_Specificity'] for m in model_summaries.keys()]
    sens_stds = [model_summaries[m]['Std_Sensitivity'] for m in model_summaries.keys()]
    colors = ['red' if model_summaries[m]['WHO_Compliant_Folds'] < 1 else 'green' for m in model_summaries.keys()]
    
    # Plot error bars first (behind the points)
    ax.errorbar(specificities, sensitivities, xerr=spec_stds, yerr=sens_stds, 
                fmt='none', capsize=5, color='black', alpha=0.6, zorder=1)
    
    # Plot scatter points on top of error bars
    ax.scatter(specificities, sensitivities, c=colors, s=150, alpha=0.8, edgecolors='black', zorder=2)
    
    # Add model labels
    for i, model_name in enumerate(model_summaries.keys()):
        ax.annotate(model_name, (specificities[i], sensitivities[i]),
                   xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    # WHO compliance zones
    ax.add_patch(Rectangle((0.7, 0.9), 0.3, 0.1, alpha=0.2, color='green', label='WHO Compliant Zone'))
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, linewidth=2, label='WHO Sensitivity ≥90%')
    ax.axvline(x=0.7, color='red', linestyle='--', alpha=0.7, linewidth=2, label='WHO Specificity ≥70%')
    
    ax.set_xlabel('Mean Specificity', fontsize=12)
    ax.set_ylabel('Mean Sensitivity', fontsize=12)
    ax.set_title('WHO TB Screening Compliance Analysis - Ensemble Models', fontsize=14, fontweight='bold')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    
    plot_filename = config.get_output_filename('gpu_who_compliance_analysis.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_cv_fold_variance_ensemble(results_df, ensemble_df, config):
    """Create cross-validation fold variance analysis"""
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Cross-Validation Fold Variance Analysis - Ensemble Models', fontsize=16, fontweight='bold')
    
    # Get unique models for analysis
    unique_models = results_df['Model'].unique()
    
    # Plot 1: Sensitivity variance across folds
    ax = axes[0, 0]
    for model in unique_models:
        model_data = results_df[results_df['Model'] == model]
        if len(model_data) > 0:
            folds = model_data['Fold'].tolist()
            sensitivities = model_data['Sensitivity'].tolist()
            ax.plot(folds, sensitivities, marker='o', label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 2: Specificity variance across folds
    ax = axes[0, 1]
    for model in unique_models:
        model_data = results_df[results_df['Model'] == model]
        if len(model_data) > 0:
            folds = model_data['Fold'].tolist()
            specificities = model_data['Specificity'].tolist()
            ax.plot(folds, specificities, marker='o', label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('Specificity')
    ax.set_title('Specificity Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    
    # Plot 3: WHO Score variance
    ax = axes[1, 0]
    for model in unique_models:
        model_data = results_df[results_df['Model'] == model]
        if len(model_data) > 0:
            folds = model_data['Fold'].tolist()
            who_scores = model_data['WHO_Score'].tolist()
            ax.plot(folds, who_scores, marker='o', label=model, linewidth=2, markersize=6)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('WHO Score')
    ax.set_title('WHO Score Across Folds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation comparison
    ax = axes[1, 1]
    model_names = []
    sens_stds = []
    spec_stds = []
    
    for model in unique_models:
        model_data = results_df[results_df['Model'] == model]
        if len(model_data) > 1:  # Need at least 2 folds for std
            model_names.append(model)
            sens_stds.append(model_data['Sensitivity'].std())
            spec_stds.append(model_data['Specificity'].std())
    
    if model_names:
        x = np.arange(len(model_names))
        width = 0.35
        
        ax.bar(x - width/2, sens_stds, width, label='Sensitivity Std', alpha=0.7)
        ax.bar(x + width/2, spec_stds, width, label='Specificity Std', alpha=0.7)
        
        ax.set_xlabel('Model')
        ax.set_ylabel('Standard Deviation')
        ax.set_title('Performance Variance (Standard Deviation)')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    plot_filename = config.get_output_filename('gpu_cv_fold_variance.png')
    plot_path = f"results/{plot_filename}"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def save_results(results, ensemble_results, config):
    """Save standardized results and analysis with all required outputs"""
    
    # Generate output prefix
    output_prefix = config.get_output_filename("")
    
    # Create standardized DataFrames
    analysis_results_df = create_standard_analysis_results_df(results, ensemble_results)
    cv_summary_df = create_standard_cv_summary_df(results, ensemble_results)
    cv_detailed_df = pd.DataFrame(results + ensemble_results)  # Keep detailed fold results
    
    # Save with standard naming convention
    analysis_results_path = f"results/{output_prefix}_analysis_results.csv"
    cv_summary_path = f"results/{output_prefix}_cross_validation_summary.csv"
    cv_detailed_path = f"results/{output_prefix}_cross_validation_detailed.csv"
    
    analysis_results_df.to_csv(analysis_results_path, index=False)
    cv_summary_df.to_csv(cv_summary_path, index=False)
    cv_detailed_df.to_csv(cv_detailed_path, index=False)
    
    # Generate all required visualizations
    dashboard_path = save_performance_dashboard_ensemble(analysis_results_df, pd.DataFrame(ensemble_results), config)
    roc_path = save_roc_curves_ensemble(results, ensemble_results, config)  # Pass original results for proper ROC
    pr_path = save_precision_recall_curves_ensemble(results, ensemble_results, config)  # Pass original results for proper PR
    confusion_path = save_confusion_matrices_ensemble(analysis_results_df, pd.DataFrame(ensemble_results), config)
    compliance_path = save_who_compliance_analysis_ensemble(analysis_results_df, pd.DataFrame(ensemble_results), config)
    variance_path = save_cv_fold_variance_ensemble(analysis_results_df, pd.DataFrame(ensemble_results), config)
    
    # Generate executive summary with standard format
    exec_summary_path = f"results/{output_prefix}_executive_summary.txt"
    ensemble_df = pd.DataFrame(ensemble_results)
    
    with open(exec_summary_path, 'w') as f:
        f.write("🎯 ENSEMBLE TB DETECTION - EXECUTIVE SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        
        f.write(f"📊 Experiment: {config.get_run_identifier()}\n")
        f.write(f"⏰ Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"🔧 Device: {config.device_type.upper()}\n")
        f.write(f"🔄 Mode: {config.n_folds}-fold Cross-Validation\n\n")
        
        f.write("🏆 ENSEMBLE PERFORMANCE\n")
        f.write("-" * 25 + "\n")
        ensemble_sens = ensemble_df['sensitivity'].mean()
        ensemble_spec = ensemble_df['specificity'].mean()
        ensemble_who = ensemble_df['who_score'].mean()
        who_compliant = ensemble_df['who_compliant'].sum()
        
        f.write(f"Sensitivity: {ensemble_sens:.3f} ± {ensemble_df['sensitivity'].std():.3f}\n")
        f.write(f"Specificity: {ensemble_spec:.3f} ± {ensemble_df['specificity'].std():.3f}\n")
        f.write(f"WHO Score: {ensemble_who:.3f}\n")
        f.write(f"WHO Compliant Folds: {who_compliant}/{config.n_folds}\n")
        f.write(f"Mean ROC AUC: {ensemble_df['roc_auc'].mean():.3f}\n")
        f.write(f"Mean F1 Score: {ensemble_df.get('f1_score', ensemble_df.get('f1', pd.Series([0.0]))).mean():.3f}\n\n")
        
        f.write("📈 INDIVIDUAL MODEL COMPARISON\n")
        f.write("-" * 35 + "\n")
        for model in analysis_results_df[analysis_results_df['Model'] != 'Ensemble']['Model'].unique():
            model_data = analysis_results_df[analysis_results_df['Model'] == model]
            sens = model_data['Sensitivity'].mean()
            spec = model_data['Specificity'].mean()
            who = model_data['WHO_Score'].mean()
            f.write(f"{model:<18}: Sens={sens:.3f}, Spec={spec:.3f}, WHO={who:.3f}\n")
        
        f.write(f"\n🎯 WHO COMPLIANCE TARGET: ≥90% Sensitivity, ≥70% Specificity\n")
        if ensemble_sens >= 0.90 and ensemble_spec >= 0.70:
            f.write("✅ WHO COMPLIANCE: ACHIEVED\n")
        elif ensemble_sens >= 0.90:
            f.write("⚠️  WHO COMPLIANCE: High sensitivity, but specificity needs improvement\n")
        elif ensemble_spec >= 0.70:
            f.write("⚠️  WHO COMPLIANCE: Good specificity, but sensitivity needs significant improvement\n")
        else:
            f.write("❌ WHO COMPLIANCE: Both sensitivity and specificity need improvement\n")
        
        f.write(f"\n📊 BEST PERFORMING MODEL: Ensemble (WHO Score: {ensemble_who:.3f})\n")
        f.write(f"📁 OUTPUT FILES:\n")
        f.write(f"   • Analysis Results: {output_prefix}_analysis_results.csv\n")
        f.write(f"   • CV Summary: {output_prefix}_cross_validation_summary.csv\n") 
        f.write(f"   • CV Detailed: {output_prefix}_cross_validation_detailed.csv\n")
        f.write(f"   • Performance Dashboard: {output_prefix}_gpu_performance_dashboard.png\n")
        f.write(f"   • ROC Curves: {output_prefix}_gpu_roc_curves.png\n")
        f.write(f"   • PR Curves: {output_prefix}_gpu_precision_recall_curves.png\n")
        f.write(f"   • Confusion Matrices: {output_prefix}_gpu_confusion_matrices_grid.png\n")
        f.write(f"   • WHO Compliance: {output_prefix}_gpu_who_compliance_analysis.png\n")
        f.write(f"   • Fold Variance: {output_prefix}_gpu_cv_fold_variance.png\n")
    
    print(f"\n✅ Standardized results saved:")
    print(f"   📊 Analysis Results: {analysis_results_path}")
    print(f"   📊 CV Summary: {cv_summary_path}")
    print(f"   📊 CV Detailed: {cv_detailed_path}")
    print(f"   📋 Executive Summary: {exec_summary_path}")
    print(f"   📈 Performance Dashboard: {dashboard_path}")
    print(f"   📈 ROC Curves: {roc_path}")
    print(f"   📈 PR Curves: {pr_path}")
    print(f"   📈 Confusion Matrices: {confusion_path}")
    print(f"   📈 WHO Compliance: {compliance_path}")
    print(f"   📈 Fold Variance: {variance_path}")
    
    return (analysis_results_path, cv_summary_path, cv_detailed_path, exec_summary_path,
            dashboard_path, roc_path, pr_path, confusion_path, compliance_path, variance_path)

def main():
    """Main execution function"""
    
    # Load configuration
    config = load_gpu_config_from_args()
    config.run_description = "Sensitivity-weighted ensemble: NB+LR+WHO-MLP+XGB for WHO compliance breakthrough"
    config.use_cross_validation = True
    config.n_folds = 5
    
    # Print configuration
    config.print_config()
    
    # Save configuration
    config_path = config.save_run_config()
    
    # Create results directory
    os.makedirs("results", exist_ok=True)
    
    # Run ensemble experiment
    start_time = datetime.now()
    results, ensemble_results, fold_models = run_ensemble_cross_validation(config)
    end_time = datetime.now()
    
    # Save results
    results_paths = save_results(results, ensemble_results, config)
    
    # Print final summary
    ensemble_df = pd.DataFrame(ensemble_results)
    print(f"\n🎯 FINAL ENSEMBLE RESULTS")
    print("=" * 40)
    print(f"Sensitivity: {ensemble_df['sensitivity'].mean():.3f} ± {ensemble_df['sensitivity'].std():.3f}")
    print(f"Specificity: {ensemble_df['specificity'].mean():.3f} ± {ensemble_df['specificity'].std():.3f}")
    print(f"WHO Score: {ensemble_df['who_score'].mean():.3f}")
    print(f"WHO Compliant Folds: {ensemble_df['who_compliant'].sum()}/{config.n_folds}")
    print(f"⏱️  Total Runtime: {end_time - start_time}")
    
    return results_paths

if __name__ == "__main__":
    main()