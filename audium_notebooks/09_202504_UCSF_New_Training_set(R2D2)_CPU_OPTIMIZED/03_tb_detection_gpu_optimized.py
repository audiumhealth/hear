#!/usr/bin/env python3
"""
GPU-Optimized TB Detection Analysis for UCSF R2D2 Training Dataset
Targets performance bottlenecks with GPU acceleration, especially WHO-SVM optimization
"""

import os
import time
import warnings
import multiprocessing
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed
import psutil

warnings.filterwarnings('ignore')

# ML imports - CPU models (keep fast ones)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

# GPU-accelerated models
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    print("⚠️  XGBoost not available - using CPU fallback for tree models")
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.tree import DecisionTreeClassifier
    XGBOOST_AVAILABLE = False

# PyTorch for MPS acceleration
try:
    import torch
    import torch.nn as nn
    PYTORCH_AVAILABLE = True
    
    class PyTorchMLP(nn.Module):
        """GPU-accelerated MLP using PyTorch"""
        
        def __init__(self, input_dim, hidden_dims=[100, 50], dropout=0.2):
            super().__init__()
            
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, 1))
            layers.append(nn.Sigmoid())
            
            self.network = nn.Sequential(*layers)
            
        def forward(self, x):
            return self.network(x).squeeze()
    
except ImportError:
    print("⚠️  PyTorch not available - using CPU fallback for neural networks")
    from sklearn.neural_network import MLPClassifier
    PYTORCH_AVAILABLE = False
    
    # Define dummy class for cases when PyTorch is not available
    class PyTorchMLP:
        pass

# Metrics imports
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    f1_score, fbeta_score, roc_auc_score, accuracy_score, 
    classification_report, average_precision_score
)
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE

# Import configuration
from config_gpu import GPUConfig, load_gpu_config_from_args, get_gpu_parser

class PyTorchMLPWrapper:
    """Scikit-learn compatible wrapper for PyTorch MLP"""
    
    def __init__(self, hidden_dims=[100, 50], epochs=100, lr=0.001, batch_size=32, device='cpu'):
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.device = device
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Initialize model
        self.model = PyTorchMLP(X_scaled.shape[1], self.hidden_dims).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = nn.BCELoss()
        
        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            # Mini-batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i+self.batch_size]
                batch_y = y_tensor[i:i+self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
        
        return self
    
    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            probs = self.model(X_tensor).cpu().numpy()
        
        # Return probabilities in sklearn format
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

def load_embeddings_and_labels(config):
    """Load embeddings and labels with enhanced error handling"""
    print("📂 Loading embeddings and labels...")
    print(f"   📁 Embeddings: {config.get_embeddings_path()}")
    print(f"   📁 Metadata: {config.get_metadata_path()}")
    print(f"   📁 Labels: {config.get_labels_path()}")
    
    # Load embeddings
    embeddings_data = np.load(config.get_embeddings_path(), allow_pickle=True)
    print(f"✅ Found {len(embeddings_data.files)} embedding files")
    
    # Convert NPZ to standard array format if needed
    if len(embeddings_data.files) == 1 and 'arr_0' in embeddings_data.files:
        # Single array format
        embeddings_array = embeddings_data['arr_0']
        print(f"✅ Converted to embeddings array: {embeddings_array.shape}")
    else:
        # File-key format - convert to single array
        all_embeddings = []
        for file_key in sorted(embeddings_data.files):
            embeddings = embeddings_data[file_key]
            if embeddings.ndim == 1:
                embeddings = embeddings.reshape(1, -1)
            all_embeddings.append(embeddings)
        embeddings_array = np.vstack(all_embeddings)
        print(f"✅ Converted to embeddings array: {embeddings_array.shape}")
    
    # Load metadata
    metadata_df = pd.read_csv(config.get_metadata_path())
    print(f"✅ Loaded metadata: {len(metadata_df)} file records")
    
    # Expand metadata to match embeddings if needed
    if len(metadata_df) != len(embeddings_array):
        if 'num_clips' in metadata_df.columns:
            expanded_metadata = []
            for _, row in metadata_df.iterrows():
                num_clips = row['num_clips']
                for clip_idx in range(num_clips):
                    clip_row = row.copy()
                    clip_row['clip_index'] = clip_idx
                    expanded_metadata.append(clip_row)
            metadata_df = pd.DataFrame(expanded_metadata)
            print(f"✅ Expanded metadata: {len(metadata_df)} clip records")
    
    # Load patient labels
    labels_df = pd.read_csv(config.get_labels_path())
    print(f"✅ Loaded patient labels: {len(labels_df)} patients")
    
    # Handle different column name formats
    label_columns = labels_df.columns.tolist()
    print(f"   📋 Labels columns: {label_columns}")
    
    if 'StudyID' in label_columns and 'Label' in label_columns:
        patient_id_col, label_col = 'StudyID', 'Label'
    elif 'patient_id' in label_columns and 'tb_status' in label_columns:
        patient_id_col, label_col = 'patient_id', 'tb_status'
    else:
        raise ValueError(f"Unknown label file format. Columns: {label_columns}")
    
    return embeddings_array, metadata_df, labels_df, patient_id_col, label_col

# Global function for multiprocessing (needs to be pickleable)
_global_embeddings = None
_global_metadata_df = None

def _process_patient_global(patient_id):
    """Global function to process patient for multiprocessing"""
    global _global_embeddings, _global_metadata_df
    
    patient_mask = _global_metadata_df['patient_id'] == patient_id
    patient_embeddings = _global_embeddings[patient_mask]
    
    if len(patient_embeddings) == 0:
        return None
        
    # Calculate multiple statistics
    stats = [
        np.mean(patient_embeddings, axis=0),      # Mean
        np.std(patient_embeddings, axis=0),       # Standard deviation
        np.median(patient_embeddings, axis=0),    # Median
        np.min(patient_embeddings, axis=0),       # Minimum
        np.max(patient_embeddings, axis=0),       # Maximum
        np.percentile(patient_embeddings, 25, axis=0),  # 25th percentile
        np.percentile(patient_embeddings, 75, axis=0)   # 75th percentile
    ]
    
    return {
        'patient_id': patient_id,
        'features': np.concatenate(stats),
        'num_clips': len(patient_embeddings)
    }

def create_patient_level_features(embeddings, metadata_df):
    """Create patient-level features with parallel processing"""
    print("👥 Creating patient-level features...")
    print(f"   💻 Using {multiprocessing.cpu_count()} cores for patient feature aggregation")
    
    global _global_embeddings, _global_metadata_df
    _global_embeddings = embeddings
    _global_metadata_df = metadata_df
    
    # Group by patient ID
    unique_patients = metadata_df['patient_id'].unique()
    
    # Process patients (using simple loop for now to avoid multiprocessing issues)
    results = []
    for patient_id in unique_patients:
        result = _process_patient_global(patient_id)
        if result is not None:
            results.append(result)
    
    # Filter out None results
    results = [r for r in results if r is not None]
    
    # Extract features and metadata
    X = np.array([r['features'] for r in results])
    patient_ids = np.array([r['patient_id'] for r in results])
    num_clips = np.array([r['num_clips'] for r in results])
    
    print(f"✅ Patient-level features: {X.shape}")
    print(f"   📊 Feature dimension per stat: {embeddings.shape[1]}")
    print(f"   📊 Total features: 7 × {embeddings.shape[1]} = {X.shape[1]}")
    
    return X, patient_ids, num_clips

def get_gpu_optimized_models(config):
    """Get GPU-optimized models based on available hardware"""
    models = {}
    
    # Always available CPU models (keep the fast ones)
    models.update({
        "Naive Bayes": GaussianNB(),
        "LDA": LinearDiscriminantAnalysis(),
        "Logistic Regression": LogisticRegression(
            penalty='l1', solver='liblinear', C=0.1,
            class_weight='balanced', random_state=config.random_state, max_iter=1000
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5)
    })
    
    # XGBoost models (GPU-accelerated if available, otherwise CPU)
    if XGBOOST_AVAILABLE:
        gpu_enabled = config.device_type in ['mps', 'cuda', 'xgboost_gpu']
        gpu_id = 0 if config.device_type in ['cuda', 'xgboost_gpu'] else None
        
        models.update({
            "XGBoost": xgb.XGBClassifier(
                tree_method=config.xgboost_tree_method,
                gpu_id=gpu_id,
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_state,
                eval_metric='logloss'
            ),
            "XGBoost RF": xgb.XGBRFClassifier(
                tree_method=config.xgboost_tree_method,
                gpu_id=gpu_id,
                n_estimators=100,
                max_depth=10,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=config.random_state,
                eval_metric='logloss'
            ),
            "XGBoost Dart": xgb.XGBClassifier(
                tree_method=config.xgboost_tree_method,
                gpu_id=gpu_id,
                booster='dart',
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=config.random_state,
                eval_metric='logloss'
            )
        })
    else:
        # CPU fallback tree models
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.tree import DecisionTreeClassifier
        models.update({
            "Random Forest": RandomForestClassifier(
                n_estimators=100, max_depth=10, class_weight='balanced',
                random_state=config.random_state, n_jobs=config.n_jobs
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                random_state=config.random_state
            ),
            "Decision Tree": DecisionTreeClassifier(
                max_depth=10, class_weight='balanced', random_state=config.random_state
            )
        })
    
    # PyTorch MLP for GPU acceleration
    if PYTORCH_AVAILABLE and config.device_type in ['mps', 'cuda', 'mps_pytorch_only', 'cuda_pytorch_only']:
        models["MLP (GPU)"] = PyTorchMLPWrapper(
            hidden_dims=[100, 50],
            epochs=100,
            lr=0.001,
            batch_size=32,
            device=config.pytorch_device
        )
    else:
        # CPU fallback MLP
        from sklearn.neural_network import MLPClassifier
        models["MLP"] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.01,
            random_state=config.random_state,
            max_iter=200
        )
    
    return models

def evaluate_model_single_fold(X_train, X_test, y_train, y_test, model, model_name, config):
    """Evaluate a single model on train/test split"""
    start_time = time.time()
    
    try:
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        
        # Probabilities
        if hasattr(model, 'predict_proba'):
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_train_proba = None
            y_test_proba = None
        
        # Calculate metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        else:
            tn = fp = fn = tp = 0
            sensitivity = specificity = precision = 0
        
        # Additional metrics
        f1 = f1_score(y_test, y_test_pred, average='weighted')
        
        # ROC AUC
        if y_test_proba is not None and len(np.unique(y_test)) > 1:
            roc_auc = roc_auc_score(y_test, y_test_proba)
        else:
            roc_auc = 0
        
        # WHO score and compliance
        who_score = sensitivity + 0.5 * specificity if sensitivity > 0 and specificity > 0 else 0
        who_compliant = sensitivity >= 0.9 and specificity >= 0.7
        
        training_time = time.time() - start_time
        
        return {
            'model_name': model_name,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'who_score': who_score,
            'who_compliant': who_compliant,
            'training_time': training_time,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp},
            'y_test': y_test,
            'y_test_proba': y_test_proba
        }
        
    except Exception as e:
        return {
            'model_name': model_name,
            'error': str(e),
            'training_time': time.time() - start_time
        }

def evaluate_models_parallel(X, y, config):
    """Evaluate models in parallel"""
    if config.use_cross_validation:
        return evaluate_models_cross_validation(X, y, config)
    else:
        return evaluate_models_single_split(X, y, config)

def evaluate_models_single_split(X, y, config):
    """Evaluate models using single train/test split"""
    print("🤖 Training models with single 80/20 split in parallel...")
    
    # Train-test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train)} patients ({np.sum(y_train)} TB+)")
    print(f"📊 Test set: {len(X_test)} patients ({np.sum(y_test)} TB+)")
    
    # Preprocessing
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Get models
    models = get_gpu_optimized_models(config)
    print(f"💻 Using {config.n_jobs} cores for model training")
    
    # Evaluate models
    def evaluate_single_model(item):
        model_name, model = item
        print(f"🔄 Training {model_name}...")
        result = evaluate_model_single_fold(
            X_train_scaled, X_test_scaled, y_train, y_test, model, model_name, config
        )
        
        if 'error' not in result:
            who_status = "✅" if result['who_compliant'] else "❌"
            print(f"✅ {model_name} completed in {result['training_time']:.1f}s - "
                  f"Test Acc: {result['test_accuracy']:.3f}, WHO: {who_status}")
        else:
            print(f"❌ {model_name} failed: {result['error']}")
        
        return result
    
    # Run in parallel
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(evaluate_single_model)(item) for item in models.items()
    )
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    print(f"✅ {len(successful_results)} models trained successfully")
    
    return successful_results, scaler

def evaluate_models_cross_validation(X, y, config):
    """Evaluate models using cross-validation"""
    print("🤖 Running cross-validation in parallel...")
    
    print(f"📊 Dataset: {len(X)} patients ({np.sum(y)} TB+, {len(y) - np.sum(y)} TB-)")
    print(f"💻 Using {config.n_jobs} cores for cross-validation")
    
    # Cross-validation setup
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    # Get models
    models = get_gpu_optimized_models(config)
    
    def evaluate_model_cv(model_name, model):
        print(f"🔄 Cross-validating {model_name}...")
        start_time = time.time()
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train_fold = X[train_idx]
            X_test_fold = X[test_idx]
            y_train_fold = y[train_idx]
            y_test_fold = y[test_idx]
            
            # Scale data
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_test_scaled = scaler.transform(X_test_fold)
            
            # Evaluate fold
            fold_result = evaluate_model_single_fold(
                X_train_scaled, X_test_scaled, y_train_fold, y_test_fold,
                model, model_name, config
            )
            
            if 'error' not in fold_result:
                fold_result['fold'] = fold
                fold_results.append(fold_result)
        
        if fold_results:
            # Calculate mean and std across folds
            metrics = ['train_accuracy', 'test_accuracy', 'sensitivity', 'specificity', 
                      'precision', 'f1_score', 'roc_auc', 'who_score']
            
            mean_results = {}
            for metric in metrics:
                values = [r[metric] for r in fold_results]
                mean_results[f'{metric}_mean'] = np.mean(values)
                mean_results[f'{metric}_std'] = np.std(values)
            
            # WHO compliance analysis
            who_compliant_folds = sum(1 for r in fold_results if r['who_compliant'])
            who_compliance_rate = who_compliant_folds / config.n_folds
            mean_results['who_compliance_rate'] = who_compliance_rate
            
            total_time = time.time() - start_time
            
            who_status = f"{who_compliant_folds}/{config.n_folds}"
            test_acc_str = f"{mean_results['test_accuracy_mean']:.3f}±{mean_results['test_accuracy_std']:.3f}"
            
            print(f"✅ {model_name} CV completed in {total_time:.1f}s - "
                  f"Test Acc: {test_acc_str}, WHO: {who_status} folds")
            
            return {
                'model_name': model_name,
                'fold_results': fold_results,
                'mean_results': mean_results,
                'who_compliant_folds': who_compliant_folds,
                'total_folds': config.n_folds,
                'training_time': total_time
            }
        else:
            return {'model_name': model_name, 'error': 'All folds failed'}
    
    # Run cross-validation in parallel
    results = Parallel(n_jobs=config.n_jobs)(
        delayed(evaluate_model_cv)(name, model) for name, model in models.items()
    )
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    print(f"✅ {len(successful_results)} models completed cross-validation")
    
    return successful_results, None

# Simplified visualization functions for GPU version
def create_comprehensive_visualizations(results, config):
    """Create comprehensive visualizations using enhanced naming"""
    print("📊 Creating comprehensive visualizations...")
    
    if config.use_cross_validation:
        create_cross_validation_visualizations(results, config)
    else:
        create_single_split_visualizations(results, config)

def create_single_split_visualizations(results, config):
    """Create visualizations for single split results"""
    
    # 1. Performance Dashboard
    create_performance_dashboard(results, config)
    
    # 2. ROC Curves
    create_roc_curves(results, config)
    
    # 3. Precision-Recall Curves  
    create_precision_recall_curves(results, config)
    
    # 4. Confusion Matrices Grid
    create_confusion_matrices_grid(results, config)
    
    # 5. WHO Compliance Analysis
    create_who_compliance_analysis(results, config)

def create_cross_validation_visualizations(results, config):
    """Create visualizations for cross-validation results"""
    
    # 1. Cross-Validation Performance Dashboard
    create_cv_performance_dashboard(results, config)
    
    # 2. Cross-Validation Fold Variance Plots
    create_cv_fold_variance_plots(results, config)
    
    # 3. Cross-Validation ROC Curves
    create_cv_roc_curves(results, config)
    
    # 4. Cross-Validation Precision-Recall Curves
    create_cv_precision_recall_curves(results, config)
    
    # 5. Cross-Validation WHO Compliance Analysis
    create_cv_who_compliance_analysis(results, config)

def create_performance_dashboard(results, config):
    """Create performance dashboard"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GPU-Optimized TB Detection Model Comparison', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Plot 1: Sensitivity with WHO target
    ax = axes[0, 0]
    sensitivities = [r['sensitivity'] for r in results]
    bars = ax.bar(model_names, sensitivities, color='lightblue', alpha=0.7)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥90%')
    ax.set_title('Sensitivity (Recall)')
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
    specificities = [r['specificity'] for r in results]
    bars = ax.bar(model_names, specificities, color='lightgreen', alpha=0.7)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥70%')
    ax.set_title('Specificity')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, specificities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Precision
    ax = axes[0, 2]
    precisions = [r['precision'] for r in results]
    bars = ax.bar(model_names, precisions, color='lightcoral', alpha=0.7)
    ax.set_title('Precision')
    ax.set_ylabel('Precision')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, precisions):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 4: F1 Score
    ax = axes[1, 0]
    f1_scores = [r['f1_score'] for r in results]
    bars = ax.bar(model_names, f1_scores, color='lightsalmon', alpha=0.7)
    ax.set_title('F1 Score')
    ax.set_ylabel('F1 Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 5: ROC AUC
    ax = axes[1, 1]
    roc_aucs = [r['roc_auc'] if r['roc_auc'] is not None else 0 for r in results]
    bars = ax.bar(model_names, roc_aucs, color='lightsteelblue', alpha=0.7)
    ax.set_title('ROC AUC')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, roc_aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 6: WHO Compliance
    ax = axes[1, 2]
    who_scores = [r['who_score'] for r in results]
    who_compliant = [r['who_compliant'] for r in results]
    colors = ['green' if compliant else 'red' for compliant in who_compliant]
    
    bars = ax.bar(model_names, who_scores, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (0.8)')
    ax.set_title('WHO Compliance Score')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    for bar, val in zip(bars, who_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot with enhanced naming
    plot_filename = config.get_output_filename('gpu_performance_dashboard.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance dashboard saved: {plot_path}")
    plt.close()

def create_roc_curves(results, config):
    """Create ROC curves plot"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        if result['y_test_proba'] is not None:
            try:
                fpr, tpr, _ = roc_curve(result['y_test'], result['y_test_proba'])
                auc_score = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f"{result['model_name']} (AUC = {auc_score:.3f})", 
                        color=colors[i], alpha=0.8, linewidth=2)
            except:
                print(f"⚠️  Could not create ROC curve for {result['model_name']}")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - GPU-Optimized TB Detection Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_roc_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ ROC curves saved: {plot_path}")
    plt.close()

def create_precision_recall_curves(results, config):
    """Create Precision-Recall curves plot"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        if result['y_test_proba'] is not None:
            try:
                precision, recall, _ = precision_recall_curve(result['y_test'], result['y_test_proba'])
                avg_precision = average_precision_score(result['y_test'], result['y_test_proba'])
                plt.plot(recall, precision, label=f"{result['model_name']} (AP = {avg_precision:.3f})", 
                        color=colors[i], alpha=0.8, linewidth=2)
            except:
                print(f"⚠️  Could not create PR curve for {result['model_name']}")
    
    # Add baseline (random classifier performance)
    pos_rate = np.mean([np.mean(r['y_test']) for r in results if r['y_test_proba'] is not None])
    plt.axhline(y=pos_rate, color='k', linestyle='--', alpha=0.5, label=f'Random Classifier (AP = {pos_rate:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - GPU-Optimized TB Detection Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_precision_recall_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Precision-Recall curves saved: {plot_path}")
    plt.close()

def create_confusion_matrices_grid(results, config):
    """Create grid of confusion matrices for all models"""
    
    n_models = len(results)
    if n_models == 0:
        return
    
    cols = min(4, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
    fig.suptitle('Confusion Matrices - GPU-Optimized TB Detection Models', fontsize=16, fontweight='bold')
    
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, result in enumerate(results):
        if i >= len(axes.flat):
            break
            
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        cm = result['confusion_matrix']
        cm_matrix = np.array([[cm['tn'], cm['fp']], [cm['fn'], cm['tp']]])
        
        # Create heatmap
        sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'])
        ax.set_title(f"{result['model_name']}\nSens: {result['sensitivity']:.3f}, Spec: {result['specificity']:.3f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide unused subplots
    for i in range(n_models, rows * cols):
        if i < len(axes.flat):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            ax.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_confusion_matrices_grid.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrices grid saved: {plot_path}")
    plt.close()

def create_cv_performance_dashboard(results, config):
    """Create cross-validation performance dashboard"""
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('GPU-Optimized Cross-Validation TB Detection Performance', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Extract mean values and std for error bars with fallbacks
    sens_means = [r['mean_results'].get('sensitivity_mean', 0) for r in results]
    sens_stds = [r['mean_results'].get('sensitivity_std', 0) for r in results]
    
    spec_means = [r['mean_results'].get('specificity_mean', 0) for r in results]  
    spec_stds = [r['mean_results'].get('specificity_std', 0) for r in results]
    
    # Plot 1: Sensitivity with error bars
    ax = axes[0, 0]
    bars = ax.bar(model_names, sens_means, yerr=sens_stds, color='lightblue', alpha=0.7, capsize=5)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥90%')
    ax.set_title('Sensitivity (Mean ± Std)')
    ax.set_ylabel('Sensitivity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Specificity with error bars
    ax = axes[0, 1]
    bars = ax.bar(model_names, spec_means, yerr=spec_stds, color='lightgreen', alpha=0.7, capsize=5)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥70%')
    ax.set_title('Specificity (Mean ± Std)')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 3: ROC AUC with error bars
    ax = axes[0, 2]
    roc_means = [r['mean_results'].get('roc_auc_mean', 0) for r in results]
    roc_stds = [r['mean_results'].get('roc_auc_std', 0) for r in results]
    bars = ax.bar(model_names, roc_means, yerr=roc_stds, color='lightyellow', alpha=0.7, capsize=5)
    ax.set_title('ROC AUC (Mean ± Std)')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: WHO Score with error bars
    ax = axes[1, 0]
    who_means = [r['mean_results'].get('who_score_mean', 0) for r in results]
    who_stds = [r['mean_results'].get('who_score_std', 0) for r in results]
    bars = ax.bar(model_names, who_means, yerr=who_stds, color='lightcoral', alpha=0.7, capsize=5)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='WHO Threshold ≥0.8')
    ax.set_title('WHO Score (Mean ± Std)')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 5: WHO Compliance Rate
    ax = axes[1, 1]
    try:
        # Try to get compliance rate from mean_results
        compliance_rates = [r['mean_results'].get('who_compliance_rate', 
                                                   r['who_compliant_folds'] / r['total_folds']) 
                           for r in results]
    except (KeyError, ZeroDivisionError):
        # Fallback: calculate from fold results
        compliance_rates = []
        for r in results:
            compliant_folds = sum(1 for fold in r['fold_results'] if fold.get('who_compliant', False))
            total_folds = len(r['fold_results']) if r['fold_results'] else 1
            compliance_rates.append(compliant_folds / total_folds)
    
    colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.5 else 'red' for rate in compliance_rates]
    bars = ax.bar(model_names, compliance_rates, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Target ≥80%')
    ax.set_title('WHO Compliance Rate')
    ax.set_ylabel('Compliance Rate')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 6: Training Time
    ax = axes[1, 2]
    train_times = [r.get('training_time', 0) for r in results]
    bars = ax.bar(model_names, train_times, color='lightpink', alpha=0.7)
    ax.set_title('Training Time (seconds)')
    ax.set_ylabel('Time (s)')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars for training time
    for bar, time_val in zip(bars, train_times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + max(train_times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_cv_performance_dashboard.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation performance dashboard saved: {plot_path}")
    plt.close()

def create_cv_fold_variance_plots(results, config):
    """Create cross-validation fold variance analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Validation Fold Variance Analysis - GPU Optimized', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    n_folds = results[0]['total_folds'] if results else config.n_folds
    
    # Plot 1: Sensitivity variance across folds
    ax = axes[0, 0]
    for i, result in enumerate(results):
        fold_sens = [fold['sensitivity'] for fold in result['fold_results']]
        folds = range(1, len(fold_sens) + 1)
        ax.plot(folds, fold_sens, marker='o', label=result['model_name'], alpha=0.7)
    
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.5, label='WHO Target ≥90%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Sensitivity')
    ax.set_title('Sensitivity Across CV Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Specificity variance across folds
    ax = axes[0, 1]
    for i, result in enumerate(results):
        fold_specs = [fold['specificity'] for fold in result['fold_results']]
        folds = range(1, len(fold_specs) + 1)
        ax.plot(folds, fold_specs, marker='s', label=result['model_name'], alpha=0.7)
    
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.5, label='WHO Target ≥70%')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Specificity')
    ax.set_title('Specificity Across CV Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 3: ROC AUC variance across folds
    ax = axes[1, 0]
    for i, result in enumerate(results):
        fold_aucs = [fold['roc_auc'] for fold in result['fold_results']]
        folds = range(1, len(fold_aucs) + 1)
        ax.plot(folds, fold_aucs, marker='^', label=result['model_name'], alpha=0.7)
    
    ax.set_xlabel('Fold')
    ax.set_ylabel('ROC AUC')
    ax.set_title('ROC AUC Across CV Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: WHO Score variance across folds
    ax = axes[1, 1]
    for i, result in enumerate(results):
        fold_who_scores = []
        for fold in result['fold_results']:
            # Calculate WHO score for each fold
            sens = fold['sensitivity']
            spec = fold['specificity']
            who_score = min(sens, spec) if sens >= 0.9 and spec >= 0.7 else 0
            fold_who_scores.append(who_score)
        
        folds = range(1, len(fold_who_scores) + 1)
        ax.plot(folds, fold_who_scores, marker='d', label=result['model_name'], alpha=0.7)
    
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.5, label='WHO Threshold ≥0.8')
    ax.set_xlabel('Fold')
    ax.set_ylabel('WHO Score')
    ax.set_title('WHO Score Across CV Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_cv_fold_variance.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation fold variance plots saved: {plot_path}")
    plt.close()

def create_who_compliance_analysis(results, config):
    """Create WHO compliance analysis for single split results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('GPU-Optimized WHO TB Screening Compliance Analysis', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Plot 1: WHO Compliance Status
    ax = axes[0]
    who_compliant = [r['who_compliant'] for r in results]
    colors = ['green' if compliant else 'red' for compliant in who_compliant]
    
    bars = ax.bar(model_names, [1 if c else 0 for c in who_compliant], color=colors, alpha=0.7)
    ax.set_ylabel('WHO Compliant (1=Yes, 0=No)')
    ax.set_title('WHO Compliance Status')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.2)
    
    # Add compliance labels
    for bar, compliant in zip(bars, who_compliant):
        height = bar.get_height()
        label = '✅ Compliant' if compliant else '❌ Non-compliant'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                label, ha='center', va='bottom', fontsize=9)
    
    # Plot 2: WHO Score Distribution
    ax = axes[1]
    who_scores = [r['who_score'] for r in results]
    bars = ax.bar(model_names, who_scores, color=colors, alpha=0.7)
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (0.8)')
    ax.set_ylabel('WHO Score')
    ax.set_title('WHO Score Distribution')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars, who_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}', ha='center', va='bottom', fontsize=8)
    
    # Plot 3: Sensitivity vs Specificity with WHO targets
    ax = axes[2]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    
    for i, (sens, spec, name, compliant) in enumerate(zip(sensitivities, specificities, model_names, who_compliant)):
        color = 'green' if compliant else 'red'
        marker = 'o' if compliant else 'x'
        ax.scatter(spec, sens, color=color, marker=marker, s=100, alpha=0.7)
        ax.annotate(name, (spec, sens), xytext=(5, 5), textcoords='offset points',
                   fontsize=8, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Add WHO target lines
    ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity Target ≥90%')
    ax.axvline(x=0.7, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity Target ≥70%')
    
    # Add WHO compliance regions
    ax.axhspan(0.9, 1.0, alpha=0.1, color='green')
    ax.axvspan(0.7, 1.0, alpha=0.1, color='green')
    
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('WHO Target Compliance')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_who_compliance_analysis.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ WHO compliance analysis saved: {plot_path}")
    plt.close()

def create_cv_roc_curves(results, config):
    """Create cross-validation ROC curves"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        model_name = result['model_name']
        
        # Collect ROC data from all folds
        all_fpr = []
        all_tpr = []
        fold_aucs = []
        
        for fold_result in result['fold_results']:
            if 'y_test_proba' in fold_result and fold_result['y_test_proba'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(fold_result['y_test'], fold_result['y_test_proba'])
                    all_fpr.append(fpr)
                    all_tpr.append(tpr)
                    fold_aucs.append(auc(fpr, tpr))
                except:
                    continue
        
        if fold_aucs:
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            
            # Plot mean ROC curve (simplified)
            mean_fpr = np.linspace(0, 1, 100)
            mean_tpr = np.mean([np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)], axis=0)
            
            plt.plot(mean_fpr, mean_tpr, 
                    label=f"{model_name} (AUC = {mean_auc:.3f} ± {std_auc:.3f})", 
                    color=colors[i], alpha=0.8, linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('Cross-Validation ROC Curves - GPU-Optimized TB Detection')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_cv_roc_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation ROC curves saved: {plot_path}")
    plt.close()

def create_cv_precision_recall_curves(results, config):
    """Create cross-validation Precision-Recall curves"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        model_name = result['model_name']
        
        # Collect PR data from all folds
        all_precision = []
        all_recall = []
        fold_aps = []
        
        for fold_result in result['fold_results']:
            if 'y_test_proba' in fold_result and fold_result['y_test_proba'] is not None:
                try:
                    precision, recall, _ = precision_recall_curve(fold_result['y_test'], fold_result['y_test_proba'])
                    all_precision.append(precision)
                    all_recall.append(recall)
                    fold_aps.append(average_precision_score(fold_result['y_test'], fold_result['y_test_proba']))
                except:
                    continue
        
        if fold_aps:
            mean_ap = np.mean(fold_aps)
            std_ap = np.std(fold_aps)
            
            # Plot mean PR curve (simplified)
            mean_recall = np.linspace(0, 1, 100)
            mean_precision = np.mean([np.interp(mean_recall[::-1], recall[::-1], precision[::-1])[::-1] 
                                    for precision, recall in zip(all_precision, all_recall)], axis=0)
            
            plt.plot(mean_recall, mean_precision, 
                    label=f"{model_name} (AP = {mean_ap:.3f} ± {std_ap:.3f})", 
                    color=colors[i], alpha=0.8, linewidth=2)
    
    # Add baseline
    if results and results[0]['fold_results']:
        pos_rates = []
        for result in results:
            for fold_result in result['fold_results']:
                if 'y_test' in fold_result:
                    pos_rates.append(np.mean(fold_result['y_test']))
        if pos_rates:
            mean_pos_rate = np.mean(pos_rates)
            plt.axhline(y=mean_pos_rate, color='k', linestyle='--', alpha=0.5, 
                       label=f'Random Classifier (AP = {mean_pos_rate:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Cross-Validation Precision-Recall Curves - GPU-Optimized TB Detection')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_cv_precision_recall_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation Precision-Recall curves saved: {plot_path}")
    plt.close()

def create_cv_who_compliance_analysis(results, config):
    """Create WHO compliance analysis for cross-validation results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Cross-Validation WHO TB Screening Compliance Analysis - GPU Optimized', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Plot 1: WHO Compliance Rate by Model
    ax = axes[0]
    compliance_rates = [r['who_compliant_folds'] / r['total_folds'] for r in results]
    colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.5 else 'red' for rate in compliance_rates]
    
    bars = ax.bar(model_names, compliance_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.8, label='80% Compliance Target')
    
    # Add value labels
    for bar, rate, result in zip(bars, compliance_rates, results):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{rate:.1%}\n({result["who_compliant_folds"]}/{result["total_folds"]})',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_ylabel('WHO Compliance Rate')
    ax.set_title('WHO Compliance Rate Across Folds')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Mean WHO Score with Error Bars
    ax = axes[1]
    who_means = [r['mean_results']['who_score_mean'] for r in results]
    who_stds = [r['mean_results']['who_score_std'] for r in results]
    
    bars = ax.bar(model_names, who_means, yerr=who_stds, color='lightblue', alpha=0.7, 
                  capsize=5, edgecolor='black')
    ax.axhline(y=0.8, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (0.8)')
    
    ax.set_ylabel('WHO Score (Mean ± Std)')
    ax.set_title('WHO Score Across CV Folds')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 3: Sensitivity vs Specificity (Mean with Error Bars)
    ax = axes[2]
    sens_means = [r['mean_results']['sensitivity_mean'] for r in results]
    sens_stds = [r['mean_results']['sensitivity_std'] for r in results]
    spec_means = [r['mean_results']['specificity_mean'] for r in results]
    spec_stds = [r['mean_results']['specificity_std'] for r in results]
    
    colors = ['green' if rate >= 0.5 else 'orange' if rate >= 0.2 else 'red' 
              for rate in compliance_rates]
    
    for i, (sens_m, sens_s, spec_m, spec_s, name, color) in enumerate(
        zip(sens_means, sens_stds, spec_means, spec_stds, model_names, colors)):
        ax.errorbar(spec_m, sens_m, xerr=spec_s, yerr=sens_s, 
                   fmt='o', color=color, alpha=0.7, markersize=10, capsize=5)
        ax.annotate(name, (spec_m, sens_m), xytext=(5, 5), textcoords='offset points',
                   fontsize=9, bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Add WHO target lines
    ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.8, label='WHO Sensitivity Target ≥90%')
    ax.axvline(x=0.7, color='blue', linestyle='--', alpha=0.8, label='WHO Specificity Target ≥70%')
    
    # Add WHO compliance regions
    ax.axhspan(0.9, 1.0, alpha=0.1, color='green')
    ax.axvspan(0.7, 1.0, alpha=0.1, color='green')
    
    ax.set_xlabel('Specificity (Mean)')
    ax.set_ylabel('Sensitivity (Mean)')
    ax.set_title('WHO Target Compliance (CV Results)')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('gpu_cv_who_compliance_analysis.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation WHO compliance analysis saved: {plot_path}")
    plt.close()

def save_results(results, config):
    """Save analysis results to files with enhanced naming"""
    print("💾 Saving results...")
    
    if config.use_cross_validation:
        return save_cross_validation_results(results, config)
    else:
        return save_single_split_results(results, config)

def save_single_split_results(results, config):
    """Save single split results to files"""
    # Create results DataFrame
    results_data = []
    for result in results:
        if 'error' not in result:
            results_data.append({
                'Model': result['model_name'],
                'Train_Accuracy': result['train_accuracy'],
                'Test_Accuracy': result['test_accuracy'], 
                'Sensitivity': result['sensitivity'],
                'Specificity': result['specificity'],
                'Precision': result['precision'],
                'F1_Score': result['f1_score'],
                'ROC_AUC': result['roc_auc'],
                'WHO_Score': result['who_score'],
                'WHO_Compliant': result['who_compliant'],
                'Training_Time': result['training_time'],
                'TP': result['confusion_matrix']['tp'],
                'TN': result['confusion_matrix']['tn'],
                'FP': result['confusion_matrix']['fp'],
                'FN': result['confusion_matrix']['fn']
            })
    
    if results_data:
        results_df = pd.DataFrame(results_data)
        results_df = results_df.sort_values('WHO_Score', ascending=False)
        
        # Save detailed results
        results_filename = config.get_output_filename('gpu_analysis_results.csv')
        results_path = os.path.join(config.results_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"✅ Detailed results saved: {results_path}")
        
        # Create executive summary
        summary_lines = [
            "GPU-OPTIMIZED TB DETECTION ANALYSIS - EXECUTIVE SUMMARY",
            "=" * 60,
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {config.get_run_identifier()}",
            f"GPU Device: {config.device_type.upper()}",
            f"Dataset: {config.dataset_name}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 30,
            f"Total Models Trained: {len(results_data)}",
            f"WHO Compliant Models: {sum(r['WHO_Compliant'] for r in results_data)}",
            f"WHO Compliance Rate: {sum(r['WHO_Compliant'] for r in results_data) / len(results_data):.1%}",
            "",
            "TOP PERFORMING MODELS",
            "-" * 30
        ]
        
        # Add top 5 models by WHO score
        top_models = results_df.head(5)
        for idx, row in top_models.iterrows():
            summary_lines.append(
                f"• {row['Model']}: WHO={row['WHO_Score']:.3f}, "
                f"Sens={row['Sensitivity']:.3f}, Spec={row['Specificity']:.3f}, "
                f"Time={row['Training_Time']:.1f}s, {'✅' if row['WHO_Compliant'] else '❌'}"
            )
        
        summary_text = "\n".join(summary_lines)
        summary_filename = config.get_output_filename('gpu_executive_summary.txt')
        summary_path = os.path.join(config.results_dir, summary_filename)
        
        with open(summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"✅ Executive summary saved: {summary_path}")
        
        return results_df
    
    return None

def save_cross_validation_results(results, config):
    """Save cross-validation results to files"""
    # Create summary results DataFrame (mean ± std)
    summary_data = []
    detailed_data = []
    
    for result in results:
        if 'error' not in result:
            mean_res = result['mean_results']
            
            # Summary results (mean ± std across folds)
            summary_data.append({
                'Model': result['model_name'],
                'Train_Accuracy_Mean': mean_res['train_accuracy_mean'],
                'Train_Accuracy_Std': mean_res['train_accuracy_std'],
                'Test_Accuracy_Mean': mean_res['test_accuracy_mean'],
                'Test_Accuracy_Std': mean_res['test_accuracy_std'],
                'Sensitivity_Mean': mean_res['sensitivity_mean'],
                'Sensitivity_Std': mean_res['sensitivity_std'],
                'Specificity_Mean': mean_res['specificity_mean'],
                'Specificity_Std': mean_res['specificity_std'],
                'Precision_Mean': mean_res['precision_mean'],
                'Precision_Std': mean_res['precision_std'],
                'F1_Score_Mean': mean_res['f1_score_mean'],
                'F1_Score_Std': mean_res['f1_score_std'],
                'ROC_AUC_Mean': mean_res['roc_auc_mean'],
                'ROC_AUC_Std': mean_res['roc_auc_std'],
                'WHO_Score_Mean': mean_res['who_score_mean'],
                'WHO_Score_Std': mean_res['who_score_std'],
                'WHO_Compliant_Folds': result['who_compliant_folds'],
                'Total_Folds': result['total_folds'],
                'WHO_Compliance_Rate': result['who_compliant_folds'] / result['total_folds'],
                'Training_Time': result['training_time']
            })
            
            # Detailed results (all folds)
            for fold_result in result['fold_results']:
                detailed_data.append({
                    'Model': result['model_name'],
                    'Fold': fold_result['fold'],
                    'Train_Accuracy': fold_result['train_accuracy'],
                    'Test_Accuracy': fold_result['test_accuracy'],
                    'Sensitivity': fold_result['sensitivity'],
                    'Specificity': fold_result['specificity'],
                    'Precision': fold_result['precision'],
                    'F1_Score': fold_result['f1_score'],
                    'ROC_AUC': fold_result['roc_auc'],
                    'WHO_Score': fold_result['who_score'],
                    'WHO_Compliant': fold_result['who_compliant'],
                    'TP': fold_result['confusion_matrix']['tp'],
                    'TN': fold_result['confusion_matrix']['tn'],
                    'FP': fold_result['confusion_matrix']['fp'],
                    'FN': fold_result['confusion_matrix']['fn']
                })
    
    if summary_data:
        # Save summary results
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('WHO_Score_Mean', ascending=False)
        
        summary_filename = config.get_output_filename('gpu_cross_validation_summary.csv')
        summary_path = os.path.join(config.results_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Cross-validation summary saved: {summary_path}")
        
        # Save detailed results
        detailed_df = pd.DataFrame(detailed_data)
        detailed_filename = config.get_output_filename('gpu_cross_validation_detailed.csv')
        detailed_path = os.path.join(config.results_dir, detailed_filename)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"✅ Cross-validation detailed results saved: {detailed_path}")
        
        # Create executive summary
        summary_lines = [
            "GPU-OPTIMIZED TB DETECTION CROSS-VALIDATION ANALYSIS - EXECUTIVE SUMMARY",
            "=" * 70,
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {config.get_run_identifier()}",
            f"GPU Device: {config.device_type.upper()}",
            f"Cross-Validation: {config.n_folds}-fold",
            f"Dataset: {config.dataset_name}",
            "",
            "PERFORMANCE SUMMARY",
            "-" * 30,
            f"Total Models Trained: {len(summary_data)}",
            f"Total Folds Evaluated: {len(detailed_data)}",
            f"Models with >50% WHO Compliance: {sum(1 for r in summary_data if r['WHO_Compliance_Rate'] > 0.5)}",
            "",
            "TOP PERFORMING MODELS (by mean WHO score)",
            "-" * 45
        ]
        
        # Add top 5 models by mean WHO score
        top_models = summary_df.head(5)
        for idx, row in top_models.iterrows():
            compliance_rate = row['WHO_Compliance_Rate']
            summary_lines.append(
                f"• {row['Model']}: WHO={row['WHO_Score_Mean']:.3f}±{row['WHO_Score_Std']:.3f}, "
                f"Sens={row['Sensitivity_Mean']:.3f}, Spec={row['Specificity_Mean']:.3f}, "
                f"Compliance={compliance_rate:.1%}, Time={row['Training_Time']:.1f}s"
            )
        
        summary_text = "\n".join(summary_lines)
        exec_summary_filename = config.get_output_filename('gpu_cross_validation_executive_summary.txt')
        exec_summary_path = os.path.join(config.results_dir, exec_summary_filename)
        
        with open(exec_summary_path, 'w') as f:
            f.write(summary_text)
        
        print(f"✅ Cross-validation executive summary saved: {exec_summary_path}")
        
        return summary_df
    
    return None

def main():
    """Main function"""
    parser = get_gpu_parser()
    args = parser.parse_args()
    config = load_gpu_config_from_args(args)
    
    # Print configuration and save it
    config.print_config()
    config.save_run_config()
    
    print(f"💾 System Memory: {psutil.virtual_memory().total // (1024**3)} GB available")
    print()
    
    try:
        # Load data
        embeddings, metadata_df, labels_df, patient_id_col, label_col = load_embeddings_and_labels(config)
        
        # Create patient-level features
        X, patient_ids, num_clips = create_patient_level_features(embeddings, metadata_df)
        
        # Create labels
        label_map = dict(zip(labels_df[patient_id_col], labels_df[label_col]))
        y = np.array([1 if label_map.get(pid, 'TB Negative') == 'TB Positive' else 0 for pid in patient_ids])
        
        print(f"📊 Final dataset: {len(embeddings)} clips from {len(patient_ids)} patients")
        print(f"   🟢 TB Positive clips: {np.sum(y)}")
        print(f"   🔴 TB Negative clips: {len(y) - np.sum(y)}")
        
        # Train and evaluate models
        results, scaler = evaluate_models_parallel(X, y, config)
        
        # Create visualizations
        create_comprehensive_visualizations(results, config)
        
        # Save results
        results_df = save_results(results, config)
        
        if results_df is not None:
            print("\n🏆 TOP PERFORMING MODELS:")
            print("=" * 50)
            top_3 = results_df.head(3)
            for idx, row in top_3.iterrows():
                # Handle both single split and cross-validation results
                if config.use_cross_validation:
                    # Cross-validation results
                    who_score = row['WHO_Score_Mean']
                    sensitivity = row['Sensitivity_Mean']
                    specificity = row['Specificity_Mean']
                    compliance_rate = row['WHO_Compliance_Rate']
                    status = f"✅ WHO COMPLIANT ({compliance_rate:.1%} folds)" if compliance_rate > 0.5 else f"❌ Low compliance ({compliance_rate:.1%} folds)"
                else:
                    # Single split results
                    who_score = row['WHO_Score']
                    sensitivity = row['Sensitivity']
                    specificity = row['Specificity']
                    who_compliant = row['WHO_Compliant']
                    status = "✅ WHO COMPLIANT" if who_compliant else "❌ Non-compliant"
                
                print(f"🥇 {row['Model']}")
                print(f"   WHO Score: {who_score:.3f} ({status})")
                print(f"   Sensitivity: {sensitivity:.3f}, Specificity: {specificity:.3f}")
                print(f"   Training Time: {row['Training_Time']:.1f}s")
                print()
        
        print("🎉 GPU-optimized analysis completed successfully!")
        print(f"🚀 Configuration saved as: {config.get_config_filename()}")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()