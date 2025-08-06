#!/usr/bin/env python3
"""
CPU-Optimized TB Detection Analysis for UCSF R2D2 Dataset
Multi-core parallel processing for faster model training and evaluation
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
from joblib import Parallel, delayed
import psutil
warnings.filterwarnings('ignore')

# Import configuration
from config import load_config_from_args, get_common_parser

# ML imports
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
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

def load_embeddings_and_labels(config):
    """Load embeddings and create labels with configurable file paths"""
    print("📂 Loading embeddings and labels...")
    
    embeddings_path = config.get_embeddings_path()
    metadata_path = config.get_metadata_path()
    labels_path = config.get_labels_path()
    
    print(f"   📁 Embeddings: {embeddings_path}")
    print(f"   📁 Metadata: {metadata_path}")
    print(f"   📁 Labels: {labels_path}")
    
    # Load embeddings
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    
    embeddings_data = np.load(embeddings_path, allow_pickle=True)
    
    # Check if embeddings are stored as individual file keys or as a single array
    if 'embeddings' in embeddings_data.files:
        # Standard format with single 'embeddings' key
        embeddings_array = embeddings_data['embeddings']
        print(f"✅ Loaded embeddings: {embeddings_array.shape}")
    else:
        # File-based format - each file path is a key
        print(f"✅ Found {len(embeddings_data.files)} embedding files")
        # Convert to single array format
        all_embeddings = []
        for file_key in embeddings_data.files:
            file_embeddings = embeddings_data[file_key]
            if len(file_embeddings) > 0:
                all_embeddings.extend(file_embeddings)
        
        if all_embeddings:
            embeddings_array = np.array(all_embeddings)
            print(f"✅ Converted to embeddings array: {embeddings_array.shape}")
        else:
            raise ValueError("No embeddings found in NPZ file")
    
    # Load metadata
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    metadata_df = pd.read_csv(metadata_path)
    print(f"✅ Loaded metadata: {len(metadata_df)} file records")
    
    # Expand metadata to match individual clips/embeddings
    expanded_metadata = []
    for _, row in metadata_df.iterrows():
        num_clips = row['num_clips']
        for clip_idx in range(num_clips):
            expanded_row = row.copy()
            expanded_row['clip_index'] = clip_idx
            expanded_metadata.append(expanded_row)
    
    metadata_df = pd.DataFrame(expanded_metadata)
    print(f"✅ Expanded metadata: {len(metadata_df)} clip records")
    
    # Load labels
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
    labels_df = pd.read_csv(labels_path)
    print(f"✅ Loaded patient labels: {len(labels_df)} patients")
    
    # Create embeddings array
    embeddings = embeddings_array
    
    # Check column names and merge metadata with labels
    print(f"   📋 Labels columns: {list(labels_df.columns)}")
    
    # Handle different possible column naming conventions
    if 'patient_id' in labels_df.columns and 'tb_status' in labels_df.columns:
        # Standard format
        merge_cols = ['patient_id', 'tb_status']
        merge_on = 'patient_id'
        status_col = 'tb_status'
    elif 'StudyID' in labels_df.columns and 'Label' in labels_df.columns:
        # UCSF format
        merge_cols = ['StudyID', 'Label']
        merge_on = 'StudyID'
        status_col = 'Label'
        # Rename to match metadata
        labels_df = labels_df.rename(columns={'StudyID': 'patient_id', 'Label': 'tb_status'})
        merge_on = 'patient_id'
        status_col = 'tb_status'
    else:
        raise ValueError(f"Unknown label file format. Columns: {list(labels_df.columns)}")
    
    metadata_with_labels = metadata_df.merge(
        labels_df[['patient_id', 'tb_status']], 
        on='patient_id', 
        how='left'
    )
    
    # Convert TB status to binary labels
    metadata_with_labels['label'] = (metadata_with_labels['tb_status'] == 'TB Positive').astype(int)
    
    # Remove rows with missing labels
    valid_mask = metadata_with_labels['label'].notna()
    metadata_clean = metadata_with_labels[valid_mask]
    embeddings_clean = embeddings[valid_mask]
    
    print(f"📊 Final dataset: {len(embeddings_clean)} clips from {len(labels_df)} patients")
    print(f"   🟢 TB Positive clips: {metadata_clean['label'].sum()}")
    print(f"   🔴 TB Negative clips: {len(metadata_clean) - metadata_clean['label'].sum()}")
    
    return embeddings_clean, metadata_clean

def create_patient_level_features(embeddings, metadata_df):
    """Create patient-level features from clip-level embeddings (optimized)"""
    print("👥 Creating patient-level features...")
    
    # Group by patient for parallel processing
    patient_groups = metadata_df.groupby('patient_id')
    
    def process_patient_parallel(args):
        """Process single patient features in parallel"""
        patient_id, patient_data = args
        patient_indices = patient_data.index.tolist()
        patient_embeddings = embeddings[patient_indices]
        
        if len(patient_embeddings) == 0:
            return None
        
        # Calculate aggregated features
        features = {
            'patient_id': patient_id,
            'num_clips': len(patient_embeddings),
            'label': patient_data['label'].iloc[0]  # All clips from same patient have same label
        }
        
        # Add statistical aggregations
        features['mean'] = np.mean(patient_embeddings, axis=0)
        features['std'] = np.std(patient_embeddings, axis=0)
        features['median'] = np.median(patient_embeddings, axis=0)
        features['q25'] = np.percentile(patient_embeddings, 25, axis=0)
        features['q75'] = np.percentile(patient_embeddings, 75, axis=0)
        features['min'] = np.min(patient_embeddings, axis=0)
        features['max'] = np.max(patient_embeddings, axis=0)
        
        return features
    
    # Process patients in parallel
    n_jobs = min(multiprocessing.cpu_count(), len(patient_groups))
    print(f"   💻 Using {n_jobs} cores for patient feature aggregation")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_patient_parallel)(group) 
        for group in patient_groups
    )
    
    # Filter out None results and combine features
    valid_results = [r for r in results if r is not None]
    
    if not valid_results:
        raise ValueError("No valid patient features created")
    
    # Combine all statistical features
    feature_dim = valid_results[0]['mean'].shape[0]
    stat_names = ['mean', 'std', 'median', 'q25', 'q75', 'min', 'max']
    
    patient_features = []
    patient_labels = []
    patient_ids = []
    patient_num_clips = []
    
    for result in valid_results:
        # Concatenate all statistical features
        combined_features = np.concatenate([result[stat] for stat in stat_names])
        
        patient_features.append(combined_features)
        patient_labels.append(result['label'])
        patient_ids.append(result['patient_id'])
        patient_num_clips.append(result['num_clips'])
    
    patient_features = np.array(patient_features)
    patient_labels = np.array(patient_labels)
    
    print(f"✅ Patient-level features: {patient_features.shape}")
    print(f"   📊 Feature dimension per stat: {feature_dim}")
    print(f"   📊 Total features: {len(stat_names)} × {feature_dim} = {patient_features.shape[1]}")
    
    return patient_features, patient_labels, patient_ids, patient_num_clips

def train_model_parallel(args):
    """Train a single model in parallel"""
    model_name, model, X_train, y_train, X_test, y_test, config = args
    
    print(f"🔄 Training {model_name}...")
    start_time = datetime.now()
    
    try:
        # Set n_jobs for models that support it
        if hasattr(model, 'n_jobs'):
            model.set_params(n_jobs=1)  # Use 1 job per model to avoid nested parallelism
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
        y_test_proba = None
        
        # Get prediction probabilities if available
        if hasattr(model, 'predict_proba'):
            try:
                y_test_proba = model.predict_proba(X_test)[:, 1]
            except:
                pass
        elif hasattr(model, 'decision_function'):
            try:
                y_test_proba = model.decision_function(X_test)
                # Normalize decision function output to [0, 1]
                y_test_proba = (y_test_proba - y_test_proba.min()) / (y_test_proba.max() - y_test_proba.min())
            except:
                pass
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # Calculate additional metrics
        f1 = f1_score(y_test, y_test_pred)
        
        roc_auc = None
        if y_test_proba is not None:
            try:
                roc_auc = roc_auc_score(y_test, y_test_proba)
            except:
                pass
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # WHO compliance check (Sensitivity + 0.5 × Specificity ≥ 1.25)
        who_score = sensitivity + 0.5 * specificity
        who_compliant = who_score >= 1.25
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        result = {
            'model_name': model_name,
            'model': model,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'who_score': who_score,
            'who_compliant': who_compliant,
            'y_test': y_test,
            'y_test_pred': y_test_pred,
            'y_test_proba': y_test_proba,
            'training_time': training_time,
            'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
        }
        
        print(f"✅ {model_name} completed in {training_time:.1f}s - Test Acc: {test_accuracy:.3f}, WHO: {'✅' if who_compliant else '❌'}")
        return result
        
    except Exception as e:
        print(f"❌ {model_name} failed: {str(e)}")
        return {
            'model_name': model_name,
            'error': str(e),
            'training_time': (datetime.now() - start_time).total_seconds()
        }

def evaluate_models_parallel(X, y, config):
    """Evaluate multiple models using either cross-validation or single split"""
    
    if config.use_cross_validation:
        print(f"🤖 Running {config.n_folds}-fold cross-validation in parallel...")
        return evaluate_models_cross_validation(X, y, config)
    else:
        print("🤖 Training models with single 80/20 split in parallel...")
        return evaluate_models_single_split(X, y, config)

def evaluate_models_single_split(X, y, config):
    """Evaluate multiple models with single train/test split"""
    
    # Split data (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.random_state, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train)} patients ({np.sum(y_train)} TB+)")
    print(f"📊 Test set: {len(X_test)} patients ({np.sum(y_test)} TB+)")
    
    # Apply feature scaling
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models with optimized parameters for parallel processing
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.random_state, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=config.random_state),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=config.random_state, n_jobs=1),
        'Logistic Regression': LogisticRegression(random_state=config.random_state, max_iter=1000, n_jobs=1),
        'SVM': SVC(random_state=config.random_state, probability=True),
        'MLP': MLPClassifier(random_state=config.random_state, max_iter=500),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'Decision Tree': DecisionTreeClassifier(random_state=config.random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }
    
    # Prepare arguments for parallel processing
    model_args = [
        (name, model, X_train_scaled, y_train, X_test_scaled, y_test, config)
        for name, model in models.items()
    ]
    
    # Train models in parallel
    print(f"💻 Using {config.n_jobs} cores for model training")
    
    # Use fewer cores for model training to avoid memory issues
    n_jobs_models = min(config.n_jobs, 4, len(models))  
    
    results = Parallel(n_jobs=n_jobs_models)(
        delayed(train_model_parallel)(args) for args in model_args
    )
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if failed_results:
        print(f"⚠️  {len(failed_results)} models failed:")
        for failed in failed_results:
            print(f"   ❌ {failed['model_name']}: {failed['error']}")
    
    print(f"✅ {len(successful_results)} models trained successfully")
    
    return successful_results, scaler

def evaluate_models_cross_validation(X, y, config):
    """Evaluate multiple models using k-fold cross-validation"""
    
    print(f"📊 Dataset: {len(X)} patients ({np.sum(y)} TB+, {len(y) - np.sum(y)} TB-)")
    
    # Apply feature scaling to entire dataset
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=config.random_state, n_jobs=1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=config.random_state),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=config.random_state, n_jobs=1),
        'Logistic Regression': LogisticRegression(random_state=config.random_state, max_iter=1000, n_jobs=1),
        'SVM': SVC(random_state=config.random_state, probability=True),
        'MLP': MLPClassifier(random_state=config.random_state, max_iter=500),
        'Naive Bayes': GaussianNB(),
        'LDA': LinearDiscriminantAnalysis(),
        'Decision Tree': DecisionTreeClassifier(random_state=config.random_state),
        'KNN': KNeighborsClassifier(n_neighbors=5, n_jobs=1)
    }
    
    # Setup cross-validation
    cv = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=config.random_state)
    
    print(f"💻 Using {config.n_jobs} cores for cross-validation")
    
    def evaluate_model_cv(args):
        """Evaluate single model with cross-validation"""
        model_name, model = args
        print(f"🔄 Cross-validating {model_name}...")
        start_time = datetime.now()
        
        try:
            # Set n_jobs for models that support it
            if hasattr(model, 'n_jobs'):
                model.set_params(n_jobs=1)  # Use 1 job per model to avoid nested parallelism
            
            fold_results = []
            
            # Perform cross-validation manually to get detailed results
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_scaled, y)):
                X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
                y_train_fold, y_test_fold = y[train_idx], y[test_idx]
                
                # Train model on this fold
                fold_model = model.__class__(**model.get_params())
                fold_model.fit(X_train_fold, y_train_fold)
                
                # Make predictions
                y_train_pred = fold_model.predict(X_train_fold)
                y_test_pred = fold_model.predict(X_test_fold)
                y_test_proba = None
                
                # Get prediction probabilities if available
                if hasattr(fold_model, 'predict_proba'):
                    try:
                        y_test_proba = fold_model.predict_proba(X_test_fold)[:, 1]
                    except:
                        pass
                elif hasattr(fold_model, 'decision_function'):
                    try:
                        y_test_proba = fold_model.decision_function(X_test_fold)
                        # Normalize decision function output to [0, 1]
                        y_test_proba = (y_test_proba - y_test_proba.min()) / (y_test_proba.max() - y_test_proba.min())
                    except:
                        pass
                
                # Calculate metrics for this fold
                train_accuracy = accuracy_score(y_train_fold, y_train_pred)
                test_accuracy = accuracy_score(y_test_fold, y_test_pred)
                f1 = f1_score(y_test_fold, y_test_pred)
                
                roc_auc = None
                if y_test_proba is not None:
                    try:
                        roc_auc = roc_auc_score(y_test_fold, y_test_proba)
                    except:
                        pass
                
                # Calculate confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_test_fold, y_test_pred).ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                
                # WHO compliance check
                who_score = sensitivity + 0.5 * specificity
                who_compliant = who_score >= 1.25
                
                fold_results.append({
                    'fold': fold_idx + 1,
                    'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy,
                    'sensitivity': sensitivity,
                    'specificity': specificity,
                    'precision': precision,
                    'f1_score': f1,
                    'roc_auc': roc_auc,
                    'who_score': who_score,
                    'who_compliant': who_compliant,
                    'y_test': y_test_fold,
                    'y_test_pred': y_test_pred,
                    'y_test_proba': y_test_proba,
                    'confusion_matrix': {'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp}
                })
            
            # Calculate mean and std across folds
            metrics = ['train_accuracy', 'test_accuracy', 'sensitivity', 'specificity', 
                      'precision', 'f1_score', 'roc_auc', 'who_score']
            
            mean_results = {}
            std_results = {}
            for metric in metrics:
                values = [fold[metric] for fold in fold_results if fold[metric] is not None]
                if values:
                    mean_results[f'{metric}_mean'] = np.mean(values)
                    mean_results[f'{metric}_std'] = np.std(values)
                else:
                    mean_results[f'{metric}_mean'] = 0
                    mean_results[f'{metric}_std'] = 0
            
            # WHO compliance across folds
            who_compliant_folds = sum(1 for fold in fold_results if fold['who_compliant'])
            
            training_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'model_name': model_name,
                'model': model,
                'fold_results': fold_results,
                'mean_results': mean_results,
                'std_results': std_results,
                'who_compliant_folds': who_compliant_folds,
                'total_folds': config.n_folds,
                'training_time': training_time
            }
            
            print(f"✅ {model_name} CV completed in {training_time:.1f}s - "
                  f"Test Acc: {mean_results['test_accuracy_mean']:.3f}±{mean_results['test_accuracy_std']:.3f}, "
                  f"WHO: {who_compliant_folds}/{config.n_folds} folds")
            
            return result
            
        except Exception as e:
            print(f"❌ {model_name} failed: {str(e)}")
            return {
                'model_name': model_name,
                'error': str(e),
                'training_time': (datetime.now() - start_time).total_seconds()
            }
    
    # Prepare arguments for parallel processing
    model_args = list(models.items())
    
    # Evaluate models in parallel
    n_jobs_models = min(config.n_jobs, 4, len(models))  # Conservative parallel jobs
    
    results = Parallel(n_jobs=n_jobs_models)(
        delayed(evaluate_model_cv)(args) for args in model_args
    )
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    failed_results = [r for r in results if 'error' in r]
    
    if failed_results:
        print(f"⚠️  {len(failed_results)} models failed:")
        for failed in failed_results:
            print(f"   ❌ {failed['model_name']}: {failed['error']}")
    
    print(f"✅ {len(successful_results)} models completed cross-validation")
    
    return successful_results, scaler

def create_comprehensive_visualizations(results, config):
    """Create comprehensive visualizations for model comparison"""
    if not results:
        print("⚠️  No results to visualize")
        return
    
    print("📊 Creating comprehensive visualizations...")
    
    if config.use_cross_validation:
        create_cross_validation_visualizations(results, config)
    else:
        create_single_split_visualizations(results, config)

def create_single_split_visualizations(results, config):
    """Create visualizations for single split results"""
    
    # 1. Performance Dashboard (6-panel)
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
    """Create 6-panel performance dashboard"""
    
    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('CPU-Optimized TB Detection Model Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data for plotting
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
    
    # Add value labels
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
    
    # Add value labels
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
    
    # Add value labels
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
    
    # Add value labels
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
    
    # Add value labels
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
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (1.25)')
    ax.set_title('WHO Compliance Score')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Add value labels
    for bar, val in zip(bars, who_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('performance_dashboard.png')
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
    plt.title('ROC Curves - TB Detection Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('roc_curves.png')
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
    plt.title('Precision-Recall Curves - TB Detection Performance')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('precision_recall_curves.png')
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
    fig.suptitle('Confusion Matrices - TB Detection Models', fontsize=16, fontweight='bold')
    
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
                   xticklabels=['Negative', 'Positive'], 
                   yticklabels=['Negative', 'Positive'])
        
        # Add metrics to title
        title = f"{result['model_name']}\n"
        title += f"Sens: {result['sensitivity']:.3f}, Spec: {result['specificity']:.3f}\n"
        title += f"{'✅ WHO' if result['who_compliant'] else '❌ Non-WHO'}"
        ax.set_title(title, fontsize=10)
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
    plot_filename = config.get_output_filename('confusion_matrices_grid.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Confusion matrices grid saved: {plot_path}")
    plt.close()

def create_who_compliance_analysis(results, config):
    """Create WHO compliance analysis visualization"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('WHO TB Screening Compliance Analysis', fontsize=16, fontweight='bold')
    
    # Extract data
    model_names = [r['model_name'] for r in results]
    sensitivities = [r['sensitivity'] for r in results]
    specificities = [r['specificity'] for r in results]
    who_scores = [r['who_score'] for r in results]
    who_compliant = [r['who_compliant'] for r in results]
    
    # Plot 1: Sensitivity vs Specificity with WHO regions
    ax = axes[0]
    colors = ['green' if compliant else 'red' for compliant in who_compliant]
    scatter = ax.scatter(specificities, sensitivities, c=colors, alpha=0.7, s=150, edgecolor='black')
    
    # Add model names as annotations
    for i, name in enumerate(model_names):
        ax.annotate(name, (specificities[i], sensitivities[i]), 
                   xytext=(5, 5), textcoords='offset points', fontsize=9,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7))
    
    # Add WHO target lines
    ax.axhline(y=0.9, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='WHO Sensitivity Target ≥90%')
    ax.axvline(x=0.7, color='blue', linestyle='--', alpha=0.8, linewidth=2, label='WHO Specificity Target ≥70%')
    
    # Add WHO compliance regions
    ax.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Sensitivity Target Zone')
    ax.axvspan(0.7, 1.0, alpha=0.1, color='green', label='Specificity Target Zone')
    
    ax.set_xlabel('Specificity')
    ax.set_ylabel('Sensitivity')
    ax.set_title('WHO Target Compliance Map')
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    # Plot 2: WHO Score comparison
    ax = axes[1]
    colors = ['green' if compliant else 'red' for compliant in who_compliant]
    bars = ax.bar(model_names, who_scores, color=colors, alpha=0.7, edgecolor='black')
    
    # Add WHO threshold line
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.8, linewidth=2, label='WHO Threshold (1.25)')
    
    # Add value labels on bars
    for bar, score, compliant in zip(bars, who_scores, who_compliant):
        height = bar.get_height()
        symbol = '✅' if compliant else '❌'
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score:.3f}\n{symbol}', ha='center', va='bottom', fontsize=9,
                fontweight='bold')
    
    ax.set_ylabel('WHO Score (Sensitivity + 0.5 × Specificity)')
    ax.set_title('WHO Compliance Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('who_compliance_analysis.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ WHO compliance analysis saved: {plot_path}")
    plt.close()

def create_cv_performance_dashboard(results, config):
    """Create cross-validation performance dashboard with error bars"""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Cross-Validation TB Detection Model Comparison', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Extract mean and std values
    metrics_data = {}
    for metric in ['sensitivity', 'specificity', 'precision', 'f1_score', 'roc_auc', 'who_score']:
        means = [r['mean_results'][f'{metric}_mean'] for r in results]
        stds = [r['mean_results'][f'{metric}_std'] for r in results]
        metrics_data[metric] = {'means': means, 'stds': stds}
    
    # Plot 1: Sensitivity with WHO target
    ax = axes[0, 0]
    means = metrics_data['sensitivity']['means']
    stds = metrics_data['sensitivity']['stds']
    bars = ax.bar(model_names, means, yerr=stds, color='lightblue', alpha=0.7, capsize=5)
    ax.axhline(y=0.9, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥90%')
    ax.set_title('Sensitivity (Cross-Validation)')
    ax.set_ylabel('Sensitivity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 2: Specificity with WHO target
    ax = axes[0, 1]
    means = metrics_data['specificity']['means']
    stds = metrics_data['specificity']['stds']
    bars = ax.bar(model_names, means, yerr=stds, color='lightgreen', alpha=0.7, capsize=5)
    ax.axhline(y=0.7, color='red', linestyle='--', alpha=0.7, label='WHO Target ≥70%')
    ax.set_title('Specificity (Cross-Validation)')
    ax.set_ylabel('Specificity')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 3: Precision
    ax = axes[0, 2]
    means = metrics_data['precision']['means']
    stds = metrics_data['precision']['stds']
    ax.bar(model_names, means, yerr=stds, color='lightcoral', alpha=0.7, capsize=5)
    ax.set_title('Precision (Cross-Validation)')
    ax.set_ylabel('Precision')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: F1 Score
    ax = axes[1, 0]
    means = metrics_data['f1_score']['means']
    stds = metrics_data['f1_score']['stds']
    ax.bar(model_names, means, yerr=stds, color='lightsalmon', alpha=0.7, capsize=5)
    ax.set_title('F1 Score (Cross-Validation)')
    ax.set_ylabel('F1 Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 5: ROC AUC
    ax = axes[1, 1]
    means = metrics_data['roc_auc']['means']
    stds = metrics_data['roc_auc']['stds']
    ax.bar(model_names, means, yerr=stds, color='lightsteelblue', alpha=0.7, capsize=5)
    ax.set_title('ROC AUC (Cross-Validation)')
    ax.set_ylabel('ROC AUC')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 6: WHO Compliance
    ax = axes[1, 2]
    means = metrics_data['who_score']['means']
    stds = metrics_data['who_score']['stds']
    who_compliant_rates = [r['who_compliant_folds'] / r['total_folds'] for r in results]
    colors = ['green' if rate >= 0.5 else 'orange' if rate >= 0.2 else 'red' for rate in who_compliant_rates]
    
    bars = ax.bar(model_names, means, yerr=stds, color=colors, alpha=0.7, capsize=5)
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (1.25)')
    ax.set_title('WHO Score (Cross-Validation)')
    ax.set_ylabel('WHO Score')
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('cv_performance_dashboard.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation performance dashboard saved: {plot_path}")
    plt.close()

def create_cv_fold_variance_plots(results, config):
    """Create cross-validation fold variance analysis plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cross-Validation Fold Variance Analysis', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    n_folds = results[0]['total_folds'] if results else 5
    
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
    
    # Plot 3: WHO Score variance across folds
    ax = axes[1, 0]
    for i, result in enumerate(results):
        fold_who_scores = [fold['who_score'] for fold in result['fold_results']]
        folds = range(1, len(fold_who_scores) + 1)
        ax.plot(folds, fold_who_scores, marker='^', label=result['model_name'], alpha=0.7)
    
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.5, label='WHO Threshold (1.25)')
    ax.set_xlabel('Fold')
    ax.set_ylabel('WHO Score')
    ax.set_title('WHO Score Across CV Folds')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.1)
    
    # Plot 4: Performance variance (coefficient of variation)
    ax = axes[1, 1]
    metrics = ['sensitivity', 'specificity', 'f1_score', 'roc_auc']
    metric_labels = ['Sensitivity', 'Specificity', 'F1 Score', 'ROC AUC']
    
    cv_data = {}
    for metric in metrics:
        cv_data[metric] = []
        for result in enumerate(results):
            fold_values = [fold[metric] for fold in result[1]['fold_results'] if fold[metric] is not None]
            if fold_values:
                cv = np.std(fold_values) / np.mean(fold_values) if np.mean(fold_values) > 0 else 0
                cv_data[metric].append(cv)
            else:
                cv_data[metric].append(0)
    
    x = np.arange(len(model_names))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax.bar(x + i * width, cv_data[metric], width, label=label, alpha=0.7)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Coefficient of Variation')
    ax.set_title('Performance Variability Across Folds')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    plot_filename = config.get_output_filename('cv_fold_variance.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation fold variance plots saved: {plot_path}")
    plt.close()

def create_cv_roc_curves(results, config):
    """Create ROC curves for cross-validation results"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        # Collect all fold ROC data
        fold_tprs = []
        fold_aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        
        for fold in result['fold_results']:
            if fold['y_test_proba'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(fold['y_test'], fold['y_test_proba'])
                    fold_aucs.append(auc(fpr, tpr))
                    # Interpolate to common FPR values
                    fold_tprs.append(np.interp(mean_fpr, fpr, tpr))
                    fold_tprs[-1][0] = 0.0  # Ensure it starts at 0
                except:
                    continue
        
        if fold_tprs:
            # Calculate mean and std
            fold_tprs = np.array(fold_tprs)
            mean_tpr = np.mean(fold_tprs, axis=0)
            std_tpr = np.std(fold_tprs, axis=0)
            mean_tpr[-1] = 1.0  # Ensure it ends at 1
            
            mean_auc = np.mean(fold_aucs)
            std_auc = np.std(fold_aucs)
            
            # Plot mean ROC curve
            plt.plot(mean_fpr, mean_tpr, color=colors[i], alpha=0.8, linewidth=2,
                    label=f"{result['model_name']} (AUC = {mean_auc:.3f} ± {std_auc:.3f})")
            
            # Plot confidence interval
            tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
            tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
            plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[i], alpha=0.2)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curves - Cross-Validation Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('cv_roc_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation ROC curves saved: {plot_path}")
    plt.close()

def create_cv_precision_recall_curves(results, config):
    """Create Precision-Recall curves for cross-validation results"""
    
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(results)))
    
    for i, result in enumerate(results):
        # Collect all fold PR data
        fold_precisions = []
        fold_aps = []
        mean_recall = np.linspace(0, 1, 100)
        
        for fold in result['fold_results']:
            if fold['y_test_proba'] is not None:
                try:
                    precision, recall, _ = precision_recall_curve(fold['y_test'], fold['y_test_proba'])
                    fold_aps.append(average_precision_score(fold['y_test'], fold['y_test_proba']))
                    # Interpolate to common recall values
                    fold_precisions.append(np.interp(mean_recall[::-1], recall[::-1], precision[::-1])[::-1])
                except:
                    continue
        
        if fold_precisions:
            # Calculate mean and std
            fold_precisions = np.array(fold_precisions)
            mean_precision = np.mean(fold_precisions, axis=0)
            std_precision = np.std(fold_precisions, axis=0)
            
            mean_ap = np.mean(fold_aps)
            std_ap = np.std(fold_aps)
            
            # Plot mean PR curve
            plt.plot(mean_recall, mean_precision, color=colors[i], alpha=0.8, linewidth=2,
                    label=f"{result['model_name']} (AP = {mean_ap:.3f} ± {std_ap:.3f})")
            
            # Plot confidence interval
            precision_upper = np.minimum(mean_precision + std_precision, 1)
            precision_lower = np.maximum(mean_precision - std_precision, 0)
            plt.fill_between(mean_recall, precision_lower, precision_upper, color=colors[i], alpha=0.2)
    
    # Add baseline
    pos_rate = np.mean([np.mean([np.mean(fold['y_test']) for fold in result['fold_results']]) 
                       for result in results])
    plt.axhline(y=pos_rate, color='k', linestyle='--', alpha=0.5, 
                label=f'Random Classifier (AP = {pos_rate:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves - Cross-Validation Results')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plot_filename = config.get_output_filename('cv_precision_recall_curves.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation Precision-Recall curves saved: {plot_path}")
    plt.close()

def create_cv_who_compliance_analysis(results, config):
    """Create WHO compliance analysis for cross-validation results"""
    
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle('Cross-Validation WHO TB Screening Compliance Analysis', fontsize=16, fontweight='bold')
    
    model_names = [r['model_name'] for r in results]
    
    # Plot 1: WHO Compliance Rate by Model
    ax = axes[0]
    compliance_rates = [r['who_compliant_folds'] / r['total_folds'] for r in results]
    colors = ['green' if rate >= 0.8 else 'orange' if rate >= 0.5 else 'red' for rate in compliance_rates]
    
    bars = ax.bar(model_names, compliance_rates, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.8, label='80% Compliance Target')
    
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
    ax.axhline(y=1.25, color='black', linestyle='--', alpha=0.8, label='WHO Threshold (1.25)')
    
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
    plot_filename = config.get_output_filename('cv_who_compliance_analysis.png')
    plot_path = os.path.join(config.results_dir, plot_filename)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✅ Cross-validation WHO compliance analysis saved: {plot_path}")
    plt.close()

def save_results(results, config):
    """Save analysis results to files"""
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
        results_filename = config.get_output_filename('cpu_optimized_analysis_results.csv')
        results_path = os.path.join(config.results_dir, results_filename)
        results_df.to_csv(results_path, index=False)
        print(f"✅ Detailed results saved: {results_path}")
        
        # Create executive summary
        summary_lines = [
            "CPU-OPTIMIZED TB DETECTION ANALYSIS - EXECUTIVE SUMMARY",
            "=" * 60,
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {config.embeddings_file}, single 80/20 train-test split",
            f"CPU Cores Used: {config.n_jobs}",
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
                f"{'✅' if row['WHO_Compliant'] else '❌'}"
            )
        
        summary_text = "\n".join(summary_lines)
        summary_filename = config.get_output_filename('cpu_optimized_executive_summary.txt')
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
    
    if summary_data and detailed_data:
        # Save summary results
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('WHO_Score_Mean', ascending=False)
        
        summary_filename = config.get_output_filename('cpu_optimized_cross_validation_summary.csv')
        summary_path = os.path.join(config.results_dir, summary_filename)
        summary_df.to_csv(summary_path, index=False)
        print(f"✅ Cross-validation summary saved: {summary_path}")
        
        # Save detailed results (all folds)
        detailed_df = pd.DataFrame(detailed_data)
        detailed_filename = config.get_output_filename('cpu_optimized_cross_validation_detailed.csv')
        detailed_path = os.path.join(config.results_dir, detailed_filename)
        detailed_df.to_csv(detailed_path, index=False)
        print(f"✅ Cross-validation detailed results saved: {detailed_path}")
        
        # Create executive summary
        summary_lines = [
            "CPU-OPTIMIZED TB DETECTION CROSS-VALIDATION ANALYSIS - EXECUTIVE SUMMARY",
            "=" * 70,
            "",
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Configuration: {config.embeddings_file}, {config.n_folds}-fold cross-validation",
            f"CPU Cores Used: {config.n_jobs}",
            "",
            "CROSS-VALIDATION SUMMARY",
            "-" * 35,
            f"Total Models Evaluated: {len(summary_data)}",
            f"Total Folds per Model: {config.n_folds}",
            f"Total Evaluations: {len(summary_data) * config.n_folds}",
            "",
            "WHO COMPLIANCE SUMMARY",
            "-" * 30,
        ]
        
        # Add WHO compliance statistics
        total_fold_evaluations = len(detailed_data)
        who_compliant_evaluations = sum(1 for d in detailed_data if d['WHO_Compliant'])
        
        summary_lines.extend([
            f"Total Fold Evaluations: {total_fold_evaluations}",
            f"WHO Compliant Evaluations: {who_compliant_evaluations}",
            f"Overall WHO Compliance Rate: {who_compliant_evaluations / total_fold_evaluations:.1%}",
            "",
            "TOP PERFORMING MODELS (by mean WHO score)",
            "-" * 45
        ])
        
        # Add top 5 models by mean WHO score
        top_models = summary_df.head(5)
        for idx, row in top_models.iterrows():
            compliance_rate = row['WHO_Compliance_Rate']
            summary_lines.append(
                f"• {row['Model']}: WHO={row['WHO_Score_Mean']:.3f}±{row['WHO_Score_Std']:.3f}, "
                f"Sens={row['Sensitivity_Mean']:.3f}±{row['Sensitivity_Std']:.3f}, "
                f"Spec={row['Specificity_Mean']:.3f}±{row['Specificity_Std']:.3f}, "
                f"Compliance: {row['WHO_Compliant_Folds']}/{row['Total_Folds']} folds ({compliance_rate:.1%})"
            )
        
        summary_text = "\n".join(summary_lines)
        summary_exec_filename = config.get_output_filename('cpu_optimized_cross_validation_executive_summary.txt')
        summary_exec_path = os.path.join(config.results_dir, summary_exec_filename)
        
        with open(summary_exec_path, 'w') as f:
            f.write(summary_text)
        
        print(f"✅ Cross-validation executive summary saved: {summary_exec_path}")
        
        return summary_df
    
    return None

def main():
    """Main function"""
    parser = get_common_parser()
    args = parser.parse_args()
    config = load_config_from_args(args)
    config.print_config()
    
    print(f"💾 System Memory: {psutil.virtual_memory().total // (1024**3)} GB available")
    print()
    
    try:
        # Load data
        embeddings, metadata = load_embeddings_and_labels(config)
        
        # Create patient-level features
        X, y, patient_ids, num_clips = create_patient_level_features(embeddings, metadata)
        
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
        
        print("🎉 CPU-optimized analysis completed successfully!")
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()