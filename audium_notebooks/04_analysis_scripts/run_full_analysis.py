#!/usr/bin/env python3
"""
Run TB detection analysis on full UCSF dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Advanced ML imports
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier, 
    VotingClassifier, AdaBoostClassifier
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score, fbeta_score, roc_auc_score,
    accuracy_score, precision_score, recall_score
)
from sklearn.utils.class_weight import compute_class_weight

# Data augmentation
from imblearn.over_sampling import SMOTE

# Feature engineering
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from scipy import stats

# XGBoost
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

print("✅ Advanced ML libraries loaded successfully")
print(f"🔧 XGBoost available: {XGBOOST_AVAILABLE}")

def extract_temporal_features(embedding_sequence):
    """
    Extract temporal features from embedding sequences
    
    Args:
        embedding_sequence: (n_clips, n_features) array
    
    Returns:
        feature_vector: concatenated temporal features
    """
    features = []
    
    # Statistical features across time
    features.extend([
        np.mean(embedding_sequence, axis=0),  # Temporal mean
        np.std(embedding_sequence, axis=0),   # Temporal std
        np.max(embedding_sequence, axis=0),   # Temporal max
        np.min(embedding_sequence, axis=0),   # Temporal min
        np.median(embedding_sequence, axis=0) # Temporal median
    ])
    
    # Temporal dynamics
    if len(embedding_sequence) > 1:
        # First and second derivatives (temporal changes)
        first_diff = np.diff(embedding_sequence, axis=0)
        features.append(np.mean(first_diff, axis=0))  # Mean change rate
        features.append(np.std(first_diff, axis=0))   # Variability of changes
        
        if len(embedding_sequence) > 2:
            second_diff = np.diff(first_diff, axis=0)
            features.append(np.mean(second_diff, axis=0))  # Acceleration
        else:
            features.append(np.zeros(512))  # Pad with zeros if insufficient data
    else:
        features.append(np.zeros(512))  # No temporal change
        features.append(np.zeros(512))  # No temporal change
        features.append(np.zeros(512))  # No temporal change
    
    # Range and percentiles
    features.append(np.ptp(embedding_sequence, axis=0))  # Range (max - min)
    features.append(np.percentile(embedding_sequence, 25, axis=0))  # Q1
    features.append(np.percentile(embedding_sequence, 75, axis=0))  # Q3
    
    # Skewness and kurtosis (shape of distribution) - with NaN handling
    try:
        skew_feat = stats.skew(embedding_sequence, axis=0, nan_policy='omit')
        # Replace any remaining NaN values with 0
        skew_feat = np.nan_to_num(skew_feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(skew_feat)
    except:
        features.append(np.zeros(512))
    
    try:
        kurt_feat = stats.kurtosis(embedding_sequence, axis=0, nan_policy='omit')
        # Replace any remaining NaN values with 0
        kurt_feat = np.nan_to_num(kurt_feat, nan=0.0, posinf=0.0, neginf=0.0)
        features.append(kurt_feat)
    except:
        features.append(np.zeros(512))
    
    # Concatenate all features
    final_features = np.concatenate(features)
    
    # Final safety check - replace any NaN/inf values
    final_features = np.nan_to_num(final_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return final_features

def load_advanced_embeddings(embedding_path, metadata_path, use_temporal=True, max_samples=None):
    """
    Load embeddings with advanced feature engineering
    NEW: Works with new dataset structure
    """
    print("🔄 Loading new UCSF embeddings with advanced features...")
    
    # Load embeddings - each audio file is a separate key
    embeddings_data = np.load(embedding_path)
    all_keys = list(embeddings_data.keys())
    print(f"📊 Loaded {len(all_keys)} embedding files")
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    if max_samples:
        metadata = metadata.head(max_samples)
    
    # Find matching keys
    common_keys = set(all_keys) & set(metadata['full_key'])
    print(f"📊 Found {len(common_keys)} matching files")
    
    # Create mapping from key to metadata
    key_to_label = dict(zip(metadata['full_key'], metadata['label']))
    key_to_patient = dict(zip(metadata['full_key'], metadata['StudyID']))
    
    # Label mapping
    label_map = {"TB Positive": 1, "TB Negative": 0}
    
    # Process embeddings - FIXED: Use lists and convert at the end
    X_list, y_list, keys_list, patient_ids_list = [], [], [], []
    
    for key in common_keys:
        if key in key_to_label and key_to_label[key] in label_map:
            emb = embeddings_data[key]  # Shape: (n_clips, n_features)
            
            if use_temporal and len(emb.shape) > 1:
                # Extract temporal features
                features = extract_temporal_features(emb)
            else:
                # Simple mean aggregation
                features = np.mean(emb, axis=0)
                # Safety check for mean aggregation too
                features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            
            X_list.append(features)
            y_list.append(label_map[key_to_label[key]])
            keys_list.append(key)
            patient_ids_list.append(key_to_patient[key])
    
    # Convert to numpy arrays - FIXED: Use vstack for 2D arrays
    X = np.vstack(X_list)
    y = np.array(y_list)
    keys = np.array(keys_list)
    patient_ids = np.array(patient_ids_list)
    
    # Final safety check on the entire dataset
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Verify no NaN values remain
    if np.isnan(X).any():
        print("⚠️ Warning: NaN values detected after processing, replacing with 0")
        X = np.nan_to_num(X, nan=0.0)
    
    if np.isinf(X).any():
        print("⚠️ Warning: Infinite values detected after processing, replacing with 0")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"✅ Processed {len(X)} samples with {X.shape[1]} features")
    print(f"📈 TB Positive: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"📉 TB Negative: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print(f"🏥 Unique patients: {len(np.unique(patient_ids))}")
    print(f"🔍 NaN check: {np.isnan(X).any()} | Inf check: {np.isinf(X).any()}")
    
    return X, y, keys, patient_ids

def main():
    # Load the full dataset
    EMBEDDING_PATH = "ucsf_new_embeddings.npz"
    METADATA_PATH = "ucsf_new_embeddings_metadata.csv"
    
    # Load dataset
    X, y, file_keys, patient_ids = load_advanced_embeddings(
        EMBEDDING_PATH, METADATA_PATH, use_temporal=True, max_samples=None
    )
    
    print(f"\\n🎯 Enhanced dataset shape: {X.shape}")
    print(f"🎯 Feature expansion: {X.shape[1]} features (was 512)")
    
    # Additional dataset summary
    print(f"\\n📊 FULL DATASET SUMMARY:")
    print(f"   Total audio files: {len(X)}")
    print(f"   Total patients: {len(np.unique(patient_ids))}")
    print(f"   Average files per patient: {len(X) / len(np.unique(patient_ids)):.1f}")
    print(f"   TB Positive files: {sum(y)} ({sum(y)/len(y)*100:.1f}%)")
    print(f"   TB Negative files: {len(y)-sum(y)} ({(len(y)-sum(y))/len(y)*100:.1f}%)")
    print(f"   Temporal features: {X.shape[1]} (13x expansion from 512)")
    
    # Check for any data quality issues
    print(f"\\n🔍 DATA QUALITY CHECKS:")
    print(f"   NaN values: {np.isnan(X).sum()}")
    print(f"   Infinite values: {np.isinf(X).sum()}")
    print(f"   Zero variance features: {np.sum(np.var(X, axis=0) == 0)}")
    print(f"   Feature range: [{np.min(X):.3f}, {np.max(X):.3f}]")
    
    print("\\n" + "="*60)
    print("🔄 FULL DATASET LOADED SUCCESSFULLY!")
    print("="*60)
    
    # Save results for notebook
    results = {
        'X': X,
        'y': y,
        'file_keys': file_keys,
        'patient_ids': patient_ids,
        'dataset_summary': {
            'total_files': len(X),
            'total_patients': len(np.unique(patient_ids)),
            'avg_files_per_patient': len(X) / len(np.unique(patient_ids)),
            'tb_positive_rate': sum(y) / len(y),
            'feature_count': X.shape[1]
        }
    }
    
    # Save to file for notebook use
    np.savez('full_dataset_processed.npz', **results)
    print("✅ Processed dataset saved to 'full_dataset_processed.npz'")
    
    return results

if __name__ == "__main__":
    main()