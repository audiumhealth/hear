#!/usr/bin/env python3
"""
Configuration file for CPU-optimized UCSF R2D2 TB Detection Pipeline
Handles all configurable parameters and system settings
"""

import os
import json
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime

class Config:
    """Configuration class for the pipeline"""
    
    def __init__(self):
        # System configuration
        self.n_jobs = multiprocessing.cpu_count()  # Use all available cores by default
        self.batch_size = max(1, self.n_jobs // 2)  # Conservative batch size
        
        # Data configuration
        self.embeddings_file = "final_embeddings.npz"
        self.metadata_file = "final_embeddings_metadata.csv"
        self.labels_file = "clean_patients_final.csv"
        
        # Training configuration
        self.use_cross_validation = False  # Cross-validation disabled by default
        self.n_folds = 5  # 5-fold cross-validation
        self.random_state = 42
        
        # Audio processing configuration
        self.sample_rate = 16000
        self.clip_duration = 2
        self.clip_overlap_percent = 10
        self.silence_threshold_db = -50
        
        # Directory paths
        self.data_dir = "data"
        self.results_dir = "results"
        self.reports_dir = "reports"
        self.models_dir = "models"
        
        # Enhanced naming configuration
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_description = ""  # User-provided description
        self.dataset_name = ""  # Will be set from embeddings file
        
        # Create configs directory
        self.configs_dir = "configs"
        os.makedirs(self.configs_dir, exist_ok=True)
        
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'embeddings_file') and args.embeddings_file:
            self.embeddings_file = args.embeddings_file
        if hasattr(args, 'cross_validation') and args.cross_validation:
            self.use_cross_validation = args.cross_validation
        if hasattr(args, 'n_folds') and args.n_folds:
            self.n_folds = args.n_folds
        if hasattr(args, 'n_jobs') and args.n_jobs:
            self.n_jobs = args.n_jobs if args.n_jobs > 0 else multiprocessing.cpu_count()
        if hasattr(args, 'batch_size') and args.batch_size:
            self.batch_size = args.batch_size
        if hasattr(args, 'random_state') and args.random_state:
            self.random_state = args.random_state
            
        # Enhanced naming arguments
        if hasattr(args, 'run_description') and args.run_description:
            self.run_description = args.run_description
        if hasattr(args, 'dataset_name') and args.dataset_name:
            self.dataset_name = args.dataset_name
        else:
            # Extract dataset name from embeddings file
            self.dataset_name = Path(self.embeddings_file).stem
    
    def get_embeddings_path(self):
        """Get full path to embeddings file"""
        return os.path.join(self.data_dir, self.embeddings_file)
    
    def get_metadata_path(self):
        """Get full path to metadata file"""
        metadata_name = self.embeddings_file.replace('.npz', '_metadata.csv')
        return os.path.join(self.data_dir, metadata_name)
    
    def get_labels_path(self):
        """Get full path to labels file"""
        return os.path.join(self.data_dir, self.labels_file)
    
    def get_embeddings_basename(self):
        """Get basename of embeddings file (without extension) for prefixing output files"""
        return Path(self.embeddings_file).stem
    
    def get_config_hash(self):
        """Generate 6-character hash of key configuration parameters"""
        config_dict = {
            'embeddings_file': self.embeddings_file,
            'use_cross_validation': self.use_cross_validation,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:6]
    
    def get_mode_identifier(self):
        """Get execution mode identifier"""
        if self.use_cross_validation:
            return f"cpu_cv{self.n_folds}fold"
        else:
            return "cpu_single"
    
    def get_run_identifier(self):
        """Generate comprehensive run identifier"""
        mode = self.get_mode_identifier()
        config_hash = self.get_config_hash()
        
        components = [
            self.dataset_name,
            mode,
            self.run_timestamp,
            config_hash
        ]
        
        if self.run_description:
            # Insert description before timestamp
            components.insert(2, self.run_description)
            
        return "_".join(components)
    
    def get_output_filename(self, base_filename):
        """Generate comprehensive output filename"""
        run_id = self.get_run_identifier()
        
        # Handle file extension
        if '.' in base_filename:
            return f"{run_id}_{base_filename}"
        else:
            return f"{run_id}_{base_filename}"
    
    def get_config_filename(self):
        """Get configuration filename for this run"""
        run_id = self.get_run_identifier()
        return f"{run_id}_config.json"
    
    def save_run_config(self):
        """Save complete configuration to JSON for reproducibility"""
        config_data = {
            'run_info': {
                'timestamp': self.run_timestamp,
                'dataset_name': self.dataset_name,
                'run_description': self.run_description,
                'config_hash': self.get_config_hash(),
                'mode': self.get_mode_identifier()
            },
            'data_config': {
                'embeddings_file': self.embeddings_file,
                'metadata_file': self.get_metadata_path(),
                'labels_file': self.get_labels_path()
            },
            'training_config': {
                'use_cross_validation': self.use_cross_validation,
                'n_folds': self.n_folds,
                'random_state': self.random_state,
                'n_jobs': self.n_jobs,
                'batch_size': self.batch_size
            },
            'audio_config': {
                'sample_rate': self.sample_rate,
                'clip_duration': self.clip_duration,
                'clip_overlap_percent': self.clip_overlap_percent,
                'silence_threshold_db': self.silence_threshold_db
            },
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'device_type': 'cpu'
            }
        }
        
        config_path = os.path.join(self.configs_dir, self.get_config_filename())
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        return config_path
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_name in [self.data_dir, self.results_dir, self.reports_dir, self.models_dir]:
            os.makedirs(dir_name, exist_ok=True)
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 CPU-Optimized Pipeline Configuration")
        print("=" * 50)
        print(f"💻 CPU Cores Available: {multiprocessing.cpu_count()}")
        print(f"⚡ CPU Cores to Use: {self.n_jobs}")
        print(f"📦 Batch Size: {self.batch_size}")
        if self.use_cross_validation:
            print(f"🔄 Cross-Validation: {self.n_folds}-fold enabled")
        else:
            print(f"📊 Evaluation: Single 80/20 train-test split")
        print(f"🗃️  Embeddings File: {self.embeddings_file}")
        print(f"🎯 Random State: {self.random_state}")
        print()

def get_common_parser():
    """Get common argument parser for all scripts"""
    parser = argparse.ArgumentParser(description="CPU-Optimized UCSF R2D2 TB Detection Pipeline")
    
    parser.add_argument('--embeddings_file', type=str, default='final_embeddings.npz',
                       help='Path to embeddings NPZ file (default: final_embeddings.npz)')
    
    parser.add_argument('--cross_validation', action='store_true',
                       help='Enable 5-fold cross-validation (default: single 80/20 split)')
    
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of CPU cores to use (-1 for all cores, default: -1)')
    
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for processing (default: auto-calculated)')
    
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    parser.add_argument('--run_description', type=str, default='',
                       help='Optional description for this run (used in output filenames)')
    
    parser.add_argument('--dataset_name', type=str, default='',
                       help='Dataset name override (default: derived from embeddings file)')
    
    return parser

def load_config_from_args(args=None):
    """Load configuration from command line arguments"""
    if args is None:
        parser = get_common_parser()
        args = parser.parse_args()
    
    config = Config()
    config.update_from_args(args)
    config.ensure_directories()
    
    return config