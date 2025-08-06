#!/usr/bin/env python3
"""
Configuration file for Leave-One-Out TB Detection Pipeline
Enhanced configuration with test dataset handling and WHO optimization
"""

import os
import json
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime

class LeaveOneOutConfig:
    """Configuration class for leave-one-out validation pipeline"""
    
    def __init__(self):
        # System configuration
        self.n_jobs = multiprocessing.cpu_count()
        self.batch_size = max(1, self.n_jobs // 2)
        
        # Leave-one-out dataset configuration
        self.training_patients_file = "data/training_patients_leave_one_out.csv"
        self.test_patients_file = "data/test_patients_leave_one_out.csv" 
        self.embeddings_file = "data/final_embeddings.npz"
        self.metadata_file = "data/final_embeddings_metadata.csv"
        
        # Training configuration
        self.use_cross_validation = True
        self.n_folds = 5
        self.random_state = 42
        
        # WHO optimization parameters
        self.who_optimization = True
        self.who_sensitivity_target = 0.90
        self.who_specificity_target = 0.70
        self.who_threshold_optimization = True
        self.sensitivity_weight = 8.0  # Favor sensitivity in loss functions
        
        # Audio processing configuration (for reference)
        self.sample_rate = 16000
        self.clip_duration = 2
        self.clip_overlap_percent = 10
        self.silence_threshold_db = -50
        
        # Directory paths
        self.data_dir = "data"
        self.results_dir = "results"
        self.reports_dir = "reports"
        self.models_dir = "models"
        self.configs_dir = "configs"
        
        # Enhanced naming configuration
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_description = "leave_one_out_validation"
        self.dataset_name = "final_embeddings"
        
        # Create directories
        for directory in [self.results_dir, self.configs_dir, self.reports_dir, self.models_dir]:
            os.makedirs(directory, exist_ok=True)
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        if hasattr(args, 'test_patients') and args.test_patients:
            # This will be used to prepare the datasets, but doesn't change internal file names
            pass
            
        if hasattr(args, 'cross_validation') and args.cross_validation is not None:
            self.use_cross_validation = args.cross_validation
            
        if hasattr(args, 'n_folds') and args.n_folds:
            self.n_folds = args.n_folds
            
        if hasattr(args, 'n_jobs') and args.n_jobs:
            self.n_jobs = args.n_jobs if args.n_jobs > 0 else multiprocessing.cpu_count()
            
        if hasattr(args, 'random_state') and args.random_state:
            self.random_state = args.random_state
            
        if hasattr(args, 'run_description') and args.run_description:
            self.run_description = args.run_description
            
        # WHO optimization parameters
        if hasattr(args, 'who_sensitivity_target') and args.who_sensitivity_target:
            self.who_sensitivity_target = args.who_sensitivity_target
            
        if hasattr(args, 'who_specificity_target') and args.who_specificity_target:
            self.who_specificity_target = args.who_specificity_target
    
    def get_config_hash(self):
        """Generate 6-character hash of key configuration parameters"""
        config_dict = {
            'embeddings_file': self.embeddings_file,
            'use_cross_validation': self.use_cross_validation,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'n_jobs': self.n_jobs,
            'who_optimization': self.who_optimization,
            'who_sensitivity_target': self.who_sensitivity_target,
            'who_specificity_target': self.who_specificity_target
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:6]
    
    def get_mode_identifier(self):
        """Get execution mode identifier"""
        if self.use_cross_validation:
            return f"leave_one_out_cv{self.n_folds}fold"
        else:
            return "leave_one_out_single"
    
    def get_run_identifier(self):
        """Generate comprehensive run identifier for leave-one-out validation"""
        mode = self.get_mode_identifier()
        config_hash = self.get_config_hash()
        
        components = [
            self.dataset_name,
            mode,
            self.run_description,
            self.run_timestamp,
            config_hash
        ]
        
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
                'mode': self.get_mode_identifier(),
                'run_type': 'leave_one_out_validation'
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
                'batch_size': self.batch_size
            },
            'who_optimization_config': {
                'who_optimization': self.who_optimization,
                'who_sensitivity_target': self.who_sensitivity_target,
                'who_specificity_target': self.who_specificity_target,
                'who_threshold_optimization': self.who_threshold_optimization,
                'sensitivity_weight': self.sensitivity_weight
            },
            'audio_config': {
                'sample_rate': self.sample_rate,
                'clip_duration': self.clip_duration,
                'clip_overlap_percent': self.clip_overlap_percent,
                'silence_threshold_db': self.silence_threshold_db
            },
            'system_info': {
                'cpu_count': multiprocessing.cpu_count(),
                'device_type': 'hybrid_cpu_gpu',
                'pipeline_type': 'leave_one_out_validation'
            }
        }
        
        config_path = os.path.join(self.configs_dir, self.get_config_filename())
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        return config_path
    
    def load_config_from_file(self, config_path):
        """Load configuration from JSON file"""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Update configuration from loaded data
        if 'training_config' in config_data:
            train_config = config_data['training_config']
            self.use_cross_validation = train_config.get('use_cross_validation', self.use_cross_validation)
            self.n_folds = train_config.get('n_folds', self.n_folds)
            self.random_state = train_config.get('random_state', self.random_state)
            self.n_jobs = train_config.get('n_jobs', self.n_jobs)
        
        if 'who_optimization_config' in config_data:
            who_config = config_data['who_optimization_config']
            self.who_optimization = who_config.get('who_optimization', self.who_optimization)
            self.who_sensitivity_target = who_config.get('who_sensitivity_target', self.who_sensitivity_target)
            self.who_specificity_target = who_config.get('who_specificity_target', self.who_specificity_target)
        
        if 'run_info' in config_data:
            run_info = config_data['run_info']
            self.run_description = run_info.get('run_description', self.run_description)
        
        return config_data
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for dir_name in [self.data_dir, self.results_dir, self.reports_dir, 
                        self.models_dir, self.configs_dir]:
            os.makedirs(dir_name, exist_ok=True)
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 Leave-One-Out Validation Pipeline Configuration")
        print("=" * 60)
        print(f"💻 CPU Cores Available: {multiprocessing.cpu_count()}")
        print(f"⚡ CPU Cores to Use: {self.n_jobs}")
        print(f"📦 Batch Size: {self.batch_size}")
        
        if self.use_cross_validation:
            print(f"🔄 Cross-Validation: {self.n_folds}-fold enabled")
        else:
            print(f"📊 Evaluation: Single train-test split")
            
        print(f"🧪 Test Dataset: {self.test_patients_file}")
        print(f"🗃️  Embeddings File: {self.embeddings_file}")
        print(f"🎯 Random State: {self.random_state}")
        
        if self.who_optimization:
            print(f"🏥 WHO Optimization: Enabled")
            print(f"   📈 Sensitivity Target: ≥{self.who_sensitivity_target*100:.0f}%")
            print(f"   📉 Specificity Target: ≥{self.who_specificity_target*100:.0f}%")
            print(f"   ⚖️  Sensitivity Weight: {self.sensitivity_weight}x")
        else:
            print(f"🏥 WHO Optimization: Disabled")
        
        print(f"🏷️  Run Description: {self.run_description}")
        print()

def get_leave_one_out_parser():
    """Get argument parser for leave-one-out validation"""
    parser = argparse.ArgumentParser(description="Leave-One-Out TB Detection Validation Pipeline")
    
    # Dataset configuration
    parser.add_argument('--test_patients', type=str,
                       default='patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv',
                       help='Path to reserved test patients CSV file')
    
    parser.add_argument('--embeddings_file', type=str, default='data/final_embeddings.npz',
                       help='Path to embeddings NPZ file (default: data/final_embeddings.npz)')
    
    parser.add_argument('--metadata_file', type=str, default='data/final_embeddings_metadata.csv',
                       help='Path to embeddings metadata CSV file')
    
    # Training configuration
    parser.add_argument('--cross_validation', action='store_true',
                       help='Enable cross-validation (default: enabled for leave-one-out)')
    
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of folds for cross-validation (default: 5)')
    
    parser.add_argument('--n_jobs', type=int, default=-1,
                       help='Number of CPU cores to use (-1 for all cores, default: -1)')
    
    parser.add_argument('--random_state', type=int, default=42,
                       help='Random state for reproducibility (default: 42)')
    
    # WHO optimization
    parser.add_argument('--who_sensitivity_target', type=float, default=0.90,
                       help='WHO sensitivity target (default: 0.90)')
    
    parser.add_argument('--who_specificity_target', type=float, default=0.70,
                       help='WHO specificity target (default: 0.70)')
    
    parser.add_argument('--disable_who_optimization', action='store_true',
                       help='Disable WHO optimization features')
    
    # Run configuration
    parser.add_argument('--run_description', type=str, default='leave_one_out_validation',
                       help='Description for this run (default: leave_one_out_validation)')
    
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose output')
    
    return parser

def load_leave_one_out_config(args=None):
    """Load configuration for leave-one-out validation"""
    if args is None:
        parser = get_leave_one_out_parser()
        args = parser.parse_args()
    
    config = LeaveOneOutConfig()
    config.update_from_args(args)
    
    # Handle WHO optimization disable
    if hasattr(args, 'disable_who_optimization') and args.disable_who_optimization:
        config.who_optimization = False
    
    config.ensure_directories()
    
    return config, args

# Backwards compatibility functions
def load_config_from_args_gpu(args=None):
    """GPU configuration loader (backwards compatibility)"""
    return load_leave_one_out_config(args)

def get_common_parser_gpu():
    """GPU parser (backwards compatibility)"""  
    return get_leave_one_out_parser()

if __name__ == "__main__":
    # Test configuration
    config, args = load_leave_one_out_config()
    config.print_config()
    
    # Test config save/load
    config_path = config.save_run_config()
    print(f"Configuration test successful: {config_path}")