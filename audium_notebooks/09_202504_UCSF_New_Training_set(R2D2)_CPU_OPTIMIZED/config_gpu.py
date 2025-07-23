#!/usr/bin/env python3
"""
GPU Configuration for CPU-optimized UCSF R2D2 TB Detection Pipeline
Enhanced with comprehensive output naming and GPU device detection
"""

import os
import json
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
import platform

# Import base config
from config import Config, get_common_parser

class GPUConfig(Config):
    """GPU-enhanced configuration class with comprehensive naming"""
    
    def __init__(self):
        super().__init__()
        
        # GPU-specific configuration
        self.device_type = self.detect_optimal_device()
        self.gpu_memory_fraction = 0.8  # Use 80% of GPU memory
        
        # Set device-specific parameters
        self._update_device_settings()
        
        # Enhanced naming configuration
        self.run_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_description = ""  # User-provided description
        self.dataset_name = ""  # Will be set from embeddings file
        
        # Create configs directory
        self.configs_dir = "configs"
        os.makedirs(self.configs_dir, exist_ok=True)
    
    def detect_optimal_device(self):
        """Detect the best available GPU acceleration device"""
        try:
            # Check for Apple Metal Performance Shaders (M1/M2 Macs)
            import torch
            if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                # Check if XGBoost has GPU support too
                if self._check_xgboost_gpu():
                    return 'mps'  # Full GPU support
                else:
                    return 'mps_pytorch_only'  # PyTorch GPU + XGBoost CPU
        except ImportError:
            pass
        
        try:
            # Check for NVIDIA CUDA
            import torch
            if torch.cuda.is_available():
                if self._check_xgboost_gpu():
                    return 'cuda'  # Full GPU support
                else:
                    return 'cuda_pytorch_only'  # PyTorch GPU + XGBoost CPU
        except ImportError:
            pass
        
        # Check standalone XGBoost GPU support
        if self._check_xgboost_gpu():
            return 'xgboost_gpu'
        
        return 'cpu'
    
    def _check_xgboost_gpu(self):
        """Check if XGBoost was compiled with GPU support"""
        try:
            import xgboost as xgb
            build_info = xgb.build_info()
            has_gpu = build_info.get('USE_CUDA', False) or build_info.get('USE_NCCL', False)
            
            if has_gpu:
                # Test actual GPU functionality
                dtrain = xgb.DMatrix([[1, 2], [3, 4]], label=[0, 1])
                params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
                xgb.train(params, dtrain, num_boost_round=1)
                return True
        except:
            pass
        return False
    
    def _update_device_settings(self):
        """Update device-specific settings based on detected device type"""
        # Update PyTorch device and XGBoost method based on detected device
        if self.device_type in ['mps', 'cuda', 'mps_pytorch_only', 'cuda_pytorch_only']:
            # Extract base device type (mps/cuda) from compound device names
            base_device = self.device_type.split('_')[0]
            self.pytorch_device = base_device
        else:
            self.pytorch_device = 'cpu'
            
        if self.device_type in ['mps', 'cuda', 'xgboost_gpu']:
            self.xgboost_tree_method = 'gpu_hist'
        else:
            self.xgboost_tree_method = 'hist'  # CPU fallback
    
    def get_device_info(self):
        """Get detailed device information"""
        info = {
            'device_type': self.device_type,
            'platform': platform.platform(),
            'cpu_count': multiprocessing.cpu_count()
        }
        
        if self.device_type == 'mps':
            try:
                import torch
                info['mps_available'] = torch.backends.mps.is_available()
                info['device_name'] = 'Apple Silicon GPU (MPS)'
            except ImportError:
                pass
        elif self.device_type == 'cuda':
            try:
                import torch
                info['cuda_version'] = torch.version.cuda
                info['device_name'] = torch.cuda.get_device_name(0)
                info['gpu_memory'] = torch.cuda.get_device_properties(0).total_memory
            except ImportError:
                pass
        elif self.device_type == 'xgboost_gpu':
            info['device_name'] = 'XGBoost GPU Support'
        
        return info
    
    def update_from_args(self, args):
        """Update configuration from command line arguments"""
        super().update_from_args(args)
        
        # GPU-specific arguments
        if hasattr(args, 'device') and args.device:
            if args.device in ['cpu', 'mps', 'cuda', 'auto']:
                if args.device == 'auto':
                    self.device_type = self.detect_optimal_device()
                else:
                    self.device_type = args.device
                    
        if hasattr(args, 'gpu_memory_fraction') and args.gpu_memory_fraction:
            self.gpu_memory_fraction = args.gpu_memory_fraction
            
        if hasattr(args, 'run_description') and args.run_description:
            self.run_description = args.run_description
            
        if hasattr(args, 'dataset_name') and args.dataset_name:
            self.dataset_name = args.dataset_name
        else:
            # Extract dataset name from embeddings file
            self.dataset_name = Path(self.embeddings_file).stem
            
        # Update device-specific settings after changes
        self._update_device_settings()
    
    def get_config_hash(self):
        """Generate 6-character hash of key configuration parameters"""
        config_dict = {
            'embeddings_file': self.embeddings_file,
            'use_cross_validation': self.use_cross_validation,
            'n_folds': self.n_folds,
            'random_state': self.random_state,
            'device_type': self.device_type,
            'n_jobs': self.n_jobs,
            'xgboost_tree_method': self.xgboost_tree_method
        }
        
        config_str = json.dumps(config_dict, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:6]
    
    def get_mode_identifier(self):
        """Get execution mode identifier"""
        if self.device_type == 'cpu':
            device_prefix = 'cpu'
        else:
            device_prefix = 'gpu'
            
        if self.use_cross_validation:
            return f"{device_prefix}_cv{self.n_folds}fold"
        else:
            return f"{device_prefix}_single"
    
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
    
    def get_output_filename(self, file_type):
        """Generate comprehensive output filename"""
        run_id = self.get_run_identifier()
        
        # Handle file extension
        if '.' in file_type:
            return f"{run_id}_{file_type}"
        else:
            return f"{run_id}_{file_type}"
    
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
            'gpu_config': {
                'device_type': self.device_type,
                'pytorch_device': self.pytorch_device,
                'xgboost_tree_method': self.xgboost_tree_method,
                'gpu_memory_fraction': self.gpu_memory_fraction
            },
            'audio_config': {
                'sample_rate': self.sample_rate,
                'clip_duration': self.clip_duration,
                'clip_overlap_percent': self.clip_overlap_percent,
                'silence_threshold_db': self.silence_threshold_db
            },
            'system_info': self.get_device_info()
        }
        
        config_path = os.path.join(self.configs_dir, self.get_config_filename())
        
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"✅ Configuration saved: {config_path}")
        return config_path
    
    def print_config(self):
        """Print current configuration with GPU information"""
        device_info = self.get_device_info()
        
        print("🔧 GPU-Optimized Pipeline Configuration")
        print("=" * 50)
        print(f"🎯 Dataset: {self.dataset_name}")
        print(f"⏰ Timestamp: {self.run_timestamp}")
        if self.run_description:
            print(f"📝 Description: {self.run_description}")
        print(f"🔑 Config Hash: {self.get_config_hash()}")
        print(f"🏃 Mode: {self.get_mode_identifier()}")
        print()
        
        print("💻 System Information")
        print("-" * 20)
        print(f"Platform: {device_info.get('platform', 'Unknown')}")
        print(f"CPU Cores: {device_info['cpu_count']}")
        print(f"CPU Cores to Use: {self.n_jobs}")
        print()
        
        print("🚀 GPU Configuration")
        print("-" * 20)
        print(f"Device Type: {self.device_type.upper()}")
        print(f"Device Name: {device_info.get('device_name', 'Unknown')}")
        if self.device_type != 'cpu':
            print(f"PyTorch Device: {self.pytorch_device}")
            print(f"XGBoost Method: {self.xgboost_tree_method}")
            print(f"GPU Memory Usage: {self.gpu_memory_fraction*100:.0f}%")
        else:
            print("GPU Status: CPU fallback mode")
        print()
        
        print("📊 Training Configuration")
        print("-" * 25)
        if self.use_cross_validation:
            print(f"Cross-Validation: {self.n_folds}-fold enabled")
        else:
            print("Evaluation: Single 80/20 train-test split")
        print(f"Embeddings File: {self.embeddings_file}")
        print(f"Random State: {self.random_state}")
        print(f"Batch Size: {self.batch_size}")
        print()

def get_gpu_parser():
    """Get GPU-enhanced argument parser"""
    parser = get_common_parser()
    
    # Add GPU-specific arguments (avoid duplicates from common parser)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto',
                       help='Device to use for acceleration (default: auto-detect)')
    
    parser.add_argument('--gpu_memory_fraction', type=float, default=0.8,
                       help='Fraction of GPU memory to use (default: 0.8)')
    
    return parser

def load_gpu_config_from_args(args=None):
    """Load GPU configuration from command line arguments"""
    if args is None:
        parser = get_gpu_parser()
        args = parser.parse_args()
    
    config = GPUConfig()
    config.update_from_args(args)
    config.ensure_directories()
    
    return config

if __name__ == "__main__":
    # Test configuration
    config = load_gpu_config_from_args()
    config.print_config()
    config.save_run_config()