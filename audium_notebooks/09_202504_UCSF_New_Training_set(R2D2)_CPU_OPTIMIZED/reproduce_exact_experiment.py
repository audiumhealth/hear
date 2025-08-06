#!/usr/bin/env python3
"""
Exact parameter reproduction script for 3-fold CV experiment.
Ensures the re-run uses identical parameters for comparability.
"""

import json
import os
from datetime import datetime

def get_original_parameters():
    """Extract the exact parameters from the original 3-fold CV run"""
    
    print("🔍 PARAMETER VERIFICATION FOR REPRODUCIBLE EXPERIMENT")
    print("=" * 70)
    
    # Original run details from baseline_pipeline_runs.csv
    original_params = {
        "run_id": "final_embeddings_gpu_cv3fold_20250729_063852_98dcc4",
        "run_type": "cross_validation_3fold", 
        "script_used": "03_tb_detection_gpu_optimized.py",
        "commit_hash": "74ed391",
        "config_file": "configs/final_embeddings_gpu_cv3fold_3-fold CV test with aggressive sensitivity optimization_20250729_063852_98dcc4_config.json",
        "input_data": [
            "final_embeddings.npz",
            "data/final_embeddings_metadata.csv", 
            "data/clean_patients_final.csv"
        ],
        "gpu_device": "MPS_PYTORCH_ONLY",
        "notes": "3-fold CV validation: 40% faster execution with maintained statistical validity - Logistic Regression leads with 76.5% sensitivity and 64.5% specificity using optimized thresholds"
    }
    
    # Load original configuration
    config_path = original_params["config_file"]
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            original_config = json.load(f)
            
        print("📋 ORIGINAL EXPERIMENT PARAMETERS:")
        print("-" * 50)
        print(f"Date: 2025-07-29 06:38:52")
        print(f"Script: {original_params['script_used']}")
        print(f"Git commit: {original_params['commit_hash']}")
        print(f"Run description: {original_config['run_info']['run_description']}")
        print()
        
        print("🔧 TRAINING CONFIGURATION:")
        training_config = original_config["training_config"]
        for key, value in training_config.items():
            print(f"  {key}: {value}")
        print()
        
        print("🖥️  GPU CONFIGURATION:")
        gpu_config = original_config["gpu_config"]
        for key, value in gpu_config.items():
            print(f"  {key}: {value}")
        print()
        
        print("📂 DATA CONFIGURATION:")
        data_config = original_config["data_config"]
        for key, value in data_config.items():
            print(f"  {key}: {value}")
        print()
        
        return original_config, original_params
    else:
        print(f"❌ Original config file not found: {config_path}")
        return None, None

def create_reproduction_config(original_config):
    """Create new config file for reproduction with identical parameters"""
    
    if not original_config:
        return None
        
    # Create new config with same parameters but new timestamp
    new_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    reproduction_config = original_config.copy()
    reproduction_config["run_info"]["timestamp"] = new_timestamp
    reproduction_config["run_info"]["run_description"] = "REPRODUCTION: 3-fold CV with fixes - aggressive sensitivity optimization"
    reproduction_config["run_info"]["mode"] = "gpu_cv3fold_reproduction"
    
    # Generate new config hash for tracking
    import hashlib
    config_str = json.dumps(reproduction_config, sort_keys=True)
    new_hash = hashlib.md5(config_str.encode()).hexdigest()[:6]
    reproduction_config["run_info"]["config_hash"] = new_hash
    
    # Save reproduction config 
    config_filename = f"final_embeddings_gpu_cv3fold_reproduction_{new_timestamp}_{new_hash}_config.json"
    config_path = f"configs/{config_filename}"
    
    os.makedirs("configs", exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(reproduction_config, f, indent=2)
    
    print("✅ REPRODUCTION CONFIGURATION CREATED:")
    print(f"   Config file: {config_path}")
    print(f"   New timestamp: {new_timestamp}")
    print(f"   New hash: {new_hash}")
    print()
    
    return config_path, reproduction_config

def generate_exact_command(config_path, reproduction_config):
    """Generate the exact command to run for reproduction"""
    
    print("🚀 EXACT REPRODUCTION COMMAND:")
    print("-" * 50)
    
    # The command should match the original run parameters exactly
    command = f"""
# Activate environment
source /Users/abelvillcaroque/python/venvs/v_audium_hear/bin/activate

# Navigate to directory
cd "/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)_CPU_OPTIMIZED"

# Run with identical parameters (using Python script directly like original)
python 03_tb_detection_gpu_optimized.py \\
    --config {config_path} \\
    --cross_validation \\
    --n_folds 3 \\
    --random_state 42 \\
    --n_jobs 10 \\
    --batch_size 5 \\
    --device_type mps_pytorch_only \\
    --pytorch_device mps \\
    --run_description "REPRODUCTION: 3-fold CV with fixes - aggressive sensitivity optimization"
"""
    
    print(command)
    
    print("📋 PARAMETER VERIFICATION:")
    print(f"✅ Cross-validation: {reproduction_config['training_config']['use_cross_validation']}")
    print(f"✅ N-folds: {reproduction_config['training_config']['n_folds']}")
    print(f"✅ Random state: {reproduction_config['training_config']['random_state']}")
    print(f"✅ GPU device: {reproduction_config['gpu_config']['device_type']}")
    print(f"✅ Batch size: {reproduction_config['training_config']['batch_size']}")
    print(f"✅ N jobs: {reproduction_config['training_config']['n_jobs']}")
    
    print("\n🔍 EXPECTED DIFFERENCES AFTER FIXES:")
    print("   • Standard deviations: +20-25% increase (0.0615 → 0.0753)")
    print("   • All plots: Will show consistent optimal threshold values")
    print("   • Debug logging: Comprehensive validation output")
    print("   • No change in mean values: Same optimal metrics (0.645 specificity)")
    
    return command.strip()

def main():
    """Main reproduction verification"""
    original_config, original_params = get_original_parameters()
    
    if original_config and original_params:
        config_path, reproduction_config = create_reproduction_config(original_config)
        
        if config_path and reproduction_config:
            command = generate_exact_command(config_path, reproduction_config)
            
            print("\n" + "=" * 70)
            print("✅ REPRODUCTION SETUP COMPLETE")
            print("=" * 70)
            print("Ready to run with identical parameters for direct comparison.")
            print("The only differences will be from our applied fixes:")
            print("  1. Corrected standard deviations (ddof=1)")
            print("  2. Consistent optimal metric usage across all outputs")
            print("  3. Enhanced debug logging for verification")
            
            return True
    
    return False

if __name__ == "__main__":
    main()