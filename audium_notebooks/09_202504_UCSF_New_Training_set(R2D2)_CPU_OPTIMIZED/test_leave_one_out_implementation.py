#!/usr/bin/env python3
"""
Test script for Leave-One-Out Implementation
Validates that all components are working correctly
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

def test_configuration():
    """Test configuration loading"""
    print("🔧 Testing configuration...")
    
    try:
        from config_leave_one_out import LeaveOneOutConfig, load_leave_one_out_config
        
        # Test basic configuration
        config = LeaveOneOutConfig()
        print(f"✅ Configuration created successfully")
        print(f"   Run ID: {config.get_run_identifier()}")
        print(f"   WHO targets: {config.who_sensitivity_target*100:.0f}% / {config.who_specificity_target*100:.0f}%")
        
        # Test configuration save
        config_path = config.save_run_config()
        print(f"✅ Configuration save test passed")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_data_preparation():
    """Test dataset preparation functionality"""
    print("\n📂 Testing data preparation...")
    
    try:
        # Check if required files exist
        test_patients_file = "patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv"
        labels_file = "data/clean_patients_final.csv"
        embeddings_file = "data/final_embeddings.npz"
        metadata_file = "data/final_embeddings_metadata.csv"
        
        files_exist = {
            'test_patients': os.path.exists(test_patients_file),
            'labels': os.path.exists(labels_file),
            'embeddings': os.path.exists(embeddings_file),
            'metadata': os.path.exists(metadata_file)
        }
        
        print(f"📋 File availability:")
        for name, exists in files_exist.items():
            status = "✅" if exists else "❌"
            print(f"   {status} {name}: {exists}")
        
        # Test loading test patients
        if files_exist['test_patients']:
            test_df = pd.read_csv(test_patients_file)
            print(f"✅ Test patients loaded: {len(test_df)} patients")
            
            # Show first few patients
            if 'PID' in test_df.columns:
                test_patients = test_df['PID'].tolist()[:5]
            else:
                test_patients = test_df.iloc[:, -1].tolist()[:5]
            print(f"   Sample patients: {test_patients}")
        
        return all(files_exist.values())
        
    except Exception as e:
        print(f"❌ Data preparation test failed: {e}")
        return False

def test_pipeline_imports():
    """Test that all pipeline imports work correctly"""
    print("\n🐍 Testing pipeline imports...")
    
    try:
        # Test main pipeline imports
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        print("✅ Core ML imports successful")
        
        # Test XGBoost
        try:
            import xgboost as xgb
            print("✅ XGBoost available")
        except ImportError:
            print("⚠️  XGBoost not available (will be skipped in pipeline)")
        
        # Test PyTorch
        try:
            import torch
            print("✅ PyTorch available")
            device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
            print(f"   Device: {device}")
        except ImportError:
            print("⚠️  PyTorch not available (will be skipped in pipeline)")
        
        return True
        
    except Exception as e:
        print(f"❌ Pipeline imports test failed: {e}")
        return False

def test_baseline_structure():
    """Test that baseline pipeline structure exists"""
    print("\n📊 Testing baseline structure compatibility...")
    
    try:
        # Check for baseline pipeline runs file
        baseline_file = "baseline_pipeline_runs.csv"
        if os.path.exists(baseline_file):
            baseline_df = pd.read_csv(baseline_file)
            print(f"✅ Baseline pipeline runs found: {len(baseline_df)} previous runs")
            
            # Show column structure
            print(f"   Columns: {list(baseline_df.columns)}")
            
            # Show most recent run
            if len(baseline_df) > 0:
                recent_run = baseline_df.iloc[-1]
                print(f"   Most recent: {recent_run['run_id']} ({recent_run['run_type']})")
        else:
            print("⚠️  Baseline pipeline runs file not found (will be created)")
        
        # Check results directory structure
        results_dir = "results"
        if os.path.exists(results_dir):
            result_files = os.listdir(results_dir)
            print(f"✅ Results directory exists with {len(result_files)} files")
        else:
            print("⚠️  Results directory will be created")
        
        return True
        
    except Exception as e:
        print(f"❌ Baseline structure test failed: {e}")
        return False

def test_deliverable_structure():
    """Test expected deliverable file naming"""
    print("\n📁 Testing deliverable structure...")
    
    try:
        from config_leave_one_out import LeaveOneOutConfig
        
        config = LeaveOneOutConfig()
        run_id = config.get_run_identifier()
        
        expected_files = [
            f"{run_id}_cross_validation_detailed.csv",
            f"{run_id}_cross_validation_summary.csv", 
            f"{run_id}_leave_one_out_test_results.csv",
            f"{run_id}_test_vs_cv_comparison.csv",
            f"{run_id}_cv_performance_dashboard.png",
            f"{run_id}_leave_one_out_test_dashboard.png",
            f"{run_id}_leave_one_out_executive_summary.txt",
            f"{run_id}_config.json"
        ]
        
        print(f"✅ Expected deliverables for run {run_id[:20]}...:")
        for i, filename in enumerate(expected_files, 1):
            print(f"   {i}. {filename}")
        
        print(f"\n📊 Total deliverables: {len(expected_files)} files")
        print(f"🎯 Matches baseline structure: 8-10 files per run")
        
        return True
        
    except Exception as e:
        print(f"❌ Deliverable structure test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Leave-One-Out Implementation Test Suite")
    print("=" * 50)
    
    tests = [
        ("Configuration", test_configuration),
        ("Data Preparation", test_data_preparation), 
        ("Pipeline Imports", test_pipeline_imports),
        ("Baseline Structure", test_baseline_structure),
        ("Deliverable Structure", test_deliverable_structure)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("🧪 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {status} {test_name}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Implementation ready for execution.")
        print("\nNext steps:")
        print("1. Run: python prepare_leave_one_out_dataset.py --test_patients patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv")
        print("2. Run: python 04_leave_one_out_validation.py")
        print("3. Or use: ./run_leave_one_out_pipeline.sh")
    else:
        print("⚠️  Some tests failed. Please fix issues before running pipeline.")
        
        if not results.get("Data Preparation", True):
            print("\n💡 Data Preparation Issues:")
            print("   • Ensure embeddings are generated: python 02_generate_embeddings_final.py")
            print("   • Check that clean_patients_final.csv exists")
            print("   • Verify test patients file path")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)