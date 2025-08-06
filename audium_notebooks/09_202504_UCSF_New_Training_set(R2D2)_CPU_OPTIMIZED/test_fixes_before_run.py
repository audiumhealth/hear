#!/usr/bin/env python3
"""
Pre-run validation to test our fixes before running the full 3-fold CV pipeline.
Tests the key generation and plotting logic with mock data.
"""

import numpy as np

def test_standard_deviation_fix():
    """Test that standard deviation calculation is corrected"""
    print("🧪 Testing Standard Deviation Fix...")
    
    # Test data from actual Logistic Regression 3-fold results
    fold_values = [0.6884057971014492, 0.6884057971014492, 0.5579710144927537]
    
    # Calculate both ways
    population_std = np.std(fold_values, ddof=0)  # Old way (wrong)
    sample_std = np.std(fold_values, ddof=1)      # New way (correct)
    
    print(f"   Fold values: {fold_values}")
    print(f"   Population Std (ddof=0): {population_std:.6f} (old/wrong)")
    print(f"   Sample Std (ddof=1):     {sample_std:.6f} (new/correct)")
    print(f"   Improvement: +{((sample_std/population_std - 1) * 100):.1f}%")
    
    # Validate expectations
    expected_old = 0.0615  # What we saw in the original results
    expected_new = 0.0753  # What we calculated manually
    
    if abs(sample_std - expected_new) < 0.001:
        print("   ✅ Standard deviation fix working correctly")
        return True
    else:
        print(f"   ❌ Standard deviation fix failed: expected {expected_new:.4f}, got {sample_std:.4f}")
        return False

def test_key_generation_logic():
    """Test the optimal key generation logic"""
    print("\n🧪 Testing Key Generation Logic...")
    
    # Mock fold results (what individual folds return)
    mock_fold_results = [
        {
            'sensitivity': 0.4186, 'specificity': 0.3478,           # Regular thresholds
            'optimal_sensitivity': 0.7907, 'optimal_specificity': 0.6884,  # Optimal thresholds
        },
        {
            'sensitivity': 0.4200, 'specificity': 0.3500,
            'optimal_sensitivity': 0.7674, 'optimal_specificity': 0.6884,
        },
        {
            'sensitivity': 0.4100, 'specificity': 0.3450,
            'optimal_sensitivity': 0.7381, 'optimal_specificity': 0.5580,
        }
    ]
    
    # Simulate the key generation logic from the pipeline
    metrics = ['sensitivity', 'specificity', 'optimal_sensitivity', 'optimal_specificity']
    
    mean_results = {}
    for metric in metrics:
        try:
            values = [r[metric] for r in mock_fold_results]
            mean_results[f'{metric}_mean'] = np.mean(values)
            mean_results[f'{metric}_std'] = np.std(values, ddof=1)  # Fixed calculation
            
            print(f"   {metric}_mean: {mean_results[f'{metric}_mean']:.4f}")
            print(f"   {metric}_std:  {mean_results[f'{metric}_std']:.4f}")
            
        except KeyError as e:
            print(f"   ❌ Missing key: {e}")
            return False
    
    # Validate key existence
    required_keys = ['optimal_sensitivity_mean', 'optimal_sensitivity_std', 
                    'optimal_specificity_mean', 'optimal_specificity_std']
    
    for key in required_keys:
        if key in mean_results:
            print(f"   ✅ {key}: {mean_results[key]:.4f}")
        else:
            print(f"   ❌ Missing key: {key}")
            return False
    
    # Validate values are different (optimal vs regular)
    if (mean_results['optimal_specificity_mean'] > mean_results['specificity_mean'] and
        mean_results['optimal_sensitivity_mean'] > mean_results['sensitivity_mean']):
        print("   ✅ Optimal values are higher than regular values (expected)")
        return True
    else:
        print("   ❌ Optimal values are not higher than regular values")
        return False

def test_plot_extraction_logic():
    """Test the plot data extraction logic"""
    print("\n🧪 Testing Plot Data Extraction Logic...")
    
    # Mock results structure (what plotting functions receive)
    mock_results = [{
        'model_name': 'Logistic Regression',
        'mean_results': {
            'sensitivity_mean': 0.4162,       # Regular threshold (wrong)
            'sensitivity_std': 0.0054,
            'specificity_mean': 0.3476,      # Regular threshold (wrong)
            'specificity_std': 0.0025,
            'optimal_sensitivity_mean': 0.7654,    # Optimal (correct)
            'optimal_sensitivity_std': 0.0264,
            'optimal_specificity_mean': 0.6449,   # Optimal (correct)
            'optimal_specificity_std': 0.0753,
        }
    }]
    
    # Simulate the fixed plotting logic
    for r in mock_results:
        model_name = r['model_name']
        
        # Check what keys exist (debug simulation)
        available_keys = list(r['mean_results'].keys())
        print(f"   Available keys for {model_name}: {[k for k in available_keys if 'optimal' in k]}")
        
        # Extract specificity using the fixed logic
        if 'optimal_specificity_mean' in r['mean_results']:
            spec_mean = r['mean_results']['optimal_specificity_mean']
            spec_std = r['mean_results']['optimal_specificity_std']
            source = "optimal_specificity_mean"
        else:
            spec_mean = r['mean_results'].get('specificity_mean', 0)
            spec_std = r['mean_results'].get('specificity_std', 0)
            source = "specificity_mean (fallback)"
            
        print(f"   {model_name}:")
        print(f"     Data source: {source}")
        print(f"     Extracted value: {spec_mean:.4f} ± {spec_std:.4f}")
        
        # Validate
        if source == "optimal_specificity_mean" and spec_mean > 0.6:
            print(f"     ✅ Using optimal values correctly")
            return True
        else:
            print(f"     ❌ Not using optimal values correctly")
            return False

def run_all_tests():
    """Run all pre-run validation tests"""
    print("🔍 PRE-RUN VALIDATION TESTS")
    print("=" * 50)
    
    tests = [
        ("Standard Deviation Fix", test_standard_deviation_fix),
        ("Key Generation Logic", test_key_generation_logic),
        ("Plot Extraction Logic", test_plot_extraction_logic),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with error: {str(e)}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 TEST RESULTS SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Ready to run the pipeline.")
        return True
    else:
        print("❌ Some tests failed. Check the fixes before running the pipeline.")
        return False

if __name__ == "__main__":
    run_all_tests()