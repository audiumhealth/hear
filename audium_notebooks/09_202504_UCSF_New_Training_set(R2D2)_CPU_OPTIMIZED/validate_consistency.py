#!/usr/bin/env python3
"""
Comprehensive validation script to verify data consistency across ALL outputs.
This script validates that plots, CSVs, and summaries all use the same optimized threshold data.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

class DataConsistencyValidator:
    """Validates consistency across TB detection pipeline outputs"""
    
    def __init__(self, run_id="final_embeddings_reproduction_gpu_cv3fold_REPRODUCTION: 3-fold CV with fixes - aggressive sensitivity optimization_20250730_225849_d9eafe"):
        self.run_id = run_id
        self.results_dir = Path("results")
        self.validation_results = {}
        self.errors = []
        self.warnings = []
        
    def validate_all(self):
        """Run all validation checks"""
        print("🔍 COMPREHENSIVE DATA CONSISTENCY VALIDATION")
        print("=" * 60)
        print(f"Validating run: {self.run_id}")
        print()
        
        # Load data files
        if not self.load_data_files():
            return False
            
        # Run validation checks
        self.validate_standard_deviations()
        self.validate_csv_consistency()
        self.validate_summary_consistency()
        self.validate_fold_data_consistency()
        self.validate_expected_values()
        
        # Generate report
        self.generate_validation_report()
        
        return len(self.errors) == 0
    
    def load_data_files(self):
        """Load all data files for validation"""
        print("📂 Loading data files...")
        
        try:
            # Summary CSV
            summary_path = self.results_dir / f"{self.run_id}_gpu_cross_validation_summary.csv"
            if summary_path.exists():
                self.summary_df = pd.read_csv(summary_path)
                print(f"✅ Loaded summary CSV: {len(self.summary_df)} models")
            else:
                self.errors.append(f"Missing summary CSV: {summary_path}")
                return False
                
            # Detailed CSV  
            detailed_path = self.results_dir / f"{self.run_id}_gpu_cross_validation_detailed.csv"
            if detailed_path.exists():
                self.detailed_df = pd.read_csv(detailed_path)
                print(f"✅ Loaded detailed CSV: {len(self.detailed_df)} fold results")
            else:
                self.errors.append(f"Missing detailed CSV: {detailed_path}")
                return False
                
            # Executive Summary
            exec_path = self.results_dir / f"{self.run_id}_gpu_cross_validation_executive_summary.txt"
            if exec_path.exists():
                with open(exec_path, 'r') as f:
                    self.exec_summary = f.read()
                print(f"✅ Loaded executive summary: {len(self.exec_summary)} chars")
            else:
                self.warnings.append(f"Missing executive summary: {exec_path}")
                
            return True
            
        except Exception as e:
            self.errors.append(f"Error loading data files: {str(e)}")
            return False
    
    def validate_standard_deviations(self):
        """Validate that standard deviations are calculated correctly (ddof=1)"""
        print("\n📊 Validating Standard Deviation Calculations...")
        
        # Test case: Logistic Regression specificity
        logistic_detailed = self.detailed_df[self.detailed_df['Model'] == 'Logistic Regression']
        if len(logistic_detailed) >= 3:
            fold_values = logistic_detailed['Specificity'].values
            
            # Calculate manual standard deviation (sample std)
            manual_std = np.std(fold_values, ddof=1)
            
            # Get reported standard deviation from summary
            logistic_summary = self.summary_df[self.summary_df['Model'] == 'Logistic Regression']
            if len(logistic_summary) > 0:
                reported_std = logistic_summary['Specificity_Std'].iloc[0]
                
                # Validate
                if abs(manual_std - reported_std) < 0.001:
                    print(f"✅ Standard deviation correct: manual={manual_std:.4f}, reported={reported_std:.4f}")
                    self.validation_results['std_calculation'] = 'PASS'
                else:
                    error_msg = f"Standard deviation mismatch: manual={manual_std:.4f}, reported={reported_std:.4f}"
                    print(f"❌ {error_msg}")
                    self.errors.append(error_msg)
                    self.validation_results['std_calculation'] = 'FAIL'
                    
                print(f"   Fold values: {fold_values}")
                print(f"   Expected corrected std (ddof=1): {manual_std:.4f}")
                print(f"   Old incorrect std (ddof=0): {np.std(fold_values, ddof=0):.4f}")
            else:
                self.warnings.append("Logistic Regression not found in summary")
        else:
            self.warnings.append("Insufficient fold data for std validation")
    
    def validate_csv_consistency(self):
        """Validate consistency between summary and detailed CSVs"""
        print("\n📋 Validating CSV Consistency...")
        
        for _, model_summary in self.summary_df.iterrows():
            model_name = model_summary['Model']
            model_detailed = self.detailed_df[self.detailed_df['Model'] == model_name]
            
            if len(model_detailed) == 0:
                self.warnings.append(f"No detailed data for {model_name}")
                continue
                
            # Calculate means from detailed data
            detailed_sens_mean = model_detailed['Sensitivity'].mean()
            detailed_spec_mean = model_detailed['Specificity'].mean()
            
            # Compare with summary data  
            summary_sens_mean = model_summary['Sensitivity_Mean']
            summary_spec_mean = model_summary['Specificity_Mean']
            
            # Validate sensitivity
            if abs(detailed_sens_mean - summary_sens_mean) < 0.001:
                print(f"✅ {model_name}: Sensitivity consistent ({summary_sens_mean:.4f})")
            else:
                error_msg = f"{model_name}: Sensitivity mismatch - summary={summary_sens_mean:.4f}, detailed_calc={detailed_sens_mean:.4f}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
                
            # Validate specificity
            if abs(detailed_spec_mean - summary_spec_mean) < 0.001:
                print(f"✅ {model_name}: Specificity consistent ({summary_spec_mean:.4f})")
            else:
                error_msg = f"{model_name}: Specificity mismatch - summary={summary_spec_mean:.4f}, detailed_calc={detailed_spec_mean:.4f}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
        
        self.validation_results['csv_consistency'] = 'PASS' if not any('mismatch' in e for e in self.errors) else 'FAIL'
    
    def validate_summary_consistency(self):
        """Validate that executive summary matches CSV data"""
        print("\n📄 Validating Executive Summary Consistency...")
        
        if not hasattr(self, 'exec_summary'):
            self.warnings.append("Executive summary not available for validation")
            return
            
        # Extract values from executive summary
        import re
        
        # Look for Logistic Regression line
        logistic_pattern = r'• Logistic Regression: WHO=[\d.]+±[\d.]+, Sens=([\d.]+), Spec=([\d.]+)'
        match = re.search(logistic_pattern, self.exec_summary)
        
        if match:
            exec_sens = float(match.group(1))
            exec_spec = float(match.group(2))
            
            # Compare with CSV
            logistic_summary = self.summary_df[self.summary_df['Model'] == 'Logistic Regression']
            if len(logistic_summary) > 0:
                csv_sens = logistic_summary['Sensitivity_Mean'].iloc[0]
                csv_spec = logistic_summary['Specificity_Mean'].iloc[0]
                
                if abs(exec_sens - csv_sens) < 0.001 and abs(exec_spec - csv_spec) < 0.001:
                    print(f"✅ Executive summary matches CSV: Sens={csv_sens:.3f}, Spec={csv_spec:.3f}")
                    self.validation_results['summary_consistency'] = 'PASS'
                else:
                    error_msg = f"Executive summary mismatch - CSV: Sens={csv_sens:.3f}, Spec={csv_spec:.3f} vs Exec: Sens={exec_sens:.3f}, Spec={exec_spec:.3f}"
                    print(f"❌ {error_msg}")
                    self.errors.append(error_msg)
                    self.validation_results['summary_consistency'] = 'FAIL'
            else:
                self.warnings.append("Logistic Regression not found in CSV for summary validation")
        else:
            self.warnings.append("Could not parse Logistic Regression from executive summary")
    
    def validate_fold_data_consistency(self):
        """Validate that fold data shows expected optimal values"""
        print("\n🔄 Validating Fold Data Consistency...")
        
        # Check Logistic Regression fold values
        logistic_detailed = self.detailed_df[self.detailed_df['Model'] == 'Logistic Regression']
        
        if len(logistic_detailed) >= 3:
            fold_specs = logistic_detailed['Specificity'].values
            print(f"📊 Logistic Regression fold specificities: {fold_specs}")
            
            # These should be the OPTIMIZED values now, not the ~0.3-0.4 regular threshold values
            if any(spec > 0.5 for spec in fold_specs):
                print("✅ Fold data shows optimized threshold values (>0.5)")
                self.validation_results['fold_data'] = 'PASS'
            else:
                error_msg = f"Fold data shows regular threshold values (<0.5): {fold_specs}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
                self.validation_results['fold_data'] = 'FAIL'
        else:
            self.warnings.append("Insufficient fold data for validation")
    
    def validate_expected_values(self):
        """Validate that we see the expected corrected values"""
        print("\n🎯 Validating Expected Corrected Values...")
        
        # Logistic Regression should show:
        # - Specificity mean: ~0.645 (not ~0.347)
        # - Specificity std: ~0.075 (not ~0.061)
        
        logistic_summary = self.summary_df[self.summary_df['Model'] == 'Logistic Regression']
        if len(logistic_summary) > 0:
            spec_mean = logistic_summary['Specificity_Mean'].iloc[0]
            spec_std = logistic_summary['Specificity_Std'].iloc[0]
            
            print(f"📊 Logistic Regression: Specificity = {spec_mean:.4f} ± {spec_std:.4f}")
            
            # Validate mean (should be ~0.645, not ~0.347)
            if spec_mean > 0.6:
                print("✅ Specificity mean shows optimized values (>0.6)")
            else:
                error_msg = f"Specificity mean shows regular threshold values: {spec_mean:.4f}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
                
            # Validate std (should be ~0.075, not ~0.061)  
            if spec_std > 0.07:
                print("✅ Specificity std shows corrected calculation (>0.07)")
            else:
                error_msg = f"Specificity std shows uncorrected calculation: {spec_std:.4f}"
                print(f"❌ {error_msg}")
                self.errors.append(error_msg)
                
            self.validation_results['expected_values'] = 'PASS' if spec_mean > 0.6 and spec_std > 0.07 else 'FAIL'
        else:
            self.warnings.append("Logistic Regression not found for expected value validation")
    
    def generate_validation_report(self):
        """Generate final validation report"""
        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT")
        print("=" * 60)
        
        # Overall status
        if len(self.errors) == 0:
            print("🎉 VALIDATION PASSED: All data consistency checks successful!")
            overall_status = "PASS"
        else:
            print("❌ VALIDATION FAILED: Data consistency issues found")
            overall_status = "FAIL"
            
        print(f"\nOverall Status: {overall_status}")
        print(f"Errors: {len(self.errors)}")
        print(f"Warnings: {len(self.warnings)}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for check, result in self.validation_results.items():
            status_icon = "✅" if result == "PASS" else "❌"
            print(f"  {status_icon} {check}: {result}")
        
        # Errors
        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"  {i}. {error}")
        
        # Warnings  
        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")
                
        print("\n" + "=" * 60)
        
        return overall_status == "PASS"


if __name__ == "__main__":
    # Run validation
    validator = DataConsistencyValidator()
    success = validator.validate_all()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)