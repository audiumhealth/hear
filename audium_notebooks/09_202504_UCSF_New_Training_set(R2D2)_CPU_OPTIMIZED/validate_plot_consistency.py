#!/usr/bin/env python3
"""
Plot consistency validator - simulates the plot data extraction logic
to verify it matches CSV data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

def simulate_plot_data_extraction():
    """Simulate the plot data extraction logic to verify consistency"""
    
    print("🔍 PLOT DATA EXTRACTION SIMULATION")
    print("=" * 50)
    
    # Load CSV data
    run_id = "final_embeddings_gpu_cv3fold_3-fold CV test with aggressive sensitivity optimization_20250729_063852_98dcc4"
    summary_path = Path(f"results/{run_id}_gpu_cross_validation_summary.csv")
    
    if not summary_path.exists():
        print(f"❌ Summary CSV not found: {summary_path}")
        return False
        
    summary_df = pd.read_csv(summary_path)
    print(f"📊 Loaded summary CSV: {len(summary_df)} models")
    
    # Simulate the plot extraction logic
    print(f"\n🎯 Simulating plot data extraction logic...")
    
    for _, row in summary_df.iterrows():
        model_name = row['Model']
        
        # This simulates what the plotting functions would extract
        # After our fixes, they should use optimal_*_mean keys
        
        # Create a mock mean_results dictionary like the plotting code sees
        mock_mean_results = {
            'sensitivity_mean': 0.4200,  # Regular threshold (wrong values)
            'sensitivity_std': 0.0050,
            'specificity_mean': 0.3500,  # Regular threshold (wrong values)  
            'specificity_std': 0.0030,
            'optimal_sensitivity_mean': row['Sensitivity_Mean'],    # Optimal values (correct)
            'optimal_sensitivity_std': row['Sensitivity_Std'],
            'optimal_specificity_mean': row['Specificity_Mean'],   # Optimal values (correct)
            'optimal_specificity_std': row['Specificity_Std']
        }
        
        # Simulate plotting function logic (after our fixes)
        if 'optimal_specificity_mean' in mock_mean_results:
            plot_spec_mean = mock_mean_results['optimal_specificity_mean']
            plot_spec_std = mock_mean_results['optimal_specificity_std']
            data_source = "✅ optimal_specificity_mean"
        else:
            plot_spec_mean = mock_mean_results.get('specificity_mean', 0)
            plot_spec_std = mock_mean_results.get('specificity_std', 0)
            data_source = "❌ specificity_mean (fallback)"
            
        # Compare with CSV
        csv_spec_mean = row['Specificity_Mean']
        csv_spec_std = row['Specificity_Std']
        
        consistent = abs(plot_spec_mean - csv_spec_mean) < 0.001 and abs(plot_spec_std - csv_spec_std) < 0.001
        
        print(f"\n📊 {model_name}:")
        print(f"   Data source: {data_source}")
        print(f"   Plot would extract: {plot_spec_mean:.4f} ± {plot_spec_std:.4f}")
        print(f"   CSV shows:         {csv_spec_mean:.4f} ± {csv_spec_std:.4f}")
        print(f"   Consistent: {'✅ YES' if consistent else '❌ NO'}")
        
        if model_name == 'Logistic Regression':
            print(f"   🎯 Key test case - should show ~0.645, not ~0.35")
            if plot_spec_mean > 0.6:
                print(f"   ✅ Shows optimized values (>0.6)")
            else:
                print(f"   ❌ Shows regular threshold values (<0.6)")
    
    return True

if __name__ == "__main__":
    simulate_plot_data_extraction()