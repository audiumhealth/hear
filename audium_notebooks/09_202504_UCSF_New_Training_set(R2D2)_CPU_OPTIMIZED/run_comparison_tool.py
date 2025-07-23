#!/usr/bin/env python3
"""
Cross-Run Comparison Tool for CPU vs GPU Optimized TB Detection Pipeline
Compares performance metrics, timing, and configuration differences across runs
"""

import os
import json
import glob
import argparse
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_run_configs(config_pattern="configs/*_config.json"):
    """Load all run configurations matching the pattern"""
    config_files = glob.glob(config_pattern)
    configs = []
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                config['config_file'] = config_file
                configs.append(config)
        except Exception as e:
            print(f"⚠️  Could not load {config_file}: {e}")
    
    return configs

def load_run_results(results_pattern="results/*_analysis_results.csv"):
    """Load all analysis results matching the pattern"""
    result_files = glob.glob(results_pattern)
    results = []
    
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            # Extract run identifier from filename
            run_id = os.path.basename(result_file).replace('_analysis_results.csv', '').replace('_cross_validation_summary.csv', '')
            df['run_id'] = run_id
            df['results_file'] = result_file
            results.append(df)
        except Exception as e:
            print(f"⚠️  Could not load {result_file}: {e}")
    
    return results

def extract_run_info(run_id):
    """Extract information from run identifier"""
    parts = run_id.split('_')
    
    info = {
        'run_id': run_id,
        'dataset': parts[0] if len(parts) > 0 else 'unknown',
        'device': 'unknown',
        'mode': 'unknown',
        'timestamp': 'unknown',
        'config_hash': 'unknown'
    }
    
    # Extract device and mode
    for i, part in enumerate(parts):
        if part in ['cpu', 'gpu']:
            info['device'] = part
            if i + 1 < len(parts):
                if parts[i + 1].startswith('cv') and parts[i + 1].endswith('fold'):
                    info['mode'] = parts[i + 1]
                elif parts[i + 1] == 'single':
                    info['mode'] = 'single'
    
    # Extract timestamp and config hash (usually last two parts)
    if len(parts) >= 2:
        if '_' in parts[-2] and len(parts[-1]) == 6:  # timestamp_hash pattern
            info['timestamp'] = parts[-2]
            info['config_hash'] = parts[-1]
        elif len(parts[-1]) == 6:  # just hash
            info['config_hash'] = parts[-1]
    
    return info

def create_performance_comparison(results, output_dir="results"):
    """Create performance comparison visualizations"""
    
    if not results:
        print("No results to compare")
        return
    
    # Combine all results
    combined_df = pd.concat(results, ignore_index=True)
    
    # Extract run information
    run_info = []
    for _, row in combined_df.iterrows():
        info = extract_run_info(row['run_id'])
        run_info.append(info)
    
    run_info_df = pd.DataFrame(run_info)
    combined_df = pd.concat([combined_df, run_info_df], axis=1)
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('CPU vs GPU Performance Comparison', fontsize=16, fontweight='bold')
    
    # Define metrics to compare
    metrics = ['Sensitivity', 'Specificity', 'WHO_Score', 'ROC_AUC', 'F1_Score', 'Training_Time']
    
    for i, metric in enumerate(metrics):
        if i >= 6:
            break
            
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        
        if metric in combined_df.columns:
            # Group by device and model
            if 'device' in combined_df.columns:
                sns.boxplot(data=combined_df, x='device', y=metric, hue='Model', ax=ax)
                ax.set_title(f'{metric} Comparison')
                ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                sns.barplot(data=combined_df, x='Model', y=metric, ax=ax)
                ax.set_title(f'{metric} Comparison')
                ax.tick_params(axis='x', rotation=45)
        else:
            ax.text(0.5, 0.5, f'{metric} not available', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{metric} (Not Available)')
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_path = os.path.join(output_dir, f'cpu_gpu_performance_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✅ Performance comparison saved: {comparison_path}")
    plt.close()
    
    return combined_df

def create_timing_comparison(configs, output_dir="results"):
    """Create timing comparison analysis"""
    
    if not configs:
        print("No configs to compare")
        return
    
    timing_data = []
    
    for config in configs:
        run_info = config.get('run_info', {})
        system_info = config.get('system_info', {})
        
        timing_data.append({
            'run_id': run_info.get('dataset', 'unknown') + '_' + run_info.get('mode', 'unknown'),
            'dataset': run_info.get('dataset', 'unknown'),
            'device': system_info.get('device_type', 'unknown'),
            'mode': run_info.get('mode', 'unknown'),
            'timestamp': run_info.get('timestamp', 'unknown'),
            'cpu_count': system_info.get('cpu_count', 0),
            'gpu_cores': system_info.get('gpu_cores', 0) if 'gpu_cores' in system_info else 0
        })
    
    if timing_data:
        timing_df = pd.DataFrame(timing_data)
        
        # Create timing comparison plot
        plt.figure(figsize=(12, 8))
        
        if 'device' in timing_df.columns and len(timing_df) > 1:
            device_counts = timing_df['device'].value_counts()
            plt.pie(device_counts.values, labels=device_counts.index, autopct='%1.1f%%')
            plt.title('Runs by Device Type')
            
            timing_path = os.path.join(output_dir, f'device_usage_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(timing_path, dpi=300, bbox_inches='tight')
            print(f"✅ Device usage comparison saved: {timing_path}")
        
        plt.close()
        
        return timing_df
    
    return None

def generate_comparison_report(combined_results, timing_data, configs, output_dir="results"):
    """Generate comprehensive comparison report"""
    
    report_lines = [
        "CROSS-RUN COMPARISON ANALYSIS REPORT",
        "=" * 50,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Runs Analyzed: {len(configs)}",
        ""
    ]
    
    if combined_results is not None and len(combined_results) > 0:
        # Device comparison
        if 'device' in combined_results.columns:
            device_counts = combined_results['device'].value_counts()
            report_lines.extend([
                "RUNS BY DEVICE TYPE",
                "-" * 20,
                *[f"{device}: {count} models" for device, count in device_counts.items()],
                ""
            ])
        
        # Performance comparison
        metrics = ['WHO_Score', 'Sensitivity', 'Specificity', 'Training_Time']
        
        for metric in metrics:
            if metric in combined_results.columns:
                report_lines.extend([
                    f"{metric.upper()} COMPARISON",
                    "-" * len(f"{metric.upper()} COMPARISON"),
                ])
                
                if 'device' in combined_results.columns:
                    for device in combined_results['device'].unique():
                        device_data = combined_results[combined_results['device'] == device][metric]
                        if len(device_data) > 0:
                            report_lines.append(
                                f"{device.upper()}: Mean={device_data.mean():.3f}, Std={device_data.std():.3f}, Best={device_data.max():.3f}"
                            )
                
                report_lines.append("")
        
        # Best performing models
        if 'WHO_Score' in combined_results.columns:
            best_models = combined_results.nlargest(5, 'WHO_Score')
            report_lines.extend([
                "TOP 5 MODELS ACROSS ALL RUNS",
                "-" * 30,
            ])
            
            for idx, row in best_models.iterrows():
                device_info = f" ({row['device'].upper()})" if 'device' in row else ""
                report_lines.append(
                    f"• {row['Model']}{device_info}: WHO={row['WHO_Score']:.3f}, "
                    f"Sens={row.get('Sensitivity', 'N/A'):.3f}, "
                    f"Spec={row.get('Specificity', 'N/A'):.3f}"
                )
            
            report_lines.append("")
    
    # Configuration analysis
    if configs:
        report_lines.extend([
            "CONFIGURATION ANALYSIS",
            "-" * 22,
        ])
        
        datasets = set()
        devices = set()
        modes = set()
        
        for config in configs:
            run_info = config.get('run_info', {})
            system_info = config.get('system_info', {})
            
            datasets.add(run_info.get('dataset', 'unknown'))
            devices.add(system_info.get('device_type', 'unknown'))
            modes.add(run_info.get('mode', 'unknown'))
        
        report_lines.extend([
            f"Datasets analyzed: {', '.join(sorted(datasets))}",
            f"Device types used: {', '.join(sorted(devices))}",
            f"Evaluation modes: {', '.join(sorted(modes))}",
            ""
        ])
    
    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(output_dir, f'cross_run_comparison_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt')
    
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"✅ Comparison report saved: {report_path}")
    
    return report_path

def main():
    parser = argparse.ArgumentParser(description='Compare TB Detection Pipeline Runs')
    parser.add_argument('--config_pattern', type=str, default='configs/*_config.json',
                       help='Pattern to match configuration files')
    parser.add_argument('--results_pattern', type=str, default='results/*_results.csv',
                       help='Pattern to match results files')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for comparison reports')
    parser.add_argument('--metric', type=str, default='WHO_Score',
                       help='Primary metric for comparison')
    
    args = parser.parse_args()
    
    print("🔍 CROSS-RUN COMPARISON ANALYSIS")
    print("=" * 40)
    print(f"Config pattern: {args.config_pattern}")
    print(f"Results pattern: {args.results_pattern}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configurations
    print("📂 Loading run configurations...")
    configs = load_run_configs(args.config_pattern)
    print(f"✅ Found {len(configs)} configuration files")
    
    # Load results
    print("📊 Loading analysis results...")
    results = load_run_results(args.results_pattern)
    print(f"✅ Found {len(results)} results files")
    
    if not configs and not results:
        print("❌ No configuration or results files found!")
        print("💡 Make sure you have run the pipeline at least once")
        return
    
    # Create comparisons
    combined_results = None
    if results:
        print("📈 Creating performance comparison...")
        combined_results = create_performance_comparison(results, args.output_dir)
    
    timing_data = None
    if configs:
        print("⏱️  Creating timing comparison...")
        timing_data = create_timing_comparison(configs, args.output_dir)
    
    # Generate comprehensive report
    print("📋 Generating comparison report...")
    report_path = generate_comparison_report(combined_results, timing_data, configs, args.output_dir)
    
    print()
    print("🎉 Cross-run comparison analysis complete!")
    print(f"📊 Check {args.output_dir}/ for comparison outputs")
    
    # Print summary
    if combined_results is not None and len(combined_results) > 0:
        print("\n📈 QUICK SUMMARY:")
        print("-" * 20)
        
        if 'WHO_Score' in combined_results.columns:
            best_overall = combined_results.loc[combined_results['WHO_Score'].idxmax()]
            device_info = f" ({best_overall['device'].upper()})" if 'device' in best_overall else ""
            print(f"🏆 Best Model: {best_overall['Model']}{device_info}")
            print(f"   WHO Score: {best_overall['WHO_Score']:.3f}")
            if 'Sensitivity' in best_overall:
                print(f"   Sensitivity: {best_overall['Sensitivity']:.3f}")
            if 'Specificity' in best_overall:
                print(f"   Specificity: {best_overall['Specificity']:.3f}")
        
        if 'device' in combined_results.columns:
            device_performance = combined_results.groupby('device')['WHO_Score'].agg(['mean', 'std', 'count'])
            print("\n📊 Device Performance:")
            for device, stats in device_performance.iterrows():
                print(f"   {device.upper()}: μ={stats['mean']:.3f} σ={stats['std']:.3f} (n={stats['count']})")

if __name__ == "__main__":
    main()