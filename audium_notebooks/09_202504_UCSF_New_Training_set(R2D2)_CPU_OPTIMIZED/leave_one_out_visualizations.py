#!/usr/bin/env python3
"""
Enhanced Visualization Module for Leave-One-Out TB Detection Pipeline
Comprehensive ROC, PRC, and WHO analysis with confidence bands and data export
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy.interpolate import interp1d
import numpy as np
import os

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

def aggregate_cv_roc_curves(cv_results):
    """Aggregate ROC curves across CV folds with confidence bands"""
    
    all_fpr = []
    all_tpr = []
    
    # Collect all curves
    for curve_data in cv_results['curve_data']['roc_curves']:
        all_fpr.append(curve_data['fpr'])
        all_tpr.append(curve_data['tpr'])
    
    # Create common FPR grid
    mean_fpr = np.linspace(0, 1, 100)
    
    # Interpolate all TPR curves to common FPR grid
    interpolated_tpr = []
    for i, (fpr, tpr) in enumerate(zip(all_fpr, all_tpr)):
        interpolated_tpr.append(np.interp(mean_fpr, fpr, tpr))
        interpolated_tpr[i][0] = 0.0  # Ensure starts at 0
    
    # Calculate mean and std
    tpr_array = np.array(interpolated_tpr)
    mean_tpr = np.mean(tpr_array, axis=0)
    std_tpr = np.std(tpr_array, axis=0, ddof=1)
    
    # Calculate AUC stats
    aucs = []
    for fpr, tpr in zip(all_fpr, all_tpr):
        aucs.append(auc(fpr, tpr))
    
    return {
        'mean_fpr': mean_fpr,
        'mean_tpr': mean_tpr,
        'std_tpr': std_tpr,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs, ddof=1)
    }

def aggregate_cv_prc_curves(cv_results):
    """Aggregate PRC curves across CV folds with confidence bands"""
    
    all_recall = []
    all_precision = []
    
    # Collect all curves
    for curve_data in cv_results['curve_data']['prc_curves']:
        all_recall.append(curve_data['recall'])
        all_precision.append(curve_data['precision'])
    
    # Create common recall grid
    mean_recall = np.linspace(0, 1, 100)
    
    # Interpolate all precision curves to common recall grid
    interpolated_precision = []
    for i, (recall, precision) in enumerate(zip(all_recall, all_precision)):
        # Reverse for interpolation (recall should be increasing)
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]
        interpolated_precision.append(np.interp(mean_recall, recall_rev, precision_rev))
    
    # Calculate mean and std
    precision_array = np.array(interpolated_precision)
    mean_precision = np.mean(precision_array, axis=0)
    std_precision = np.std(precision_array, axis=0, ddof=1)
    
    # Calculate AP (Average Precision) stats
    aps = []
    for recall, precision in zip(all_recall, all_precision):
        aps.append(auc(recall, precision))
    
    return {
        'mean_recall': mean_recall,
        'mean_precision': mean_precision,
        'std_precision': std_precision,
        'mean_ap': np.mean(aps),
        'std_ap': np.std(aps, ddof=1)
    }

def create_comprehensive_roc_analysis(cv_results, test_results, config):
    """Create comprehensive ROC analysis with confidence bands"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: CV ROC Curves with Confidence Bands (Top-left)
    ax1 = axes[0, 0]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(cv_results)))
    
    for i, (model_name, model_results) in enumerate(cv_results.items()):
        roc_stats = aggregate_cv_roc_curves(model_results)
        
        color = colors[i]
        
        # Plot mean curve
        ax1.plot(roc_stats['mean_fpr'], roc_stats['mean_tpr'], 
                color=color, linewidth=2,
                label=f'{model_name} (AUC: {roc_stats["mean_auc"]:.3f}±{roc_stats["std_auc"]:.3f})')
        
        # Add confidence band
        ax1.fill_between(roc_stats['mean_fpr'],
                        roc_stats['mean_tpr'] - roc_stats['std_tpr'],
                        roc_stats['mean_tpr'] + roc_stats['std_tpr'],
                        color=color, alpha=0.2)
    
    # Diagonal reference line
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax1.set_title('Cross-Validation ROC Curves\n(5-fold, 481 patients, with confidence bands)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Leave-One-Out Test ROC Curves (Top-right)
    ax2 = axes[0, 1]
    
    for i, (model_name, model_results) in enumerate(test_results.items()):
        roc_data = model_results['curve_data']['roc_curve']
        test_auc = auc(roc_data['fpr'], roc_data['tpr'])
        
        ax2.plot(roc_data['fpr'], roc_data['tpr'],
                linewidth=2, linestyle='--', color=colors[i],
                label=f'{model_name} (AUC: {test_auc:.3f})')
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')
    
    ax2.set_title('Leave-One-Out Test ROC Curves\n(Independent validation, 61 patients)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: CV vs Test ROC Comparison (Bottom-left)
    ax3 = axes[1, 0]
    
    for i, model_name in enumerate(cv_results.keys()):
        cv_roc_stats = aggregate_cv_roc_curves(cv_results[model_name])
        test_roc_data = test_results[model_name]['curve_data']['roc_curve']
        test_auc = auc(test_roc_data['fpr'], test_roc_data['tpr'])
        
        color = colors[i]
        
        # CV curve with confidence band
        ax3.plot(cv_roc_stats['mean_fpr'], cv_roc_stats['mean_tpr'], 
                color=color, linewidth=1, alpha=0.7,
                label=f'{model_name} CV (AUC: {cv_roc_stats["mean_auc"]:.3f})')
        ax3.fill_between(cv_roc_stats['mean_fpr'],
                        cv_roc_stats['mean_tpr'] - cv_roc_stats['std_tpr'],
                        cv_roc_stats['mean_tpr'] + cv_roc_stats['std_tpr'],
                        color=color, alpha=0.1)
        
        # Test curve (clean line)
        ax3.plot(test_roc_data['fpr'], test_roc_data['tpr'],
                color=color, linewidth=3, linestyle='--',
                label=f'{model_name} Test (AUC: {test_auc:.3f})')
    
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_title('CV vs Test ROC Comparison\n(Generalization Analysis)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Best Model Focus (Bottom-right)
    ax4 = axes[1, 1]
    
    # Find best CV and test models
    best_cv_model = max(cv_results.keys(), 
                       key=lambda m: cv_results[m]['mean_results']['who_score_mean'])
    best_test_model = max(test_results.keys(),
                         key=lambda m: test_results[m]['test_who_score'])
    
    # Plot best CV model
    best_cv_roc = aggregate_cv_roc_curves(cv_results[best_cv_model])
    ax4.plot(best_cv_roc['mean_fpr'], best_cv_roc['mean_tpr'],
            'b-', linewidth=2, label=f'Best CV: {best_cv_model}')
    ax4.fill_between(best_cv_roc['mean_fpr'],
                    best_cv_roc['mean_tpr'] - best_cv_roc['std_tpr'],
                    best_cv_roc['mean_tpr'] + best_cv_roc['std_tpr'],
                    color='blue', alpha=0.2)
    
    # Plot best test model
    best_test_roc = test_results[best_test_model]['curve_data']['roc_curve']
    ax4.plot(best_test_roc['fpr'], best_test_roc['tpr'],
            'r--', linewidth=3, label=f'Best Test: {best_test_model}')
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax4.set_title('Best Model ROC Analysis\n(CV vs Test Champions)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('False Positive Rate')
    ax4.set_ylabel('True Positive Rate')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = config.get_output_filename("roc_curves.png")
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ ROC curves visualization saved: {filename}")

def create_comprehensive_prc_analysis(cv_results, test_results, config):
    """Create comprehensive PRC analysis with confidence bands"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: CV PRC Curves with Confidence Bands (Top-left)
    ax1 = axes[0, 0]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(cv_results)))
    
    for i, (model_name, model_results) in enumerate(cv_results.items()):
        prc_stats = aggregate_cv_prc_curves(model_results)
        
        color = colors[i]
        
        # Plot mean curve
        ax1.plot(prc_stats['mean_recall'], prc_stats['mean_precision'], 
                color=color, linewidth=2,
                label=f'{model_name} (AP: {prc_stats["mean_ap"]:.3f}±{prc_stats["std_ap"]:.3f})')
        
        # Add confidence band
        ax1.fill_between(prc_stats['mean_recall'],
                        prc_stats['mean_precision'] - prc_stats['std_precision'],
                        prc_stats['mean_precision'] + prc_stats['std_precision'],
                        color=color, alpha=0.2)
    
    # Random baseline (TB prevalence)
    tb_prevalence_cv = 0.232
    ax1.axhline(y=tb_prevalence_cv, color='k', linestyle='--', alpha=0.5, 
               label=f'Random (TB prev: {tb_prevalence_cv:.1%})')
    
    ax1.set_title('Cross-Validation PRC Curves\n(5-fold, 481 patients, with confidence bands)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Recall (Sensitivity)')
    ax1.set_ylabel('Precision (PPV)')
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Plot 2: Leave-One-Out Test PRC Curves (Top-right)
    ax2 = axes[0, 1]
    
    for i, (model_name, model_results) in enumerate(test_results.items()):
        prc_data = model_results['curve_data']['prc_curve']
        test_ap = auc(prc_data['recall'], prc_data['precision'])
        
        ax2.plot(prc_data['recall'], prc_data['precision'],
                linewidth=2, linestyle='--', color=colors[i],
                label=f'{model_name} (AP: {test_ap:.3f})')
    
    tb_prevalence_test = 0.279
    ax2.axhline(y=tb_prevalence_test, color='k', linestyle='--', alpha=0.5,
               label=f'Random (TB prev: {tb_prevalence_test:.1%})')
    
    ax2.set_title('Leave-One-Out Test PRC Curves\n(Independent validation, 61 patients)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Recall (Sensitivity)')
    ax2.set_ylabel('Precision (PPV)')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    
    # Plot 3: CV vs Test PRC Comparison (Bottom-left)
    ax3 = axes[1, 0]
    
    for i, model_name in enumerate(cv_results.keys()):
        cv_prc_stats = aggregate_cv_prc_curves(cv_results[model_name])
        test_prc_data = test_results[model_name]['curve_data']['prc_curve']
        test_ap = auc(test_prc_data['recall'], test_prc_data['precision'])
        
        color = colors[i]
        
        # CV curve with confidence band
        ax3.plot(cv_prc_stats['mean_recall'], cv_prc_stats['mean_precision'], 
                color=color, linewidth=1, alpha=0.7,
                label=f'{model_name} CV (AP: {cv_prc_stats["mean_ap"]:.3f})')
        ax3.fill_between(cv_prc_stats['mean_recall'],
                        cv_prc_stats['mean_precision'] - cv_prc_stats['std_precision'],
                        cv_prc_stats['mean_precision'] + cv_prc_stats['std_precision'],
                        color=color, alpha=0.1)
        
        # Test curve (clean line)
        ax3.plot(test_prc_data['recall'], test_prc_data['precision'],
                color=color, linewidth=3, linestyle='--',
                label=f'{model_name} Test (AP: {test_ap:.3f})')
    
    ax3.axhline(y=tb_prevalence_cv, color='k', linestyle='--', alpha=0.3)
    ax3.set_title('CV vs Test PRC Comparison\n(Generalization Analysis)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Recall (Sensitivity)')
    ax3.set_ylabel('Precision (PPV)')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Class Balance Impact Analysis (Bottom-right)
    ax4 = axes[1, 1]
    
    # Show how class balance affects PRC performance
    prevalences = [0.1, 0.2, 0.232, 0.279, 0.3, 0.4, 0.5]
    
    # Simulate PRC performance under different prevalences
    for i, model_name in enumerate(list(cv_results.keys())[:3]):  # Top 3 models
        test_sensitivity = test_results[model_name]['test_sensitivity'] 
        test_specificity = test_results[model_name]['test_specificity']
        
        ppvs = []
        for prev in prevalences:
            # Calculate PPV given sensitivity, specificity, and prevalence
            ppv = (test_sensitivity * prev) / (test_sensitivity * prev + (1 - test_specificity) * (1 - prev))
            ppvs.append(ppv)
        
        ax4.plot(prevalences, ppvs, 'o-', color=colors[i], 
                label=f'{model_name} (Sens: {test_sensitivity:.2f})')
    
    # Mark actual prevalences
    ax4.axvline(x=tb_prevalence_cv, color='blue', linestyle=':', alpha=0.7, 
               label=f'CV TB prev: {tb_prevalence_cv:.1%}')
    ax4.axvline(x=tb_prevalence_test, color='red', linestyle=':', alpha=0.7,
               label=f'Test TB prev: {tb_prevalence_test:.1%}')
    
    ax4.set_title('Class Balance Impact on Precision\n(PPV vs TB prevalence)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('TB Prevalence')
    ax4.set_ylabel('Positive Predictive Value (PPV)')
    ax4.legend(loc='lower right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.05, 0.55])
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    filename = config.get_output_filename("precision_recall_curves.png")
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ PRC curves visualization saved: {filename}")

def create_comprehensive_who_analysis(cv_results, test_results, config):
    """Create comprehensive WHO compliance analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: CV WHO Compliance Scatter with Error Bars (Top-left)
    ax1 = axes[0, 0]
    
    # WHO compliance boundary
    ax1.axhline(y=config.who_specificity_target, color='red', linestyle='--', alpha=0.7, 
               linewidth=2, label=f'WHO Specificity ≥{config.who_specificity_target*100:.0f}%')
    ax1.axvline(x=config.who_sensitivity_target, color='red', linestyle='--', alpha=0.7, 
               linewidth=2, label=f'WHO Sensitivity ≥{config.who_sensitivity_target*100:.0f}%')
    
    # WHO compliance zone
    ax1.fill_between([config.who_sensitivity_target, 1.0], 
                    [config.who_specificity_target, config.who_specificity_target], 
                    [1.0, 1.0], 
                    color='green', alpha=0.1, label='WHO Compliant Zone')
    
    # Plot CV results with error bars
    colors = plt.cm.Set1(np.linspace(0, 1, len(cv_results)))
    
    for i, (model_name, model_results) in enumerate(cv_results.items()):
        fold_results = model_results['fold_results']
        sensitivities = [fr['optimal_sensitivity'] for fr in fold_results]
        specificities = [fr['optimal_specificity'] for fr in fold_results]
        
        sens_mean, sens_std = np.mean(sensitivities), np.std(sensitivities, ddof=1)
        spec_mean, spec_std = np.mean(specificities), np.std(specificities, ddof=1)
        
        # Plot mean with error bars
        ax1.errorbar(sens_mean, spec_mean, 
                    xerr=sens_std, yerr=spec_std,
                    color=colors[i], capsize=5, capthick=2, 
                    marker='o', markersize=8, alpha=0.8,
                    label=f'{model_name} CV')
    
    ax1.set_title('Cross-Validation WHO Compliance\n(5-fold results with error bars)', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Sensitivity (Recall)')
    ax1.set_ylabel('Specificity')
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.0])
    ax1.legend(loc='lower left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Leave-One-Out Test WHO Compliance (Top-right)
    ax2 = axes[0, 1]
    
    # WHO compliance boundary
    ax2.axhline(y=config.who_specificity_target, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.axvline(x=config.who_sensitivity_target, color='red', linestyle='--', alpha=0.7, linewidth=2)
    ax2.fill_between([config.who_sensitivity_target, 1.0], 
                    [config.who_specificity_target, config.who_specificity_target], 
                    [1.0, 1.0], 
                    color='green', alpha=0.1)
    
    # Plot test results
    for i, (model_name, model_results) in enumerate(test_results.items()):
        sensitivity = model_results['test_sensitivity']
        specificity = model_results['test_specificity'] 
        who_compliant = model_results['test_who_compliant']
        
        color = 'green' if who_compliant else colors[i]
        marker_size = 150 if who_compliant else 100
        marker = 's' if who_compliant else 'o'
        
        ax2.scatter([sensitivity], [specificity], 
                   color=color, s=marker_size, marker=marker, alpha=0.8,
                   label=f'{model_name} {"✓" if who_compliant else "✗"}')
    
    ax2.set_title('Leave-One-Out Test WHO Compliance\n(Independent validation, 61 patients)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sensitivity (Recall)')
    ax2.set_ylabel('Specificity')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: WHO Threshold Analysis (Bottom-left)
    ax3 = axes[1, 0]
    
    # Create threshold analysis for best models
    best_cv_model = max(cv_results.keys(), 
                       key=lambda m: cv_results[m]['mean_results']['who_score_mean'])
    best_test_model = max(test_results.keys(),
                         key=lambda m: test_results[m]['test_who_score'])
    
    thresholds = np.linspace(0.01, 0.99, 50)
    
    # Simulate threshold impact (using test data from best test model)
    if 'threshold_analysis' in test_results[best_test_model].get('curve_data', {}):
        threshold_data = test_results[best_test_model]['curve_data']['threshold_analysis']
        
        ax3.plot(threshold_data['thresholds'], threshold_data['sensitivities'], 
                'b-', linewidth=2, label='Sensitivity')
        ax3.plot(threshold_data['thresholds'], threshold_data['specificities'],
                'r-', linewidth=2, label='Specificity')
        ax3.plot(threshold_data['thresholds'], threshold_data['who_scores'],
                'g-', linewidth=3, label='WHO Score')
        
        # Mark WHO targets
        ax3.axhline(y=config.who_sensitivity_target, color='blue', linestyle=':', alpha=0.7)
        ax3.axhline(y=config.who_specificity_target, color='red', linestyle=':', alpha=0.7)
        
        # Mark optimal threshold
        optimal_idx = np.argmax(threshold_data['who_scores'])
        optimal_threshold = threshold_data['thresholds'][optimal_idx]
        ax3.axvline(x=optimal_threshold, color='green', linestyle='--', alpha=0.7,
                   label=f'Optimal threshold: {optimal_threshold:.3f}')
    
    ax3.set_title(f'WHO Threshold Optimization\n(Best test model: {best_test_model})', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Decision Threshold')
    ax3.set_ylabel('Performance Metric')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Clinical Deployment Assessment (Bottom-right)
    ax4 = axes[1, 1]
    
    # Create deployment readiness heatmap
    models = list(cv_results.keys())
    metrics = ['CV WHO Score', 'Test WHO Score', 'CV-Test Gap', 'WHO Compliant']
    
    deployment_matrix = []
    for model_name in models:
        cv_who_score = cv_results[model_name]['mean_results']['who_score_mean']
        test_who_score = test_results[model_name]['test_who_score']
        gap = abs(cv_who_score - test_who_score)
        who_compliant = 1.0 if test_results[model_name]['test_who_compliant'] else 0.0
        
        deployment_matrix.append([cv_who_score, test_who_score, 1.0 - gap, who_compliant])
    
    deployment_df = pd.DataFrame(deployment_matrix, 
                                index=[m.replace(' ', '\n') for m in models], 
                                columns=metrics)
    
    sns.heatmap(deployment_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                center=0.5, ax=ax4, cbar_kws={'label': 'Performance Score'})
    
    ax4.set_title('Clinical Deployment Readiness\n(Higher = Better for deployment)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Deployment Metrics')
    ax4.set_ylabel('Models')
    
    plt.tight_layout()
    filename = config.get_output_filename("who_compliance_analysis.png")
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ WHO compliance analysis saved: {filename}")

def create_confusion_matrices_grid(cv_results, test_results, config):
    """Create confusion matrices grid for test dataset"""
    
    n_models = len(test_results)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, (model_name, model_results) in enumerate(test_results.items()):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        cm = model_results['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['TB Negative', 'TB Positive'],
                   yticklabels=['TB Negative', 'TB Positive'])
        
        # Add performance metrics
        sensitivity = model_results['test_sensitivity']
        specificity = model_results['test_specificity']
        who_score = model_results['test_who_score']
        who_compliant = "✓" if model_results['test_who_compliant'] else "✗"
        
        ax.set_title(f'{model_name}\nSens: {sensitivity:.3f}, Spec: {specificity:.3f}\nWHO: {who_score:.3f} {who_compliant}', 
                    fontsize=12)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(len(test_results), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    filename = config.get_output_filename("confusion_matrices_grid.png")
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Confusion matrices grid saved: {filename}")

def create_cv_fold_variance_plot(cv_results, config):
    """Create cross-validation fold variance analysis"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Performance variance across folds
    ax1 = axes[0, 0]
    
    models = list(cv_results.keys())
    metrics = ['accuracy', 'roc_auc', 'optimal_sensitivity', 'optimal_specificity']
    
    fold_data = {metric: {model: [] for model in models} for metric in metrics}
    
    for model_name, model_results in cv_results.items():
        for fold_result in model_results['fold_results']:
            for metric in metrics:
                fold_data[metric][model_name].append(fold_result[metric])
    
    # Box plot for each metric
    x_pos = np.arange(len(models))
    colors = plt.cm.Set1(np.linspace(0, 1, len(metrics)))
    
    for i, metric in enumerate(metrics):
        metric_data = [fold_data[metric][model] for model in models]
        bp = ax1.boxplot(metric_data, positions=x_pos + i*0.2, widths=0.15, 
                        patch_artist=True, boxprops=dict(facecolor=colors[i]))
        
        if i == 0:  # Only add labels once
            ax1.set_xticks(x_pos + 0.3)
            ax1.set_xticklabels([m.replace(' ', '\n') for m in models], rotation=0)
    
    ax1.set_title('Cross-Validation Performance Variance\n(5-fold stability analysis)', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Performance Score')
    ax1.legend([plt.Rectangle((0,0),1,1, facecolor=colors[i]) for i in range(len(metrics))], 
              metrics, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1])
    
    # Plot 2: WHO Score variance
    ax2 = axes[0, 1]
    
    who_score_data = []
    model_labels = []
    
    for model_name, model_results in cv_results.items():
        who_scores = [fr['who_score'] for fr in model_results['fold_results']]
        who_score_data.append(who_scores)
        
        mean_score = np.mean(who_scores)
        std_score = np.std(who_scores, ddof=1)
        model_labels.append(f'{model_name}\n({mean_score:.3f}±{std_score:.3f})')
    
    bp = ax2.boxplot(who_score_data, labels=model_labels, patch_artist=True)
    
    # Color boxes by mean WHO score
    colors = plt.cm.RdYlGn([np.mean(data) for data in who_score_data])
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_title('WHO Score Stability Across Folds\n(Lower variance = more reliable)', 
                  fontsize=14, fontweight='bold')
    ax2.set_ylabel('WHO Score')
    ax2.grid(True, alpha=0.3)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    # Plot 3: Sensitivity-Specificity tradeoff variance
    ax3 = axes[1, 0]
    
    for i, (model_name, model_results) in enumerate(cv_results.items()):
        sensitivities = [fr['optimal_sensitivity'] for fr in model_results['fold_results']]
        specificities = [fr['optimal_specificity'] for fr in model_results['fold_results']]
        
        # Plot individual fold points
        ax3.scatter(sensitivities, specificities, alpha=0.6, s=50, 
                   label=f'{model_name} (folds)')
        
        # Plot mean with error ellipse
        sens_mean, spec_mean = np.mean(sensitivities), np.mean(specificities)
        sens_std, spec_std = np.std(sensitivities, ddof=1), np.std(specificities, ddof=1)
        
        ax3.scatter([sens_mean], [spec_mean], s=200, marker='s', alpha=0.8,
                   label=f'{model_name} (mean)')
        
        # Add error ellipse
        ellipse = patches.Ellipse((sens_mean, spec_mean), sens_std*2, spec_std*2,
                                 alpha=0.3, facecolor=plt.cm.Set1(i/len(cv_results)))
        ax3.add_patch(ellipse)
    
    # WHO targets
    ax3.axhline(y=config.who_specificity_target, color='red', linestyle='--', alpha=0.7)
    ax3.axvline(x=config.who_sensitivity_target, color='red', linestyle='--', alpha=0.7)
    
    ax3.set_title('Sensitivity-Specificity Tradeoff Variance\n(Ellipses show 2σ confidence)', 
                  fontsize=14, fontweight='bold')
    ax3.set_xlabel('Sensitivity')
    ax3.set_ylabel('Specificity')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    
    # Plot 4: Model consistency ranking
    ax4 = axes[1, 1]
    
    consistency_metrics = []
    for model_name, model_results in cv_results.items():
        who_scores = [fr['who_score'] for fr in model_results['fold_results']]
        sensitivities = [fr['optimal_sensitivity'] for fr in model_results['fold_results']]
        specificities = [fr['optimal_specificity'] for fr in model_results['fold_results']]
        
        # Calculate coefficient of variation (CV = std/mean)
        who_cv = np.std(who_scores, ddof=1) / np.mean(who_scores) if np.mean(who_scores) > 0 else float('inf')
        sens_cv = np.std(sensitivities, ddof=1) / np.mean(sensitivities)
        spec_cv = np.std(specificities, ddof=1) / np.mean(specificities)
        
        # Overall consistency score (lower = more consistent)
        consistency_score = np.mean([who_cv, sens_cv, spec_cv])
        consistency_metrics.append((model_name, consistency_score, np.mean(who_scores)))
    
    # Sort by consistency
    consistency_metrics.sort(key=lambda x: x[1])
    
    models_sorted = [x[0] for x in consistency_metrics]
    consistency_scores = [x[1] for x in consistency_metrics]
    mean_who_scores = [x[2] for x in consistency_metrics]
    
    # Create scatter plot
    scatter = ax4.scatter(consistency_scores, mean_who_scores, s=150, alpha=0.7,
                         c=range(len(models_sorted)), cmap='viridis')
    
    # Add model labels
    for i, (model, consistency, who_score) in enumerate(consistency_metrics):
        ax4.annotate(model.replace(' ', '\n'), (consistency, who_score), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', alpha=0.7))
    
    ax4.set_title('Model Consistency vs Performance\n(Top-right = High performance + High consistency)', 
                  fontsize=14, fontweight='bold')
    ax4.set_xlabel('Consistency Score (lower = more consistent)')
    ax4.set_ylabel('Mean WHO Score')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = config.get_output_filename("cv_fold_variance.png")
    plt.savefig(f"results/{filename}", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ CV fold variance analysis saved: {filename}")