#!/usr/bin/env python3
"""
Prepare Leave-One-Out Dataset Split for TB Detection Pipeline
Creates training/test splits based on reserved patient list
"""

import pandas as pd
import numpy as np
import os
import argparse
from pathlib import Path

def load_reserved_test_patients(test_patients_file):
    """Load reserved test patient IDs from CSV file"""
    print(f"📂 Loading reserved test patients from: {test_patients_file}")
    
    if not os.path.exists(test_patients_file):
        raise FileNotFoundError(f"Test patients file not found: {test_patients_file}")
    
    test_df = pd.read_csv(test_patients_file)
    print(f"✅ Loaded {len(test_df)} reserved test patients")
    
    # Extract patient IDs (assuming column is 'PID' based on the file structure)
    if 'PID' in test_df.columns:
        test_patients = test_df['PID'].tolist()
    elif 'StudyID' in test_df.columns:
        test_patients = test_df['StudyID'].tolist()
    else:
        # Try first column
        test_patients = test_df.iloc[:, -1].tolist()  # Last column likely contains patient IDs
    
    # Remove any NaN values
    test_patients = [p for p in test_patients if pd.notna(p)]
    
    print(f"✅ Extracted {len(test_patients)} valid test patient IDs")
    print(f"📋 Sample test patients: {test_patients[:5]}")
    
    # Check if patients follow R2D2 pattern
    r2d2_pattern = [str(p).startswith('R2D2') for p in test_patients]
    r2d2_count = sum(r2d2_pattern)
    print(f"🔍 Test patients matching R2D2 pattern: {r2d2_count}/{len(test_patients)}")
    
    return test_patients

def load_full_patient_dataset(labels_file):
    """Load full patient dataset with TB labels"""
    print(f"📂 Loading full patient dataset from: {labels_file}")
    
    if not os.path.exists(labels_file):
        raise FileNotFoundError(f"Labels file not found: {labels_file}")
    
    labels_df = pd.read_csv(labels_file)
    print(f"✅ Loaded {len(labels_df)} patients from full dataset")
    
    # Display label distribution
    if 'Label' in labels_df.columns:
        label_counts = labels_df['Label'].value_counts()
        print(f"📊 Label distribution:")
        for label, count in label_counts.items():
            print(f"   {label}: {count} patients ({count/len(labels_df)*100:.1f}%)")
    
    return labels_df

def create_leave_one_out_splits(full_dataset, test_patients, output_dir):
    """Create training and test splits for leave-one-out validation"""
    
    print(f"\n🔄 Creating leave-one-out dataset splits...")
    
    # Convert to sets for efficient operations
    test_patient_set = set(test_patients)
    full_patient_set = set(full_dataset['StudyID'].tolist())
    
    # Validate test patients exist in full dataset
    test_in_full = test_patient_set.intersection(full_patient_set)
    test_not_in_full = test_patient_set - full_patient_set
    
    print(f"📊 Dataset alignment:")
    print(f"   Reserved test patients: {len(test_patients)}")
    print(f"   Test patients in full dataset: {len(test_in_full)}")
    print(f"   Test patients NOT in full dataset: {len(test_not_in_full)}")
    
    if test_not_in_full:
        print(f"⚠️  Warning: {len(test_not_in_full)} test patients not found in full dataset:")
        print(f"   {sorted(list(test_not_in_full))[:10]}")
    
    # Use only test patients that exist in full dataset
    valid_test_patients = list(test_in_full)
    
    # Create training dataset (exclude test patients)
    training_dataset = full_dataset[~full_dataset['StudyID'].isin(valid_test_patients)].copy()
    test_dataset = full_dataset[full_dataset['StudyID'].isin(valid_test_patients)].copy()
    
    print(f"\n📋 Final dataset splits:")
    print(f"   Training patients: {len(training_dataset)}")
    print(f"   Test patients: {len(test_dataset)}")
    print(f"   Total patients: {len(full_dataset)}")
    
    # Analyze TB distribution in splits
    if 'Label' in training_dataset.columns:
        print(f"\n📊 Training dataset TB distribution:")
        train_tb_counts = training_dataset['Label'].value_counts()
        for label, count in train_tb_counts.items():
            print(f"   {label}: {count} patients ({count/len(training_dataset)*100:.1f}%)")
        
        print(f"\n📊 Test dataset TB distribution:")
        test_tb_counts = test_dataset['Label'].value_counts()
        for label, count in test_tb_counts.items():
            print(f"   {label}: {count} patients ({count/len(test_dataset)*100:.1f}%)")
        
        # Check for balance preservation
        train_tb_rate = (training_dataset['Label'] == 'TB Positive').mean()
        test_tb_rate = (test_dataset['Label'] == 'TB Positive').mean()
        print(f"\n📈 TB prevalence comparison:")
        print(f"   Training set TB rate: {train_tb_rate:.3f}")
        print(f"   Test set TB rate: {test_tb_rate:.3f}")
        print(f"   Difference: {abs(train_tb_rate - test_tb_rate):.3f}")
    
    # Save datasets
    os.makedirs(output_dir, exist_ok=True)
    
    training_file = os.path.join(output_dir, 'training_patients_leave_one_out.csv')
    test_file = os.path.join(output_dir, 'test_patients_leave_one_out.csv')
    
    training_dataset.to_csv(training_file, index=False)
    test_dataset.to_csv(test_file, index=False)
    
    print(f"\n✅ Dataset splits saved:")
    print(f"   Training dataset: {training_file}")
    print(f"   Test dataset: {test_file}")
    
    return training_dataset, test_dataset, training_file, test_file

def validate_embeddings_availability(training_file, test_file, embeddings_file, metadata_file):
    """Validate that embeddings exist for all patients in splits"""
    
    print(f"\n🔍 Validating embeddings availability...")
    
    # Load embeddings metadata
    if not os.path.exists(metadata_file):
        print(f"⚠️  Embeddings metadata file not found: {metadata_file}")
        print(f"   Please generate embeddings first before running leave-one-out validation")
        return False
    
    metadata_df = pd.read_csv(metadata_file)
    embedding_patients = set(metadata_df['patient_id'].unique()) if 'patient_id' in metadata_df.columns else set()
    
    # Load dataset splits
    training_df = pd.read_csv(training_file)
    test_df = pd.read_csv(test_file)
    
    training_patients = set(training_df['StudyID'].tolist())
    test_patients = set(test_df['StudyID'].tolist())
    
    # Check availability
    train_with_embeddings = training_patients.intersection(embedding_patients)
    train_missing_embeddings = training_patients - embedding_patients
    test_with_embeddings = test_patients.intersection(embedding_patients)
    test_missing_embeddings = test_patients - embedding_patients
    
    print(f"📊 Embeddings availability:")
    print(f"   Training patients with embeddings: {len(train_with_embeddings)}/{len(training_patients)}")
    print(f"   Training patients missing embeddings: {len(train_missing_embeddings)}")
    print(f"   Test patients with embeddings: {len(test_with_embeddings)}/{len(test_patients)}")
    print(f"   Test patients missing embeddings: {len(test_missing_embeddings)}")
    
    if train_missing_embeddings:
        print(f"⚠️  Training patients missing embeddings: {sorted(list(train_missing_embeddings))[:10]}")
    
    if test_missing_embeddings:
        print(f"⚠️  Test patients missing embeddings: {sorted(list(test_missing_embeddings))[:10]}")
    
    # Check embeddings file exists
    embeddings_exists = os.path.exists(embeddings_file)
    print(f"📁 Embeddings file exists: {embeddings_exists}")
    
    if embeddings_exists:
        try:
            embeddings_data = np.load(embeddings_file, allow_pickle=True)
            print(f"✅ Embeddings file loaded successfully with {len(embeddings_data.files)} files")
        except Exception as e:
            print(f"❌ Error loading embeddings file: {e}")
            return False
    
    # Validation result
    validation_success = (
        len(train_missing_embeddings) == 0 and
        len(test_missing_embeddings) == 0 and
        embeddings_exists
    )
    
    if validation_success:
        print(f"✅ All patients have embeddings available - ready for leave-one-out validation!")
    else:
        print(f"❌ Missing embeddings detected - please generate embeddings for missing patients first")
    
    return validation_success

def generate_summary_report(training_file, test_file, output_dir):
    """Generate summary report for the leave-one-out split"""
    
    training_df = pd.read_csv(training_file)
    test_df = pd.read_csv(test_file)
    
    # Create summary
    summary = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'training_patients': len(training_df),
        'test_patients': len(test_df),
        'total_patients': len(training_df) + len(test_df),
        'test_percentage': len(test_df) / (len(training_df) + len(test_df)) * 100
    }
    
    if 'Label' in training_df.columns:
        # Training set statistics
        train_tb_pos = (training_df['Label'] == 'TB Positive').sum()
        train_tb_neg = (training_df['Label'] == 'TB Negative').sum()
        summary.update({
            'training_tb_positive': train_tb_pos,
            'training_tb_negative': train_tb_neg,
            'training_tb_prevalence': train_tb_pos / len(training_df)
        })
        
        # Test set statistics
        test_tb_pos = (test_df['Label'] == 'TB Positive').sum()
        test_tb_neg = (test_df['Label'] == 'TB Negative').sum()
        summary.update({
            'test_tb_positive': test_tb_pos,
            'test_tb_negative': test_tb_neg,
            'test_tb_prevalence': test_tb_pos / len(test_df)
        })
    
    # Save summary
    summary_file = os.path.join(output_dir, 'leave_one_out_split_summary.csv')
    pd.DataFrame([summary]).to_csv(summary_file, index=False)
    
    # Generate text report
    report_file = os.path.join(output_dir, 'leave_one_out_preparation_report.txt')
    with open(report_file, 'w') as f:
        f.write("Leave-One-Out Dataset Preparation Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Generated: {summary['timestamp']}\n\n")
        f.write("Dataset Split Summary:\n")
        f.write(f"  Training patients: {summary['training_patients']}\n")
        f.write(f"  Test patients: {summary['test_patients']}\n")
        f.write(f"  Total patients: {summary['total_patients']}\n")
        f.write(f"  Test set percentage: {summary['test_percentage']:.1f}%\n\n")
        
        if 'training_tb_prevalence' in summary:
            f.write("TB Distribution:\n")
            f.write(f"  Training set TB prevalence: {summary['training_tb_prevalence']:.3f}\n")
            f.write(f"  Training TB positive: {summary['training_tb_positive']}\n")
            f.write(f"  Training TB negative: {summary['training_tb_negative']}\n\n")
            f.write(f"  Test set TB prevalence: {summary['test_tb_prevalence']:.3f}\n")
            f.write(f"  Test TB positive: {summary['test_tb_positive']}\n")
            f.write(f"  Test TB negative: {summary['test_tb_negative']}\n\n")
        
        f.write("Files Generated:\n")
        f.write(f"  Training dataset: {os.path.basename(training_file)}\n")
        f.write(f"  Test dataset: {os.path.basename(test_file)}\n")
        f.write(f"  Summary CSV: {os.path.basename(summary_file)}\n")
        f.write(f"  This report: {os.path.basename(report_file)}\n\n")
        
        f.write("Next Steps:\n")
        f.write("  1. Validate embeddings availability for all patients\n")
        f.write("  2. Run leave-one-out validation pipeline\n")
        f.write("  3. Evaluate models on reserved test dataset\n")
        f.write("  4. Generate clinical validation report\n")
    
    print(f"\n✅ Summary report generated:")
    print(f"   Summary CSV: {summary_file}")
    print(f"   Text report: {report_file}")
    
    return summary_file, report_file

def main():
    parser = argparse.ArgumentParser(description='Prepare leave-one-out dataset splits')
    parser.add_argument('--test_patients', type=str, required=True,
                       help='Path to reserved test patients CSV file')
    parser.add_argument('--labels_file', type=str,
                       default='data/clean_patients_final.csv',
                       help='Path to full patient labels file')
    parser.add_argument('--embeddings_file', type=str,
                       default='data/final_embeddings.npz',
                       help='Path to embeddings NPZ file')
    parser.add_argument('--metadata_file', type=str,
                       default='data/final_embeddings_metadata.csv',
                       help='Path to embeddings metadata CSV file')
    parser.add_argument('--output_dir', type=str,
                       default='data',
                       help='Output directory for generated files')
    parser.add_argument('--validate_embeddings', action='store_true',
                       help='Validate embeddings availability for all patients')
    
    args = parser.parse_args()
    
    print("🚀 Leave-One-Out Dataset Preparation")
    print("=" * 50)
    
    try:
        # Step 1: Load reserved test patients
        test_patients = load_reserved_test_patients(args.test_patients)
        
        # Step 2: Load full patient dataset
        full_dataset = load_full_patient_dataset(args.labels_file)
        
        # Step 3: Create training/test splits
        training_df, test_df, training_file, test_file = create_leave_one_out_splits(
            full_dataset, test_patients, args.output_dir
        )
        
        # Step 4: Validate embeddings (if requested)
        if args.validate_embeddings:
            embeddings_valid = validate_embeddings_availability(
                training_file, test_file, args.embeddings_file, args.metadata_file
            )
            if not embeddings_valid:
                print("\n⚠️  Warning: Some patients missing embeddings. Generate embeddings first.")
        
        # Step 5: Generate summary report
        summary_file, report_file = generate_summary_report(
            training_file, test_file, args.output_dir
        )
        
        print(f"\n🎉 Leave-one-out dataset preparation complete!")
        print(f"Ready to run: python 04_leave_one_out_validation.py")
        
    except Exception as e:
        print(f"❌ Error during dataset preparation: {e}")
        raise

if __name__ == "__main__":
    main()