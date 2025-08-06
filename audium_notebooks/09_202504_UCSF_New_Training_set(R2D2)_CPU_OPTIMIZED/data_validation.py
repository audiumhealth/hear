#!/usr/bin/env python3
"""
Data Validation Script for New UCSF R2D2 Training Dataset
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def main():
    # Set up paths
    metadata_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2 lung sounds metadata_TRAIN_2025.05.08_v3.csv'
    audio_data_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    output_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    print("="*60)
    print("NEW UCSF R2D2 TRAINING DATASET VALIDATION")
    print("="*60)
    
    # 1. Load and inspect metadata
    print("\n1. METADATA ANALYSIS:")
    print(f"Loading metadata from: {metadata_path}")
    
    if not os.path.exists(metadata_path):
        print("❌ ERROR: Metadata file not found!")
        return
    
    metadata = pd.read_csv(metadata_path)
    print(f"✅ Metadata loaded: {metadata.shape[0]} patients, {metadata.shape[1]} columns")
    
    # Check StudyID format
    r2d2_pattern = metadata['StudyID'].str.match(r'^R2D2\d{5}$')
    print(f"✅ StudyIDs matching R2D2NNNNN pattern: {r2d2_pattern.sum()}/{len(metadata)}")
    
    # Check TB labels
    label_counts = metadata['Microbiologicreferencestandard'].value_counts()
    print(f"✅ Label distribution: {dict(label_counts)}")
    
    # Filter valid labels
    valid_labels = metadata[metadata['Microbiologicreferencestandard'].isin(['TB Negative', 'TB Positive'])]
    tb_prevalence = (valid_labels['Microbiologicreferencestandard'] == 'TB Positive').mean()
    print(f"✅ Valid patients (excl. Indeterminate): {len(valid_labels)}")
    print(f"✅ TB Prevalence: {tb_prevalence:.3f}")
    
    # 2. Audio file validation
    print("\n2. AUDIO FILE ANALYSIS:")
    print(f"Checking audio directory: {audio_data_path}")
    
    if not os.path.exists(audio_data_path):
        print("❌ ERROR: Audio data directory not found!")
        return
    
    # Get patient directories
    patient_dirs = [d for d in os.listdir(audio_data_path) if os.path.isdir(os.path.join(audio_data_path, d))]
    patient_dirs.sort()
    print(f"✅ Patient directories found: {len(patient_dirs)}")
    
    # Count audio files per patient (including nested directories)
    file_counts = {}
    total_files = 0
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(audio_data_path, patient_dir)
        # Use recursive search to find all wav files
        wav_files = glob.glob(os.path.join(patient_path, '**/*.wav'), recursive=True)
        file_counts[patient_dir] = len(wav_files)
        total_files += len(wav_files)
    
    avg_files = np.mean(list(file_counts.values()))
    print(f"✅ Total audio files: {total_files}")
    print(f"✅ Average files per patient: {avg_files:.2f}")
    print(f"✅ Min/Max files per patient: {min(file_counts.values())}/{max(file_counts.values())}")
    
    # 3. Data alignment check
    print("\n3. DATA ALIGNMENT CHECK:")
    metadata_patients = set(metadata['StudyID'])
    audio_patients = set(patient_dirs)
    
    print(f"✅ Patients in metadata: {len(metadata_patients)}")
    print(f"✅ Patients with audio: {len(audio_patients)}")
    
    aligned_patients = metadata_patients.intersection(audio_patients)
    missing_audio = metadata_patients - audio_patients
    missing_metadata = audio_patients - metadata_patients
    
    print(f"✅ Perfectly aligned: {len(aligned_patients)}")
    print(f"⚠️  Missing audio: {len(missing_audio)}")
    print(f"⚠️  Missing metadata: {len(missing_metadata)}")
    
    # 4. Sanity checks
    print("\n4. SANITY CHECKS:")
    expected_avg = 20
    
    checks = [
        ("Average files per patient (~20)", abs(avg_files - expected_avg) <= 5),
        ("Perfect alignment", len(missing_audio) == 0 and len(missing_metadata) == 0),
        ("R2D2 format compliance", r2d2_pattern.all()),
        ("Valid TB labels available", len(valid_labels) > 0),
        ("All patients have audio files", min(file_counts.values()) > 0)
    ]
    
    all_passed = True
    for check_name, passed in checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False
    
    # 5. Create small test dataset
    print("\n5. SMALL TEST DATASET CREATION:")
    tb_positive_patients = valid_labels[valid_labels['Microbiologicreferencestandard'] == 'TB Positive']['StudyID'].tolist()
    tb_negative_patients = valid_labels[valid_labels['Microbiologicreferencestandard'] == 'TB Negative']['StudyID'].tolist()
    
    # Select patients that have audio files
    tb_positive_with_audio = [p for p in tb_positive_patients if p in audio_patients]
    tb_negative_with_audio = [p for p in tb_negative_patients if p in audio_patients]
    
    # Create small balanced test set
    n_positive = min(5, len(tb_positive_with_audio))
    n_negative = min(5, len(tb_negative_with_audio))
    
    small_test_patients = (
        tb_positive_with_audio[:n_positive] + 
        tb_negative_with_audio[:n_negative]
    )
    
    print(f"✅ Small test dataset: {len(small_test_patients)} patients")
    print(f"   - TB Positive: {n_positive}")
    print(f"   - TB Negative: {n_negative}")
    print(f"   - Test patients: {small_test_patients}")
    
    # 6. Save validation results
    print("\n6. SAVING RESULTS:")
    
    # Save validation summary
    validation_summary = {
        'metadata_patients': len(metadata_patients),
        'audio_patients': len(audio_patients),
        'aligned_patients': len(aligned_patients),
        'total_audio_files': total_files,
        'avg_files_per_patient': avg_files,
        'expected_avg_files': expected_avg,
        'avg_deviation': abs(avg_files - expected_avg),
        'tb_positive_patients': (valid_labels['Microbiologicreferencestandard'] == 'TB Positive').sum(),
        'tb_negative_patients': (valid_labels['Microbiologicreferencestandard'] == 'TB Negative').sum(),
        'indeterminate_patients': (metadata['Microbiologicreferencestandard'] == 'Indeterminate').sum(),
        'tb_prevalence': tb_prevalence,
        'perfect_alignment': len(missing_audio) == 0 and len(missing_metadata) == 0,
        'missing_audio': len(missing_audio),
        'missing_metadata': len(missing_metadata),
        'all_sanity_checks_passed': all_passed
    }
    
    summary_df = pd.DataFrame([validation_summary])
    summary_path = os.path.join(output_dir, 'data', 'validation_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Validation summary saved: {summary_path}")
    
    # Save small test patient list
    small_test_df = pd.DataFrame({
        'StudyID': small_test_patients,
        'Label': [valid_labels[valid_labels['StudyID'] == p]['Microbiologicreferencestandard'].iloc[0] for p in small_test_patients]
    })
    test_path = os.path.join(output_dir, 'data', 'small_test_patients.csv')
    small_test_df.to_csv(test_path, index=False)
    print(f"✅ Small test patients saved: {test_path}")
    
    # 7. Final assessment
    print("\n" + "="*60)
    print("FINAL ASSESSMENT")
    print("="*60)
    
    if all_passed:
        print("🎉 ALL VALIDATION CHECKS PASSED!")
        print("✅ Dataset structure is valid and ready for pipeline execution.")
        print("✅ Small test dataset created for iterative development.")
        print("✅ Ready to proceed with embedding generation.")
    else:
        print("⚠️  SOME VALIDATION CHECKS FAILED!")
        print("❌ Review data structure issues before proceeding.")
        print("❌ Check missing files or metadata alignment.")
    
    print("="*60)
    return all_passed

if __name__ == "__main__":
    main()