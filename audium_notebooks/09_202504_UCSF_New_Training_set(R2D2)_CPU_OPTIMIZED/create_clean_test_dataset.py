#!/usr/bin/env python3
"""
Create a clean small test dataset excluding problematic patients
"""

import pandas as pd
import numpy as np
import os
import glob

def main():
    # Set up paths
    metadata_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2 lung sounds metadata_TRAIN_2025.05.08_v3.csv'
    audio_data_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    output_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Filter valid labels
    valid_labels = metadata[metadata['Microbiologicreferencestandard'].isin(['TB Negative', 'TB Positive'])]
    
    # Get patient directories
    patient_dirs = [d for d in os.listdir(audio_data_path) if os.path.isdir(os.path.join(audio_data_path, d))]
    
    # Count audio files per patient
    file_counts = {}
    for patient_dir in patient_dirs:
        patient_path = os.path.join(audio_data_path, patient_dir)
        wav_files = glob.glob(os.path.join(patient_path, '**/*.wav'), recursive=True)
        file_counts[patient_dir] = len(wav_files)
    
    # Define criteria for "clean" patients
    # - Has audio files (>0)
    # - Has reasonable number of files (<=30 to avoid the 8975 outlier)
    # - Has metadata
    clean_patients = []
    for patient_id in valid_labels['StudyID']:
        if patient_id in file_counts:
            file_count = file_counts[patient_id]
            if 0 < file_count <= 30:
                clean_patients.append(patient_id)
    
    print(f"Clean patients identified: {len(clean_patients)}")
    
    # Split by TB status
    tb_positive_clean = []
    tb_negative_clean = []
    
    for patient_id in clean_patients:
        label = valid_labels[valid_labels['StudyID'] == patient_id]['Microbiologicreferencestandard'].iloc[0]
        if label == 'TB Positive':
            tb_positive_clean.append(patient_id)
        else:
            tb_negative_clean.append(patient_id)
    
    print(f"TB Positive clean patients: {len(tb_positive_clean)}")
    print(f"TB Negative clean patients: {len(tb_negative_clean)}")
    
    # Create small balanced test set
    n_test = 5  # 5 of each
    small_test_patients = (
        tb_positive_clean[:n_test] + 
        tb_negative_clean[:n_test]
    )
    
    print(f"\\nSmall test dataset ({len(small_test_patients)} patients):")
    for patient_id in small_test_patients:
        label = valid_labels[valid_labels['StudyID'] == patient_id]['Microbiologicreferencestandard'].iloc[0]
        file_count = file_counts[patient_id]
        print(f"  {patient_id}: {label} ({file_count} files)")
    
    # Save clean small test dataset
    small_test_df = pd.DataFrame({
        'StudyID': small_test_patients,
        'Label': [valid_labels[valid_labels['StudyID'] == p]['Microbiologicreferencestandard'].iloc[0] for p in small_test_patients],
        'FileCount': [file_counts[p] for p in small_test_patients]
    })
    
    test_path = os.path.join(output_dir, 'data', 'clean_small_test_patients.csv')
    small_test_df.to_csv(test_path, index=False)
    print(f"\\nClean small test dataset saved: {test_path}")
    
    # Create medium test set (50 patients)
    n_medium = 25  # 25 of each
    medium_test_patients = (
        tb_positive_clean[:n_medium] + 
        tb_negative_clean[:n_medium]
    )
    
    medium_test_df = pd.DataFrame({
        'StudyID': medium_test_patients,
        'Label': [valid_labels[valid_labels['StudyID'] == p]['Microbiologicreferencestandard'].iloc[0] for p in medium_test_patients],
        'FileCount': [file_counts[p] for p in medium_test_patients]
    })
    
    medium_path = os.path.join(output_dir, 'data', 'clean_medium_test_patients.csv')
    medium_test_df.to_csv(medium_path, index=False)
    print(f"Clean medium test dataset saved: {medium_path}")
    
    # Save all clean patients for eventual full analysis
    all_clean_df = pd.DataFrame({
        'StudyID': clean_patients,
        'Label': [valid_labels[valid_labels['StudyID'] == p]['Microbiologicreferencestandard'].iloc[0] for p in clean_patients],
        'FileCount': [file_counts[p] for p in clean_patients]
    })
    
    all_clean_path = os.path.join(output_dir, 'data', 'all_clean_patients.csv')
    all_clean_df.to_csv(all_clean_path, index=False)
    print(f"All clean patients saved: {all_clean_path}")
    
    # Summary statistics
    print(f"\\nClean dataset summary:")
    print(f"  Total clean patients: {len(clean_patients)}")
    print(f"  TB Positive: {len(tb_positive_clean)} ({len(tb_positive_clean)/len(clean_patients):.3f})")
    print(f"  TB Negative: {len(tb_negative_clean)} ({len(tb_negative_clean)/len(clean_patients):.3f})")
    print(f"  Average files per patient: {np.mean([file_counts[p] for p in clean_patients]):.2f}")
    print(f"  Total files: {sum([file_counts[p] for p in clean_patients])}")

if __name__ == "__main__":
    main()