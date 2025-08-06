#!/usr/bin/env python3
"""
Investigate patient outliers in the dataset
"""

import pandas as pd
import numpy as np
import os
import glob
from collections import Counter

def main():
    # Set up paths
    metadata_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2 lung sounds metadata_TRAIN_2025.05.08_v3.csv'
    audio_data_path = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    
    # Load metadata
    metadata = pd.read_csv(metadata_path)
    
    # Get patient directories
    patient_dirs = [d for d in os.listdir(audio_data_path) if os.path.isdir(os.path.join(audio_data_path, d))]
    patient_dirs.sort()
    
    # Count audio files per patient (including nested directories)
    file_counts = {}
    for patient_dir in patient_dirs:
        patient_path = os.path.join(audio_data_path, patient_dir)
        wav_files = glob.glob(os.path.join(patient_path, '**/*.wav'), recursive=True)
        file_counts[patient_dir] = len(wav_files)
    
    print("PATIENT OUTLIER INVESTIGATION")
    print("="*50)
    
    # Find patients with 0 files
    zero_files = [p for p, count in file_counts.items() if count == 0]
    print(f"\nPatients with 0 audio files: {len(zero_files)}")
    for patient in zero_files:
        print(f"  - {patient}")
    
    # Find patients with unusually high file counts
    file_counts_list = list(file_counts.values())
    q75 = np.percentile(file_counts_list, 75)
    q25 = np.percentile(file_counts_list, 25)
    iqr = q75 - q25
    outlier_threshold = q75 + 1.5 * iqr
    
    print(f"\nFile count statistics:")
    print(f"  Q25: {q25:.1f}")
    print(f"  Q75: {q75:.1f}")
    print(f"  IQR: {iqr:.1f}")
    print(f"  Outlier threshold (Q75 + 1.5*IQR): {outlier_threshold:.1f}")
    
    high_outliers = [(p, count) for p, count in file_counts.items() if count > outlier_threshold]
    print(f"\nPatients with unusually high file counts ({len(high_outliers)}):")
    for patient, count in sorted(high_outliers, key=lambda x: x[1], reverse=True):
        print(f"  - {patient}: {count} files")
    
    # Check which patients are missing from metadata
    metadata_patients = set(metadata['StudyID'])
    audio_patients = set(patient_dirs)
    
    missing_audio = metadata_patients - audio_patients
    missing_metadata = audio_patients - metadata_patients
    
    print(f"\nMissing audio files (in metadata but no audio): {len(missing_audio)}")
    for patient in missing_audio:
        print(f"  - {patient}")
    
    print(f"\nMissing metadata (audio files but no metadata): {len(missing_metadata)}")
    for patient in missing_metadata:
        print(f"  - {patient}")
    
    # Distribution of file counts
    print(f"\nFile count distribution:")
    file_count_dist = Counter(file_counts.values())
    for count in sorted(file_count_dist.keys()):
        if file_count_dist[count] > 1:  # Only show counts that occur more than once
            print(f"  {count} files: {file_count_dist[count]} patients")
    
    # Check the problematic patient directory
    if high_outliers:
        worst_patient = max(high_outliers, key=lambda x: x[1])
        print(f"\nInvestigating worst outlier: {worst_patient[0]} ({worst_patient[1]} files)")
        worst_path = os.path.join(audio_data_path, worst_patient[0])
        print(f"Directory structure:")
        os.system(f"find '{worst_path}' -type d | head -10")
        print(f"Sample files:")
        os.system(f"find '{worst_path}' -name '*.wav' | head -5")

if __name__ == "__main__":
    main()