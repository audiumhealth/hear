#!/usr/bin/env python3
"""
Investigate potential double-counting of audio files
"""

import os
import glob
import pandas as pd
from collections import defaultdict

def investigate_double_counting():
    """Investigate if files are being counted multiple times"""
    
    base_audio_dir = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    
    print("🔍 INVESTIGATING POTENTIAL DOUBLE-COUNTING")
    print("=" * 60)
    
    # Get patient directories
    patient_dirs = [d for d in os.listdir(base_audio_dir) if os.path.isdir(os.path.join(base_audio_dir, d))]
    patient_dirs.sort()
    
    # Let's examine a few specific patients in detail
    problem_patients = ['R2D201001', 'R2D201002', 'R2D201013']
    
    for patient_id in problem_patients:
        if patient_id not in patient_dirs:
            continue
            
        print(f"\n📂 DETAILED ANALYSIS: {patient_id}")
        print("-" * 40)
        
        patient_path = os.path.join(base_audio_dir, patient_id)
        
        # Get all wav files recursively
        wav_files = glob.glob(os.path.join(patient_path, '**/*.wav'), recursive=True)
        
        print(f"Total files found: {len(wav_files)}")
        print(f"First 5 files:")
        for i, file_path in enumerate(wav_files[:5]):
            rel_path = os.path.relpath(file_path, patient_path)
            print(f"  {i+1}. {rel_path}")
        
        # Check for duplicate filenames
        filenames = [os.path.basename(f) for f in wav_files]
        filename_counts = defaultdict(int)
        for filename in filenames:
            filename_counts[filename] += 1
        
        # Find duplicates
        duplicates = {fname: count for fname, count in filename_counts.items() if count > 1}
        if duplicates:
            print(f"⚠️  DUPLICATE FILENAMES FOUND: {len(duplicates)}")
            for fname, count in list(duplicates.items())[:5]:  # Show first 5
                print(f"     {fname}: {count} times")
        else:
            print("✅ No duplicate filenames found")
        
        # Check directory structure
        print(f"Directory structure:")
        for root, dirs, files in os.walk(patient_path):
            level = root.replace(patient_path, '').count(os.sep)
            indent = ' ' * 2 * level
            rel_root = os.path.relpath(root, patient_path)
            if rel_root == '.':
                rel_root = patient_id
            print(f"{indent}{rel_root}/ ({len(files)} files)")
        
        # Check for nested patient directories
        print(f"Checking for nested patient directories:")
        nested_patients = []
        for root, dirs, files in os.walk(patient_path):
            for dirname in dirs:
                if dirname.startswith('R2D2') and dirname != patient_id:
                    nested_patients.append(dirname)
                    nested_path = os.path.join(root, dirname)
                    nested_files = glob.glob(os.path.join(nested_path, '**/*.wav'), recursive=True)
                    print(f"  🚨 Found nested patient {dirname} with {len(nested_files)} files")
        
        if nested_patients:
            print(f"⚠️  This patient contains {len(nested_patients)} other patients' data!")
        else:
            print("✅ No nested patient directories found")
    
    print(f"\n" + "=" * 60)
    print("SUMMARY ANALYSIS")
    print("=" * 60)
    
    # Overall statistics
    total_files = 0
    total_unique_files = 0
    patients_with_nested = 0
    
    all_file_paths = set()
    
    for patient_id in patient_dirs:
        patient_path = os.path.join(base_audio_dir, patient_id)
        wav_files = glob.glob(os.path.join(patient_path, '**/*.wav'), recursive=True)
        
        total_files += len(wav_files)
        
        # Check for nested patients
        has_nested = False
        for root, dirs, files in os.walk(patient_path):
            for dirname in dirs:
                if dirname.startswith('R2D2') and dirname != patient_id:
                    has_nested = True
                    break
            if has_nested:
                break
        
        if has_nested:
            patients_with_nested += 1
        
        # Add to global set to check for duplicates across patients
        for file_path in wav_files:
            all_file_paths.add(file_path)
    
    total_unique_files = len(all_file_paths)
    
    print(f"Total files counted: {total_files}")
    print(f"Unique file paths: {total_unique_files}")
    print(f"Potential duplicates: {total_files - total_unique_files}")
    print(f"Patients with nested data: {patients_with_nested}")
    print(f"Average files per patient: {total_files / len(patient_dirs):.2f}")
    
    if total_files > total_unique_files:
        print(f"🚨 CONFIRMED: {total_files - total_unique_files} files are being double-counted!")
    else:
        print("✅ No double-counting detected")
    
    # Solution recommendation
    print(f"\n📋 RECOMMENDED SOLUTION:")
    print("1. Exclude files that belong to nested patient directories")
    print("2. Only count files that are directly associated with the patient")
    print("3. Create a file ownership mapping to prevent cross-contamination")

if __name__ == "__main__":
    investigate_double_counting()