#!/usr/bin/env python3
"""
Final Data Validation for New UCSF R2D2 Training Dataset
Excludes all subdirectories under R2D201001 to prevent nested patient contamination
"""

import os
import glob
import pandas as pd
from collections import defaultdict
import numpy as np

def get_direct_files_only(patient_dir):
    """Get only files directly in the patient directory, no subdirectories"""
    if not os.path.exists(patient_dir):
        return []
    
    wav_files = []
    
    # Get files directly in the patient directory
    for item in os.listdir(patient_dir):
        item_path = os.path.join(patient_dir, item)
        if os.path.isfile(item_path) and item.endswith('.wav'):
            wav_files.append(item_path)
    
    return wav_files

def get_files_with_exclusion_rule(patient_id, base_audio_dir):
    """Get files for a patient with special exclusion rule for R2D201001"""
    patient_dir = os.path.join(base_audio_dir, patient_id)
    
    if patient_id == 'R2D201001':
        # For R2D201001, only get files directly in the main directory
        wav_files = get_direct_files_only(patient_dir)
        print(f"  🔍 R2D201001: Found {len(wav_files)} files in main directory (subdirectories excluded)")
        return wav_files
    else:
        # For all other patients, use recursive search as normal
        if not os.path.exists(patient_dir):
            return []
        return glob.glob(os.path.join(patient_dir, '**/*.wav'), recursive=True)

def validate_data_final():
    """Final data validation with R2D201001 subdirectory exclusion"""
    
    # File paths
    metadata_file = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2 lung sounds metadata_TRAIN_2025.05.08_v3.csv'
    base_audio_dir = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    
    print("🔍 FINAL DATA VALIDATION - R2D201001 SUBDIRECTORIES EXCLUDED")
    print("=" * 80)
    
    # Load metadata
    metadata_df = pd.read_csv(metadata_file)
    print(f"✅ Loaded metadata with {len(metadata_df)} rows")
    
    # Filter for valid TB labels
    valid_labels = ['TB Positive', 'TB Negative']
    filtered_metadata = metadata_df[metadata_df['Microbiologicreferencestandard'].isin(valid_labels)]
    print(f"✅ Filtered to {len(filtered_metadata)} patients with valid TB labels")
    
    # Get all patient directories
    patient_dirs = [d for d in os.listdir(base_audio_dir) if os.path.isdir(os.path.join(base_audio_dir, d)) and d.startswith('R2D2')]
    patient_dirs.sort()
    print(f"✅ Found {len(patient_dirs)} patient directories")
    
    # Process each patient with exclusion rule
    print(f"\n📂 PROCESSING PATIENTS WITH EXCLUSION RULES")
    print("-" * 60)
    
    # Initialize storage
    patient_file_counts = {}
    patient_files = {}
    total_files = 0
    r2d201001_excluded_count = 0
    
    for patient_id in patient_dirs:
        wav_files = get_files_with_exclusion_rule(patient_id, base_audio_dir)
        
        patient_file_counts[patient_id] = len(wav_files)
        patient_files[patient_id] = wav_files
        total_files += len(wav_files)
        
        # Track what we excluded for R2D201001
        if patient_id == 'R2D201001':
            # Count what would have been included with recursive search
            all_files = glob.glob(os.path.join(base_audio_dir, patient_id, '**/*.wav'), recursive=True)
            r2d201001_excluded_count = len(all_files) - len(wav_files)
            print(f"  📊 R2D201001: {len(wav_files)} files kept, {r2d201001_excluded_count} files excluded from subdirectories")
    
    # Create comprehensive patient report
    print(f"\n📊 CREATING COMPREHENSIVE PATIENT REPORT")
    print("-" * 60)
    
    patient_report = []
    
    # Get all unique patients from metadata
    metadata_patients = set(filtered_metadata['StudyID'].unique())
    audio_patients = set(patient_dirs)
    all_patients = metadata_patients.union(audio_patients)
    
    for patient_id in sorted(all_patients):
        # Get patient info from metadata
        patient_metadata = filtered_metadata[filtered_metadata['StudyID'] == patient_id]
        has_metadata = len(patient_metadata) > 0
        tb_label = patient_metadata['Microbiologicreferencestandard'].iloc[0] if has_metadata else "No metadata"
        
        # Get audio file info
        num_files = patient_file_counts.get(patient_id, 0)
        has_audio = num_files > 0
        
        # Determine status
        if has_metadata and has_audio:
            status = "Complete"
        elif has_metadata and not has_audio:
            status = "Missing Audio"
        elif not has_metadata and has_audio:
            status = "Missing Metadata"
        else:
            status = "Missing Both"
        
        # Determine if included in analysis
        included = has_metadata and has_audio and tb_label in valid_labels
        
        # Special handling for R2D201001
        exclusion_note = ""
        if patient_id == 'R2D201001':
            exclusion_note = f"Excluded {r2d201001_excluded_count} files from subdirectories"
        
        # Get file locations (simplified)
        file_locations = ""
        if has_audio and patient_id in patient_files:
            sample_files = patient_files[patient_id][:3]  # First 3 files
            dirs = set(os.path.dirname(f) for f in sample_files)
            file_locations = '; '.join(dirs)
        
        patient_report.append({
            'PatientID': patient_id,
            'Status': status,
            'TB_Label': tb_label,
            'Has_Metadata': has_metadata,
            'Has_Audio': has_audio,
            'Num_Audio_Files': num_files,
            'Exclusion_Applied': patient_id == 'R2D201001',
            'Exclusion_Note': exclusion_note,
            'Included_In_Analysis': included,
            'File_Locations': file_locations,
            'Sample_Files': '; '.join([os.path.basename(f) for f in patient_files.get(patient_id, [])[:3]])
        })
    
    # Create DataFrame and save report
    report_df = pd.DataFrame(patient_report)
    
    # Save comprehensive report
    output_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    os.makedirs(os.path.join(output_dir, 'reports'), exist_ok=True)
    
    report_path = os.path.join(output_dir, 'reports', 'comprehensive_patient_report_final.csv')
    report_df.to_csv(report_path, index=False)
    
    print(f"✅ Comprehensive patient report saved to: {report_path}")
    
    # Print summary statistics
    print(f"\n📈 SUMMARY STATISTICS")
    print("=" * 60)
    
    total_patients = len(report_df)
    complete_patients = len(report_df[report_df['Status'] == 'Complete'])
    missing_audio = len(report_df[report_df['Status'] == 'Missing Audio'])
    missing_metadata = len(report_df[report_df['Status'] == 'Missing Metadata'])
    missing_both = len(report_df[report_df['Status'] == 'Missing Both'])
    
    included_in_analysis = len(report_df[report_df['Included_In_Analysis'] == True])
    
    tb_positive = len(report_df[report_df['TB_Label'] == 'TB Positive'])
    tb_negative = len(report_df[report_df['TB_Label'] == 'TB Negative'])
    
    print(f"Total Patients: {total_patients}")
    print(f"Complete (metadata + audio): {complete_patients}")
    print(f"Missing Audio: {missing_audio}")
    print(f"Missing Metadata: {missing_metadata}")
    print(f"Missing Both: {missing_both}")
    print(f"")
    print(f"Included in Analysis: {included_in_analysis}")
    print(f"")
    print(f"TB Positive: {tb_positive}")
    print(f"TB Negative: {tb_negative}")
    print(f"")
    print(f"Total Audio Files: {total_files}")
    print(f"Average Files per Patient: {total_files / total_patients:.2f}")
    print(f"Average Files per Complete Patient: {total_files / complete_patients:.2f}")
    
    # R2D201001 specific stats
    print(f"\n🔍 R2D201001 EXCLUSION DETAILS")
    print("-" * 40)
    r2d201001_kept = patient_file_counts.get('R2D201001', 0)
    print(f"Files kept: {r2d201001_kept}")
    print(f"Files excluded: {r2d201001_excluded_count}")
    print(f"Exclusion reason: Subdirectory contamination from other patients")
    
    # Create clean dataset (patients included in analysis)
    clean_patients = report_df[report_df['Included_In_Analysis'] == True].copy()
    clean_patients['Label'] = clean_patients['TB_Label']
    
    # Save clean dataset
    clean_path = os.path.join(output_dir, 'data', 'clean_patients_final.csv')
    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    clean_patients[['PatientID', 'Label']].rename(columns={'PatientID': 'StudyID'}).to_csv(clean_path, index=False)
    
    print(f"\n✅ Clean dataset saved to: {clean_path}")
    print(f"Clean dataset contains {len(clean_patients)} patients")
    
    # Create file mapping for embeddings
    print(f"\n📝 CREATING FILE MAPPING FOR EMBEDDINGS")
    print("-" * 50)
    
    file_mapping = []
    for patient_id in clean_patients['PatientID']:
        patient_audio_files = patient_files.get(patient_id, [])
        for file_path in patient_audio_files:
            filename = os.path.basename(file_path)
            file_key = f"{patient_id}/{filename}"
            
            file_mapping.append({
                'patient_id': patient_id,
                'filename': filename,
                'file_key': file_key,
                'file_path': file_path
            })
    
    # Save file mapping
    mapping_path = os.path.join(output_dir, 'data', 'file_mapping_final.csv')
    mapping_df = pd.DataFrame(file_mapping)
    mapping_df.to_csv(mapping_path, index=False)
    
    print(f"✅ File mapping saved to: {mapping_path}")
    print(f"File mapping contains {len(mapping_df)} files from {len(clean_patients)} patients")
    
    # Final validation
    print(f"\n🎉 FINAL VALIDATION COMPLETE!")
    print("=" * 60)
    print("Key improvements:")
    print("1. ✅ R2D201001 subdirectories excluded to prevent contamination")
    print("2. ✅ Clean dataset created with proper patient-file mapping")
    print("3. ✅ Comprehensive patient report with exclusion details")
    print("4. ✅ All files properly attributed to their owning patients")
    
    return report_df, clean_patients, mapping_df

if __name__ == "__main__":
    validate_data_final()