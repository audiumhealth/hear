#!/usr/bin/env python3
"""
Update comprehensive patient report with embedding generation status and analysis inclusion details
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def update_comprehensive_patient_report():
    """Update the comprehensive patient report with embedding and analysis status"""
    
    base_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    print("🔄 UPDATING COMPREHENSIVE PATIENT REPORT")
    print("=" * 60)
    
    # Load existing comprehensive patient report
    report_path = os.path.join(base_dir, 'reports', 'comprehensive_patient_report_final.csv')
    
    if not os.path.exists(report_path):
        print(f"❌ Base report not found: {report_path}")
        print("Please run data_validation_final.py first")
        return
    
    report_df = pd.read_csv(report_path)
    print(f"✅ Loaded base report with {len(report_df)} patients")
    
    # Load file mapping
    mapping_path = os.path.join(base_dir, 'data', 'file_mapping_final.csv')
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        print(f"✅ Loaded file mapping with {len(mapping_df)} files")
    else:
        mapping_df = pd.DataFrame()
        print("⚠️  File mapping not found")
    
    # Load clean patients dataset
    clean_patients_path = os.path.join(base_dir, 'data', 'clean_patients_final.csv')
    if os.path.exists(clean_patients_path):
        clean_patients_df = pd.read_csv(clean_patients_path)
        print(f"✅ Loaded clean patients dataset with {len(clean_patients_df)} patients")
    else:
        clean_patients_df = pd.DataFrame()
        print("⚠️  Clean patients dataset not found")
    
    # Check for embeddings
    embeddings_path = os.path.join(base_dir, 'data', 'final_embeddings.npz')
    embeddings_metadata_path = os.path.join(base_dir, 'data', 'final_embeddings_metadata.csv')
    
    embeddings_exist = os.path.exists(embeddings_path)
    embeddings_metadata_exist = os.path.exists(embeddings_metadata_path)
    
    print(f"📊 Embeddings file exists: {embeddings_exist}")
    print(f"📊 Embeddings metadata exists: {embeddings_metadata_exist}")
    
    # Load embeddings information if available
    embedding_patients = set()
    embedding_file_counts = {}
    embedding_clip_counts = {}
    
    if embeddings_exist:
        try:
            embeddings_data = np.load(embeddings_path, allow_pickle=True)
            print(f"✅ Loaded embeddings for {len(embeddings_data.files)} files")
            
            # Extract patient info from embeddings
            for file_key in embeddings_data.files:
                patient_id = file_key.split('/')[0]
                embedding_patients.add(patient_id)
                
                # Count files per patient
                if patient_id not in embedding_file_counts:
                    embedding_file_counts[patient_id] = 0
                    embedding_clip_counts[patient_id] = 0
                
                embedding_file_counts[patient_id] += 1
                
                # Count clips
                embeddings = embeddings_data[file_key]
                embedding_clip_counts[patient_id] += len(embeddings)
                
        except Exception as e:
            print(f"⚠️  Error loading embeddings: {e}")
    
    # Load embeddings metadata if available
    if embeddings_metadata_exist:
        try:
            embeddings_metadata_df = pd.read_csv(embeddings_metadata_path)
            print(f"✅ Loaded embeddings metadata for {len(embeddings_metadata_df)} files")
        except Exception as e:
            print(f"⚠️  Error loading embeddings metadata: {e}")
            embeddings_metadata_df = pd.DataFrame()
    else:
        embeddings_metadata_df = pd.DataFrame()
    
    # Check for analysis results
    analysis_results_path = os.path.join(base_dir, 'results', 'final_analysis_results.csv')
    analysis_exists = os.path.exists(analysis_results_path)
    print(f"📊 Analysis results exist: {analysis_exists}")
    
    # Update the report with new information
    print(f"\n🔄 Updating patient report with embedding and analysis status...")
    
    # Add new columns
    report_df['Embeddings_Generated'] = False
    report_df['Embeddings_Files_Count'] = 0
    report_df['Embeddings_Clips_Count'] = 0
    report_df['Analysis_Ready'] = False
    report_df['Analysis_Included'] = False
    report_df['Exclusion_Reason'] = ''
    report_df['Last_Updated'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Update status for each patient
    for idx, row in report_df.iterrows():
        patient_id = row['PatientID']
        
        # Check if embeddings were generated
        if patient_id in embedding_patients:
            report_df.at[idx, 'Embeddings_Generated'] = True
            report_df.at[idx, 'Embeddings_Files_Count'] = embedding_file_counts.get(patient_id, 0)
            report_df.at[idx, 'Embeddings_Clips_Count'] = embedding_clip_counts.get(patient_id, 0)
        
        # Check if patient is analysis ready (in clean dataset)
        if not clean_patients_df.empty:
            analysis_ready = patient_id in clean_patients_df['StudyID'].values
            report_df.at[idx, 'Analysis_Ready'] = analysis_ready
            
            # If analysis was run, check if this patient was included
            if analysis_exists and analysis_ready:
                report_df.at[idx, 'Analysis_Included'] = True
            elif analysis_exists and not analysis_ready:
                report_df.at[idx, 'Analysis_Included'] = False
        
        # Determine exclusion reasons
        exclusion_reasons = []
        
        if row['Status'] == 'Missing Audio':
            exclusion_reasons.append('No audio files')
        elif row['Status'] == 'Missing Metadata':
            exclusion_reasons.append('No metadata')
        elif row['Status'] == 'Missing Both':
            exclusion_reasons.append('No audio files, No metadata')
        
        if row['TB_Label'] not in ['TB Positive', 'TB Negative']:
            exclusion_reasons.append('Invalid TB label')
        
        if patient_id == 'R2D201001' and row['Exclusion_Applied']:
            exclusion_reasons.append('Subdirectory contamination excluded')
        
        if not report_df.at[idx, 'Embeddings_Generated'] and report_df.at[idx, 'Analysis_Ready']:
            exclusion_reasons.append('Embeddings not generated')
        
        report_df.at[idx, 'Exclusion_Reason'] = '; '.join(exclusion_reasons) if exclusion_reasons else 'None'
    
    # Calculate summary statistics
    total_patients = len(report_df)
    complete_patients = len(report_df[report_df['Status'] == 'Complete'])
    analysis_ready = len(report_df[report_df['Analysis_Ready'] == True])
    embeddings_generated = len(report_df[report_df['Embeddings_Generated'] == True])
    analysis_included = len(report_df[report_df['Analysis_Included'] == True])
    
    print(f"\n📊 UPDATED REPORT STATISTICS")
    print("-" * 40)
    print(f"Total Patients: {total_patients}")
    print(f"Complete Patients: {complete_patients}")
    print(f"Analysis Ready: {analysis_ready}")
    print(f"Embeddings Generated: {embeddings_generated}")
    print(f"Analysis Included: {analysis_included}")
    
    # Save updated report
    updated_report_path = os.path.join(base_dir, 'reports', 'comprehensive_patient_report_final.csv')
    report_df.to_csv(updated_report_path, index=False)
    print(f"\n✅ Updated report saved to: {updated_report_path}")
    
    # Create summary reports
    print(f"\n📋 CREATING SUMMARY REPORTS")
    print("-" * 40)
    
    # 1. Patients missing embeddings
    missing_embeddings = report_df[
        (report_df['Analysis_Ready'] == True) & 
        (report_df['Embeddings_Generated'] == False)
    ].copy()
    
    if len(missing_embeddings) > 0:
        missing_embeddings_path = os.path.join(base_dir, 'reports', 'patients_missing_embeddings.csv')
        missing_embeddings[['PatientID', 'TB_Label', 'Num_Audio_Files', 'Exclusion_Reason']].to_csv(
            missing_embeddings_path, index=False
        )
        print(f"⚠️  {len(missing_embeddings)} patients missing embeddings - saved to patients_missing_embeddings.csv")
    else:
        print("✅ All analysis-ready patients have embeddings generated")
    
    # 2. Patients excluded from analysis
    excluded_patients = report_df[report_df['Analysis_Ready'] == False].copy()
    excluded_summary_path = os.path.join(base_dir, 'reports', 'patients_excluded_from_analysis.csv')
    excluded_patients[['PatientID', 'Status', 'TB_Label', 'Exclusion_Reason']].to_csv(
        excluded_summary_path, index=False
    )
    print(f"📊 {len(excluded_patients)} patients excluded from analysis - saved to patients_excluded_from_analysis.csv")
    
    # 3. Analysis summary
    analysis_summary = {
        'Total_Patients': total_patients,
        'Complete_Patients': complete_patients,
        'Analysis_Ready': analysis_ready,
        'Embeddings_Generated': embeddings_generated,
        'Analysis_Included': analysis_included,
        'Missing_Embeddings': len(missing_embeddings),
        'Excluded_Patients': len(excluded_patients),
        'TB_Positive_Included': len(report_df[(report_df['Analysis_Included'] == True) & (report_df['TB_Label'] == 'TB Positive')]),
        'TB_Negative_Included': len(report_df[(report_df['Analysis_Included'] == True) & (report_df['TB_Label'] == 'TB Negative')]),
        'Last_Updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    analysis_summary_df = pd.DataFrame([analysis_summary])
    analysis_summary_path = os.path.join(base_dir, 'reports', 'analysis_summary.csv')
    analysis_summary_df.to_csv(analysis_summary_path, index=False)
    print(f"📊 Analysis summary saved to analysis_summary.csv")
    
    # 4. Embedding generation progress
    if embeddings_generated > 0:
        embedding_progress = report_df[report_df['Embeddings_Generated'] == True].copy()
        embedding_progress_path = os.path.join(base_dir, 'reports', 'embedding_generation_progress.csv')
        embedding_progress[['PatientID', 'TB_Label', 'Num_Audio_Files', 'Embeddings_Files_Count', 'Embeddings_Clips_Count']].to_csv(
            embedding_progress_path, index=False
        )
        print(f"✅ Embedding progress saved to embedding_generation_progress.csv")
    
    print(f"\n🎉 COMPREHENSIVE PATIENT REPORT UPDATE COMPLETE!")
    print("=" * 60)
    print("Updated files:")
    print(f"  📊 Main report: comprehensive_patient_report_final.csv")
    print(f"  📊 Missing embeddings: patients_missing_embeddings.csv")
    print(f"  📊 Excluded patients: patients_excluded_from_analysis.csv")
    print(f"  📊 Analysis summary: analysis_summary.csv")
    print(f"  📊 Embedding progress: embedding_generation_progress.csv")
    
    return report_df

if __name__ == "__main__":
    update_comprehensive_patient_report()