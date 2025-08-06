#!/usr/bin/env python3
"""
Generate HeAR embeddings for all complete patients in the fixed dataset
Based on 01_data_processing/quick_embedding_generation.py
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

def setup_hear_model():
    """Set up the HeAR model for embedding generation"""
    print("🔄 Loading HeAR model...")
    
    from huggingface_hub import snapshot_download
    from keras.layers import TFSMLayer
    
    # Download model from Hugging Face
    model_path = snapshot_download("google/hear")
    
    # Load the SavedModel using TFSMLayer
    loaded_model = TFSMLayer(model_path, call_endpoint="serving_default")
    
    # Create inference function
    infer_func = lambda audio_batch: loaded_model(audio_batch)
    
    print("✅ HeAR model loaded successfully")
    return infer_func

def process_audio_file(file_path, infer_func, sample_rate=16000, clip_duration=2, 
                      clip_overlap_percent=10, silence_threshold_db=-50):
    """Process a single audio file and generate embeddings"""
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
        
        if len(audio) == 0:
            print(f"⚠️  Empty audio file: {file_path}")
            return []
        
        # Calculate clip parameters
        clip_samples = int(clip_duration * sample_rate)
        hop_samples = int(clip_samples * (1 - clip_overlap_percent / 100))
        
        # Generate clips
        clips = []
        start = 0
        while start + clip_samples <= len(audio):
            clip = audio[start:start + clip_samples]
            
            # Check if clip is not mostly silent
            clip_power_db = 10 * np.log10(np.mean(clip**2) + 1e-10)
            if clip_power_db > silence_threshold_db:
                clips.append(clip)
            
            start += hop_samples
        
        # Handle remaining audio if file is shorter than clip duration
        if len(clips) == 0 and len(audio) > 0:
            # Pad or crop to exact clip duration
            if len(audio) < clip_samples:
                clip = np.pad(audio, (0, clip_samples - len(audio)), 'constant')
            else:
                clip = audio[:clip_samples]
            
            # Check silence threshold
            clip_power_db = 10 * np.log10(np.mean(clip**2) + 1e-10)
            if clip_power_db > silence_threshold_db:
                clips.append(clip)
        
        if len(clips) == 0:
            print(f"⚠️  No valid clips found in: {file_path}")
            return []
        
        # Generate embeddings for all clips
        embeddings = []
        
        # Process clips in batches for efficiency
        batch_size = 32
        for i in range(0, len(clips), batch_size):
            batch_clips = clips[i:i + batch_size]
            
            # Stack clips into batch
            batch_array = np.stack(batch_clips, axis=0)
            
            # Generate embeddings using HeAR model
            batch_embeddings = infer_func(batch_array)
            
            # Extract embeddings (HeAR returns dict with 'output_0' key)
            if isinstance(batch_embeddings, dict):
                batch_embeddings = batch_embeddings['output_0']
            
            # Convert to numpy and add to list
            batch_embeddings_np = np.array(batch_embeddings)
            for j in range(len(batch_clips)):
                embeddings.append(batch_embeddings_np[j])
        
        return embeddings
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return []

def generate_embeddings_for_dataset():
    """Generate embeddings for all complete patients"""
    
    # Set up paths
    base_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    # Load file mapping (corrected version)
    file_mapping_path = os.path.join(base_dir, 'data', 'file_mapping_fixed.csv')
    clean_patients_path = os.path.join(base_dir, 'data', 'clean_patients_fixed.csv')
    
    print("📂 Loading file mapping and patient data...")
    file_mapping_df = pd.read_csv(file_mapping_path)
    clean_patients_df = pd.read_csv(clean_patients_path)
    
    print(f"✅ Loaded {len(file_mapping_df)} files from {len(clean_patients_df)} patients")
    
    # Set up HeAR model
    infer_func = setup_hear_model()
    
    # Prepare output paths
    output_dir = os.path.join(base_dir, 'data')
    embeddings_path = os.path.join(output_dir, 'complete_embeddings.npz')
    metadata_path = os.path.join(output_dir, 'complete_embeddings_metadata.csv')
    
    # Initialize storage
    embeddings_dict = {}
    metadata_records = []
    
    # Process files by patient for better organization
    print(f"\n🎯 Processing {len(clean_patients_df)} patients...")
    
    patient_summary = []
    total_files = 0
    total_clips = 0
    
    for idx, patient_row in tqdm(clean_patients_df.iterrows(), total=len(clean_patients_df), desc="Processing patients"):
        patient_id = patient_row['StudyID']
        
        # Get files for this patient
        patient_files = file_mapping_df[file_mapping_df['patient_id'] == patient_id]
        
        patient_clips = 0
        patient_processed_files = 0
        
        for _, file_row in patient_files.iterrows():
            file_path = file_row['file_path']
            filename = file_row['filename']
            file_key = file_row['file_key']
            
            # Check if file exists
            if not os.path.exists(file_path):
                print(f"⚠️  File not found: {file_path}")
                continue
            
            # Generate embeddings
            embeddings = process_audio_file(file_path, infer_func)
            
            if embeddings:
                # Store embeddings
                embeddings_dict[file_key] = np.array(embeddings)
                
                # Store metadata
                metadata_records.append({
                    'patient_id': patient_id,
                    'filename': filename,
                    'file_key': file_key,
                    'file_path': file_path,
                    'num_clips': len(embeddings)
                })
                
                patient_clips += len(embeddings)
                patient_processed_files += 1
                total_files += 1
                total_clips += len(embeddings)
        
        # Patient summary
        patient_summary.append({
            'patient_id': patient_id,
            'tb_label': patient_row['Label'],
            'total_files': len(patient_files),
            'processed_files': patient_processed_files,
            'total_clips': patient_clips
        })
        
        # Progress update every 50 patients
        if (idx + 1) % 50 == 0:
            print(f"✅ Processed {idx + 1}/{len(clean_patients_df)} patients")
            print(f"   Files processed: {total_files}, Clips generated: {total_clips}")
    
    # Save embeddings
    print(f"\n💾 Saving embeddings...")
    np.savez_compressed(embeddings_path, **embeddings_dict)
    print(f"✅ Embeddings saved to: {embeddings_path}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Save patient summary
    summary_df = pd.DataFrame(patient_summary)
    summary_path = os.path.join(base_dir, 'results', 'patient_processing_summary.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    print(f"✅ Patient summary saved to: {summary_path}")
    
    # Final statistics
    print(f"\n🎉 EMBEDDING GENERATION COMPLETE!")
    print("=" * 50)
    print(f"Total Patients: {len(clean_patients_df)}")
    print(f"Total Files Processed: {total_files}")
    print(f"Total Clips Generated: {total_clips}")
    print(f"Average Files per Patient: {total_files / len(clean_patients_df):.2f}")
    print(f"Average Clips per Patient: {total_clips / len(clean_patients_df):.2f}")
    print(f"Average Clips per File: {total_clips / total_files:.2f}")
    
    # TB distribution
    tb_positive = len(summary_df[summary_df['tb_label'] == 'TB Positive'])
    tb_negative = len(summary_df[summary_df['tb_label'] == 'TB Negative'])
    print(f"\nTB Distribution:")
    print(f"  TB Positive: {tb_positive} ({tb_positive/len(clean_patients_df)*100:.1f}%)")
    print(f"  TB Negative: {tb_negative} ({tb_negative/len(clean_patients_df)*100:.1f}%)")
    
    # Performance statistics
    avg_clips_positive = summary_df[summary_df['tb_label'] == 'TB Positive']['total_clips'].mean()
    avg_clips_negative = summary_df[summary_df['tb_label'] == 'TB Negative']['total_clips'].mean()
    print(f"\nAverage Clips per Patient:")
    print(f"  TB Positive: {avg_clips_positive:.2f}")
    print(f"  TB Negative: {avg_clips_negative:.2f}")
    
    return embeddings_path, metadata_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Generate HeAR embeddings for complete dataset')
    parser.add_argument('--force', action='store_true', help='Force regeneration even if files exist')
    args = parser.parse_args()
    
    # Check if embeddings already exist
    base_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    embeddings_path = os.path.join(base_dir, 'data', 'complete_embeddings.npz')
    
    if os.path.exists(embeddings_path) and not args.force:
        print(f"✅ Embeddings already exist at: {embeddings_path}")
        print("Use --force to regenerate")
        return
    
    print("🎯 COMPLETE EMBEDDINGS GENERATION")
    print("=" * 50)
    
    # Generate embeddings
    embeddings_path, metadata_path, summary_path = generate_embeddings_for_dataset()
    
    print(f"\n📁 Output files:")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Summary: {summary_path}")

if __name__ == "__main__":
    main()