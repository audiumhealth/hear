#!/usr/bin/env python3
"""
Generate HeAR embeddings for the final corrected dataset
Uses the clean dataset with R2D201001 subdirectories excluded
"""

import os
import sys
import numpy as np
import pandas as pd
import librosa
import time
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
            return []
        
        # Generate embeddings for all clips
        embeddings = []
        
        # Process clips in batches for efficiency
        batch_size = 16  # Reduced batch size to avoid memory issues
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

def generate_embeddings_batch(start_patient=0, batch_size=50, save_every=25):
    """Generate embeddings in batches with regular saving"""
    
    # Set up paths
    base_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    # Load file mapping (final corrected version)
    file_mapping_path = os.path.join(base_dir, 'data', 'file_mapping_final.csv')
    clean_patients_path = os.path.join(base_dir, 'data', 'clean_patients_final.csv')
    
    print("📂 Loading file mapping and patient data...")
    file_mapping_df = pd.read_csv(file_mapping_path)
    clean_patients_df = pd.read_csv(clean_patients_path)
    
    print(f"✅ Loaded {len(file_mapping_df)} files from {len(clean_patients_df)} patients")
    
    # Set up HeAR model
    infer_func = setup_hear_model()
    
    # Prepare output paths
    output_dir = os.path.join(base_dir, 'data')
    embeddings_path = os.path.join(output_dir, 'final_embeddings.npz')
    metadata_path = os.path.join(output_dir, 'final_embeddings_metadata.csv')
    
    # Initialize or load existing data
    embeddings_dict = {}
    metadata_records = []
    
    # Check if we have existing embeddings to continue from
    if os.path.exists(embeddings_path):
        print("📂 Loading existing embeddings...")
        existing_embeddings = np.load(embeddings_path, allow_pickle=True)
        embeddings_dict = {key: existing_embeddings[key] for key in existing_embeddings.files}
        print(f"✅ Loaded {len(embeddings_dict)} existing embeddings")
    
    if os.path.exists(metadata_path):
        print("📂 Loading existing metadata...")
        existing_metadata = pd.read_csv(metadata_path)
        metadata_records = existing_metadata.to_dict('records')
        print(f"✅ Loaded {len(metadata_records)} existing metadata records")
    
    # Process patients in batches
    total_patients = len(clean_patients_df)
    end_patient = min(start_patient + batch_size, total_patients)
    
    print(f"\n🎯 Processing patients {start_patient+1}-{end_patient} of {total_patients}...")
    
    patient_summary = []
    total_files = 0
    total_clips = 0
    
    start_time = time.time()
    
    for idx in range(start_patient, end_patient):
        patient_row = clean_patients_df.iloc[idx]
        patient_id = patient_row['StudyID']
        
        # Get files for this patient
        patient_files = file_mapping_df[file_mapping_df['patient_id'] == patient_id]
        
        patient_clips = 0
        patient_processed_files = 0
        
        print(f"  📁 Processing patient {patient_id} ({idx+1}/{total_patients})")
        
        for _, file_row in patient_files.iterrows():
            file_path = file_row['file_path']
            filename = file_row['filename']
            file_key = file_row['file_key']
            
            # Skip if already processed
            if file_key in embeddings_dict:
                continue
            
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
        
        # Save progress every save_every patients
        if (idx - start_patient + 1) % save_every == 0:
            print(f"  💾 Saving progress... ({idx+1}/{total_patients})")
            
            # Save embeddings
            np.savez_compressed(embeddings_path, **embeddings_dict)
            
            # Save metadata
            metadata_df = pd.DataFrame(metadata_records)
            metadata_df.to_csv(metadata_path, index=False)
            
            elapsed_time = time.time() - start_time
            avg_time_per_patient = elapsed_time / (idx - start_patient + 1)
            remaining_patients = total_patients - idx - 1
            eta = avg_time_per_patient * remaining_patients
            
            print(f"  ⏱️  Elapsed: {elapsed_time/60:.1f}m, ETA: {eta/60:.1f}m")
            print(f"  📊 Progress: {total_files} files, {total_clips} clips generated")
    
    # Final save
    print(f"\n💾 Saving final results...")
    np.savez_compressed(embeddings_path, **embeddings_dict)
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_records)
    metadata_df.to_csv(metadata_path, index=False)
    
    # Save patient summary
    summary_df = pd.DataFrame(patient_summary)
    summary_path = os.path.join(base_dir, 'results', 'patient_processing_summary_final.csv')
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    
    # Final statistics
    elapsed_time = time.time() - start_time
    print(f"\n🎉 BATCH PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Processed Patients: {end_patient - start_patient}")
    print(f"Total Files Processed: {total_files}")
    print(f"Total Clips Generated: {total_clips}")
    print(f"Total Time: {elapsed_time/60:.1f} minutes")
    print(f"Average Time per Patient: {elapsed_time/(end_patient - start_patient):.1f} seconds")
    
    return embeddings_path, metadata_path, summary_path

def main():
    parser = argparse.ArgumentParser(description='Generate HeAR embeddings for final corrected dataset')
    parser.add_argument('--start', type=int, default=0, help='Start patient index')
    parser.add_argument('--batch_size', type=int, default=50, help='Number of patients to process')
    parser.add_argument('--save_every', type=int, default=25, help='Save progress every N patients')
    
    args = parser.parse_args()
    
    print("🎯 FINAL EMBEDDINGS GENERATION")
    print("=" * 50)
    print("Using corrected dataset with R2D201001 subdirectories excluded")
    print("")
    
    # Generate embeddings
    embeddings_path, metadata_path, summary_path = generate_embeddings_batch(
        start_patient=args.start,
        batch_size=args.batch_size,
        save_every=args.save_every
    )
    
    print(f"\n📁 Output files:")
    print(f"  Embeddings: {embeddings_path}")
    print(f"  Metadata: {metadata_path}")
    print(f"  Summary: {summary_path}")

if __name__ == "__main__":
    main()