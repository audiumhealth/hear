#!/usr/bin/env python3
"""
Generate HeAR embeddings for New UCSF R2D2 Training Dataset
Based on the existing embedding generation code from 01_data_processing/
"""

import os
import pandas as pd
import numpy as np
import librosa
import warnings
import glob
from tqdm import tqdm
import argparse

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, module="soundfile")
warnings.filterwarnings("ignore", module="librosa")

def load_hear_model():
    """Load the HeAR model from Hugging Face"""
    print("Loading HeAR model from Hugging Face...")
    
    from huggingface_hub import snapshot_download
    from keras.layers import TFSMLayer
    
    # Download model from Hugging Face
    model_path = snapshot_download("google/hear")
    
    # Load the SavedModel using TFSMLayer
    loaded_model = TFSMLayer(model_path, call_endpoint="serving_default")
    
    # Create inference function
    infer = lambda audio_batch: loaded_model(audio_batch)
    
    print("✅ HeAR model loaded successfully")
    return infer

def find_wav_files_for_patient(patient_id, base_audio_dir):
    """Find all wav files for a given patient, handling nested directory structure"""
    patient_dir = os.path.join(base_audio_dir, patient_id)
    if not os.path.exists(patient_dir):
        return []
    
    # Use recursive search to find all wav files
    wav_files = glob.glob(os.path.join(patient_dir, '**/*.wav'), recursive=True)
    return wav_files

def process_audio_file(file_path, infer_func, sample_rate=16000, clip_duration=2, 
                      clip_overlap_percent=10, silence_threshold_db=-50):
    """Process a single audio file and generate embeddings"""
    
    clip_length = sample_rate * clip_duration
    
    try:
        # Load audio file
        audio, sr = librosa.load(file_path, sr=sample_rate, mono=True)
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None
    
    # Segment into clips
    clip_batch = []
    overlap_samples = int(clip_length * (clip_overlap_percent / 100))
    step_size = clip_length - overlap_samples
    num_clips = max(1, (len(audio) - overlap_samples) // step_size)
    
    clips_kept = 0
    
    for i in range(num_clips):
        start_sample = i * step_size
        end_sample = start_sample + clip_length
        clip = audio[start_sample:end_sample]
        
        # Pad clip if needed
        if end_sample > len(audio):
            clip = np.pad(clip, (0, clip_length - len(clip)), 'constant')
        
        # Calculate loudness
        power = np.mean(clip**2)
        rms_loudness = -np.inf if power == 0 else round(20 * np.log10(np.sqrt(power)))
        
        # Skip if too quiet
        if rms_loudness < silence_threshold_db:
            continue
        
        clip_batch.append(clip)
        clips_kept += 1
    
    # Generate embeddings if we have valid clips
    if len(clip_batch) == 0:
        print(f"⚠️  No valid clips found for {os.path.basename(file_path)}")
        return None
    
    try:
        clip_batch = np.asarray(clip_batch)
        embedding_batch = infer_func(clip_batch)['output_0'].numpy()
        return embedding_batch
    except Exception as e:
        print(f"❌ Error generating embeddings for {file_path}: {e}")
        return None

def generate_embeddings_for_patients(patient_list, base_audio_dir, output_dir, test_mode=False):
    """Generate embeddings for a list of patients"""
    
    # Load HeAR model
    infer_func = load_hear_model()
    
    # Storage for embeddings and metadata
    all_embeddings = {}
    metadata_list = []
    
    # Statistics
    total_files = 0
    total_clips = 0
    failed_files = 0
    
    print(f"\n🔄 Processing {len(patient_list)} patients...")
    
    for patient_id in tqdm(patient_list, desc="Processing patients"):
        # Find all wav files for this patient
        wav_files = find_wav_files_for_patient(patient_id, base_audio_dir)
        
        if not wav_files:
            print(f"⚠️  No audio files found for patient {patient_id}")
            continue
        
        patient_embeddings = []
        
        print(f"\\n👤 Processing patient {patient_id} ({len(wav_files)} files)")
        
        for file_path in tqdm(wav_files, desc=f"Files for {patient_id}", leave=False):
            total_files += 1
            filename = os.path.basename(file_path)
            
            # Generate embeddings for this file
            embeddings = process_audio_file(file_path, infer_func)
            
            if embeddings is None:
                failed_files += 1
                continue
            
            # Store embeddings with unique key
            file_key = f"{patient_id}/{filename}"
            all_embeddings[file_key] = embeddings
            
            # Store metadata
            metadata_list.append({
                'patient_id': patient_id,
                'filename': filename,
                'file_key': file_key,
                'file_path': file_path,
                'num_clips': len(embeddings)
            })
            
            total_clips += len(embeddings)
            
            # For test mode, limit processing
            if test_mode and total_files >= 20:
                print(f"\\n🛑 Test mode: Stopping after {total_files} files")
                break
        
        if test_mode and total_files >= 20:
            break
    
    # Save embeddings
    embeddings_path = os.path.join(output_dir, 'data', 'new_ucsf_embeddings.npz')
    np.savez_compressed(embeddings_path, **all_embeddings)
    print(f"\\n✅ Embeddings saved to: {embeddings_path}")
    
    # Save metadata
    metadata_df = pd.DataFrame(metadata_list)
    metadata_path = os.path.join(output_dir, 'data', 'new_ucsf_embeddings_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✅ Metadata saved to: {metadata_path}")
    
    # Print summary
    print(f"\\n📊 PROCESSING SUMMARY:")
    print(f"  Patients processed: {len(patient_list)}")
    print(f"  Total files processed: {total_files}")
    print(f"  Failed files: {failed_files}")
    print(f"  Total clips generated: {total_clips}")
    print(f"  Average clips per file: {total_clips/max(1, total_files-failed_files):.2f}")
    
    return embeddings_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description='Generate HeAR embeddings for New UCSF R2D2 dataset')
    parser.add_argument('--dataset', choices=['small', 'medium', 'mini', 'all'], default='small',
                       help='Dataset size to process')
    parser.add_argument('--test-mode', action='store_true',
                       help='Test mode: process only first 20 files')
    
    args = parser.parse_args()
    
    # Set up paths
    base_audio_dir = '/Users/abelvillcaroque/data/Audium/202504_UCSF_New_Trainig_set(R2D2)/R2D2_Train_Data'
    output_dir = '/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set(R2D2)'
    
    # Load patient list based on dataset size
    if args.dataset == 'small':
        patient_file = 'clean_small_test_patients.csv'
    elif args.dataset == 'medium':
        patient_file = 'clean_medium_test_patients.csv'
    elif args.dataset == 'mini':
        patient_file = 'mini_test_patients.csv'
    else:  # all
        patient_file = 'all_clean_patients.csv'
    
    patient_list_path = os.path.join(output_dir, 'data', patient_file)
    
    if not os.path.exists(patient_list_path):
        print(f"❌ Patient list file not found: {patient_list_path}")
        print("Please run create_clean_test_dataset.py first")
        return
    
    # Load patient list
    patients_df = pd.read_csv(patient_list_path)
    patient_list = patients_df['StudyID'].tolist()
    
    print(f"🎯 Processing {args.dataset} dataset: {len(patient_list)} patients")
    if args.test_mode:
        print("🧪 Running in test mode")
    
    # Generate embeddings
    embeddings_path, metadata_path = generate_embeddings_for_patients(
        patient_list, base_audio_dir, output_dir, test_mode=args.test_mode
    )
    
    print(f"\\n🎉 Embedding generation complete!")
    print(f"Next steps:")
    print(f"  1. Run analysis on the generated embeddings")
    print(f"  2. Train and evaluate TB detection models")
    print(f"  3. Compare with existing pipeline results")

if __name__ == "__main__":
    main()