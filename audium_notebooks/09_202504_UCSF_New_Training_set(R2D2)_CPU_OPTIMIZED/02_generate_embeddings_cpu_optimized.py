#!/usr/bin/env python3
"""
CPU-Optimized HeAR Embeddings Generation for UCSF R2D2 Dataset
Multi-core parallel processing for faster embedding generation
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
import multiprocessing
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import psutil
warnings.filterwarnings('ignore')

# Import configuration
from config import load_config_from_args, get_common_parser

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

def process_audio_file_optimized(args_tuple):
    """
    Optimized audio processing function for multiprocessing
    Args: (file_path, patient_id, file_id, model_path, processing_params)
    """
    file_path, patient_id, file_id, model_path, params = args_tuple
    
    try:
        # Load model in each process (TensorFlow models need process-local loading)
        from huggingface_hub import snapshot_download
        from keras.layers import TFSMLayer
        
        # Load model locally in this process
        loaded_model = TFSMLayer(model_path, call_endpoint="serving_default")
        infer_func = lambda audio_batch: loaded_model(audio_batch)
        
        # Process audio file
        embeddings_data = process_audio_file(
            file_path, infer_func, 
            sample_rate=params['sample_rate'],
            clip_duration=params['clip_duration'],
            clip_overlap_percent=params['clip_overlap_percent'],
            silence_threshold_db=params['silence_threshold_db']
        )
        
        return {
            'patient_id': patient_id,
            'file_id': file_id,
            'file_path': file_path,
            'embeddings': embeddings_data,
            'num_clips': len(embeddings_data),
            'status': 'success'
        }
        
    except Exception as e:
        return {
            'patient_id': patient_id,
            'file_id': file_id,
            'file_path': file_path,
            'embeddings': [],
            'num_clips': 0,
            'status': 'error',
            'error': str(e)
        }

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
        
        # Extract overlapping clips
        clips = []
        for start_idx in range(0, len(audio) - clip_samples + 1, hop_samples):
            end_idx = start_idx + clip_samples
            clip = audio[start_idx:end_idx]
            
            # Apply silence threshold
            clip_rms = librosa.feature.rms(y=clip)[0]
            clip_db = librosa.amplitude_to_db(clip_rms)
            
            if np.mean(clip_db) > silence_threshold_db:
                clips.append(clip)
        
        if not clips:
            return []
        
        # Batch process clips through HeAR model
        clips_array = np.array(clips)
        embeddings = infer_func(clips_array)
        
        # Convert to numpy if needed
        if hasattr(embeddings, 'numpy'):
            embeddings = embeddings.numpy()
        
        return embeddings.tolist()
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        return []

def process_patients_parallel(patients_df, config, start_patient=0, batch_size=None):
    """
    Process patients in parallel using multiprocessing
    """
    if batch_size is None:
        batch_size = len(patients_df)
    
    # Determine the batch of patients to process
    end_patient = min(start_patient + batch_size, len(patients_df))
    batch_patients = patients_df.iloc[start_patient:end_patient]
    
    print(f"🔄 Processing patients {start_patient+1} to {end_patient} ({len(batch_patients)} patients)")
    print(f"💻 Using {config.n_jobs} CPU cores")
    
    # Setup model path (download once, reuse in all processes)
    from huggingface_hub import snapshot_download
    model_path = snapshot_download("google/hear")
    
    # Prepare processing parameters
    processing_params = {
        'sample_rate': config.sample_rate,
        'clip_duration': config.clip_duration,
        'clip_overlap_percent': config.clip_overlap_percent,
        'silence_threshold_db': config.silence_threshold_db
    }
    
    # Prepare file list for processing
    files_to_process = []
    for _, patient in batch_patients.iterrows():
        patient_files = patient['files']
        for file_path in patient_files:
            if os.path.exists(file_path):
                files_to_process.append((
                    file_path,
                    patient['patient_id'], 
                    os.path.basename(file_path),
                    model_path,
                    processing_params
                ))
    
    print(f"📁 Total files to process: {len(files_to_process)}")
    
    # Process files in parallel
    embeddings_results = []
    metadata_results = []
    
    # Use ProcessPoolExecutor for true parallelism
    with ProcessPoolExecutor(max_workers=config.n_jobs) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_audio_file_optimized, args): args[0] 
            for args in files_to_process
        }
        
        # Process results as they complete
        for future in tqdm(as_completed(future_to_file), 
                          total=len(files_to_process), 
                          desc="Processing audio files"):
            
            result = future.result()
            
            if result['status'] == 'success' and result['embeddings']:
                # Store embeddings
                embeddings_results.extend(result['embeddings'])
                
                # Store metadata for each clip
                for clip_idx in range(result['num_clips']):
                    metadata_results.append({
                        'patient_id': result['patient_id'],
                        'file_id': result['file_id'],
                        'file_path': result['file_path'],
                        'clip_index': clip_idx,
                        'embedding_index': len(metadata_results)
                    })
            
            elif result['status'] == 'error':
                print(f"❌ Failed to process {result['file_path']}: {result['error']}")
    
    return embeddings_results, metadata_results

def save_embeddings_batch(embeddings_list, metadata_list, config, batch_num=None):
    """Save embeddings and metadata in batches"""
    
    if not embeddings_list:
        print("⚠️  No embeddings to save")
        return
    
    # Create embeddings array
    embeddings_array = np.array(embeddings_list)
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    
    # Generate filenames
    if batch_num is not None:
        embeddings_file = f"batch_{batch_num:03d}_embeddings.npz"
        metadata_file = f"batch_{batch_num:03d}_metadata.csv"
    else:
        embeddings_file = config.embeddings_file
        metadata_file = config.embeddings_file.replace('.npz', '_metadata.csv')
    
    embeddings_path = os.path.join(config.data_dir, embeddings_file)
    metadata_path = os.path.join(config.data_dir, metadata_file)
    
    # Save embeddings
    np.savez_compressed(embeddings_path, embeddings=embeddings_array)
    
    # Save metadata
    metadata_df.to_csv(metadata_path, index=False)
    
    print(f"✅ Saved {len(embeddings_list)} embeddings to {embeddings_file}")
    print(f"✅ Saved metadata to {metadata_file}")
    
    return embeddings_path, metadata_path

def merge_batch_files(config, num_batches):
    """Merge all batch files into final embeddings file"""
    print(f"🔗 Merging {num_batches} batch files...")
    
    all_embeddings = []
    all_metadata = []
    
    for batch_num in range(num_batches):
        embeddings_file = f"batch_{batch_num:03d}_embeddings.npz"
        metadata_file = f"batch_{batch_num:03d}_metadata.csv"
        
        embeddings_path = os.path.join(config.data_dir, embeddings_file)
        metadata_path = os.path.join(config.data_dir, metadata_file)
        
        if os.path.exists(embeddings_path) and os.path.exists(metadata_path):
            # Load batch embeddings
            batch_embeddings = np.load(embeddings_path)['embeddings']
            all_embeddings.append(batch_embeddings)
            
            # Load batch metadata
            batch_metadata = pd.read_csv(metadata_path)
            all_metadata.append(batch_metadata)
            
            # Remove batch files
            os.remove(embeddings_path)
            os.remove(metadata_path)
    
    if all_embeddings:
        # Combine all embeddings
        final_embeddings = np.vstack(all_embeddings)
        final_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Update embedding indices
        final_metadata['embedding_index'] = range(len(final_metadata))
        
        # Save final files
        final_embeddings_path = os.path.join(config.data_dir, config.embeddings_file)
        final_metadata_path = os.path.join(config.data_dir, config.embeddings_file.replace('.npz', '_metadata.csv'))
        
        np.savez_compressed(final_embeddings_path, embeddings=final_embeddings)
        final_metadata.to_csv(final_metadata_path, index=False)
        
        print(f"✅ Final embeddings saved: {final_embeddings.shape}")
        print(f"✅ Final metadata saved: {len(final_metadata)} records")

def main():
    """Main function"""
    parser = get_common_parser()
    parser.add_argument('--start', type=int, default=0, help='Starting patient index')
    parser.add_argument('--patients_batch_size', type=int, default=None, help='Number of patients to process')
    parser.add_argument('--save_every', type=int, default=25, help='Save embeddings every N patients')
    
    args = parser.parse_args()
    config = load_config_from_args(args)
    config.print_config()
    
    print(f"💾 System Memory: {psutil.virtual_memory().total // (1024**3)} GB available")
    print()
    
    # Load patient data
    print("📂 Loading patient data...")
    labels_path = config.get_labels_path()
    if not os.path.exists(labels_path):
        print(f"❌ Labels file not found: {labels_path}")
        return
    
    patients_df = pd.read_csv(labels_path)
    print(f"✅ Loaded {len(patients_df)} patients")
    
    # Process patients in batches
    start_patient = args.start
    patients_batch_size = args.patients_batch_size or len(patients_df)
    save_every = args.save_every
    
    batch_count = 0
    all_embeddings = []
    all_metadata = []
    
    while start_patient < len(patients_df):
        current_batch_size = min(save_every, patients_batch_size, len(patients_df) - start_patient)
        
        print(f"\n🔄 Processing batch starting at patient {start_patient + 1}")
        
        # Process current batch
        batch_embeddings, batch_metadata = process_patients_parallel(
            patients_df, config, start_patient, current_batch_size
        )
        
        if batch_embeddings:
            # Save batch immediately
            save_embeddings_batch(batch_embeddings, batch_metadata, config, batch_count)
            batch_count += 1
        
        start_patient += current_batch_size
        
        # Break if we've processed the requested number of patients
        if start_patient >= args.start + patients_batch_size:
            break
    
    # Merge all batch files into final file
    if batch_count > 0:
        merge_batch_files(config, batch_count)
        print("\n🎉 CPU-optimized embedding generation completed!")
    else:
        print("\n⚠️  No embeddings were generated")

if __name__ == "__main__":
    main()