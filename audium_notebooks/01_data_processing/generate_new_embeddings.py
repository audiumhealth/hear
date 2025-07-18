#!/usr/bin/env python3
"""
Generate labeled embeddings from the new UCSF TB dataset.
This script processes the R2D2 lung sounds metadata and generates embeddings
using the HeAR model for the TB detection task.
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
import argparse
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from scipy.io import wavfile
from scipy import signal

# Import HeAR model utilities
try:
    from huggingface_hub import from_pretrained_keras
    HEAR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: HeAR model not available: {e}")
    HEAR_AVAILABLE = False

# Constants
SAMPLE_RATE = 16000  # Samples per second (Hz)
CLIP_DURATION = 2    # Duration of the audio clip in seconds
CLIP_LENGTH = SAMPLE_RATE * CLIP_DURATION  # Total number of samples

def load_metadata(metadata_path):
    """Load and process the metadata file."""
    print(f"Loading metadata from: {metadata_path}")
    
    df = pd.read_csv(metadata_path)
    
    # Filter for patients with lung sounds collected
    df = df[df['Lungsoundscollected'] == 'Yes'].copy()
    
    print(f"Total patients with lung sounds: {len(df)}")
    print(f"Countries: {df['Country'].value_counts().to_dict()}")
    
    # Use microbiological reference standard as primary label
    label_counts = df['Microbiologicreferencestandard'].value_counts()
    print(f"Label distribution: {label_counts.to_dict()}")
    
    # Filter out indeterminate cases for clear binary classification
    df = df[df['Microbiologicreferencestandard'].isin(['TB Positive', 'TB Negative'])].copy()
    
    print(f"After filtering indeterminate: {len(df)} patients")
    print(f"TB Positive: {sum(df['Microbiologicreferencestandard'] == 'TB Positive')}")
    print(f"TB Negative: {sum(df['Microbiologicreferencestandard'] == 'TB Negative')}")
    
    return df

def find_audio_files(audio_base_path, study_ids):
    """Find audio files for given study IDs."""
    print(f"Searching for audio files in: {audio_base_path}")
    
    audio_files = []
    missing_patients = []
    
    for study_id in tqdm(study_ids, desc="Finding audio files"):
        # Search in all subdirectories for the patient folder
        patient_found = False
        
        # Check all possible paths
        for root, dirs, files in os.walk(audio_base_path):
            if study_id in dirs:
                patient_dir = os.path.join(root, study_id)
                wav_files = [f for f in os.listdir(patient_dir) if f.endswith('.wav')]
                
                for wav_file in wav_files:
                    audio_files.append({
                        'StudyID': study_id,
                        'filename': wav_file,
                        'full_path': os.path.join(patient_dir, wav_file),
                        'relative_path': os.path.relpath(os.path.join(patient_dir, wav_file), audio_base_path)
                    })
                
                patient_found = True
                break
        
        if not patient_found:
            missing_patients.append(study_id)
    
    print(f"Found {len(audio_files)} audio files for {len(set([f['StudyID'] for f in audio_files]))} patients")
    print(f"Missing patients: {len(missing_patients)}")
    
    if missing_patients:
        print(f"First 10 missing: {missing_patients[:10]}")
    
    return audio_files, missing_patients

def resample_audio_and_convert_to_mono(
    audio_array: np.ndarray,
    sampling_rate: float,
    new_sampling_rate: float = SAMPLE_RATE,
) -> np.ndarray:
    """
    Resamples an audio array to 16kHz and converts it to mono if it has multiple channels.
    """
    # Convert to mono if it's multi-channel
    if audio_array.ndim > 1:
        audio_mono = np.mean(audio_array, axis=1)
    else:
        audio_mono = audio_array

    # Resample
    original_sample_count = audio_mono.shape[0]
    new_sample_count = int(round(original_sample_count * (new_sampling_rate / sampling_rate)))
    resampled_audio_mono = signal.resample(audio_mono, new_sample_count)

    return resampled_audio_mono

def process_audio_file(audio_path, model, sample_rate=16000, max_duration=30):
    """Process a single audio file and generate embeddings."""
    try:
        # Load audio
        with open(audio_path, 'rb') as f:
            original_sampling_rate, audio_array = wavfile.read(f)
        
        # Convert to float32 and normalize
        if audio_array.dtype == np.int16:
            audio_array = audio_array.astype(np.float32) / 32768.0
        elif audio_array.dtype == np.int32:
            audio_array = audio_array.astype(np.float32) / 2147483648.0
        
        # Resample and convert to mono
        audio_array = resample_audio_and_convert_to_mono(audio_array, original_sampling_rate, sample_rate)
        
        # Limit duration
        max_samples = int(max_duration * sample_rate)
        if len(audio_array) > max_samples:
            audio_array = audio_array[:max_samples]
        
        # Generate embeddings by processing in 2-second chunks
        embeddings_list = []
        for start in range(0, len(audio_array), CLIP_LENGTH):
            end = min(start + CLIP_LENGTH, len(audio_array))
            chunk = audio_array[start:end]
            
            # Pad if chunk is too short
            if len(chunk) < CLIP_LENGTH:
                chunk = np.pad(chunk, (0, CLIP_LENGTH - len(chunk)), mode='constant')
            
            # Add batch dimension
            input_tensor = np.expand_dims(chunk, axis=0)
            
            # Get embedding
            output = model(tf.constant(input_tensor, dtype=tf.float32))
            # Handle different output formats
            if isinstance(output, dict):
                # If output is a dictionary, get the embedding from the appropriate key
                if 'output_0' in output:
                    embedding = output['output_0'].numpy().flatten()
                else:
                    # Try to get the first value if structure is different
                    embedding = list(output.values())[0].numpy().flatten()
            else:
                # If output is a tensor directly
                embedding = output.numpy().flatten()
            embeddings_list.append(embedding)
        
        # Stack embeddings into a 2D array (time_steps, embedding_dim)
        embeddings = np.stack(embeddings_list)
        
        return embeddings
        
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def generate_embeddings(metadata_df, audio_files, model, output_path):
    """Generate embeddings for all audio files."""
    print("Generating embeddings...")
    
    # Create lookup for metadata
    metadata_lookup = {row['StudyID']: row for _, row in metadata_df.iterrows()}
    
    # Prepare storage
    embeddings_dict = {}
    failed_files = []
    
    # Process each audio file
    for audio_info in tqdm(audio_files, desc="Processing audio files"):
        study_id = audio_info['StudyID']
        filename = audio_info['filename']
        full_path = audio_info['full_path']
        
        # Generate key for storage
        key = f"{study_id}/{filename}"
        
        # Process audio
        embeddings = process_audio_file(full_path, model)
        
        if embeddings is not None:
            embeddings_dict[key] = embeddings
        else:
            failed_files.append(key)
    
    print(f"Successfully processed: {len(embeddings_dict)} files")
    print(f"Failed files: {len(failed_files)}")
    
    # Save embeddings
    print(f"Saving embeddings to: {output_path}")
    np.savez_compressed(output_path, **embeddings_dict)
    
    # Create corresponding metadata CSV
    metadata_output = output_path.replace('.npz', '_metadata.csv')
    
    # Generate metadata for each audio file
    metadata_records = []
    for key in embeddings_dict.keys():
        study_id = key.split('/')[0]
        filename = key.split('/')[1]
        
        if study_id in metadata_lookup:
            patient_info = metadata_lookup[study_id]
            metadata_records.append({
                'StudyID': study_id,
                'filename': filename,
                'full_key': key,
                'label': patient_info['Microbiologicreferencestandard'],
                'country': patient_info['Country'],
                'sex': patient_info['Sex'],
                'age': patient_info['Age'],
                'hiv_status': patient_info['HIVstatus']
            })
    
    metadata_df_output = pd.DataFrame(metadata_records)
    metadata_df_output.to_csv(metadata_output, index=False)
    
    print(f"Saved metadata to: {metadata_output}")
    
    # Generate statistics
    print("\n=== EMBEDDING STATISTICS ===")
    if len(embeddings_dict) > 0:
        sample_key = list(embeddings_dict.keys())[0]
        sample_embedding = embeddings_dict[sample_key]
        print(f"Sample embedding shape: {sample_embedding.shape}")
        print(f"Embedding dimension: {sample_embedding.shape[1]}")
        print(f"Average time steps: {np.mean([emb.shape[0] for emb in embeddings_dict.values()]):.1f}")
        
        label_counts = metadata_df_output['label'].value_counts()
        print(f"Label distribution in embeddings: {label_counts.to_dict()}")
        
        country_counts = metadata_df_output['country'].value_counts()
        print(f"Country distribution: {country_counts.to_dict()}")
    else:
        print("No embeddings were successfully generated!")
    
    return embeddings_dict, metadata_df_output, failed_files

def main():
    parser = argparse.ArgumentParser(description="Generate embeddings for UCSF TB dataset")
    parser.add_argument("--metadata", 
                        default="/Users/abelvillcaroque/data/Audium/UCSF-TB-Project/R2D2 lung sounds metadata (Stethee)/R2D2 lung sounds metadata_TRAIN_2023.02.08.csv",
                        help="Path to metadata CSV file")
    parser.add_argument("--audio_path", 
                        default="/Users/abelvillcaroque/data/Audium/UCSF-TB-Project/UCSF_adult",
                        help="Path to audio files directory")
    parser.add_argument("--output", 
                        default="/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/ucsf_new_embeddings.npz",
                        help="Output path for embeddings")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to process (for testing)")
    
    args = parser.parse_args()
    
    # Check if HeAR model is available
    if not HEAR_AVAILABLE:
        print("ERROR: HeAR model not available. Please ensure the hear package is installed.")
        return
    
    # Load metadata
    metadata_df = load_metadata(args.metadata)
    
    # Find audio files
    study_ids = metadata_df['StudyID'].tolist()
    audio_files, missing_patients = find_audio_files(args.audio_path, study_ids)
    
    if not audio_files:
        print("ERROR: No audio files found!")
        return
    
    # Limit files for testing if requested
    if args.max_files:
        audio_files = audio_files[:args.max_files]
        print(f"Limited to {len(audio_files)} files for testing")
    
    # Load HeAR model
    print("Loading HeAR model...")
    
    # Download model files from HuggingFace
    from huggingface_hub import snapshot_download
    import keras
    
    model_path = snapshot_download(repo_id="google/hear")
    print(f"Model downloaded to: {model_path}")
    
    # Load as TFSMLayer
    model = keras.layers.TFSMLayer(model_path, call_endpoint='serving_default')
    
    # Generate embeddings
    embeddings_dict, metadata_df_output, failed_files = generate_embeddings(
        metadata_df, audio_files, model, args.output
    )
    
    # Final summary
    print("\n=== FINAL SUMMARY ===")
    print(f"Total embeddings generated: {len(embeddings_dict)}")
    print(f"Total patients: {metadata_df_output['StudyID'].nunique()}")
    print(f"Failed files: {len(failed_files)}")
    print(f"Output saved to: {args.output}")
    
    if failed_files:
        print(f"First 10 failed files: {failed_files[:10]}")

if __name__ == "__main__":
    main()