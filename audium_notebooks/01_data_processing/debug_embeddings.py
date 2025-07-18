#!/usr/bin/env python3

import numpy as np
import pandas as pd
import os

# Change to the audium_notebooks directory
os.chdir('/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks')

print("=== DEBUGGING EMBEDDINGS FILE ===")

# Check if files exist
embedding_path = 'audium_UCSF_embeddings.npz'
csv_path = 'r2d2_audio_index_with_labels.csv'

print(f"Embedding file exists: {os.path.exists(embedding_path)}")
print(f"CSV file exists: {os.path.exists(csv_path)}")

if os.path.exists(embedding_path):
    # Load and inspect the .npz file
    embeddings_data = np.load(embedding_path)
    print(f"\nAvailable keys in {embedding_path}: {list(embeddings_data.keys())}")
    
    for key in embeddings_data.keys():
        data = embeddings_data[key]
        print(f"Key '{key}': shape={data.shape}, dtype={data.dtype}")
        
        # Show first few items if it's a string array (likely file keys)
        if data.dtype.kind in ['U', 'S']:  # Unicode or byte strings
            print(f"  First 5 items: {data[:5]}")
        elif len(data.shape) == 1 and data.shape[0] < 20:  # Small 1D array
            print(f"  First 5 items: {data[:5]}")
        else:
            print(f"  Sample values: min={data.min():.4f}, max={data.max():.4f}, mean={data.mean():.4f}")

if os.path.exists(csv_path):
    # Load and inspect the CSV file
    csv_df = pd.read_csv(csv_path)
    print(f"\nCSV file shape: {csv_df.shape}")
    print(f"CSV columns: {list(csv_df.columns)}")
    print(f"First few rows:")
    print(csv_df.head())
    
    # Show the constructed keys
    csv_df['full_key'] = csv_df['patientID'] + '/' + csv_df['filename']
    print(f"\nFirst few full keys from CSV:")
    print(csv_df['full_key'].head())

print("\n=== END DEBUGGING ===")