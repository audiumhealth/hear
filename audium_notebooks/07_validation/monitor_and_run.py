#!/usr/bin/env python3
"""
Monitor embedding generation and run notebook when complete
"""

import os
import sys
import time
import subprocess
import psutil
import numpy as np
import pandas as pd
from pathlib import Path

def check_process_status(pid):
    """Check if process is still running"""
    try:
        proc = psutil.Process(pid)
        return proc.is_running()
    except:
        return False

def get_file_stats(filepath):
    """Get file size and modification time"""
    try:
        stat = os.stat(filepath)
        return stat.st_size, stat.st_mtime
    except:
        return 0, 0

def load_embeddings_info(embedding_path):
    """Load embeddings and get basic info"""
    try:
        data = np.load(embedding_path)
        keys = list(data.keys())
        return len(keys), keys[:5]  # Count and first 5 keys
    except:
        return 0, []

def run_sanity_checks(embedding_path, metadata_path):
    """Run sanity checks on the dataset"""
    print("\n🔍 RUNNING SANITY CHECKS")
    print("=" * 50)
    
    # Load embeddings
    try:
        embeddings_data = np.load(embedding_path)
        all_keys = list(embeddings_data.keys())
        print(f"✅ Total embedding files: {len(all_keys)}")
        
        # Check embedding shape
        sample_key = all_keys[0]
        sample_embedding = embeddings_data[sample_key]
        print(f"✅ Sample embedding shape: {sample_embedding.shape}")
        print(f"✅ Expected format: (n_clips, 512)")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return False
    
    # Load metadata
    try:
        metadata = pd.read_csv(metadata_path)
        print(f"✅ Metadata rows: {len(metadata)}")
        print(f"✅ Metadata columns: {list(metadata.columns)}")
        
        # Files per patient statistics
        files_per_patient = metadata.groupby('StudyID').size()
        print(f"✅ Total patients: {len(files_per_patient)}")
        print(f"✅ Files per patient - Mean: {files_per_patient.mean():.1f}")
        print(f"✅ Files per patient - Median: {files_per_patient.median():.1f}")
        print(f"✅ Files per patient - Min: {files_per_patient.min()}")
        print(f"✅ Files per patient - Max: {files_per_patient.max()}")
        
        # Label distribution
        label_counts = metadata['label'].value_counts()
        print(f"✅ Label distribution:")
        for label, count in label_counts.items():
            percentage = count / len(metadata) * 100
            print(f"   {label}: {count} ({percentage:.1f}%)")
        
        # Country distribution
        country_counts = metadata['country'].value_counts()
        print(f"✅ Country distribution:")
        for country, count in country_counts.items():
            percentage = count / len(metadata) * 100
            print(f"   {country}: {count} ({percentage:.1f}%)")
        
        # Patient-level label distribution
        patient_labels = metadata.groupby('StudyID')['label'].first()
        patient_label_counts = patient_labels.value_counts()
        print(f"✅ Patient-level label distribution:")
        for label, count in patient_label_counts.items():
            percentage = count / len(patient_labels) * 100
            print(f"   {label}: {count} patients ({percentage:.1f}%)")
        
        # Check for matching files
        common_keys = set(all_keys) & set(metadata['full_key'])
        print(f"✅ Matching files (embeddings ∩ metadata): {len(common_keys)}")
        
        missing_embeddings = set(metadata['full_key']) - set(all_keys)
        print(f"⚠️  Missing embeddings: {len(missing_embeddings)}")
        
        if len(missing_embeddings) > 0:
            print(f"   First 5 missing: {list(missing_embeddings)[:5]}")
        
    except Exception as e:
        print(f"❌ Error loading metadata: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ SANITY CHECKS COMPLETE")
    print("=" * 50)
    return True

def run_notebook():
    """Run the notebook programmatically"""
    print("\n🚀 RUNNING NOTEBOOK")
    print("=" * 50)
    
    try:
        # Convert notebook to script and run it
        notebook_path = "UCSF_Advanced_TB_Detection_NEW_DATASET.ipynb"
        
        # Use jupyter nbconvert to run the notebook
        cmd = [
            "jupyter", "nbconvert", "--to", "notebook", 
            "--execute", "--inplace", notebook_path
        ]
        
        print(f"📝 Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Notebook executed successfully!")
            print("📊 Results should be in the notebook output cells")
        else:
            print(f"❌ Notebook execution failed:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error running notebook: {e}")

def main():
    """Main monitoring function"""
    EMBEDDING_PID = 42391
    EMBEDDING_PATH = "ucsf_new_embeddings.npz"
    METADATA_PATH = "ucsf_new_embeddings_metadata.csv"
    
    print("🔄 MONITORING EMBEDDING GENERATION")
    print("=" * 50)
    print(f"Process PID: {EMBEDDING_PID}")
    print(f"Target embedding file: {EMBEDDING_PATH}")
    print(f"Target metadata file: {METADATA_PATH}")
    
    # Initial status
    last_size = 0
    last_count = 0
    
    while True:
        # Check if process is still running
        if not check_process_status(EMBEDDING_PID):
            print("\n🎉 EMBEDDING GENERATION COMPLETE!")
            break
        
        # Check file progress
        size, mtime = get_file_stats(EMBEDDING_PATH)
        count, sample_keys = load_embeddings_info(EMBEDDING_PATH)
        
        if size != last_size or count != last_count:
            print(f"📊 Progress: {count} files, {size/1024/1024:.1f}MB ({time.strftime('%H:%M:%S')})")
            last_size = size
            last_count = count
        
        # Wait before next check
        time.sleep(30)  # Check every 30 seconds
    
    # Process completed - run sanity checks
    if run_sanity_checks(EMBEDDING_PATH, METADATA_PATH):
        # Run the notebook
        run_notebook()
    else:
        print("❌ Sanity checks failed - not running notebook")

if __name__ == "__main__":
    main()