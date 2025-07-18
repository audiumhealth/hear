#!/usr/bin/env python3
"""
Script to update file paths in notebooks after directory reorganization.
"""

import os
import json
import re
from pathlib import Path

def update_notebook_paths(notebook_path, replacements):
    """Update paths in a Jupyter notebook."""
    print(f"Updating {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    updated = False
    
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                for i, line in enumerate(source):
                    original_line = line
                    for old_path, new_path in replacements.items():
                        if old_path in line:
                            line = line.replace(old_path, new_path)
                    if line != original_line:
                        source[i] = line
                        updated = True
            elif isinstance(source, str):
                original_source = source
                for old_path, new_path in replacements.items():
                    if old_path in source:
                        source = source.replace(old_path, new_path)
                if source != original_source:
                    cell['source'] = source
                    updated = True
    
    if updated:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(notebook, f, indent=2)
        print(f"✅ Updated {notebook_path}")
    else:
        print(f"No changes needed for {notebook_path}")

def update_python_paths(script_path, replacements):
    """Update paths in a Python script."""
    print(f"Updating {script_path}...")
    
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    for old_path, new_path in replacements.items():
        content = content.replace(old_path, new_path)
    
    if content != original_content:
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"✅ Updated {script_path}")
    else:
        print(f"No changes needed for {script_path}")

def main():
    # Define path replacements for different directory contexts
    
    # For data processing notebooks (01_data_processing/)
    data_processing_replacements = {
        '"r2d2_audio_index.csv"': '"data/r2d2_audio_index.csv"',
        '"audium_UCSF_embeddings.npz"': '"data/audium_UCSF_embeddings.npz"',
        '"full_dataset_processed.npz"': '"data/full_dataset_processed.npz"',
        '"ucsf_new_embeddings.npz"': '"data/ucsf_new_embeddings.npz"',
        '"ucsf_new_embeddings_test.npz"': '"data/ucsf_new_embeddings_test.npz"',
        'CSV_PATH = "r2d2_audio_index.csv"': 'CSV_PATH = "data/r2d2_audio_index.csv"',
        'np.savez_compressed("audium_UCSF_embeddings.npz"': 'np.savez_compressed("data/audium_UCSF_embeddings.npz"',
    }
    
    # For cough classification notebooks (02_cough_classification/)
    cough_classification_replacements = {
        '"audium.pkl"': '"../shared_data/embeddings/audium.pkl"',
        '"saved_models_multiclass"': '"results/saved_models_multiclass"',
        'f"saved_models_multiclass/{': 'f"results/saved_models_multiclass/{',
        'os.makedirs("saved_models_multiclass"': 'os.makedirs("results/saved_models_multiclass"',
    }
    
    # For TB detection notebooks (03_tb_detection/)
    tb_detection_replacements = {
        '"audium_UCSF_embeddings.npz"': '"../01_data_processing/data/audium_UCSF_embeddings.npz"',
        '"r2d2_audio_index_with_labels.csv"': '"../r2d2_audio_index_with_labels.csv"',
        'EMBEDDING_PATH = "audium_UCSF_embeddings.npz"': 'EMBEDDING_PATH = "../01_data_processing/data/audium_UCSF_embeddings.npz"',
        'METADATA_PATH = "r2d2_audio_index_with_labels.csv"': 'METADATA_PATH = "../r2d2_audio_index_with_labels.csv"',
        'embedding_path = "audium_UCSF_embeddings.npz"': 'embedding_path = "../01_data_processing/data/audium_UCSF_embeddings.npz"',
        'metadata_path = "r2d2_audio_index_with_labels.csv"': 'metadata_path = "../r2d2_audio_index_with_labels.csv"',
        'csv_path = \'r2d2_audio_index_with_labels.csv\'': 'csv_path = "../r2d2_audio_index_with_labels.csv"',
        'embedding_path = \'audium_UCSF_embeddings.npz\'': 'embedding_path = "../01_data_processing/data/audium_UCSF_embeddings.npz"',
    }
    
    # For validation notebooks (07_validation/)
    validation_replacements = {
        '"../01_data_processing/data/audium_UCSF_embeddings.npz"': '"../01_data_processing/data/audium_UCSF_embeddings.npz"',
        '"../r2d2_audio_index_with_labels.csv"': '"../r2d2_audio_index_with_labels.csv"',
    }
    
    # For analysis scripts (04_analysis_scripts/)
    analysis_script_replacements = {
        '"audium_UCSF_embeddings.npz"': '"../01_data_processing/data/audium_UCSF_embeddings.npz"',
        '"full_dataset_processed.npz"': '"../01_data_processing/data/full_dataset_processed.npz"',
        '"r2d2_audio_index_with_labels.csv"': '"../r2d2_audio_index_with_labels.csv"',
        '"quick_tb_results.csv"': '"results/quick_tb_results.csv"',
        '"tb_analysis_results.png"': '"results/tb_analysis_results.png"',
    }
    
    # For WHO optimization scripts (05_who_optimization/)
    who_optimization_replacements = {
        '"audium_UCSF_embeddings.npz"': '"../01_data_processing/data/audium_UCSF_embeddings.npz"',
        '"full_dataset_processed.npz"': '"../01_data_processing/data/full_dataset_processed.npz"',
        '"r2d2_audio_index_with_labels.csv"': '"../r2d2_audio_index_with_labels.csv"',
        '"who_compliance_final_results.csv"': '"results/who_compliance_final_results.csv"',
        '"who_compliance_analysis.png"': '"results/who_compliance_analysis.png"',
        '"who_compliance_final.png"': '"results/who_compliance_final.png"',
    }
    
    # For visualization scripts (06_visualization/)
    visualization_replacements = {
        '"audium_UCSF_embeddings.npz"': '"../01_data_processing/data/audium_UCSF_embeddings.npz"',
        '"tsne_ucsf_embeddings.png"': '"results/tsne_ucsf_embeddings.png"',
        '"tsne_tb_focused.png"': '"results/tsne_tb_focused.png"',
        '"tsne_results.csv"': '"results/tsne_results.csv"',
    }
    
    # Update files in each directory
    base_dir = Path(".")
    
    directories_and_replacements = [
        ("01_data_processing", data_processing_replacements),
        ("02_cough_classification", cough_classification_replacements),
        ("03_tb_detection", tb_detection_replacements),
        ("04_analysis_scripts", analysis_script_replacements),
        ("05_who_optimization", who_optimization_replacements),
        ("06_visualization", visualization_replacements),
        ("07_validation", validation_replacements),
    ]
    
    for dir_name, replacements in directories_and_replacements:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            print(f"\n📁 Processing {dir_name}...")
            
            # Update notebooks
            for notebook_file in dir_path.glob("*.ipynb"):
                update_notebook_paths(notebook_file, replacements)
            
            # Update Python scripts
            for script_file in dir_path.glob("*.py"):
                update_python_paths(script_file, replacements)
    
    print("\n✅ Path updates complete!")

if __name__ == "__main__":
    main()