#!/usr/bin/env python3
"""
t-SNE Visualization of UCSF Embeddings
Generate 2D t-SNE plot colored by TB status
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_embeddings_and_metadata():
    """Load embeddings and metadata"""
    print("Loading embeddings and metadata...")
    
    # Load embeddings
    embeddings_data = np.load('ucsf_new_embeddings.npz', allow_pickle=True)
    
    # Load metadata
    metadata_df = pd.read_csv('ucsf_new_embeddings_metadata.csv')
    
    # Extract embeddings and create mapping
    embeddings_dict = {}
    for key in embeddings_data.keys():
        embeddings_dict[key] = embeddings_data[key]
    
    print(f"Loaded {len(embeddings_dict)} audio files")
    print(f"Metadata for {len(metadata_df)} files")
    
    return embeddings_dict, metadata_df

def process_embeddings(embeddings_dict, metadata_df):
    """Process embeddings and align with metadata"""
    print("Processing embeddings...")
    
    # Create lists for embeddings and labels
    all_embeddings = []
    file_keys = []
    
    # Process each audio file
    for key, embedding_sequence in embeddings_dict.items():
        # Each file has multiple embeddings (10 clips of 512 dimensions each)
        # We'll average them to get one embedding per file
        if len(embedding_sequence.shape) == 2:
            avg_embedding = np.mean(embedding_sequence, axis=0)
            all_embeddings.append(avg_embedding)
            file_keys.append(key)
    
    # Convert to numpy array
    X = np.array(all_embeddings)
    
    # Create dataframe for easier manipulation
    results_df = pd.DataFrame({
        'full_key': file_keys,
        'embedding_idx': range(len(file_keys))
    })
    
    # Merge with metadata
    results_df = results_df.merge(metadata_df[['full_key', 'label', 'StudyID', 'sex', 'age', 'country']], 
                                  on='full_key', how='left')
    
    # Clean up labels
    results_df['tb_status'] = results_df['label'].map({
        'TB Positive': 1,
        'TB Negative': 0
    })
    
    print(f"Final dataset: {len(X)} samples with {X.shape[1]} features")
    print(f"TB Positive: {sum(results_df['tb_status'] == 1)}")
    print(f"TB Negative: {sum(results_df['tb_status'] == 0)}")
    
    return X, results_df

def apply_tsne(X, perplexity=30, n_components=2, random_state=42):
    """Apply t-SNE dimensionality reduction"""
    print(f"Applying t-SNE with perplexity={perplexity}...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply t-SNE
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        random_state=random_state,
        n_iter=1000,
        verbose=1
    )
    
    X_tsne = tsne.fit_transform(X_scaled)
    
    print("t-SNE completed successfully")
    return X_tsne

def create_tsne_plot(X_tsne, results_df, save_path='tsne_ucsf_embeddings.png'):
    """Create t-SNE visualization"""
    print("Creating t-SNE visualization...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: TB Status
    ax1 = axes[0, 0]
    colors = ['blue', 'red']
    labels = ['TB Negative', 'TB Positive']
    
    for i, (color, label) in enumerate(zip(colors, labels)):
        mask = results_df['tb_status'] == i
        ax1.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=color, alpha=0.6, s=30, label=label)
    
    ax1.set_title('t-SNE: UCSF Embeddings by TB Status', fontsize=14, fontweight='bold')
    ax1.set_xlabel('t-SNE Component 1')
    ax1.set_ylabel('t-SNE Component 2')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: By Patient (StudyID)
    ax2 = axes[0, 1]
    unique_patients = results_df['StudyID'].unique()
    patient_colors = plt.cm.tab20(np.linspace(0, 1, len(unique_patients)))
    
    for i, patient in enumerate(unique_patients[:20]):  # Show first 20 patients
        mask = results_df['StudyID'] == patient
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                   c=[patient_colors[i]], alpha=0.6, s=30, label=patient)
    
    ax2.set_title('t-SNE: By Patient ID (First 20)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('t-SNE Component 1')
    ax2.set_ylabel('t-SNE Component 2')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: By Sex
    ax3 = axes[1, 0]
    sex_colors = {'Male': 'blue', 'Female': 'red'}
    
    for sex, color in sex_colors.items():
        mask = results_df['sex'] == sex
        if mask.any():
            ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=color, alpha=0.6, s=30, label=sex)
    
    ax3.set_title('t-SNE: By Sex', fontsize=14, fontweight='bold')
    ax3.set_xlabel('t-SNE Component 1')
    ax3.set_ylabel('t-SNE Component 2')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: By Country
    ax4 = axes[1, 1]
    unique_countries = results_df['country'].unique()
    country_colors = plt.cm.Set3(np.linspace(0, 1, len(unique_countries)))
    
    for i, country in enumerate(unique_countries):
        mask = results_df['country'] == country
        if mask.any():
            ax4.scatter(X_tsne[mask, 0], X_tsne[mask, 1], 
                       c=[country_colors[i]], alpha=0.6, s=30, label=country)
    
    ax4.set_title('t-SNE: By Country', fontsize=14, fontweight='bold')
    ax4.set_xlabel('t-SNE Component 1')
    ax4.set_ylabel('t-SNE Component 2')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as {save_path}")

def create_focused_tb_plot(X_tsne, results_df, save_path='tsne_tb_focused.png'):
    """Create focused TB visualization"""
    print("Creating focused TB visualization...")
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot TB Negative first (background)
    mask_neg = results_df['tb_status'] == 0
    ax.scatter(X_tsne[mask_neg, 0], X_tsne[mask_neg, 1], 
               c='lightblue', alpha=0.5, s=40, label='TB Negative')
    
    # Plot TB Positive on top (foreground)
    mask_pos = results_df['tb_status'] == 1
    ax.scatter(X_tsne[mask_pos, 0], X_tsne[mask_pos, 1], 
               c='red', alpha=0.8, s=60, label='TB Positive', edgecolor='darkred')
    
    ax.set_title('t-SNE Visualization: UCSF Embeddings\nTB Detection Analysis', 
                 fontsize=16, fontweight='bold')
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    n_pos = sum(mask_pos)
    n_neg = sum(mask_neg)
    n_total = len(results_df)
    
    stats_text = f'Total: {n_total} samples\nTB Positive: {n_pos} ({n_pos/n_total*100:.1f}%)\nTB Negative: {n_neg} ({n_neg/n_total*100:.1f}%)'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Focused TB plot saved as {save_path}")

def main():
    """Main execution function"""
    print("🎯 t-SNE Visualization of UCSF Embeddings")
    print("="*50)
    
    # Load data
    embeddings_dict, metadata_df = load_embeddings_and_metadata()
    
    # Process embeddings
    X, results_df = process_embeddings(embeddings_dict, metadata_df)
    
    # Apply t-SNE
    X_tsne = apply_tsne(X, perplexity=30)
    
    # Create visualizations
    create_tsne_plot(X_tsne, results_df)
    create_focused_tb_plot(X_tsne, results_df)
    
    # Save results
    results_df['tsne_x'] = X_tsne[:, 0]
    results_df['tsne_y'] = X_tsne[:, 1]
    results_df.to_csv('tsne_results.csv', index=False)
    
    print("\n" + "="*50)
    print("✅ t-SNE visualization completed successfully!")
    print("📊 Files generated:")
    print("   - tsne_ucsf_embeddings.png")
    print("   - tsne_tb_focused.png")
    print("   - tsne_results.csv")

if __name__ == "__main__":
    main()