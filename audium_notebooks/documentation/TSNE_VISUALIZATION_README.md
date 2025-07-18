# t-SNE Visualization of UCSF Embeddings

## Overview
This document describes the t-SNE visualization analysis performed on the UCSF embeddings dataset for TB detection research.

## Dataset Information
- **Total Samples**: 9,268 audio samples
- **Feature Dimensions**: 512-dimensional HeAR embeddings (averaged from 10 clips per audio file)
- **TB Distribution**: 
  - TB Positive: 1,975 samples (21.3%)
  - TB Negative: 7,293 samples (78.7%)

## Files Generated

### 1. `tsne_ucsf_embeddings.png`
**Comprehensive 4-panel visualization showing:**
- **Panel 1**: TB status (main plot) - TB Positive (red) vs TB Negative (blue)
- **Panel 2**: Patient ID groupings (first 20 patients)
- **Panel 3**: Sex distribution (Male vs Female)
- **Panel 4**: Country distribution

### 2. `tsne_tb_focused.png`
**Focused TB detection plot featuring:**
- TB Positive samples highlighted in red with dark red borders
- TB Negative samples in light blue background
- Statistics overlay with sample counts and percentages
- Optimized for TB detection analysis

### 3. `tsne_results.csv`
**Data file containing:**
- Original metadata (StudyID, filename, label, country, sex, age, HIV status)
- t-SNE coordinates (tsne_x, tsne_y)
- Processed TB status (0/1)
- Full embedding indices

## Technical Details

### t-SNE Parameters
- **Perplexity**: 30
- **Components**: 2D
- **Iterations**: 1000
- **Random State**: 42 (for reproducibility)

### Preprocessing
- **Standardization**: Applied StandardScaler before t-SNE
- **Embedding Aggregation**: Averaged 10 clips per audio file to single 512D vector
- **Feature Scaling**: Robust scaling applied to handle outliers

## Key Insights
1. **Clustering Patterns**: TB positive and negative samples show distinct clustering patterns in 2D space
2. **Patient Grouping**: Individual patients' audio samples tend to cluster together
3. **Demographic Distribution**: Clear visualization of sex and country distributions
4. **Separability**: The embeddings demonstrate reasonable separability between TB positive and negative cases

## Usage Instructions

### To Reproduce the Analysis:
```bash
# Activate the audium_hear environment
source ~/python/venvs/v_audium_hear/bin/activate

# Run the visualization script
python tsne_visualization.py
```

### Required Files:
- `ucsf_new_embeddings.npz` - Embedding data
- `ucsf_new_embeddings_metadata.csv` - Metadata with TB labels

### Environment Setup:
- Uses virtual environment: `~/python/venvs/v_audium_hear/bin/activate`
- Required packages: pandas, matplotlib, seaborn, scikit-learn, numpy

## Analysis Context
This visualization was created to explore the separability of TB positive and negative cases in the HeAR embedding space. The t-SNE reduction helps identify:
- Natural clustering patterns in the acoustic data
- Potential outliers or misclassified samples
- Demographic and geographic patterns in the data
- Quality of the HeAR embeddings for TB detection

## Next Steps
The t-SNE visualization provides a foundation for:
1. **Model Development**: Understanding data structure for classifier design
2. **Feature Engineering**: Identifying potential feature improvements
3. **Data Quality Assessment**: Spotting outliers and data issues
4. **Performance Analysis**: Correlating visualization clusters with model predictions

## File Locations
- **Script**: `/Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/tsne_visualization.py`
- **Outputs**: Same directory as script
- **Data**: `ucsf_new_embeddings.npz` and `ucsf_new_embeddings_metadata.csv`

---
*Generated: July 16, 2025*
*Analysis performed using t-SNE on 9,268 UCSF audio samples for TB detection research*