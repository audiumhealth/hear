# GPU-Optimized TB Detection Pipeline - Getting Started Guide

**For New Team Members & Collaborators**

## 🎯 What This Is

A GPU-accelerated machine learning pipeline for TB detection from cough audio using Google's HeAR (Health Acoustic Representations) model. **Goal**: Achieve WHO TB screening standards (≥90% sensitivity, ≥70% specificity).

## 📁 Data Preparation Pipeline

**Complete workflow from raw audio to embeddings. Follow in order:**

### Step 1: Check for Required Data Files
```bash
# Check for critical prerequisite files
ls -la data/clean_patients_final.csv     # Patient labels (TB status)
ls -la data/file_mapping_final.csv       # Audio file → patient mapping

# Check for embeddings
ls -la data/*.npz

# If you have embeddings (.npz files), skip to Quick Start section
# If missing prerequisite files, continue with data preparation below
```

### Step 2: Generate Prerequisite Files (If Missing)

**If you're missing the critical data files above, run the data validation script:**

```bash
# Generate clean patient dataset and file mapping
python data_validation_final.py

# This creates:
# - data/clean_patients_final.csv (543 patients with TB labels)  
# - data/file_mapping_final.csv (10,682 audio files → patient mapping)
```

**Requirements for data validation:**
- **Raw audio files** organized in patient directories (e.g., `R2D200001/audio.wav`)
- **Patient metadata** with TB diagnosis labels
- **Directory structure** following UCSF R2D2 format

### Step 3: Check Data Validation Results
```bash
# Verify the generated files
echo "Clean patients: $(wc -l < data/clean_patients_final.csv) patients"
echo "File mapping: $(wc -l < data/file_mapping_final.csv) audio files"

# Expected output:
# Clean patients: 544 patients (including header)
# File mapping: 10,683 audio files (including header)
```

### Step 4: Generate Embeddings from Audio Files

#### Option 1: CPU-Optimized Generation (Recommended for most users)
```bash
# For standard datasets
python 02_generate_embeddings_cpu_optimized.py --input_dir /path/to/audio/files --output_prefix final_embeddings

# For large datasets (batch processing)
python 02_generate_embeddings_batch.py --input_dir /path/to/audio/files --batch_size 100
```

#### Option 2: Full Pipeline Generation (For complete workflows)
```bash
# Comprehensive processing with validation
python 02_generate_embeddings_final.py --input_dir /path/to/audio/files
```

### Audio File Requirements
- **Format**: `.wav`, `.mp3`, `.webm`, or other common audio formats
- **Duration**: Any length (will be processed in 2-second clips with 10% overlap)
- **Sample Rate**: Any (will be resampled to 16kHz automatically)
- **Naming**: Patient-based naming recommended (e.g., `PatientID_timestamp.wav`)

### Expected Embedding Generation Output
```
data/
├── final_embeddings.npz              # HeAR embeddings (512-dim vectors)
├── final_embeddings_metadata.csv     # File mapping and metadata
├── file_mapping_final.csv            # Audio file to embedding mapping
└── validation_summary.csv            # Processing statistics
```

### Embedding Generation Time Estimates
- **Small dataset** (100 audio files): ~5-10 minutes
- **Medium dataset** (1,000 audio files): ~30-60 minutes  
- **Large dataset** (10,000+ audio files): ~3-6 hours

### Troubleshooting Embedding Generation
1. **"Audio file not supported"**
   - Install additional codecs: `pip install pydub[mp3]`
   - Convert to WAV format first

2. **"Out of memory"**
   - Reduce batch size: `--batch_size 50`
   - Use CPU-optimized version instead

3. **"HeAR model download failed"**
   - Ensure internet connection for first run
   - Model will be cached locally after first download

## 🚀 Quick Start (5 minutes)

### Prerequisites
- macOS with Apple Silicon (M1/M2/M3) or Linux with NVIDIA GPU
- Python virtual environment with required packages
- Audio embeddings data (`.npz` files) - **generate above if needed**

### Run Your First Analysis
```bash
# Navigate to pipeline directory
cd audium_notebooks/09_202504_UCSF_New_Training_set\(R2D2\)_CPU_OPTIMIZED/

# Basic run (single train-test split)
./run_gpu_optimized_pipeline.sh

# Cross-validation for robust results
CROSS_VALIDATION=true ./run_gpu_optimized_pipeline.sh

# With custom description for tracking
RUN_DESCRIPTION="my_experiment" ./run_gpu_optimized_pipeline.sh
```

### Expected Outputs
- **Results**: `results/` folder with CSV files and visualizations
- **Configs**: `configs/` folder with run configurations for reproducibility
- **Runtime**: ~2-5 minutes (single split), ~10-15 minutes (5-fold CV)

## 📊 What You'll Get

### Performance Metrics
- **8 ML models** compared: Naive Bayes, LDA, Logistic Regression, KNN, XGBoost variants, GPU MLP
- **WHO compliance analysis**: Which models meet clinical screening standards
- **Comprehensive visualizations**: ROC curves, confusion matrices, performance dashboards

### Output Files Examples
```
results/
├── final_embeddings_gpu_single_20250722_143052_a1b2c3_gpu_analysis_results.csv
├── final_embeddings_gpu_single_20250722_143052_a1b2c3_gpu_performance_dashboard.png
├── final_embeddings_gpu_single_20250722_143052_a1b2c3_gpu_roc_curves.png
└── final_embeddings_gpu_single_20250722_143052_a1b2c3_gpu_executive_summary.txt
```

## 🔧 Configuration Options

### Basic Options
```bash
# Use different dataset
EMBEDDINGS_FILE="your_data.npz" ./run_gpu_optimized_pipeline.sh

# Force CPU mode (disable GPU)
DEVICE=cpu ./run_gpu_optimized_pipeline.sh

# Verbose output
VERBOSE=true ./run_gpu_optimized_pipeline.sh
```

### Advanced Options
```bash
# 10-fold cross-validation
CROSS_VALIDATION=true N_FOLDS=10 ./run_gpu_optimized_pipeline.sh

# Custom GPU memory usage
GPU_MEMORY_FRACTION=0.9 ./run_gpu_optimized_pipeline.sh

# Multiple options combined
DATASET_NAME="pilot_study" \
RUN_DESCRIPTION="baseline_test" \
CROSS_VALIDATION=true \
VERBOSE=true \
./run_gpu_optimized_pipeline.sh
```

## 📈 Understanding Results

### Key Files to Check
1. **`*_analysis_results.csv`** - Detailed performance metrics for all models
2. **`*_executive_summary.txt`** - Quick overview of best performing models
3. **`*_performance_dashboard.png`** - Visual comparison of all models
4. **`*_who_compliance_analysis.png`** - Clinical compliance analysis

### Key Metrics to Look For
- **Sensitivity**: % of TB cases correctly identified (target: ≥90%)
- **Specificity**: % of non-TB cases correctly identified (target: ≥70%)
- **WHO_Compliant**: Boolean indicating if model meets clinical standards
- **WHO_Score**: min(sensitivity, specificity) for compliant models

## 🎯 Current Performance & Next Steps

### Typical Baseline Results
- **Best performing models**: Usually XGBoost variants and MLP (GPU)
- **Current sensitivity**: ~60-80% (needs improvement for WHO compliance)
- **Current specificity**: ~70-90% (typically meets WHO standards)
- **WHO compliance**: 0-30% of models typically meet both targets

### Immediate Improvement Opportunities
1. **Threshold optimization** - Quick win for sensitivity boost
2. **Ensemble methods** - Combine best models
3. **Custom loss functions** - Train specifically for WHO compliance

## 📚 Deep Dive Resources

### For Detailed Understanding
- **`WHO_Compliance_Improvement_Recommendations.txt`** - 48-page comprehensive guide
  - Implementation roadmap (8-week plan)
  - Advanced techniques (custom loss functions, feature engineering)
  - Expected performance gains per technique

### For Technical Implementation
- **`config_gpu.py`** - Configuration system with GPU optimization
- **`03_tb_detection_gpu_optimized.py`** - Main pipeline implementation
- **`run_comparison_tool.py`** - Cross-experiment analysis utility

## 🤝 Collaboration Workflow

### Sharing Results
1. **Run experiments** with descriptive `RUN_DESCRIPTION`
2. **Share results files** from `results/` folder
3. **Include config file** from `configs/` for reproducibility
4. **Use comparison tool** to analyze multiple runs

### Experiment Tracking
```bash
# Example naming convention
RUN_DESCRIPTION="baseline_v1" ./run_gpu_optimized_pipeline.sh
RUN_DESCRIPTION="threshold_optimized_v1" ./run_gpu_optimized_pipeline.sh
RUN_DESCRIPTION="ensemble_test_v1" ./run_gpu_optimized_pipeline.sh

# Compare results
python run_comparison_tool.py --config_pattern="configs/*baseline*"
```

## 🆘 Troubleshooting

### Common Issues
1. **"Virtual environment not found"** 
   - Check that `~/python/venvs/v_audium_hear/` exists
   - Activate manually: `source ~/python/venvs/v_audium_hear/bin/activate`

2. **"XGBoost GPU support not available"**
   - Expected behavior - XGBoost falls back to CPU
   - PyTorch MLP still uses GPU acceleration

3. **"Embeddings file not found"**
   - Ensure `data/final_embeddings.npz` exists
   - Or specify different file: `EMBEDDINGS_FILE="your_file.npz"`

4. **Missing visualizations**
   - Ensure matplotlib and seaborn are installed
   - Check that results directory has write permissions

### Getting Help
- Check pipeline logs for specific error messages
- Review configuration files in `configs/` directory
- Compare with working examples in `results/` directory

## 🎯 Success Metrics

### Pipeline Working Correctly When:
- ✅ All 8 models complete training without errors
- ✅ Visualizations generate (6+ PNG files per run)
- ✅ CSV results files contain performance metrics
- ✅ Executive summary shows model rankings
- ✅ Cross-validation completes within 15 minutes

### Ready for Clinical Application When:
- ✅ At least one model achieves WHO compliance (≥90% sens, ≥70% spec)
- ✅ Consistent performance across cross-validation folds
- ✅ Clear understanding of false positive/negative tradeoffs
- ✅ Validated on hold-out test set

## 📞 Next Steps

### For New Users:
1. **Run baseline** with current pipeline
2. **Review results** and identify performance gaps
3. **Read improvement recommendations** document
4. **Implement quick wins** (threshold optimization, ensembles)

### For Advanced Users:
1. **Implement custom loss functions** for WHO optimization
2. **Add new model architectures** (CatBoost, improved MLP)
3. **Integrate clinical features** and domain knowledge
4. **Develop patient-level aggregation** strategies

---

**Questions? Issues? Improvements?**
- Check existing results in `results/` for examples
- Review `WHO_Compliance_Improvement_Recommendations.txt` for detailed guidance
- Test with small datasets first using `RUN_DESCRIPTION="test"`

**Pipeline Status**: ✅ Production ready with GPU acceleration and comprehensive evaluation
**Last Updated**: July 2025
**Pipeline Version**: GPU-Optimized v1.0