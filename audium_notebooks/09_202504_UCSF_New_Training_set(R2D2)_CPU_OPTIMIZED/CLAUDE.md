# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a specialized TB (tuberculosis) detection pipeline built on Google's Health Acoustic Representations (HeAR) model for analyzing audio recordings from the UCSF R2D2 dataset. The pipeline focuses on achieving WHO-specified clinical performance targets (≥90% sensitivity, ≥70% specificity) for TB screening applications.

## Environment

### Default Python Virtual Environment
- By default, use this environment: `/Users/abelvillcaroque/python/venvs/v_audium_hear`

## Core Architecture

### Pipeline Stages
1. **Data Validation**: `data_validation_final.py` - Validates UCSF R2D2 dataset structure, excludes contaminated files (R2D201001), creates clean patient mappings
2. **Embedding Generation**: `02_generate_embeddings_*.py` - Processes audio files through HeAR model to create 1024-dim embeddings
3. **TB Detection**: `03_tb_detection_*.py` - Multi-algorithm ML pipeline with WHO compliance optimization
4. **Results Analysis**: Comprehensive evaluation with visualizations and clinical metrics

### Key Components

#### Configuration System
- **`config.py`**: CPU-optimized configuration with multi-core support
- **`config_gpu.py`**: GPU-accelerated configuration for enhanced performance
- Supports flexible file naming, experiment tracking, and reproducible runs

#### Execution Modes
- **CPU-Optimized** (`*_cpu_optimized.py`): Multi-core parallel processing for standard hardware
- **GPU-Optimized** (`*_gpu_optimized.py`): CUDA acceleration with PyTorch MLP and enhanced algorithms
- **Cross-Validation**: 3-fold or 5-fold CV for robust performance estimation
- **Single Split**: Fast 80/20 train-test evaluation

## Data Structure

### Expected Data Files
- **Embeddings**: `data/final_embeddings.npz` - HeAR model embeddings (1024-dimensional)
- **Metadata**: `data/final_embeddings_metadata.csv` - File-to-patient mapping with clip counts
- **Labels**: `data/clean_patients_final.csv` - Patient TB status (543 patients after R2D201001 exclusion)

### Dataset Characteristics
- **543 clean patients** (167 TB+, 376 TB-) after excluding R2D201001 contamination
- **10,682 audio files** with proper patient attribution
- **~110,000 audio clips** (2-second clips with 10% overlap, silence filtered at -50dB)

## Common Commands

### Quick Start (Most Common Workflow)
```bash
# Activate environment (if needed)
source ~/python/venvs/v_audium_hear/bin/activate

# Basic GPU-optimized run (recommended)
./run_gpu_optimized_pipeline.sh

# CPU-optimized alternative
./run_cpu_optimized_pipeline.sh

# Cross-validation for robust results
CROSS_VALIDATION=true ./run_gpu_optimized_pipeline.sh
```

### Data Preparation (If Starting from Raw Data)
```bash
# Validate dataset and exclude contaminated files
python data_validation_final.py

# Generate embeddings (CPU-optimized)
python 02_generate_embeddings_cpu_optimized.py --input_dir /path/to/audio --output_prefix final_embeddings

# Generate embeddings (GPU-accelerated - if available)
python 02_generate_embeddings_final.py --input_dir /path/to/audio
```

### Advanced Analysis Options
```bash
# Custom experiment with description
RUN_DESCRIPTION="sensitivity_optimization" CROSS_VALIDATION=true ./run_gpu_optimized_pipeline.sh

# Compare multiple runs
python run_comparison_tool.py --config_pattern="configs/*baseline*"

# Different embeddings file
EMBEDDINGS_FILE="other_embeddings.npz" ./run_gpu_optimized_pipeline.sh

# Force CPU mode (disable GPU)
DEVICE=cpu ./run_gpu_optimized_pipeline.sh
```

### Analysis and Debugging
```bash
# Validate data consistency
python validate_consistency.py

# Test pipeline before full run
python test_fixes_before_run.py

# Investigate specific patient data
python investigate_outliers.py
```

## ML Algorithm Suite

### Core Models (Both CPU and GPU Versions)
- **Naive Bayes**: Fast baseline with Gaussian distribution assumption
- **Linear Discriminant Analysis (LDA)**: Linear dimensionality reduction classifier
- **Logistic Regression**: L1/L2 regularized with balanced class weights
- **K-Nearest Neighbors (KNN)**: Distance-based classification
- **XGBoost**: Gradient boosting (CPU-only, falls back from GPU gracefully)

### Enhanced Models (GPU Version Only)
- **PyTorch MLP**: GPU-accelerated neural network with WHO-optimized architecture
- **Advanced Ensemble**: Sensitivity-weighted combination of multiple models
- **Clinical Hybrid**: Integration of audio features with clinical metadata

### WHO Optimization Features
- **Custom Loss Functions**: Sensitivity-prioritized training objectives
- **Threshold Optimization**: Automated clinical decision threshold tuning
- **Ensemble Weighting**: Sensitivity-weighted model combination
- **Clinical Compliance Scoring**: WHO_Score = min(sensitivity, specificity) for qualifying models

## Performance Targets

### Clinical Requirements (WHO Standards)
- **Primary Target**: ≥90% sensitivity, ≥70% specificity
- **Current Performance**: ~60-80% sensitivity, ~70-90% specificity
- **WHO Compliance Rate**: 0-30% of models typically achieve both targets

### Evaluation Methodology
- **Patient-level splits**: Prevents data leakage (all clips from a patient in same split)
- **Multiple aggregation strategies**: Any Positive, Majority Vote, Threshold-based
- **Cross-validation**: 3-fold or 5-fold for robust performance estimation
- **Comprehensive metrics**: Sensitivity, specificity, precision, NPV, F1, ROC AUC, PR AUC

## Output Structure

### Results Directory (`results/`)
- **Analysis CSV**: Detailed model performance metrics
- **Executive Summary**: Text summary of best models and WHO compliance
- **Visualizations**: ROC curves, confusion matrices, performance dashboards
- **Cross-validation**: Fold-by-fold analysis and variance assessment

### Configuration Tracking (`configs/`)
- **JSON configs**: Complete run parameters for reproducibility
- **Experiment tracking**: Timestamped runs with unique identifiers
- **Hash-based naming**: Consistent identification of identical configurations

## Key Files to Monitor

### Critical Pipeline Components
- `03_tb_detection_gpu_optimized.py:1847` - Main analysis entry point
- `config_gpu.py:156` - GPU configuration and WHO optimization settings
- `run_gpu_optimized_pipeline.sh:121` - Primary execution script

### Data Validation and Quality
- `data_validation_final.py:89` - Clean dataset creation with R2D201001 exclusion
- `validate_consistency.py:78` - Cross-run consistency validation

### Performance Analysis
- `run_comparison_tool.py:245` - Multi-experiment comparison and trending
- `WHO_Compliance_Improvement_Recommendations.txt` - 48-page optimization guide

## Troubleshooting

### Common Issues
1. **Missing embeddings**: Run `02_generate_embeddings_*.py` first
2. **R2D201001 contamination**: Use `data_validation_final.py` to exclude problematic files
3. **Memory issues**: Reduce batch sizes or use CPU-optimized versions
4. **GPU not available**: Pipeline gracefully falls back to CPU processing

### Performance Optimization
- **Multi-core**: Use `N_JOBS=-1` for all available CPU cores
- **GPU acceleration**: Ensure CUDA-compatible PyTorch installation
- **Cross-validation**: Balance robustness (CV) vs speed (single split)
- **Batch processing**: Adjust batch sizes based on available memory

## Development Notes

### Configuration Management
- All scripts use centralized configuration system (`config.py`, `config_gpu.py`)
- Experiment tracking with unique run identifiers and JSON config storage
- Flexible naming system supports different datasets and descriptions

### WHO Compliance Focus
- Pipeline specifically optimized for clinical TB screening requirements
- Multiple sensitivity optimization techniques implemented
- Comprehensive compliance reporting and model ranking

### Reproducibility
- All runs generate configuration files for exact reproduction
- Random states controlled for consistent results
- Complete environment and system information captured

This pipeline represents a production-ready TB detection system with comprehensive evaluation, WHO compliance optimization, and flexible deployment options for both CPU and GPU environments.