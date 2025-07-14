# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is the Health Acoustic Representations (HeAR) repository - a Google Health AI Developer Foundations ML model that produces embeddings from health acoustics. The model is trained on 300+ million two-second audio clips and can be used to build AI models for health acoustic analysis with less data and compute.

## Project Structure

### Core Components

- **`python/serving/`** - Production serving implementation for deploying the model on Vertex AI
  - `predictor.py` - Main predictor that processes audio input and calls the model
  - `serving_framework/` - Framework for HTTP serving with inline prediction execution
  - `vertex_schemata/` - YAML files defining PredictSchemata for Vertex AI endpoints
  - `Dockerfile` and `entrypoint.sh` - Container configuration for deployment

- **`python/data_processing/`** - Data processing utilities
  - `audio_utils.py` - Audio processing utilities
  - `data_processing_lib.py` - Core data processing functions

- **`notebooks/`** - Jupyter notebooks demonstrating model usage
  - Quick start examples for Hugging Face (TensorFlow and PyTorch)
  - Vertex AI Model Garden integration examples
  - Data efficient classifier training examples
  - Health event detector demo

- **`audium_notebooks/`** - Additional analysis notebooks (appears to be Audium-specific extensions)

## Development Commands

### Testing
- Tests use `absl.testing` framework (Google's testing library)
- Run individual test files: `python -m absl.testing.absltest <test_file>`
- Test files follow `*_test.py` naming convention

### Dependencies
- Python dependencies are managed via `requirements.txt` files
- Main requirements: `python/requirements.txt` (references `serving/requirements.txt`)
- Serving requirements: `python/serving/requirements.txt`

## Architecture Notes

### Model Serving Architecture
- Follows Google Health AI Developer Foundations container architecture
- Uses `InlinePredictionExecutor` pattern for wrapping predictor functions
- Supports both online (REST endpoint) and batch prediction workflows
- Model input accepts audio in multiple formats: `input_bytes`, `gcs_uri`, or `input_array`

### Audio Processing
- Processes health-related audio clips (typically 2-second clips)
- Uses scipy for audio signal processing
- Generates embeddings that can be used for downstream health acoustic tasks

### Integration Points
- Hugging Face model hosting: `google/hear`
- Vertex AI Model Garden deployment
- Custom container deployment on Vertex AI

## Model Formats
- Available in both TensorFlow and PyTorch formats
- Default input/output keys: `x` (input), `output_0` (output)
- Processes audio clips into embedding vectors

## Audium Health-Specific Components

### **Primary Business Workflows**

1. **Multi-label Cough Classification**
   - **Input**: `.webm` audio files with timestamp-based naming
   - **Processing**: 2-second clips with 10% overlap, silence filtering (-50dB threshold)
   - **Output**: Binary classification (vocal vs lung cough types)
   - **Models**: Multiple scikit-learn classifiers with performance comparison

2. **UCSF/R2D2 TB Detection**
   - **Dataset**: R2D2 patient audio files with TB labels
   - **Structure**: Patient ID format `R2D2NNNNN` with `.wav` files
   - **Labels**: TB Positive/Negative from microbiological reference standard
   - **Pipeline**: Audio indexing → label merging → classification

3. **Seagull Account Analysis**
   - **Purpose**: Customer revenue analysis and segmentation
   - **Methods**: Pareto analysis, customer clustering
   - **Output**: Revenue-based account prioritization

### **Key Technical Patterns**

- **Audio Processing**: Uses librosa (16kHz, mono) with scipy signal processing
- **Embedding Generation**: HeAR model produces 512-dimensional vectors
- **Model Training**: Comparative evaluation of SVM, Logistic Regression, Random Forest, MLP, Gradient Boosting
- **Evaluation Metrics**: Confusion matrices, ROC curves, precision-recall analysis
- **Data Persistence**: Pickle files for embedding cache, CSV for metadata and results

### **Data Structures**

- **Audio Files**: `.webm` format with generated codes: `YYYYMMDDHHMMSS_CODE_TYPE.webm`
- **Embeddings**: Cached in `audium.pkl` and `audium_UCSF_embeddings.npz`
- **Labels**: Multi-class ("vocal", "lung") and binary (TB classification)
- **Models**: Serialized with joblib in `saved_models_multiclass/` directory

### **Development Notes**

- Embeddings are cached to avoid reprocessing large datasets
- Silent clip filtering improves model performance
- Multiple model comparison enables optimal classifier selection
- Evaluation includes both per-clip and per-file (majority vote) accuracy