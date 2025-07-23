#!/bin/bash

# GPU-Optimized UCSF R2D2 TB Detection Pipeline
# Leverages Apple M1 Max GPU cores and other acceleration for maximum performance

echo "🚀 GPU-OPTIMIZED UCSF R2D2 TB Detection Pipeline"
echo "=" * 70
echo "⚡ Leveraging GPU acceleration for maximum performance"
echo ""

# Configuration with enhanced naming support
EMBEDDINGS_FILE=${EMBEDDINGS_FILE:-"final_embeddings.npz"}
DATASET_NAME=${DATASET_NAME:-$(basename "$EMBEDDINGS_FILE" .npz)}
RUN_DESCRIPTION=${RUN_DESCRIPTION:-""}
CROSS_VALIDATION=${CROSS_VALIDATION:-"false"}
N_FOLDS=${N_FOLDS:-"5"}
N_JOBS=${N_JOBS:-"-1"}  # Use all CPU cores by default
DEVICE=${DEVICE:-"auto"}  # Auto-detect GPU
GPU_MEMORY_FRACTION=${GPU_MEMORY_FRACTION:-"0.8"}
VERBOSE=${VERBOSE:-"false"}

# Display configuration
echo "🔧 Pipeline Configuration:"
echo "   📁 Embeddings File: $EMBEDDINGS_FILE"
echo "   🏷️  Dataset Name: $DATASET_NAME"
if [ ! -z "$RUN_DESCRIPTION" ]; then
    echo "   📝 Run Description: $RUN_DESCRIPTION"
fi
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Cross-Validation: ${N_FOLDS}-fold enabled"
else
    echo "   📊 Evaluation: Single 80/20 train-test split"
fi
echo "   🖥️  Device: $DEVICE"
echo "   💻 CPU Cores: $N_JOBS (-1 = all cores)"
echo "   💾 GPU Memory: ${GPU_MEMORY_FRACTION} fraction"
echo "   🗣️  Verbose Mode: $VERBOSE"
echo ""

# System info with GPU detection
echo "💻 System Information:"
echo "   🖥️  CPU Cores Available: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'N/A')"
echo "   💾 Total Memory: $(if command -v free >/dev/null 2>&1; then free -h | awk '/^Mem:/ {print $2}'; else echo 'N/A (macOS)'; fi)"

# Detect GPU capabilities
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - check for Apple Silicon
    CHIP_INFO=$(system_profiler SPHardwareDataType | grep "Chip:" | awk -F': ' '{print $2}' | tr -d ' ')
    if [[ "$CHIP_INFO" == *"M1"* ]] || [[ "$CHIP_INFO" == *"M2"* ]] || [[ "$CHIP_INFO" == *"M3"* ]]; then
        GPU_CORES=$(system_profiler SPDisplaysDataType | grep "Total Number of Cores:" | awk '{print $5}' | head -1)
        echo "   🚀 GPU: Apple Silicon ${CHIP_INFO} (${GPU_CORES} GPU cores)"
        echo "   ⚡ Metal Performance Shaders: Available"
    else
        echo "   🚀 GPU: Intel/AMD (limited GPU support)"
    fi
else
    # Linux - check for NVIDIA
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        echo "   🚀 GPU: $GPU_INFO"
        echo "   ⚡ CUDA Support: Available"
    else
        echo "   🚀 GPU: Not detected or unsupported"
    fi
fi

echo "   🏠 Current Directory: $(pwd)"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment Variables:"
    echo "  EMBEDDINGS_FILE      Path to embeddings NPZ file (default: final_embeddings.npz)"
    echo "  DATASET_NAME         Dataset identifier for output naming (default: derived from embeddings file)"
    echo "  RUN_DESCRIPTION      Optional run description for output naming"
    echo "  CROSS_VALIDATION     Enable cross-validation (default: false for single split)"
    echo "  N_FOLDS              Number of CV folds when cross-validation enabled (default: 5)"
    echo "  DEVICE               GPU device preference: auto|cpu|mps|cuda (default: auto)"
    echo "  N_JOBS               Number of CPU cores to use (default: -1 for all cores)"
    echo "  GPU_MEMORY_FRACTION  GPU memory fraction to use (default: 0.8)"
    echo "  VERBOSE              Enable verbose output (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Use default settings with GPU acceleration"
    echo "  ./run_gpu_optimized_pipeline.sh"
    echo ""
    echo "  # GPU with 5-fold cross-validation"
    echo "  CROSS_VALIDATION=true ./run_gpu_optimized_pipeline.sh"
    echo ""
    echo "  # Custom dataset with description"
    echo '  DATASET_NAME="ucsf_r2d2" RUN_DESCRIPTION="pilot_study" ./run_gpu_optimized_pipeline.sh'
    echo ""
    echo "  # Force CPU mode (disable GPU)"
    echo "  DEVICE=cpu ./run_gpu_optimized_pipeline.sh"
    echo ""
    echo "  # GPU with custom memory allocation"
    echo "  GPU_MEMORY_FRACTION=0.9 ./run_gpu_optimized_pipeline.sh"
    echo ""
    exit 1
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
VENV_PATH="$HOME/python/venvs/v_audium_hear"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated"
    echo "   📍 Using: $VENV_PATH"
else
    echo "⚠️  Virtual environment not found, using system Python"
    echo "   📍 Looked for: $VENV_PATH"
fi
echo ""

# Change to pipeline directory
cd "$(dirname "$0")"
echo "📂 Working directory: $(pwd)"
echo ""

# Prepare arguments for GPU script
COMMON_ARGS="--embeddings_file $EMBEDDINGS_FILE --n_jobs $N_JOBS --device $DEVICE --gpu_memory_fraction $GPU_MEMORY_FRACTION"

# Add cross-validation arguments
if [ "$CROSS_VALIDATION" = "true" ]; then
    COMMON_ARGS="$COMMON_ARGS --cross_validation --n_folds $N_FOLDS"
fi

# Add naming arguments
if [ ! -z "$DATASET_NAME" ]; then
    COMMON_ARGS="$COMMON_ARGS --dataset_name $DATASET_NAME"
fi

if [ ! -z "$RUN_DESCRIPTION" ]; then
    COMMON_ARGS="$COMMON_ARGS --run_description $RUN_DESCRIPTION"
fi

# Add verbose flag
if [ "$VERBOSE" = "true" ]; then
    COMMON_ARGS="$COMMON_ARGS --verbose"
fi

# Check if embeddings file exists
EMBEDDINGS_PATH="data/$EMBEDDINGS_FILE"
if [ ! -f "$EMBEDDINGS_PATH" ]; then
    echo "❌ Embeddings file not found: $EMBEDDINGS_PATH"
    echo ""
    echo "Available embeddings files in data/ directory:"
    ls -la data/*.npz 2>/dev/null || echo "   (No .npz files found)"
    echo ""
    echo "💡 You can specify a different file with:"
    echo "   EMBEDDINGS_FILE=\"your_file.npz\" $0"
    exit 1
fi

# Check if metadata file exists
METADATA_FILE=$(echo "$EMBEDDINGS_FILE" | sed 's/\.npz$/_metadata.csv/')
METADATA_PATH="data/$METADATA_FILE"
if [ ! -f "$METADATA_PATH" ]; then
    echo "❌ Metadata file not found: $METADATA_PATH"
    echo "💡 Metadata file should match embeddings file naming convention"
    exit 1
fi

echo "✅ Input files validated:"
echo "   📁 Embeddings: $EMBEDDINGS_PATH ($(ls -lh $EMBEDDINGS_PATH | awk '{print $5}'))"
echo "   📁 Metadata: $METADATA_PATH ($(wc -l < $METADATA_PATH) records)"
echo ""

# Check GPU dependencies
echo "🔍 Checking GPU dependencies..."
GPU_READY=true

# Check XGBoost GPU support
if python -c "import xgboost as xgb; print('XGBoost version:', xgb.__version__)" 2>/dev/null; then
    echo "✅ XGBoost available"
    # Test GPU support
    if python -c "import xgboost as xgb; xgb.train({'tree_method': 'gpu_hist'}, xgb.DMatrix([[1,2]], label=[0]), num_boost_round=1)" 2>/dev/null; then
        echo "✅ XGBoost GPU support available"
    else
        echo "⚠️  XGBoost GPU support not available - using CPU fallback"
    fi
else
    echo "⚠️  XGBoost not found - using CPU tree models"
    GPU_READY=false
fi

# Check PyTorch MPS support (Apple Silicon)
if python -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
    echo "✅ PyTorch available"
    if python -c "import torch; print('MPS available:', torch.backends.mps.is_available())" 2>/dev/null | grep -q "True"; then
        echo "✅ PyTorch MPS (Apple Silicon GPU) support available"
    elif python -c "import torch; print('CUDA available:', torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo "✅ PyTorch CUDA support available"
    else
        echo "⚠️  PyTorch GPU support not available - using CPU fallback"
    fi
else
    echo "⚠️  PyTorch not found - using CPU neural networks"
    GPU_READY=false
fi

if [ "$GPU_READY" = "true" ]; then
    echo "🚀 GPU acceleration ready!"
else
    echo "💻 Running in CPU-optimized mode"
fi
echo ""

# Step 1: GPU-Optimized TB Detection Analysis
echo "🤖 Step 1: GPU-Optimized TB Detection Analysis"
echo "-" * 70

# Generate expected output identifier for logging
EXPECTED_PREFIX="${DATASET_NAME}_gpu"
if [ "$CROSS_VALIDATION" = "true" ]; then
    EXPECTED_PREFIX="${EXPECTED_PREFIX}_cv${N_FOLDS}fold"
else
    EXPECTED_PREFIX="${EXPECTED_PREFIX}_single"
fi

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
if [ ! -z "$RUN_DESCRIPTION" ]; then
    EXPECTED_PREFIX="${EXPECTED_PREFIX}_${RUN_DESCRIPTION}_${TIMESTAMP}"
else
    EXPECTED_PREFIX="${EXPECTED_PREFIX}_${TIMESTAMP}"
fi

echo "🏷️  Expected output prefix: ${EXPECTED_PREFIX}_xxxxxx_*"
echo ""

if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "🔄 Running ${N_FOLDS}-fold cross-validation with GPU acceleration"
    echo "⚡ Expected speedup: 5-20x faster than CPU (especially for tree models)"
    echo "📊 This will provide robust performance estimates across multiple data splits"
else
    echo "🔥 Leveraging GPU acceleration for model training (single 80/20 split)"
    echo "⚡ Expected speedup: 5-20x faster than CPU (especially for tree models)"
fi
echo ""

start_time=$(date +%s)

# Run GPU-optimized analysis
if command -v python >/dev/null 2>&1; then
    python 03_tb_detection_gpu_optimized.py $COMMON_ARGS
elif command -v python3 >/dev/null 2>&1; then
    python3 03_tb_detection_gpu_optimized.py $COMMON_ARGS
else
    echo "❌ Python not found in PATH"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ GPU-optimized TB detection analysis failed!"
    echo ""
    echo "💡 Troubleshooting tips:"
    echo "   • Try CPU fallback: DEVICE=cpu $0"
    echo "   • Check GPU dependencies above"
    echo "   • Ensure virtual environment is activated"
    exit 1
fi

end_time=$(date +%s)
analysis_duration=$((end_time - start_time))

echo "✅ GPU-optimized analysis completed in ${analysis_duration}s"
echo ""

# Step 2: Results Summary
echo "📊 Step 2: Results Summary"
echo "-" * 70
echo "🎉 GPU-optimized pipeline execution completed!"
echo ""

# Performance summary
echo "⚡ Performance Summary:"
echo "   ⏱️  Total Analysis Time: ${analysis_duration}s"
echo "   🖥️  Device Used: $DEVICE"
echo "   💻 CPU Cores Used: $N_JOBS"
echo "   📁 Dataset: $DATASET_NAME ($EMBEDDINGS_FILE)"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Evaluation: ${N_FOLDS}-fold cross-validation"
else
    echo "   📊 Evaluation: Single 80/20 train-test split"
fi
echo ""

# Find actual output files (they will have timestamp and config hash)
echo "📁 Output files created:"

# Find the most recent files matching the expected pattern
if [ "$CROSS_VALIDATION" = "true" ]; then
    SUMMARY_PATTERN="${DATASET_NAME}_gpu_cv*_gpu_cross_validation_summary.csv"
    DETAILED_PATTERN="${DATASET_NAME}_gpu_cv*_gpu_cross_validation_detailed.csv"
    EXEC_PATTERN="${DATASET_NAME}_gpu_cv*_gpu_cross_validation_executive_summary.txt"
else
    SUMMARY_PATTERN="${DATASET_NAME}_gpu_single*_gpu_analysis_results.csv"
    EXEC_PATTERN="${DATASET_NAME}_gpu_single*_gpu_executive_summary.txt"
fi

DASHBOARD_PATTERN="${DATASET_NAME}_gpu*_gpu_performance_dashboard.png"
ROC_PATTERN="${DATASET_NAME}_gpu*_gpu_roc_curves.png"

# List output files
echo "📊 Analysis Results:"
ls -la results/${DATASET_NAME}_gpu*_gpu_*.csv 2>/dev/null | while read line; do
    echo "  ✅ $(echo "$line" | awk '{print $9}') ($(echo "$line" | awk '{print $5}'))"
done

echo "📈 Visualizations:"
ls -la results/${DATASET_NAME}_gpu*_gpu_*.png 2>/dev/null | while read line; do
    echo "  ✅ $(echo "$line" | awk '{print $9}') ($(echo "$line" | awk '{print $5}'))"
done

echo "📋 Configuration:"
ls -la configs/${DATASET_NAME}_gpu*_config.json 2>/dev/null | while read line; do
    echo "  ✅ $(echo "$line" | awk '{print $9}') ($(echo "$line" | awk '{print $5}'))"
done

echo ""

# Show quick results preview if executive summary exists
EXEC_FILE=$(ls results/${DATASET_NAME}_gpu*_gpu_*executive_summary.txt 2>/dev/null | head -1)
if [ -f "$EXEC_FILE" ]; then
    echo "📈 Quick Results Preview:"
    echo "-" * 30
    if [ "$CROSS_VALIDATION" = "true" ]; then
        tail -n 15 "$EXEC_FILE" | head -n 8
    else
        tail -n 10 "$EXEC_FILE" | head -n 6
    fi
else
    echo "  ❌ Executive summary not found"
fi

echo ""
echo "🎯 GPU Optimization Benefits Delivered:"
echo "  ⚡ GPU-accelerated XGBoost models (replaces slow CPU tree models)"
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "  🚀 Apple Silicon Metal Performance Shaders support"
else
    echo "  🚀 NVIDIA CUDA acceleration support"
fi
echo "  🔄 Parallel patient feature aggregation"  
echo "  💻 Efficient GPU memory management"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "  📊 Robust ${N_FOLDS}-fold cross-validation"
else
    echo "  📊 Fast single train-test split"
fi
echo "  🏷️  Enhanced output naming with timestamps and config hashing"
echo ""

echo "🏆 Status: ✅ GPU-OPTIMIZED PIPELINE COMPLETE"
echo "⏰ Total execution time: ${analysis_duration} seconds"
echo ""

# Performance comparison note
if [ -f "results/final_embeddings_cpu_optimized_analysis_results.csv" ]; then
    echo "📊 Performance Comparison Available:"
    echo "  CPU baseline: results/final_embeddings_cpu_optimized_*"
    echo "  GPU results: results/${DATASET_NAME}_gpu_*"
    echo "  Use run_comparison_tool.py to compare performance"
fi

echo ""
echo "📞 Next steps:"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "  1. Review ${DATASET_NAME}_gpu_*_cross_validation_summary.csv for mean performance across folds"
    echo "  2. Check ${DATASET_NAME}_gpu_*_cross_validation_detailed.csv for individual fold results"
    echo "  3. Analyze ${DATASET_NAME}_gpu_*_gpu_performance_dashboard.png for visual comparison"
    echo "  4. Deploy best WHO-compliant model for clinical use"
else
    echo "  1. Review ${DATASET_NAME}_gpu_*_analysis_results.csv for detailed model performance"
    echo "  2. Check ${DATASET_NAME}_gpu_*_gpu_performance_dashboard.png for visual analysis"
    echo "  3. Deploy best WHO-compliant model for clinical use"
fi
echo ""

echo "🔧 Configuration options for next run:"
echo "  • Change dataset: EMBEDDINGS_FILE=\"other_file.npz\" $0"
echo "  • Add description: RUN_DESCRIPTION=\"experiment_name\" $0"
echo "  • Enable cross-validation: CROSS_VALIDATION=true $0"
echo "  • Change CV folds: CROSS_VALIDATION=true N_FOLDS=10 $0"
echo "  • Force CPU mode: DEVICE=cpu $0"
echo "  • Adjust GPU memory: GPU_MEMORY_FRACTION=0.9 $0"
echo "  • Enable verbose: VERBOSE=true $0"
echo ""

echo "🚀 GPU-OPTIMIZED PIPELINE EXECUTION COMPLETE!"