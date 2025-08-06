#!/bin/bash

# CPU-Optimized UCSF R2D2 TB Detection Pipeline
# Multi-core parallel processing for faster execution

echo "🚀 CPU-OPTIMIZED UCSF R2D2 TB Detection Pipeline"
echo "=" * 70
echo "⚡ Utilizing full CPU power for maximum performance"
echo ""

# Configuration
EMBEDDINGS_FILE=${EMBEDDINGS_FILE:-"final_embeddings.npz"}
CROSS_VALIDATION=${CROSS_VALIDATION:-"false"}
N_FOLDS=${N_FOLDS:-"5"}
N_JOBS=${N_JOBS:-"-1"}  # Use all CPU cores by default
VERBOSE=${VERBOSE:-"false"}

# Display configuration
echo "🔧 Pipeline Configuration:"
echo "   📁 Embeddings File: $EMBEDDINGS_FILE"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Cross-Validation: ${N_FOLDS}-fold enabled"
else
    echo "   📊 Evaluation: Single 80/20 train-test split"
fi
echo "   💻 CPU Cores: $N_JOBS (-1 = all cores)"
echo "   🗣️  Verbose Mode: $VERBOSE"
echo ""

# System info
echo "💻 System Information:"
echo "   🖥️  CPU Cores Available: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'N/A')"
echo "   💾 Total Memory: $(if command -v free >/dev/null 2>&1; then free -h | awk '/^Mem:/ {print $2}'; else echo 'N/A (macOS)'; fi)"
echo "   🏠 Current Directory: $(pwd)"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
if [ -d "~/python/venvs/v_audium_hear" ]; then
    source ~/python/venvs/v_audium_hear/bin/activate
    echo "✅ Virtual environment activated"
else
    echo "⚠️  Virtual environment not found, using system Python"
fi
echo ""

# Change to pipeline directory
cd "$(dirname "$0")"
echo "📂 Working directory: $(pwd)"
echo ""

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Environment Variables:"
    echo "  EMBEDDINGS_FILE    Path to embeddings NPZ file (default: final_embeddings.npz)"
    echo "  CROSS_VALIDATION   Enable cross-validation (default: false for single split)"
    echo "  N_FOLDS            Number of CV folds when cross-validation enabled (default: 5)"
    echo "  N_JOBS             Number of CPU cores to use (default: -1 for all cores)"
    echo "  VERBOSE            Enable verbose output (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Use default settings (single 80/20 split)"
    echo "  ./run_cpu_optimized_pipeline.sh"
    echo ""
    echo "  # Enable 5-fold cross-validation"
    echo "  CROSS_VALIDATION=true ./run_cpu_optimized_pipeline.sh"
    echo ""
    echo "  # Use different embeddings with 3-fold cross-validation"
    echo "  EMBEDDINGS_FILE=\"my_embeddings.npz\" CROSS_VALIDATION=true N_FOLDS=3 ./run_cpu_optimized_pipeline.sh"
    echo ""
    echo "  # Use 8 CPU cores with verbose output"
    echo "  N_JOBS=8 VERBOSE=true ./run_cpu_optimized_pipeline.sh"
    echo ""
    exit 1
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
fi

# Prepare common arguments
COMMON_ARGS="--embeddings_file $EMBEDDINGS_FILE --n_jobs $N_JOBS --n_folds $N_FOLDS"
if [ "$CROSS_VALIDATION" = "true" ]; then
    COMMON_ARGS="$COMMON_ARGS --cross_validation"
fi
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

# Step 1: CPU-Optimized TB Detection Analysis
echo "🤖 Step 1: CPU-Optimized TB Detection Analysis"
echo "-" * 70
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "🔄 Running ${N_FOLDS}-fold cross-validation with parallel processing"
    echo "⚡ Expected speedup: 2-5x faster than sequential processing"
    echo "📊 This will provide robust performance estimates across multiple data splits"
else
    echo "🔥 Leveraging parallel processing for model training (single 80/20 split)"
    echo "⚡ Expected speedup: 2-5x faster than sequential processing"
fi
echo ""

start_time=$(date +%s)

if command -v python >/dev/null 2>&1; then
    python 03_tb_detection_cpu_optimized.py $COMMON_ARGS
elif command -v python3 >/dev/null 2>&1; then
    python3 03_tb_detection_cpu_optimized.py $COMMON_ARGS
else
    echo "❌ Python not found in PATH"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ CPU-optimized TB detection analysis failed!"
    exit 1
fi

end_time=$(date +%s)
analysis_duration=$((end_time - start_time))

echo "✅ CPU-optimized analysis completed in ${analysis_duration}s"
echo ""

# Step 2: Results Summary
echo "📊 Step 2: Results Summary"
echo "-" * 70
echo "🎉 CPU-optimized pipeline execution completed!"
echo ""

# Performance summary
echo "⚡ Performance Summary:"
echo "   ⏱️  Total Analysis Time: ${analysis_duration}s"
echo "   💻 CPU Cores Used: $N_JOBS"
echo "   📁 Dataset: $EMBEDDINGS_FILE"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Evaluation: ${N_FOLDS}-fold cross-validation"
else
    echo "   📊 Evaluation: Single 80/20 train-test split"
fi
echo ""

# Get embeddings basename for filename prefixing
EMBEDDINGS_BASENAME=$(basename "$EMBEDDINGS_FILE" .npz)

echo "📁 Output files created:"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "  📊 CV Summary: results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_summary.csv"
    echo "  📋 CV Detailed: results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_detailed.csv"
    echo "  📈 Visualizations: results/${EMBEDDINGS_BASENAME}_cpu_optimized_model_comparison.png"
    echo "  📄 Executive Summary: results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_executive_summary.txt"
else
    echo "  📊 Analysis Results: results/${EMBEDDINGS_BASENAME}_cpu_optimized_analysis_results.csv"
    echo "  📈 Visualizations: results/${EMBEDDINGS_BASENAME}_cpu_optimized_model_comparison.png"
    echo "  📋 Executive Summary: results/${EMBEDDINGS_BASENAME}_cpu_optimized_executive_summary.txt"
fi
echo ""

# Check if output files exist
echo "📋 File validation:"
if [ "$CROSS_VALIDATION" = "true" ]; then
    CV_SUMMARY_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_summary.csv"
    CV_DETAILED_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_detailed.csv"
    CV_EXEC_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_executive_summary.txt"
    
    if [ -f "$CV_SUMMARY_FILE" ]; then
        echo "  ✅ Cross-validation summary created ($(wc -l < $CV_SUMMARY_FILE) models)"
    else
        echo "  ❌ Cross-validation summary missing"
    fi
    
    if [ -f "$CV_DETAILED_FILE" ]; then
        echo "  ✅ Cross-validation detailed results created ($(wc -l < $CV_DETAILED_FILE) fold results)"
    else
        echo "  ❌ Cross-validation detailed results missing"
    fi
    
    if [ -f "$CV_EXEC_FILE" ]; then
        echo "  ✅ Executive summary created"
        echo ""
        echo "📈 Quick Results Preview:"
        echo "-" * 30
        tail -n 15 "$CV_EXEC_FILE" | head -n 8
    else
        echo "  ❌ Executive summary missing"
    fi
else
    ANALYSIS_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_analysis_results.csv"
    EXEC_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_executive_summary.txt"
    
    if [ -f "$ANALYSIS_FILE" ]; then
        echo "  ✅ Analysis results created ($(wc -l < $ANALYSIS_FILE) models)"
    else
        echo "  ❌ Analysis results missing"
    fi
    
    if [ -f "$EXEC_FILE" ]; then
        echo "  ✅ Executive summary created"
        echo ""
        echo "📈 Quick Results Preview:"
        echo "-" * 30
        tail -n 10 "$EXEC_FILE" | head -n 6
    else
        echo "  ❌ Executive summary missing"
    fi
fi

VIZ_FILE="results/${EMBEDDINGS_BASENAME}_cpu_optimized_model_comparison.png"
if [ -f "$VIZ_FILE" ]; then
    echo "  ✅ Visualization created ($(ls -lh $VIZ_FILE | awk '{print $5}'))"
else
    echo "  ❌ Visualization missing"
fi

echo ""
echo "🎯 CPU Optimization Benefits:"
echo "  ⚡ Multi-core model training"
echo "  🔄 Parallel patient feature aggregation"  
echo "  💻 Efficient resource utilization"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "  📊 Robust ${N_FOLDS}-fold cross-validation"
else
    echo "  📊 Fast single train-test split"
fi
echo "  🎛️  Flexible embeddings file selection"
echo ""

echo "🏆 Status: ✅ CPU-OPTIMIZED PIPELINE COMPLETE"
echo "⏰ Total execution time: ${analysis_duration} seconds"
echo ""

echo "📞 Next steps:"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "  1. Review results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_summary.csv for mean performance across folds"
    echo "  2. Check results/${EMBEDDINGS_BASENAME}_cpu_optimized_cross_validation_detailed.csv for individual fold results"
    echo "  3. Analyze results/${EMBEDDINGS_BASENAME}_cpu_optimized_model_comparison.png for visual comparison"
    echo "  4. Deploy best WHO-compliant model for clinical use"
else
    echo "  1. Review results/${EMBEDDINGS_BASENAME}_cpu_optimized_analysis_results.csv for detailed model performance"
    echo "  2. Check results/${EMBEDDINGS_BASENAME}_cpu_optimized_model_comparison.png for visual analysis"
    echo "  3. Deploy best WHO-compliant model for clinical use"
fi
echo ""

echo "🔧 Configuration options for next run:"
echo "  • Change embeddings: EMBEDDINGS_FILE=\"other_file.npz\" $0"
echo "  • Enable cross-validation: CROSS_VALIDATION=true $0"
echo "  • Change CV folds: CROSS_VALIDATION=true N_FOLDS=10 $0"
echo "  • Set CPU cores: N_JOBS=8 $0 (to use 8 cores)"
echo "  • Enable verbose: VERBOSE=true $0"
echo ""

echo "🚀 CPU-OPTIMIZED PIPELINE EXECUTION COMPLETE!"