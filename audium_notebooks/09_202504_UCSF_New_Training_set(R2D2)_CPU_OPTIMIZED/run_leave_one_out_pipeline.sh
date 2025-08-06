#!/bin/bash

# Leave-One-Out TB Detection Validation Pipeline
# Complete pipeline with reserved test dataset validation

echo "🚀 Leave-One-Out TB Detection Validation Pipeline"
echo "="*70
echo "🧪 Comprehensive validation with reserved test dataset for clinical deployment"
echo ""

# Configuration
TEST_PATIENTS_FILE=${TEST_PATIENTS_FILE:-"patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv"}
RUN_DESCRIPTION=${RUN_DESCRIPTION:-"leave_one_out_validation"}
CROSS_VALIDATION=${CROSS_VALIDATION:-"true"}
N_FOLDS=${N_FOLDS:-"5"}
VERBOSE=${VERBOSE:-"false"}

# Display configuration
echo "🔧 Pipeline Configuration:"
echo "   📁 Reserved Test Patients: $TEST_PATIENTS_FILE"
echo "   🏷️  Run Description: $RUN_DESCRIPTION"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Cross-Validation: ${N_FOLDS}-fold enabled"
else
    echo "   📊 Evaluation: Single train-test split"
fi
echo "   🗣️  Verbose Mode: $VERBOSE"
echo ""

# System info
echo "💻 System Information:"
echo "   🖥️  CPU Cores Available: $(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 'N/A')"
echo "   🏠 Current Directory: $(pwd)"
echo ""

# Activate virtual environment
echo "🔄 Activating virtual environment..."
VENV_PATH="/Users/abelvillcaroque/python/venvs/v_audium_hear"
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "✅ Virtual environment activated: $VENV_PATH"
    which python
else
    echo "⚠️  Virtual environment not found at $VENV_PATH, using system Python"
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
    echo "  TEST_PATIENTS_FILE    Path to reserved test patients CSV (default: patient_test_dataset_leave_one_out/Audium_Health_patient_id_test_split.csv)"
    echo "  RUN_DESCRIPTION       Description for this run (default: leave_one_out_validation)"
    echo "  CROSS_VALIDATION      Enable cross-validation (default: true)"
    echo "  N_FOLDS              Number of CV folds (default: 5)"
    echo "  VERBOSE              Enable verbose output (default: false)"
    echo ""
    echo "Examples:"
    echo "  # Basic run with default settings"
    echo "  ./run_leave_one_out_pipeline.sh"
    echo ""
    echo "  # Custom description and 3-fold CV"
    echo "  RUN_DESCRIPTION=\"clinical_validation\" N_FOLDS=3 ./run_leave_one_out_pipeline.sh"
    echo ""
    echo "  # Different test patients file"
    echo "  TEST_PATIENTS_FILE=\"my_test_patients.csv\" ./run_leave_one_out_pipeline.sh"
    echo ""
    exit 1
}

# Check for help flag
if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    show_usage
fi

# Check if test patients file exists
if [ ! -f "$TEST_PATIENTS_FILE" ]; then
    echo "❌ Reserved test patients file not found: $TEST_PATIENTS_FILE"
    echo ""
    echo "Available files in patient_test_dataset_leave_one_out/:"
    ls -la patient_test_dataset_leave_one_out/ 2>/dev/null || echo "   (Directory not found)"
    echo ""
    echo "💡 You can specify a different file with:"
    echo "   TEST_PATIENTS_FILE=\"path/to/your/file.csv\" $0"
    exit 1
fi

echo "✅ Reserved test patients file found: $TEST_PATIENTS_FILE"
echo ""

# Step 1: Prepare dataset splits (if needed)
TRAINING_FILE="data/training_patients_leave_one_out.csv"
TEST_FILE="data/test_patients_leave_one_out.csv"

if [ ! -f "$TRAINING_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "🔄 Step 1: Preparing Leave-One-Out Dataset Splits"
    echo "-"*50
    echo "📋 Creating training/test splits based on reserved patient list..."
    echo ""
    
    start_time=$(date +%s)
    
    PYTHON_CMD="$VENV_PATH/bin/python"
    if [ -x "$PYTHON_CMD" ]; then
        "$PYTHON_CMD" prepare_leave_one_out_dataset.py \
            --test_patients "$TEST_PATIENTS_FILE" \
            --validate_embeddings
    elif command -v python >/dev/null 2>&1; then
        python prepare_leave_one_out_dataset.py \
            --test_patients "$TEST_PATIENTS_FILE" \
            --validate_embeddings
    elif command -v python3 >/dev/null 2>&1; then
        python3 prepare_leave_one_out_dataset.py \
            --test_patients "$TEST_PATIENTS_FILE" \
            --validate_embeddings
    else
        echo "❌ Python not found in PATH"
        exit 1
    fi
    
    if [ $? -ne 0 ]; then
        echo "❌ Dataset preparation failed!"
        echo "Please ensure embeddings are generated and patient files are valid"
        exit 1
    fi
    
    end_time=$(date +%s)
    prep_duration=$((end_time - start_time))
    echo "✅ Dataset preparation completed in ${prep_duration}s"
    echo ""
else
    echo "✅ Dataset splits already exist, skipping preparation step"
    echo ""
fi

# Validate dataset splits exist
if [ ! -f "$TRAINING_FILE" ] || [ ! -f "$TEST_FILE" ]; then
    echo "❌ Dataset splits not found after preparation step!"
    exit 1
fi

echo "📊 Dataset split validation:"
echo "   Training patients: $(wc -l < $TRAINING_FILE | tr -d ' ') patients"
echo "   Test patients: $(wc -l < $TEST_FILE | tr -d ' ') patients"
echo ""

# Step 2: Leave-One-Out Validation Pipeline
echo "🤖 Step 2: Leave-One-Out TB Detection Validation"
echo "-"*50
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "🔄 Running ${N_FOLDS}-fold cross-validation on training data"
    echo "🧪 Then evaluating all models on reserved test dataset (65 patients)"
    echo "📊 This provides robust clinical deployment validation"
else
    echo "📊 Running single train-test split validation"
fi
echo ""

# Prepare common arguments
COMMON_ARGS="--test_patients $TEST_PATIENTS_FILE --run_description $RUN_DESCRIPTION --n_folds $N_FOLDS"
if [ "$CROSS_VALIDATION" = "true" ]; then
    COMMON_ARGS="$COMMON_ARGS --cross_validation"
fi

start_time=$(date +%s)

PYTHON_CMD="$VENV_PATH/bin/python"
if [ -x "$PYTHON_CMD" ]; then
    "$PYTHON_CMD" 04_leave_one_out_validation.py $COMMON_ARGS
elif command -v python >/dev/null 2>&1; then
    python 04_leave_one_out_validation.py $COMMON_ARGS
elif command -v python3 >/dev/null 2>&1; then
    python3 04_leave_one_out_validation.py $COMMON_ARGS
else
    echo "❌ Python not found in PATH"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "❌ Leave-one-out validation failed!"
    exit 1
fi

end_time=$(date +%s)
validation_duration=$((end_time - start_time))

echo "✅ Leave-one-out validation completed in ${validation_duration}s"
echo ""

# Step 3: Results Summary
echo "📊 Step 3: Results Summary"
echo "-"*50
echo "🎉 Leave-one-out validation pipeline execution completed!"
echo ""

# Performance summary
echo "⚡ Performance Summary:"
echo "   ⏱️  Total Validation Time: ${validation_duration}s"
echo "   📁 Reserved Test Dataset: $TEST_PATIENTS_FILE"
if [ "$CROSS_VALIDATION" = "true" ]; then
    echo "   🔄 Evaluation: ${N_FOLDS}-fold cross-validation + independent test validation"
else
    echo "   📊 Evaluation: Single train-test split + independent test validation"
fi
echo ""

# Find most recent results (based on timestamp in filename)
LATEST_SUMMARY=$(ls -t results/*leave_one_out*executive_summary.txt 2>/dev/null | head -n 1)
LATEST_CONFIG=$(ls -t configs/*leave_one_out*config.json 2>/dev/null | head -n 1)

if [ -f "$LATEST_SUMMARY" ]; then
    echo "📁 Latest output files:"
    echo "  📄 Executive Summary: $(basename "$LATEST_SUMMARY")"
    
    # Extract run ID from filename
    RUN_ID=$(basename "$LATEST_SUMMARY" | sed 's/_leave_one_out_executive_summary.txt//')
    
    echo "  📊 CV Summary: ${RUN_ID}_cross_validation_summary.csv"
    echo "  📊 CV Detailed: ${RUN_ID}_cross_validation_detailed.csv"
    echo "  🧪 Test Results: ${RUN_ID}_leave_one_out_test_results.csv"
    echo "  📈 CV vs Test: ${RUN_ID}_test_vs_cv_comparison.csv"
    echo "  📊 CV Dashboard: ${RUN_ID}_cv_performance_dashboard.png"
    echo "  🧪 Test Dashboard: ${RUN_ID}_leave_one_out_test_dashboard.png"
    
    if [ -f "$LATEST_CONFIG" ]; then
        echo "  🔧 Configuration: $(basename "$LATEST_CONFIG")"
    fi
    
    echo ""
    echo "📈 Quick Results Preview:"
    echo "-"*30
    if [ "$VERBOSE" = "true" ]; then
        cat "$LATEST_SUMMARY"
    else
        # Show key sections
        grep -A 5 "TEST DATASET VALIDATION:" "$LATEST_SUMMARY" 2>/dev/null || echo "Results summary not available"
        echo ""
        grep "CLINICAL DEPLOYMENT RECOMMENDATION:" "$LATEST_SUMMARY" 2>/dev/null || echo "Recommendation not available"
    fi
else
    echo "⚠️  Results summary not found - check for errors above"
fi

echo ""
echo "🎯 Leave-One-Out Validation Benefits:"
echo "  🧪 Independent test dataset validation (65 patients, 5 countries)"
echo "  🔄 Robust cross-validation on training data"
echo "  📊 WHO compliance validation on truly held-out data"
echo "  🎯 Clinical deployment readiness assessment"
echo "  📈 Generalization gap analysis (CV vs test performance)"
echo "  🏥 Multi-country validation for global applicability"
echo ""

echo "🏆 Status: ✅ LEAVE-ONE-OUT VALIDATION COMPLETE"
echo "⏰ Total execution time: $((validation_duration + ${prep_duration:-0})) seconds"
echo ""

echo "📞 Next steps:"
echo "  1. Review executive summary for WHO compliance results"
echo "  2. Analyze test dataset performance vs cross-validation performance"
echo "  3. Check generalization gap for model stability"
echo "  4. Deploy WHO-compliant models for clinical use"
echo ""

echo "🔧 Configuration options for next run:"
echo "  • Different test set: TEST_PATIENTS_FILE=\"other_patients.csv\" $0"
echo "  • Custom description: RUN_DESCRIPTION=\"clinical_trial\" $0"
echo "  • Different CV folds: N_FOLDS=3 $0"
echo "  • Verbose output: VERBOSE=true $0"
echo ""

echo "🚀 LEAVE-ONE-OUT VALIDATION PIPELINE COMPLETE!"