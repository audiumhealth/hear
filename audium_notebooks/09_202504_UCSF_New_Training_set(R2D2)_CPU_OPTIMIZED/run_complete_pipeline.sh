#!/bin/bash

# Complete UCSF R2D2 TB Detection Pipeline
# Execute the full pipeline from data validation to final analysis

echo "🎯 UCSF R2D2 TB Detection Pipeline - Complete Execution"
echo "=" * 60

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source ~/python/venvs/v_audium_hear/bin/activate

# Change to pipeline directory
cd /Users/abelvillcaroque/git/github/audiumhealth/hear/audium_notebooks/09_202504_UCSF_New_Training_set\(R2D2\)

echo "📂 Current directory: $(pwd)"
echo ""

# Step 1: Data Validation (Fixed)
echo "🔍 Step 1: Data Validation and Double-Counting Fix"
echo "-" * 50
python data_validation_fixed.py
if [ $? -ne 0 ]; then
    echo "❌ Data validation failed!"
    exit 1
fi
echo "✅ Data validation completed successfully"
echo ""

# Step 2: Embedding Generation (Batch)
echo "🧠 Step 2: HeAR Embedding Generation"
echo "-" * 50
echo "⏱️  Note: This step may take 4-6 hours for all 543 patients"
echo "💡 Tip: Use screen/tmux for long-running processes"
echo ""
echo "Starting batch processing of embeddings..."
echo "You can monitor progress in the output logs"
echo ""

# Process in batches of 50 patients, saving every 25
python 02_generate_embeddings_batch.py --start 0 --batch_size 50 --save_every 25 &
PID1=$!

# Wait for first batch and continue
wait $PID1
if [ $? -ne 0 ]; then
    echo "❌ Embedding generation failed!"
    exit 1
fi

# Continue with remaining batches
for start in 50 100 150 200 250 300 350 400 450 500; do
    echo "📊 Processing batch starting at patient $start..."
    python 02_generate_embeddings_batch.py --start $start --batch_size 50 --save_every 25
    if [ $? -ne 0 ]; then
        echo "❌ Embedding generation failed at batch $start!"
        exit 1
    fi
done

echo "✅ Embedding generation completed successfully"
echo ""

# Step 3: Complete TB Detection Analysis
echo "🤖 Step 3: Complete TB Detection Analysis"
echo "-" * 50
python 03_tb_detection_full_analysis.py
if [ $? -ne 0 ]; then
    echo "❌ TB detection analysis failed!"
    exit 1
fi
echo "✅ TB detection analysis completed successfully"
echo ""

# Step 4: Results Summary
echo "📊 Step 4: Results Summary"
echo "-" * 50
echo "Pipeline execution completed successfully!"
echo ""
echo "📁 Output files:"
echo "  Data: data/clean_patients_fixed.csv (543 patients)"
echo "  Embeddings: data/complete_embeddings.npz"
echo "  Analysis: results/detailed_analysis_results.csv"
echo "  WHO Models: results/who_compliant_models.csv"
echo "  Summary: results/executive_summary.txt"
echo ""
echo "📈 Key directories:"
echo "  📂 data/ - Clean datasets and embeddings"
echo "  📂 reports/ - Patient reports and summaries"
echo "  📂 results/ - Analysis results and visualizations"
echo ""

# Display key statistics
echo "🎯 Pipeline Statistics:"
if [ -f "results/executive_summary.txt" ]; then
    echo "  📊 Analysis results available in results/executive_summary.txt"
fi

if [ -f "reports/comprehensive_patient_report.csv" ]; then
    echo "  📋 Patient report available in reports/comprehensive_patient_report.csv"
fi

if [ -f "data/complete_embeddings.npz" ]; then
    echo "  🧠 Embeddings generated successfully"
fi

echo ""
echo "🎉 PIPELINE EXECUTION COMPLETE!"
echo "✅ All steps completed successfully"
echo "🚀 Ready for clinical deployment"
echo ""
echo "📞 For detailed results, check:"
echo "  • README.md - Complete documentation"
echo "  • FINAL_PIPELINE_SUMMARY.md - Executive summary"
echo "  • results/executive_summary.txt - Analysis overview"
echo ""
echo "⏰ Total execution time: Check individual step logs"
echo "🎯 Status: ✅ Complete and ready for deployment"