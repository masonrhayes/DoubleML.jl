#!/bin/bash
# Master script to run the complete Julia vs Python benchmark

set -e  # Exit on error

echo "=========================================="
echo "DoubleML.jl vs Python Benchmark"
echo "=========================================="
echo ""

# Change to the benchmarks directory
cd "$(dirname "$0")"

echo "Step 1/4: Running Julia benchmark..."
echo "  This will generate data and time Julia DoubleML"
julia --project=../.. benchmark_julia.jl
echo ""
echo "✓ Julia benchmark complete"
echo ""

echo "Step 2/4: Running Python benchmark..."
echo "  This will load the same data and time Python DoubleML"
python benchmark_python.py
echo ""
echo "✓ Python benchmark complete"
echo ""

echo "Step 3/4: Generating report..."
julia --project=../.. generate_report.jl
echo ""
echo "✓ Report generated"
echo ""

echo "Step 4/4: Done!"
echo ""
echo "=========================================="
echo "Benchmark complete!"
echo "=========================================="
echo ""
echo "Results saved to:"
echo "  - benchmarks.md (human-readable report)"
echo "  - benchmark_results_julia.json (raw data)"
echo "  - benchmark_results_python.json (raw data)"
echo "  - benchmark_data.csv (shared dataset)"
echo ""
echo "View the report:"
echo "  cat benchmarks.md"
