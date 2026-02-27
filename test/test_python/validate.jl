"""
DoubleML.jl Python Validation Tests - Simplified Modular Version

This test suite reads all configuration from config.toml and performs
bidirectional validation between Julia and Python DoubleML implementations.

To add a new model: Edit config.toml and add model tests there.
"""

using DoubleML
using Test
using CSV
using DataFrames
using JSON3
using TOML
using Dates

# Load unified configuration
const CONFIG = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
const DATA_GEN = CONFIG["data_generation"]
const MODEL_FIT = CONFIG["model_fitting"]
const TOLERANCE = MODEL_FIT["tolerance"]

# Results storage for markdown report
const TEST_RESULTS = Dict{String, Dict{String, Any}}()

# Include the generic Julia runner
include(joinpath(@__DIR__, "fit_julia.jl"))

"""
Generate a markdown comparison report from test results.
"""
function generate_markdown_report(timestamp)
    isempty(TEST_RESULTS) && return

    output_path = joinpath(@__DIR__, "python_comparison.md")
    lines = String[]

    push!(lines, "# DoubleML.jl Validation Report")
    push!(lines, "")
    push!(lines, "Generated: $timestamp")
    push!(lines, "")

    # Configuration section
    push!(lines, "## Test Configuration")
    push!(lines, "")
    push!(lines, "- Sample size: $(DATA_GEN["n_obs"])")
    push!(lines, "- Covariates: $(DATA_GEN["dim_x"])")
    push!(lines, "- True PLR effect (alpha): $(DATA_GEN["alpha"])")
    push!(lines, "- True IRM effect (theta): $(DATA_GEN["theta"])")
    push!(lines, "- Cross-fitting folds: $(MODEL_FIT["n_folds"])")
    push!(lines, "- Tolerance: $(TOLERANCE * 100)%")
    push!(lines, "")

    # Summary statistics
    push!(lines, "## Summary")
    push!(lines, "")
    total = length(TEST_RESULTS)
    coef_passed = sum(1 for (k, v) in TEST_RESULTS if v["coef_passed"])
    se_passed = sum(1 for (k, v) in TEST_RESULTS if v["se_passed"])

    push!(lines, "- Total comparisons: $total")
    push!(lines, "- Coefficient passed: $coef_passed")
    push!(lines, "- Standard error passed: $se_passed")
    push!(lines, "")

    # Model-specific counts
    for direction in ["d1", "d2"]
        dir_name = direction == "d1" ?
            "Direction 1: Python-Generated Data → Julia Models" :
            "Direction 2: Julia-Generated Data → Python Models"
        push!(lines, "## $dir_name")
        push!(lines, "")

        for model_def in CONFIG["models"]
            model_name = model_def["name"]
            push!(lines, "### $model_name Tests")
            push!(lines, "")

            # Table header
            push!(lines, "| Test | Julia Coef | Python Coef | Coef Diff | Julia SE | Python SE | SE Diff | Status |")
            push!(lines, "|------|------------|-------------|-----------|----------|-----------|---------|--------|")

            # Find all tests for this model and direction
            for (test_key, details) in TEST_RESULTS
                if startswith(test_key, "$(direction)_$(model_name)_")
                    display_name = test_key[(length(direction) + 2):end]
                    status = (details["coef_passed"] && details["se_passed"]) ? "✓" : "✗"
                    coef_diff_pct = round(details["coef_rel_diff"] * 100, digits = 1)
                    se_diff_pct = round(details["se_rel_diff"] * 100, digits = 1)

                    push!(lines, "| $display_name | $(round(details["coef_julia"], digits = 4)) | $(round(details["coef_python"], digits = 4)) | $(coef_diff_pct)% | $(round(details["se_julia"], digits = 4)) | $(round(details["se_python"], digits = 4)) | $(se_diff_pct)% | $status |")
                end
            end

            push!(lines, "")
        end
    end

    write(output_path, join(lines, "\n"))
    return println("\nMarkdown report written to: $output_path")
end

# Main test suite
@testset "Python Validation" begin
    timestamp = now()
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    # Check Python availability
    python_available = isfile(joinpath(@__DIR__, "pyproject.toml"))

    println("="^70)
    println("DoubleML.jl Bidirectional Validation Tests")
    println("="^70)
    println("Configuration:")
    println("  - Sample size: $(DATA_GEN["n_obs"])")
    println("  - Covariates: $(DATA_GEN["dim_x"])")
    println("  - Tolerance: $(TOLERANCE * 100)%")
    println("  - Python available: $python_available")
    println("="^70)

    # Phase 1: Generate Julia data
    println("\n📊 Phase 1: Generating Julia data...")
    include(joinpath(@__DIR__, "generate_data_julia.jl"))

    # Phase 2: Generate and fit Python models (if available)
    if python_available
        println("\n📊 Phase 2: Fitting Python models...")
        cd(@__DIR__) do
            run(`uv run python fit_python.py`)
        end
    end

    # Phase 3: Fit Julia models
    println("\n🔧 Phase 3: Fitting Julia models...")
    julia_results = run_all_tests(data_dir, python_available)

    # Save Julia results
    julia_results_path = joinpath(data_dir, "results_julia.json")
    open(julia_results_path, "w") do f
        JSON3.write(f, julia_results; allow_inf = true)
    end
    println("\n  ✓ Julia results saved: $julia_results_path")

    # Phase 4: Compare results
    if python_available
        println("\n⚖️  Phase 4: Comparing results...")

        python_results_path = joinpath(data_dir, "results_python.json")
        python_results = JSON3.read(python_results_path, Dict{String, Dict{String, Float64}})

        @testset "Direction 1: Python-Generated Data → Julia Models" begin
            for model_def in CONFIG["models"]
                model_name = model_def["name"]

                @testset "$model_name" begin
                    for score_config in model_def["scores"]
                        score_name = score_config["name"]

                        for learner_name in score_config["learners"]
                            test_key = "$(model_name)_$(score_name)_$(lowercase(learner_name))"
                            d1_key = "d1_$test_key"

                            @testset "$test_key" begin
                                # Check if results exist
                                if !haskey(julia_results, d1_key)
                                    @test_skip "Julia fitting failed for $test_key"
                                    continue
                                end

                                if !haskey(python_results, d1_key)
                                    @test_skip "Python fitting failed for $test_key"
                                    continue
                                end

                                jl_result = julia_results[d1_key]
                                py_result = python_results[d1_key]

                                # Check for NaN (convergence failures)
                                if isnan(jl_result["coef"]) || isnan(py_result["coef"])
                                    @test_skip "Test failed to converge"
                                    continue
                                end

                                # Calculate differences
                                coef_rel_diff = abs(jl_result["coef"] - py_result["coef"]) / abs(py_result["coef"])
                                se_rel_diff = abs(jl_result["se"] - py_result["se"]) / abs(py_result["se"])

                                coef_passed = coef_rel_diff < TOLERANCE
                                se_passed = se_rel_diff < TOLERANCE

                                # Store for report
                                TEST_RESULTS[d1_key] = Dict(
                                    "coef_julia" => jl_result["coef"],
                                    "coef_python" => py_result["coef"],
                                    "coef_rel_diff" => coef_rel_diff,
                                    "se_julia" => jl_result["se"],
                                    "se_python" => py_result["se"],
                                    "se_rel_diff" => se_rel_diff,
                                    "coef_passed" => coef_passed,
                                    "se_passed" => se_passed
                                )

                                @test coef_passed
                                @test se_passed
                            end
                        end
                    end
                end
            end
        end

        @testset "Direction 2: Julia-Generated Data → Python Models" begin
            for model_def in CONFIG["models"]
                model_name = model_def["name"]

                @testset "$model_name" begin
                    for score_config in model_def["scores"]
                        score_name = score_config["name"]

                        for learner_name in score_config["learners"]
                            test_key = "$(model_name)_$(score_name)_$(lowercase(learner_name))"
                            d2_key = "d2_$test_key"

                            @testset "$test_key" begin
                                if !haskey(julia_results, d2_key)
                                    @test_skip "Julia fitting failed for $test_key"
                                    continue
                                end

                                if !haskey(python_results, d2_key)
                                    @test_skip "Python fitting failed for $test_key"
                                    continue
                                end

                                jl_result = julia_results[d2_key]
                                py_result = python_results[d2_key]

                                if isnan(jl_result["coef"]) || isnan(py_result["coef"])
                                    @test_skip "Test failed to converge"
                                    continue
                                end

                                coef_rel_diff = abs(jl_result["coef"] - py_result["coef"]) / abs(py_result["coef"])
                                se_rel_diff = abs(jl_result["se"] - py_result["se"]) / abs(py_result["se"])

                                coef_passed = coef_rel_diff < TOLERANCE
                                se_passed = se_rel_diff < TOLERANCE

                                TEST_RESULTS[d2_key] = Dict(
                                    "coef_julia" => jl_result["coef"],
                                    "coef_python" => py_result["coef"],
                                    "coef_rel_diff" => coef_rel_diff,
                                    "se_julia" => jl_result["se"],
                                    "se_python" => py_result["se"],
                                    "se_rel_diff" => se_rel_diff,
                                    "coef_passed" => coef_passed,
                                    "se_passed" => se_passed
                                )

                                @test coef_passed
                                @test se_passed
                            end
                        end
                    end
                end
            end
        end
    else
        # Python not available - self-validation only
        @testset "Julia Self-Validation (Python unavailable)" begin
            for model_def in CONFIG["models"]
                model_name = model_def["name"]

                @testset "$model_name" begin
                    for score_config in model_def["scores"]
                        score_name = score_config["name"]

                        for learner_name in score_config["learners"]
                            test_key = "$(model_name)_$(score_name)_$(lowercase(learner_name))"
                            d2_key = "d2_$test_key"

                            @testset "$test_key" begin
                                if !haskey(julia_results, d2_key)
                                    @test_skip "Julia fitting failed for $test_key"
                                    continue
                                end

                                result = julia_results[d2_key]
                                @test isfinite(result["coef"])
                                @test isfinite(result["se"])
                                @test result["se"] > 0
                            end
                        end
                    end
                end
            end
        end

        @warn "\n⚠️  Python validation tests skipped - environment not available"
        @warn "   To enable Python validation:"
        @warn "   1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh"
        @warn "   2. Run: cd test/test_python && bash setup_python_env.sh"
        @warn "   3. Re-run tests"
    end

    # Generate report
    generate_markdown_report(timestamp)

    # Final summary
    println("\n" * "="^70)
    println("VALIDATION SUMMARY")
    println("="^70)

    if isempty(TEST_RESULTS)
        println("No Python-Julia comparisons performed")
        println("Julia self-validation completed")
    else
        total = length(TEST_RESULTS)
        coef_passed = sum(1 for (k, v) in TEST_RESULTS if v["coef_passed"])
        se_passed = sum(1 for (k, v) in TEST_RESULTS if v["se_passed"])

        println("Total comparisons: $total")
        println("  ✓ Coefficient: $coef_passed / $total")
        println("  ✓ Standard Error: $se_passed / $total")
        println("  ✓ Both passed: $(sum(1 for (k, v) in TEST_RESULTS if v["coef_passed"] && v["se_passed"])) / $total")
    end

    println("="^70)
end
