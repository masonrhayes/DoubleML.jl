"""
Generate benchmark report comparing Julia vs Python DoubleML results
"""

# Activate the test environment
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))  # Activate test environment (2 levels up from benchmarks)

using JSON3
using Dates

const BENCHMARKS_DIR = @__DIR__
const JULIA_RESULTS_FILE = joinpath(BENCHMARKS_DIR, "benchmark_results_julia.json")
const PYTHON_RESULTS_FILE = joinpath(BENCHMARKS_DIR, "benchmark_results_python.json")
const REPORT_FILE = joinpath(BENCHMARKS_DIR, "benchmarks.md")

# Hardware detection
function get_hardware_info()
    # CPU info
    cpu_model = "Unknown"
    cpu_cores = Sys.CPU_THREADS

    try
        if Sys.islinux()
            # Read CPU model from /proc/cpuinfo
            cpuinfo = read("/proc/cpuinfo", String)
            for line in split(cpuinfo, '\n')
                if startswith(line, "model name")
                    cpu_model = strip(split(line, ':')[2])
                    break
                end
            end
        elseif Sys.isapple()
            # macOS - use sysctl
            cpu_model = strip(read(`sysctl -n machdep.cpu.brand_string`, String))
        elseif Sys.iswindows()
            # Windows - use wmic
            cpu_model = strip(read(`wmic cpu get Name /value`, String))
            cpu_model = replace(cpu_model, "Name=" => "")
        end
    catch e
        @warn "Could not detect CPU model: $e"
    end

    # Memory info
    total_ram = "Unknown"
    try
        if Sys.islinux()
            meminfo = read("/proc/meminfo", String)
            for line in split(meminfo, '\n')
                if startswith(line, "MemTotal")
                    # Parse memory in kB and convert to GB
                    kb_str = strip(split(split(line, ':')[2], 'k')[1])
                    kb = parse(Int, kb_str)
                    gb = round(kb / 1024 / 1024)
                    total_ram = "$(Int(gb)) GB"
                    break
                end
            end
        elseif Sys.isapple()
            mem_str = strip(read(`sysctl -n hw.memsize`, String))
            bytes = parse(Int, mem_str)
            gb = round(bytes / 1024^3)
            total_ram = "$(Int(gb)) GB"
        end
    catch e
        @warn "Could not detect RAM: $e"
    end

    # OS detection
    os_name = Sys.islinux() ? "Linux" : Sys.isapple() ? "macOS" : Sys.iswindows() ? "Windows" : "Unknown"

    return (
        cpu = cpu_model,
        cores = cpu_cores,
        ram = total_ram,
        os = os_name,
    )
end

println("Generating benchmark report...")

# Detect hardware
hw = get_hardware_info()
println("  Hardware: $(hw.cpu) ($(hw.cores) threads, $(hw.ram), $(hw.os))")

# Check if result files exist
if !isfile(JULIA_RESULTS_FILE)
    error("Julia results not found. Run benchmark_julia.jl first.")
end

if !isfile(PYTHON_RESULTS_FILE)
    error("Python results not found. Run benchmark_python.py first.")
end

# Load results
julia_results = open(JSON3.read, JULIA_RESULTS_FILE)
python_results = open(JSON3.read, PYTHON_RESULTS_FILE)

# Extract data
n_obs = julia_results.n_obs
dim_x = julia_results.dim_x
alpha = julia_results.alpha
n_folds = julia_results.n_folds

# Julia results
jl_time_median = julia_results.benchmark_median_sec
jl_coef = julia_results.coefficient
jl_se = julia_results.std_error

# Python results
py_time_median = python_results.benchmark_median_sec
py_coef = python_results.coefficient
py_se = python_results.std_error

# Calculate comparisons
time_ratio = py_time_median / jl_time_median
coef_diff = abs(jl_coef - py_coef)
coef_pct_diff = (coef_diff / alpha) * 100
se_diff = abs(jl_se - py_se)
se_pct_diff = (se_diff / ((jl_se + py_se) / 2)) * 100

# Determine winner
time_winner = jl_time_median < py_time_median ? "Julia" : "Python"

# Generate report
report = """
# DoubleML Performance Benchmark: Julia vs Python

**Generated:** $(now())

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | $(hw.cpu) ($(hw.cores) threads) |
| RAM | $(hw.ram) |
| OS | $(hw.os) |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Observations | $n_obs |
| Covariates | $dim_x |
| Treatment Effect (true) | $alpha |
| Cross-fitting Folds | $n_folds |

## Learners Used

| Language | Learner |
|----------|---------|
| Julia | EvoTreeRegressor (100 trees, max_depth=6) |
| Python | XGBRegressor (100 trees, max_depth=6) |

## Timing Results

### Julia (DoubleML.jl)

| Metric | Time (seconds) |
|--------|----------------|
| Median | $(round(jl_time_median, digits = 2))s |
### Python (DoubleML)

| Metric | Time (seconds) |
|--------|----------------|
| Median | $(round(py_time_median, digits = 2))s |

### Performance Comparison

**Winner: $time_winner**

- **Speedup Factor:** $(round(time_ratio, digits = 2))x ($(time_winner) is faster)
- **Time Difference:** $(round(abs(py_time_median - jl_time_median), digits = 2))s

## Coefficient Estimates

| Metric | Julia | Python | Difference |
|--------|-------|--------|------------|
| Coefficient | $(round(jl_coef, digits = 6)) | $(round(py_coef, digits = 6)) | $(round(coef_diff, digits = 6)) ($(round(coef_pct_diff, digits = 2))%) |
| Std Error | $(round(jl_se, digits = 6)) | $(round(py_se, digits = 6)) | $(round(se_diff, digits = 6)) ($(round(se_pct_diff, digits = 2))%) |

**True Treatment Effect:** $alpha

### Accuracy Assessment

- **Coefficient Accuracy:** Estimates differ by $(round(coef_pct_diff, digits = 2))% from each other
- **Standard Error Agreement:** SEs differ by $(round(se_pct_diff, digits = 2))%

$(coef_pct_diff < 5 ? "✓ Coefficient estimates are in good agreement (< 5% difference)" : "⚠ Coefficient estimates differ by > 5%")
$(se_pct_diff < 10 ? "✓ Standard errors are in good agreement (< 10% difference)" : "⚠ Standard errors differ by > 10%")

## Summary

This benchmark compares DoubleML.jl (Julia) against the Python DoubleML package on identical data:
- $(n_obs) observations with $(dim_x) covariates
- Using gradient boosted trees (EvoTrees in Julia, XGBoost in Python)
- Both implementations use the same partialling_out score function

### Key Findings

1. **Performance:** $(time_winner) is $(round(time_ratio, digits = 2))x faster
2. **Accuracy:** Both implementations produce similar coefficient estimates ($(round(coef_pct_diff, digits = 2))% difference)
3. **Inference:** Standard errors are consistent between implementations ($(round(se_pct_diff, digits = 2))% difference)

## Raw Data Files

- Julia results: `benchmark_results_julia.json`
- Python results: `benchmark_results_python.json`
- Shared data: `benchmark_data.csv`

---

*Note: Benchmarks run on $(n_obs) observations with $(dim_x) dimensions. Times measured using BenchmarkTools (Julia) and time.perf_counter() (Python) with 5 repetitions each.*
"""

# Write report
open(REPORT_FILE, "w") do io
    write(io, report)
end

println("✓ Benchmark report generated: $REPORT_FILE")
println()
println("Key Results:")
println("  Julia time:   $(round(jl_time_median, digits = 2))s")
println("  Python time:  $(round(py_time_median, digits = 2))s")
println("  Speedup:      $(round(time_ratio, digits = 2))x ($(time_winner) is faster)")
println("  Coef diff:    $(round(coef_pct_diff, digits = 2))%")
