"""
Benchmark script for DoubleML.jl vs Python DoubleML
Generates shared data and runs Julia benchmark using BenchmarkTools
"""

# Activate the test environment
using Pkg
Pkg.activate(joinpath(@__DIR__, "../.."))  # Activate test environment (2 levels up from benchmarks)

using CSV
using DataFrames
using DoubleML
using MLJ
using StableRNGs
using EvoTrees
using BenchmarkTools
using JSON3

# Configuration
const N_OBS = 100_000
const DIM_X = 1000
const ALPHA = 0.5
const N_FOLDS = 5
const DATA_FILE = joinpath(@__DIR__, "benchmark_data.csv")
const RESULTS_FILE = joinpath(@__DIR__, "benchmark_results_julia.json")

println("="^60)
println("DoubleML.jl Benchmark")
println("="^60)
println("Configuration:")
println("  N observations: $N_OBS")
println("  N dimensions: $DIM_X")
println("  Treatment effect: $ALPHA")
println("  Cross-fitting folds: $N_FOLDS")
println("  Learner: EvoTreeRegressor")
println()

# Step 1: Generate data
println("Step 1: Generating data...")
rng = StableRNG(42)
df = make_plr_CCDDHNR2018(N_OBS; dim_x = DIM_X, alpha = ALPHA, return_type = DataFrame, rng = rng)


CSV.write(DATA_FILE, df)

println("  ✓ Data saved ($(size(df, 1)) rows, $(size(df, 2)) columns)")
println()

# Step 2: Setup MLJ learners
println("Step 2: Setting up EvoTreeRegressor...")
EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0

ml_l = EvoTreeRegressor(
    nrounds = 100,
    eta = 0.1,
    max_depth = 4
)

ml_m = EvoTreeRegressor(
    nrounds = 100,
    eta = 0.1,
    max_depth = 4
)
println("  ✓ Learners configured")
println()

# Step 3: Create DoubleML model
println("Step 3: Creating DoubleMLPLR model...")

data = DoubleMLData(df, :y, :d)

model = DoubleMLPLR(data, ml_l, ml_m, n_folds = N_FOLDS, score = :partialling_out)
println("  ✓ Model created")
println()

# Step 4: Run benchmark with BenchmarkTools
println("Step 4: Running benchmark with BenchmarkTools...")
println("  (This may take several minutes...)")

# Warmup first
println("  Warming up...")
fit!(model)

# Benchmark with BenchmarkTools
println("  Running 5 samples...")
b = @benchmark fit!(m, force = true, verbose = 1) setup = (m = DoubleMLPLR($data, $ml_l, $ml_m, n_folds = $N_FOLDS))

median_time = median(b.times) / 1.0e9  # Convert ns to seconds


println()
println("  Benchmark Results:")
println("    Median: $(round(median_time, digits = 2))s")
println()

# Step 5: Final fit to get coef and SE
println("Step 5: Final fit for coefficient estimation...")

coef_jl = coef(model)[1]
se_jl = stderror(model)[1]
n_obs_jl = nobs(model)

println("  Results:")
println("    Coefficient: $coef_jl")
println("    Std Error:   $se_jl")
println("    N obs:       $n_obs_jl")
println()

# Step 6: Save results
println("Step 6: Saving results...")
results = Dict(
    "language" => "Julia",
    "n_obs" => N_OBS,
    "dim_x" => DIM_X,
    "alpha" => ALPHA,
    "learner" => "EvoTreeRegressor",
    "n_folds" => N_FOLDS,
    "benchmark_median_sec" => median_time,
    "coefficient" => coef_jl,
    "std_error" => se_jl,
    "data_file" => DATA_FILE,
)

open(RESULTS_FILE, "w") do io
    JSON3.write(io, results)
end
println("  ✓ Results saved to: $RESULTS_FILE")
println()
println("="^60)
println("Julia benchmark complete!")
println("="^60)
