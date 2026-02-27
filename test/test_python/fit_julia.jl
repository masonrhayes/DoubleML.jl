"""
Generic Julia model runner for Python validation tests.
Reads configuration from config.toml and runs all specified tests.
"""

using DoubleML
using MLJ
using MLJScikitLearnInterface
using StableRNGs
using CSV
using DataFrames
using JSON3
using TOML

# Load configuration
const CONFIG = TOML.parsefile(joinpath(@__DIR__, "config.toml"))
const RNG = StableRNGs.StableRNG(CONFIG["data_generation"]["rng_seed"])

# Load required models via MLJScikitLearnInterface (sklearn models)
RandomForestRegressor = @load RandomForestRegressor pkg = MLJScikitLearnInterface verbosity = 0
RandomForestClassifier = @load RandomForestClassifier pkg = MLJScikitLearnInterface verbosity = 0
HistGradientBoostingRegressor = @load HistGradientBoostingRegressor pkg = MLJScikitLearnInterface verbosity = 0
HistGradientBoostingClassifier = @load HistGradientBoostingClassifier pkg = MLJScikitLearnInterface verbosity = 0

# Learner registry - maps config names to constructors
const LEARNER_TYPES = Dict{String, Tuple{Type, Union{Type, Nothing}}}(
    "RF" => (RandomForestRegressor, RandomForestClassifier),
    "GB" => (HistGradientBoostingRegressor, HistGradientBoostingClassifier)
)

# Data generators - maps config names to functions
const DATA_GENERATORS = Dict{String, Function}(
    "make_plr_CCDDHNR2018" => (n_obs; dim_x, rng) -> make_plr_CCDDHNR2018(n_obs; dim_x = dim_x, alpha = CONFIG["data_generation"]["alpha"], rng = rng, return_type = :DataFrame),
    "make_irm_data" => (n_obs; dim_x, rng) -> make_irm_data(n_obs; dim_x = dim_x, theta = CONFIG["data_generation"]["theta"], rng = rng, return_type = :DataFrame),
    "make_lplr_LZZ2020" => (n_obs; dim_x, rng) -> make_lplr_LZZ2020(n_obs; dim_x = dim_x, alpha = CONFIG["data_generation"]["lplr_alpha"], rng = rng, return_type = :DataFrame, return_p = false)
)

# Helper: instantiate learners with correct hyperparameters
function instantiate_learners(learner_name::String)
    RegressorType, ClassifierType = LEARNER_TYPES[learner_name]
    learners_config = CONFIG["learners"][learner_name]

    # Build kwargs from config
    julia_params = learners_config["julia"]
    kwargs = Dict{Symbol, Any}()

    # Add numeric parameters (skip package and type)
    for (k, v) in julia_params
        if k ∉ ["package", "type"]
            kwargs[Symbol(k)] = v
        end
    end

    return RegressorType, ClassifierType, kwargs
end

# Model constructors based on config
function construct_model(model_name::String, data, score_config::Dict, learner_name::String)
    model_def = nothing
    for m in CONFIG["models"]
        if m["name"] == model_name
            model_def = m
            break
        end
    end

    @assert model_def !== nothing "Model $model_name not found in config"

    RegressorType, ClassifierType, kwargs = instantiate_learners(learner_name)

    if model_name == "PLR"
        ml_l = RegressorType(; kwargs...)
        ml_m = RegressorType(; kwargs...)

        score = Symbol(score_config["julia_score"])
        n_folds = CONFIG["model_fitting"]["n_folds"]
        n_rep = CONFIG["model_fitting"]["n_rep"]

        if get(score_config, "requires_ml_g", false)
            ml_g = RegressorType(; kwargs...)
            return DoubleMLPLR(data, ml_l, ml_m; ml_g = ml_g, score = score, n_folds = n_folds, n_rep = n_rep)
        else
            return DoubleMLPLR(data, ml_l, ml_m; score = score, n_folds = n_folds, n_rep = n_rep)
        end

    elseif model_name == "IRM"
        ml_g = RegressorType(; kwargs...)
        ml_m = ClassifierType !== nothing ? ClassifierType(; kwargs...) : RegressorType(; kwargs...)

        score = Symbol(score_config["julia_score"])
        n_folds = CONFIG["model_fitting"]["n_folds"]
        n_rep = CONFIG["model_fitting"]["n_rep"]
        normalize = get(score_config, "normalize_ipw", [false])[1]  # Default false

        return DoubleMLIRM(data, ml_g, ml_m; score = score, n_folds = n_folds, n_rep = n_rep, normalize_ipw = normalize)

    elseif model_name == "LPLR"
        # LPLR has 3 learners: ml_M (classifier), ml_t (regressor), ml_m (regressor)
        ml_M = ClassifierType !== nothing ? ClassifierType(; kwargs...) : RegressorType(; kwargs...)
        ml_t = RegressorType(; kwargs...)
        ml_m = RegressorType(; kwargs...)

        score = Symbol(score_config["julia_score"])
        n_folds = CONFIG["model_fitting"]["n_folds"]
        n_rep = CONFIG["model_fitting"]["n_rep"]

        return DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = score, n_folds = n_folds, n_rep = n_rep)
    end

    error("Unknown model: $model_name")
end

# Run a single test
function run_test(model_name::String, data, score_config::Dict, learner_name::String)
    model = construct_model(model_name, data, score_config, learner_name)

    try
        fit!(model)
        return Dict("coef" => Float64(model.coef), "se" => Float64(model.se))
    catch e
        @warn "Test failed" model_name = model_name learner = learner_name score = score_config["name"] exception = e
        return Dict("coef" => NaN, "se" => NaN)
    end
end

# Generate data for a specific model
function generate_model_data(model_def::Dict, data_gen::Dict, rng)
    gen_name = model_def["data_generator_julia"]
    gen_func = DATA_GENERATORS[gen_name]

    n_obs = data_gen["n_obs"]
    dim_x = data_gen["dim_x"]

    return gen_func(n_obs; dim_x = dim_x, rng = rng)
end

# Main function to run all tests
function run_all_tests(data_dir::String, python_available::Bool)
    results = Dict{String, Dict{String, Float64}}()

    # Load or generate data for each model type
    data_cache = Dict{String, DataFrame}()

    for model_def in CONFIG["models"]
        model_name = model_def["name"]
        gen_name = model_def["data_generator_julia"]

        # Load Julia data
        jl_file = joinpath(data_dir, "$(gen_name)_jl.csv")
        if !haskey(data_cache, gen_name)
            println("  Loading data for $model_name ($gen_name)...")
            data_cache[gen_name] = CSV.read(jl_file, DataFrame)
        end

        df_jl = data_cache[gen_name]
        x_cols = filter(n -> startswith(string(n), "X"), names(df_jl))

        # Extract y_col and d_col based on model type
        # Note: All models use 'y' for outcome and 'd' for treatment
        if model_name in ["PLR", "IRM", "LPLR"]
            y_col, d_col = :y, :d
        else
            error("Unknown model type: $model_name")
        end

        data_jl = DoubleMLData(df_jl; y_col = y_col, d_col = d_col, x_cols = Symbol.(x_cols))

        # Load Python data if available
        data_py = nothing
        py_file = joinpath(data_dir, "$(gen_name)_py.csv")
        if python_available && isfile(py_file)
            df_py = CSV.read(py_file, DataFrame)
            data_py = DoubleMLData(df_py; y_col = y_col, d_col = d_col, x_cols = Symbol.(x_cols))
        end

        # Run tests for each score and learner
        for score_config in model_def["scores"]
            score_name = score_config["name"]

            for learner_name in score_config["learners"]
                test_key = "$(model_name)_$(score_name)_$(lowercase(learner_name))"

                # Direction 1: Python data → Julia models
                if data_py !== nothing
                    println("  d1_$test_key")
                    results["d1_$test_key"] = run_test(model_name, data_py, score_config, learner_name)
                end

                # Direction 2: Julia data → Python models (for comparison)
                println("  d2_$test_key")
                results["d2_$test_key"] = run_test(model_name, data_jl, score_config, learner_name)
            end
        end
    end

    return results
end

# Standalone execution
if abspath(PROGRAM_FILE) == @__FILE__
    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    # Generate all data using the existing script
    println("Generating Julia data...")
    include(joinpath(@__DIR__, "generate_data_julia.jl"))

    # Check for Python data
    python_available = any(
        isfile(joinpath(data_dir, "$(m["data_generator_julia"])_py.csv"))
            for m in CONFIG["models"]
    )

    # Run tests
    println("\nRunning tests...")
    results = run_all_tests(data_dir, python_available)

    # Save results
    results_path = joinpath(data_dir, "results_julia.json")
    open(results_path, "w") do f
        JSON3.write(f, results)
    end

    println("\n✓ Julia results saved to: $results_path")
    println("  Total tests: $(length(results))")
end
