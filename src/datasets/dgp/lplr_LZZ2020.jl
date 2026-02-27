using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Random
using StableRNGs
using StatsFuns: logistic
using CategoricalArrays

"""
    make_lplr_LZZ2020(n_obs; dim_x=20, alpha=0.5, balanced_r0=true, 
                      treatment="continuous", return_type=:DoubleMLData, rng=nothing)

Generate synthetic data for a Logistic Partially Linear Regression (LPLR) model,
as in Liu et al. (2021).

# Arguments
- `n_obs::Int`: Number of observations to generate
- `dim_x::Int=20`: Number of covariates
- `alpha::Real=0.5`: Value of the causal parameter (treatment effect on log-odds)
- `balanced_r0::Bool=true`: Use balanced r_0 specification (smaller magnitude). 
  If false, uses unbalanced specification with larger share of Y=0.
- `treatment::String="continuous"`: Treatment type - "continuous", "binary", or "binary_unbalanced"
- `return_type::Symbol=:DoubleMLData`: Output format (`:DoubleMLData` or `:DataFrame`)
- `rng::Union{AbstractRNG, Nothing}=nothing`: Random number generator

# Returns
- `DoubleMLData` object if `return_type=:DoubleMLData` (default)
- `DataFrame` if `return_type=:DataFrame`

# Data Generating Process
- Covariates: X ~ N(0, Σ) where Σ_kj = 0.2^|j-k|, clipped to [-2, 2]
- Treatment: d = a_0(x) (continuous) or d ~ Bernoulli(sigmoid(a_0(x)))
- Propensity: p = σ(α·d + r_0(x)) where σ is logistic function
- Outcome: y ~ Bernoulli(p)

# Nuisance Functions
- a_0(x) = 2/(1+exp(x₁)) - 2/(1+exp(x₂)) + sin(x₃) + cos(x₄) 
           + 0.5·I(x₅>0) - 0.5·I(x₆>0) + 0.2·x₇·x₈ - 0.2·x₉·x₁₀

- r_0(x) = 0.1·x₁·x₂·x₃ + 0.1·x₄·x₅ + 0.1·x₆³ - 0.5·sin²(x₇) 
           + 0.5·cos(x₈) + 1/(1+x₉²) - 1/(1+exp(x₁₀))
           + 0.25·I(x₁₁>0) - 0.25·I(x₁₃>0)  (balanced)
           Or with different coefficients for unbalanced

# Examples
```julia
using DoubleML

# Generate default dataset
data = make_lplr_LZZ2020(1000)

# Generate with custom parameters
data = make_lplr_LZZ2020(500, dim_x=10, alpha=1.0, treatment="binary")

# Get DataFrame instead
df = make_lplr_LZZ2020(1000, return_type=:DataFrame)

# Use with specific RNG
using StableRNGs
rng = StableRNG(123)
data = make_lplr_LZZ2020(1000, rng=rng)
```

# References
Liu, L., Zhang, Y. and Zhou, D. (2021). 
"Double/Debiased Machine Learning for Logistic Partially Linear Model." 
The Econometrics Journal, 24(3): 559-588.
doi: 10.1093/ectj/utab019

See also: [`DoubleMLLPLR`](@ref), [`DoubleMLData`](@ref)
"""
function make_lplr_LZZ2020(
        n_obs;
        dim_x = 20,
        alpha = 0.5,
        balanced_r0 = true,
        treatment = "continuous",
        return_type = :DoubleMLData,
        return_p = false,
        rng = Random.default_rng()
    )

    # AR(1) covariance matrix: Sigma_kj = 0.2^|j-k|
    sigma = [0.2^abs(j - k) for j in 1:dim_x, k in 1:dim_x]
    dist_x = MvNormal(zeros(dim_x), sigma)
    X = rand(rng, dist_x, n_obs)'

    # Clip X to [-2, 2]
    clamp!(X, -2, 2)

    # Compute a_0(x) - treatment function
    a_0_vals = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = X[i, 1], X[i, 2], X[i, 3], X[i, 4],
            X[i, 5], X[i, 6], X[i, 7], X[i, 8], X[i, 9], X[i, 10]
        a_0_vals[i] = 2 / (1 + exp(x1)) - 2 / (1 + exp(x2)) + sin(x3) + cos(x4) +
            0.5 * (x5 > 0 ? 1 : 0) - 0.5 * (x6 > 0 ? 1 : 0) +
            0.2 * x7 * x8 - 0.2 * x9 * x10
    end

    # Compute r_0(x) - outcome function
    r_0_vals = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13 = X[i, 1], X[i, 2], X[i, 3],
            X[i, 4], X[i, 5], X[i, 6], X[i, 7], X[i, 8], X[i, 9], X[i, 10], X[i, 11], X[i, 12], X[i, 13]

        if balanced_r0
            r_0_vals[i] = 0.1 * x1 * x2 * x3 + 0.1 * x4 * x5 + 0.1 * x6^3 -
                0.5 * sin(x7)^2 + 0.5 * cos(x8) +
                1 / (1 + x9^2) - 1 / (1 + exp(x10)) +
                0.25 * (x11 > 0 ? 1 : 0) - 0.25 * (x13 > 0 ? 1 : 0)
        else
            r_0_vals[i] = 0.1 * x1 * x2 * x3 + 0.1 * x4 * x5 + 0.1 * x6^3 -
                0.5 * sin(x7)^2 + 0.5 * cos(x8) +
                4 / (1 + x9^2) - 1 / (1 + exp(x10)) +
                1.5 * (x11 > 0 ? 1 : 0) - 0.25 * (x13 > 0 ? 1 : 0)
        end
    end

    # Generate treatment based on treatment type
    D = Vector{Float64}(undef, n_obs)
    if treatment == "continuous"
        D .= a_0_vals
    elseif treatment == "binary"
        d_centered = a_0_vals .- mean(a_0_vals)
        prob_d = logistic.(d_centered)
        for i in 1:n_obs
            D[i] = Float64(rand(rng, Binomial(1, prob_d[i])))
        end
    elseif treatment == "binary_unbalanced"
        prob_d = logistic.(a_0_vals)
        for i in 1:n_obs
            D[i] = Float64(rand(rng, Binomial(1, prob_d[i])))
        end
    else
        throw(ArgumentError("Invalid treatment type: $treatment. Must be one of: continuous, binary, binary_unbalanced"))
    end

    # Generate outcome
    p = logistic.(alpha .* D .+ r_0_vals)
    Y = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        Y[i] = Float64(rand(rng, Binomial(1, p[i])))
    end

    # Create DataFrame
    df = DataFrame(Float32.(X), [Symbol("X$i") for i in 1:dim_x])
    df.y = Float32.(Y)
    df.d = Float32.(D)

    # Optionally include propensity score
    if return_p
        df.p = Float32.(p)
    end

    if return_type == :DoubleMLData
        return DoubleMLData(df, y_col = :y, d_col = :d, x_cols = [Symbol("X$i") for i in 1:dim_x])
    else
        return df
    end
end
