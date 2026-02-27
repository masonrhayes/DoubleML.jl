using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Random
using StableRNGs

"""
    make_plr_CCDDHNR2018(
        n_obs;
        dim_x=20,
        alpha=0.5,
        a_0=1.0, a_1=0.25,
        b_0=1.0, b_1=0.25,
        s_1=1.0, s_2=1.0,
        return_type=:DoubleMLData,
        rng=nothing
    ) -> Union{DoubleMLData, DataFrame}

Generate synthetic data for a Partially Linear Regression (PLR) model based on
Chernozhukov et al. (2018).

This function generates data from the model:
```
Y = α·D + g(X) + ζ    (Outcome equation)
D = m(X) + v          (Treatment equation)
```
where:
- `g(X) = b₀·exp(X₁)/(1 + exp(X₁)) + b₁·X₃`
- `m(X) = a₀·X₁ + a₁·exp(X₃)/(1 + exp(X₃))`

The covariates X follow a multivariate normal distribution with Toeplitz
covariance structure (correlation 0.7^|j-k| between Xⱼ and Xₖ).

# Arguments
- `n_obs::Int`: Number of observations to generate
- `dim_x::Int=20`: Number of covariates (dimension of X)
- `alpha::Real=0.5`: True treatment effect parameter (α)
- `a_0::Real=1.0`: Parameter for treatment equation linear term
- `a_1::Real=0.25`: Parameter for treatment equation nonlinear term
- `b_0::Real=1.0`: Parameter for outcome equation nonlinear term
- `b_1::Real=0.25`: Parameter for outcome equation linear term
- `s_1::Real=1.0`: Standard deviation of treatment error (v)
- `s_2::Real=1.0`: Standard deviation of outcome error (ζ)
- `return_type::Symbol=:DoubleMLData`: Output format (`:DoubleMLData` or `:DataFrame`)
- `rng::Union{AbstractRNG, Nothing}=nothing`: Random number generator (uses global RNG if nothing)

# Returns
- `DoubleMLData` object if `return_type=:DoubleMLData` (default)
- `DataFrame` if `return_type=:DataFrame`

# Generated Data Structure
The generated data contains:
- `y`: Outcome variable (Vector{Float32})
- `d`: Treatment variable (Vector{Float32})
- `X1`, `X2`, ..., `X{dim_x}`: Covariates (Float32)

# Examples
```julia
using DoubleML

# Generate default dataset
data = make_plr_CCDDHNR2018(1000)

# Generate with custom parameters
data = make_plr_CCDDHNR2018(
    500,
    dim_x=10,
    alpha=1.0,
    return_type=:DoubleMLData
)

# Get DataFrame instead
df = make_plr_CCDDHNR2018(1000, return_type=:DataFrame)

# Use with specific RNG
using StableRNGs
rng = StableRNG(123)
data = make_plr_CCDDHNR2018(1000, rng=rng)
```

# References
Chernozhukov et al. (2018): "Double/Debiased Machine Learning for Treatment and Causal Parameters"

See also: [`DoubleMLPLR`](@ref), [`DoubleMLData`](@ref)
"""
function make_plr_CCDDHNR2018(
        n_obs;
        dim_x = 20,
        alpha = 0.5,
        a_0 = 1.0, a_1 = 0.25,
        b_0 = 1.0, b_1 = 0.25,
        s_1 = 1.0, s_2 = 1.0,
        return_type = :DoubleMLData,
        rng = Random.default_rng()
    )

    # Toeplitz covariance matrix
    sigma = [0.7^abs(j - k) for j in 1:dim_x, k in 1:dim_x]
    dist_x = MultivariateNormal(zeros(dim_x), sigma)
    X = rand(rng, dist_x, n_obs)'

    v = rand(rng, Normal(0, s_1), n_obs)
    zeta = rand(rng, Normal(0, s_2), n_obs)

    # Compute m_0(X) and g_0(X) without closures
    # m_0(x) = a_0 * x[1] + a_1 * exp(x[3]) / (1 + exp(x[3]))
    # g_0(x) = b_0 * exp(x[1]) / (1 + exp(x[1])) + b_1 * x[3]
    m_0_vals = Vector{Float64}(undef, n_obs)
    g_0_vals = Vector{Float64}(undef, n_obs)
    for i in 1:n_obs
        x1 = X[i, 1]
        x3 = X[i, 3]
        m_0_vals[i] = a_0 * x1 + a_1 * exp(x3) / (1 + exp(x3))
        g_0_vals[i] = b_0 * exp(x1) / (1 + exp(x1)) + b_1 * x3
    end

    D = Float32.(m_0_vals) + v
    Y = Float32(alpha) .* D + Float32.(g_0_vals) + zeta

    df = DataFrame(Float32.(X), [Symbol("X$i") for i in 1:dim_x])
    df.y = Float32.(Y)
    df.d = Float32.(D)

    if return_type == :DoubleMLData
        return DoubleMLData(df, y_col = :y, d_col = :d, x_cols = [Symbol("X$i") for i in 1:dim_x])
    else
        return df
    end
end
