using LinearAlgebra
using Statistics
using Distributions
using DataFrames
using Random
using CategoricalArrays

"""
    make_irm_data(
        n_obs;
        dim_x=20,
        theta=0.0,
        R2_d=0.5,
        R2_y=0.5,
        return_type=:DoubleMLData,
        rng=nothing
    ) -> Union{DoubleMLData, DataFrame}

Generate synthetic data for an Interactive Regression Model (IRM) based on
Belloni et al. (2017).

This function generates data from the model:
```
d = 1{exp(c_d * X' * beta) / (1 + exp(c_d * X' * beta)) > v}    (Treatment)
y = theta * d + c_y * X' * beta * d + zeta                        (Outcome)
```
where:
- `v ~ U(0, 1)` (uniform error for treatment)
- `zeta ~ N(0, 1)` (standard normal error for outcome)
- `X ~ N(0, Sigma)` with `Sigma_kj = 0.5^|j-k|` (covariates with Toeplitz covariance)
- `beta_j = 1/j^2` for j = 1, ..., dim_x
- Constants:
  - `c_y = sqrt(R2_y / ((1 - R2_y) * beta' * Sigma * beta))`
  - `c_d = sqrt((pi^2 / 3) * R2_d / ((1 - R2_d) * beta' * Sigma * beta))`

The data generating process is inspired by the simulation experiment in
Appendix P of Belloni et al. (2017).

# Arguments
- `n_obs::Int`: Number of observations to generate
- `dim_x::Int=20`: Number of covariates (dimension of X)
- `theta::Real=0.0`: True treatment effect parameter
- `R2_d::Real=0.5`: The value of the parameter R²_d
- `R2_y::Real=0.5`: The value of the parameter R²_y
- `return_type::Symbol=:DoubleMLData`: Output format (`:DoubleMLData` or `:DataFrame`)
- `rng::Union{AbstractRNG, Nothing}=nothing`: Random number generator (uses global RNG if nothing)

# Returns
- `DoubleMLData` object if `return_type=:DoubleMLData` (default)
- `DataFrame` if `return_type=:DataFrame`

# Generated Data Structure
The generated data contains:
- `y`: Outcome variable (Vector{Float32})
- `d`: Treatment variable (Vector{Float32}, binary 0/1)
- `X1`, `X2`, ..., `X{dim_x}`: Covariates (Float32)

# Examples
```julia
using DoubleML

# Generate default dataset
data = make_irm_data(1000)

# Generate with custom parameters
data = make_irm_data(
    500,
    dim_x=10,
    theta=1.0,
    R2_d=0.6,
    R2_y=0.4,
    return_type=:DoubleMLData
)

# Get DataFrame instead
df = make_irm_data(1000, return_type=:DataFrame)

# Use with specific RNG
using StableRNGs
rng = StableRNG(123)
data = make_irm_data(1000, rng=rng)
```

# References
Belloni, A., Chernozhukov, V., Fernández-Val, I. and Hansen, C. (2017).
"Program Evaluation and Causal Inference With High-Dimensional Data."
Econometrica, 85: 233-298.

See also: [`DoubleMLIRM`](@ref), [`DoubleMLData`](@ref)
"""
function make_irm_data(
        n_obs;
        dim_x = 20,
        theta = 0.0,
        R2_d = 0.5,
        R2_y = 0.5,
        return_type = :DoubleMLData,
        rng = Random.default_rng()
    )

    # Uniform error for treatment
    v = rand(rng, Uniform(0, 1), n_obs)
    # Standard normal error for outcome
    zeta = rand(rng, Normal(0, 1), n_obs)

    # Toeplitz covariance matrix: Sigma_kj = 0.5^|j-k|
    sigma = [0.5^abs(j - k) for j in 1:dim_x, k in 1:dim_x]
    dist_x = MultivariateNormal(zeros(dim_x), sigma)
    X = rand(rng, dist_x, n_obs)'

    # beta_j = 1/j^2 for j = 1, ..., dim_x
    beta = [1.0 / (j^2) for j in 1:dim_x]

    # Compute beta' * Sigma * beta
    b_sigma_b = dot(beta, sigma * beta)

    # Compute constants c_y and c_d
    c_y = sqrt(R2_y / ((1.0 - R2_y) * b_sigma_b))
    c_d = sqrt((pi^2 / 3.0) * R2_d / ((1.0 - R2_d) * b_sigma_b))

    # Compute treatment: d = 1{exp(c_d * X' * beta) / (1 + exp(c_d * X' * beta)) > v}
    x_beta_cd = X * (beta .* c_d)
    prob_d = exp.(x_beta_cd) ./ (1.0 .+ exp.(x_beta_cd))
    D = Float32.(prob_d .> v)

    # Compute outcome: y = theta * d + c_y * X' * beta * d + zeta
    x_beta_cy = X * (beta .* c_y)
    Y = Float32(theta) .* D .+ D .* Float32.(x_beta_cy) .+ Float32.(zeta)

    df = DataFrame(Float32.(X), [Symbol("X$i") for i in 1:dim_x])
    df.y = Y
    df.d = categorical(D, levels = [0, 1])

    if return_type == :DoubleMLData
        return DoubleMLData(df, y_col = :y, d_col = :d, x_cols = [Symbol("X$i") for i in 1:dim_x])
    else
        return df
    end
end
