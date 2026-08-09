"""
    make_plr_DTL2025(n::Int, p::Int, sigma_epsilon::Real; 
                     alpha=2.0, rng=Random.default_rng(), return_type=:DoubleMLData)

Generate synthetic data matching the simulation design from 
DiTraglia & Liu (2025), Section 6, Table 1.

This DGP corresponds to the "fixed" design described in Equations (20)-(21)
of the paper, which is used to generate the results reported in Table 1.

# Data Generating Process (Equations 20-21)

**Covariates:**
    ``X_i \\sim \\text{iid Normal}_p(0, I_p)``

**Errors:**
    ``(\\epsilon_i, V_i)' | X \\sim \\text{iid Normal}([0; 0], [\\sigma^2_\\epsilon, 0; 0, 1])``
    
where:
-  ``\\epsilon_i``: Structural error (uncorrelated with ``V_i``)
-  ``V_i``: Treatment error (variance = 1)
-  ``\\sigma_\\epsilon``: Standard deviation of ``\\epsilon`` (varies across simulations)

**Coefficient distribution:**
    ``\\beta \\sim \\text{Normal}_p(\\mu_\\beta, \\sigma^2_\\beta \\cdot I_p)``

**Fixed parameters (per paper):**
-   ``\\alpha = 2``                          (true causal effect)
-   ``\\gamma = \\iota_p / \\sqrt{p}``                   (treatment coefficients)
-   ``\\mu_\\beta = -\\gamma/2``                     (mean of outcome coefficients)
-   ``\\sigma^2_\\beta = 1/p``                     (variance of outcome coefficients)

**Construction:**
-  ``D_i = X_i'\\gamma + V_i``                (Equation 4: Treatment reduced form)
-  ``Y_i = \\alpha \\cdot D_i + X_i'\\beta + \\epsilon_i``       (Equation 5: Outcome structural)

# Arguments
- `n::Int`: Number of observations (paper uses: 200)
- `p::Int`: Number of covariates (paper uses: 100)
- `sigma_epsilon::Real`: Std dev of structural error ``\\epsilon \\in \\{1, 2, 4\\}``
- `alpha::Real`: True causal effect (default: 2.0)
- `rng::AbstractRNG`: Random number generator for reproducibility (default: `Random.default_rng()`)
- `return_type::Symbol`: Return type, either `:DoubleMLData` (default) or `:DataFrame`

# Returns
- `data::DoubleMLData`: A DoubleMLData object ready for DML analysis (default)
- `df::DataFrame`: A DataFrame with columns `X1,...,Xp`, `y`, `d` (if `return_type=:DataFrame`)

The true causal effect is always `alpha` (default: 2.0). Data are stored as `Float32` 
for memory efficiency. Ground truth parameters (γ, β, errors) are not returned but 
can be reconstructed from the DGP specification if needed for simulation studies.

# Paper Reference
Section 6, "Simulation Study", Equations (20)-(21):
> DiTraglia, F.J. & Liu, L. (2025). "Bayesian Double Machine Learning for 
> Causal Inference", arXiv:2508.12688v1.

# Table 1 Settings
The paper reports results for three values of ``\\sigma_\\epsilon``:

| ``\\sigma_\\epsilon`` | BDML-Hier Coverage | BDML-Hier RMSE | BDML-Basic Coverage | BDML-Basic RMSE |
|-----|-------------------|----------------|---------------------|-----------------|
| 1   | 0.94              | 0.09           | 0.93                | 0.11            |
| 2   | 0.94              | 0.18           | 0.91                | 0.22            |
| 4   | 0.94              | 0.35           | 0.92                | 0.46            |

All settings use n=200, p=100, and ``\\alpha=2``.

# Examples
```julia
using Random
using DoubleML

# Generate data as DoubleMLData (default)
data = make_plr_DTL2025(200, 100, 2.0)

# Generate data as DataFrame
df = make_plr_DTL2025(200, 100, 2.0; return_type=:DataFrame)

# Reproducible generation
rng = Xoshiro(123)
data = make_plr_DTL2025(200, 100, 2.0; rng=rng)
```

# Implementation Notes

**Data types:**
- Data are stored as `Float32` for memory efficiency
- The true causal effect `alpha` is 2.0 (default) or user-specified

**Randomness:**
- The function generates new random draws for ``X``, ``\beta``, ``\epsilon``, and ``V`` in each call
- Pass an `rng` argument (e.g., `Xoshiro(seed)`) for reproducibility
- Following the paper, each replication should use a fresh seed or different seed

**Coefficient generation:**
- ``\\gamma`` is fixed at ``\\iota_p/\\sqrt{p}`` (as specified in paper)
- ``\\beta`` is randomly drawn from ``N(\\mu_\\beta, \\sigma^2_\\beta\\cdot I)`` for each replication
- This mimics the paper's design where ``\\beta`` varies across replications

**Error structure:**
- ``\\epsilon`` and ``V`` are independent (covariance matrix is diagonal)
- ``\\text{Var}(V) = 1`` (normalized)
- ``\\text{Var}(\\epsilon) = \\sigma^2_\\epsilon`` (varies across simulation settings)

**Confounding:**
- ``X`` affects both ``D`` (through ``\\gamma``) and ``Y`` (through ``\\beta``)
- The specific structure (``\\gamma = \\iota_p/\\sqrt{p}``, ``\\mu_\\beta = -\\gamma/2``) creates realistic correlation

See also: [`make_plr_DTL2025`](@ref)
"""
function make_plr_DTL2025(
        n::Int, p::Int, sigma_epsilon::Real; alpha = 2.0, rng = Random.default_rng(),
        return_type = :DoubleMLData
    )
    # Validate inputs
    @assert n > 0 "n must be positive"
    @assert p > 0 "p must be positive"
    @assert sigma_epsilon > 0 "sigma_epsilon must be positive"

    # Fixed parameters from paper (Equation 21)
    gamma = ones(p) ./ sqrt(p)          # γ = ι_p / √p  (treatment coefficients)
    mu_beta = -gamma ./ 2              # μ_β = -γ/2    (mean of outcome coefficients)
    sigma2_beta = 1.0 / p              # σ²_β = 1/p    (variance of outcome coefficients)

    # Generate covariates (Equation 20): X_i ~ N(0, I_p)
    X = randn(rng, n, p)

    # Generate outcome coefficients (Equation 20): β ~ N(μ_β, σ²_β · I_p)
    # These are drawn fresh for each replication as per paper
    beta = randn(rng, p) .* sqrt(sigma2_beta) .+ mu_beta

    # Generate errors (Equation 20)
    V = randn(rng, n)                                  # Treatment error: V ~ N(0, 1)
    epsilon = randn(rng, n) .* sigma_epsilon           # Structural error: ε ~ N(0, σ²_ε)

    # Construct treatment D (Equation 4): D = X'γ + V
    D = X * gamma + V

    # Construct outcome Y (Equation 5): Y = α·D + X'β + ε
    Y = alpha .* D + X * beta + epsilon

    df = DataFrame(Float32.(X), [Symbol("X$i") for i in 1:p])
    df.y = Float32.(Y)
    df.d = Float32.(D)

    if return_type == :DoubleMLData
        return DoubleMLData(df, y_col = :y, d_col = :d, x_cols = [Symbol("X$i") for i in 1:p])
    else
        return df
    end
end
