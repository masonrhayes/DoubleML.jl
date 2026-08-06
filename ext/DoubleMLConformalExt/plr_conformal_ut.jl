"""
    DoubleMLPLRConformalUT{T, L, M}

Experimental: Double Machine Learning for Partially Linear Regression
with Conformal Prediction and Unscented Transform (UT) uncertainty propagation.

This variant replaces Monte Carlo sampling with a deterministic unscented
transform that propagates conformal prediction interval uncertainty through
the nonlinear DML2 score function at second-order accuracy. It uses closed-form
moment propagation (O(n) per evaluation) combined with a 2D unscented transform
for the ratio estimator θ̂ = -mean(ψ_b)/mean(ψ_a). Correlation uncertainty is
handled via Gauss-Hermite quadrature over the Fisher z-transform.

Key advantages over MC sampling:
- **Deterministic:** identical inputs → identical outputs (no RNG dependence)
- **Fast:** ~15 deterministic evaluations vs. 100+ MC iterations

# Type Parameters
- `T<:AbstractFloat`: Numeric type
- `L<:Supervised`: Type of ml_l learner (must be a conformal-wrapped model)
- `M<:Supervised`: Type of ml_m learner (must be a conformal-wrapped model)

# Constructor Options
- `n_folds::Int=1`: Number of cross-fitting folds (default 1 = no cross-fitting)
- `n_rep::Int=1`: Number of sample splitting repetitions (used when n_folds > 1)
- `score::Symbol=:partialling_out`: Score type (`:partialling_out` only for now)
- `ut_alpha::T=1.0`: UT spread parameter
- `ut_beta::T=2.0`: UT prior-knowledge parameter (optimal for Gaussian = 2)
- `ut_kappa::T=1.0`: UT secondary scaling parameter (for 2D state, κ=3-n=1)
- `n_gh::Int=3`: Gauss-Hermite quadrature points for correlation uncertainty (3 or 5)

# Example
```julia
using DoubleML, ConformalPrediction, MLJ

# Wrap models with conformal prediction
base_l = @load LinearRegressor pkg=MLJLinearModels
base_m = @load LinearRegressor pkg=MLJLinearModels
ml_l = conformal_model(base_l; method=:simple_inductive, coverage=0.95)
ml_m = conformal_model(base_m; method=:simple_inductive, coverage=0.95)

# Create and fit UT model
data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
model = DoubleMLPLRConformalUT(data, ml_l, ml_m)
fit!(model)  # Deterministic: no rng needed

# Results
coef(model)           # UT mean (includes Jensen correction)
stderror(model)       # UT standard error
confint(model)        # Normal-approximation CI
```
"""
mutable struct DoubleMLPLRConformalUT{
        T <: AbstractFloat,
        L <: Supervised,
        M <: Supervised,
    } <: AbstractDoubleML{T}

    # Data reference
    data::DoubleMLData{T}

    # Learners
    ml_l::L
    ml_m::M

    # Conformal configuration
    conformal_method::Symbol
    coverage::T

    # Score object
    score_obj::AbstractScore

    # Prediction intervals storage: (n_obs, 2 for lower/upper)
    l_intervals::Matrix{T}
    m_intervals::Matrix{T}

    # Correlation between l and m nuisance function uncertainties
    lm_correlation::T
    lm_corr_std::T

    # Results (primary)
    coef::T
    se::T

    # Standard DML results (for comparison)
    standard_dml_coef::T
    standard_dml_se::T

    # UT conformal-only results (diagnostics)
    ut_mean::T
    ut_var::T

    # Cross-fitting configuration (1 = no cross-fitting)
    n_folds::Int
    n_rep::Int

    # UT configuration
    ut_alpha::T
    ut_beta::T
    ut_kappa::T
    n_gh::Int

    # Diagnostics (populated after fit!)
    score_mean::Vector{T}   # [μ_A, μ_B]
    score_cov::Matrix{T}    # 2×2 covariance matrix
end

"""
    DoubleMLPLRConformalUT(data, ml_l, ml_m; n_folds=1, n_rep=1, score=:partialling_out,
                           ut_alpha=1.0, ut_beta=2.0, ut_kappa=1.0, n_gh=3)

Create a DoubleML PLR model with conformal prediction and unscented transform
uncertainty propagation.

Users must wrap their base MLJ models with `conformal_model()` from
ConformalPrediction.jl before passing them to this constructor.

# Keyword Arguments
- `n_folds::Int=1`: Number of cross-fitting folds. Default 1 means no cross-fitting.
- `n_rep::Int=1`: Number of sample splitting repetitions (used when n_folds > 1)
- `n_mc_samples::Int=100`: (ignored; kept for API compatibility with conformal types)
- `score::Symbol=:partialling_out`: Score type (`:partialling_out` only for now)
- `ut_alpha::Real=1.0`: UT spread parameter (controls sigma point distance from mean)
- `ut_beta::Real=2.0`: UT prior-knowledge parameter (optimal = 2 for Gaussian inputs)
- `ut_kappa::Real=1.0`: UT secondary scaling (for 2D state, standard κ = 3 - n_dim = 1)
- `n_gh::Int=3`: Gauss-Hermite quadrature points for correlation uncertainty (3 or 5)

# Examples
```julia
using DoubleML, DoubleML.Experimental, ConformalPrediction, MLJ

# Generate data
data = make_plr_CCDDHNR2018(500)

# Wrap models with conformal prediction
base_l = @load LinearRegressor pkg=MLJLinearModels
base_m = @load LinearRegressor pkg=MLJLinearModels
ml_l = conformal_model(base_l; method=:cv_minmax, coverage=0.95)
ml_m = conformal_model(base_m; method=:cv_minmax, coverage=0.95)

# Create and fit UT model (no cross-fitting)
model = DoubleMLPLRConformalUT(data, ml_l, ml_m)
fit!(model)

# Create model with 5-fold cross-fitting
model_cf = DoubleMLPLRConformalUT(data, ml_l, ml_m; n_folds=5, n_rep=2)
fit!(model_cf)

# Get results with UT uncertainty
coef(model)
confint(model)
```
"""
function DoubleMLPLRConformalUT(
        data::DoubleMLData{T},
        ml_l::L,
        ml_m::M;
        n_folds::Int = 1,
        n_rep::Int = 1,
        score::Symbol = :partialling_out,
        ut_alpha::Real = 1.0,
        ut_beta::Real = 2.0,
        ut_kappa::Real = 1.0,
        n_gh::Int = 3,
    ) where {T <: AbstractFloat, L <: ConformalPrediction.ConformalModel, M <: ConformalPrediction.ConformalModel}

    # Validate cross-fitting parameters
    n_folds >= 1 || throw(ArgumentError("n_folds must be >= 1, got $n_folds"))
    n_rep >= 1 || throw(ArgumentError("n_rep must be >= 1, got $n_rep"))

    # Determine conformal method and coverage from both conformal models
    conformal_method_l, coverage_l = _get_conformal_info(ml_l)
    conformal_method_m, coverage_m = _get_conformal_info(ml_m)

    # Enforce matching conformal methods
    if conformal_method_l != conformal_method_m
        throw(
            ArgumentError(
                "ml_l and ml_m must use the same conformal method. " *
                    "Got ml_l method=$(conformal_method_l) and ml_m method=$(conformal_method_m). " *
                    "Please ensure both learners use the same method parameter when wrapping with conformal_model()."
            )
        )
    end

    # Enforce matching coverage levels
    if coverage_l != coverage_m
        throw(
            ArgumentError(
                "ml_l and ml_m must have the same coverage level. " *
                    "Got ml_l coverage=$(coverage_l) and ml_m coverage=$(coverage_m). " *
                    "Please ensure both learners use the same coverage parameter when wrapping with conformal_model()."
            )
        )
    end
    coverage = coverage_l

    # Warn if coverage is below recommended 0.95
    if coverage < 0.95
        @warn "Coverage level $(coverage) is below the recommended 0.95. " *
            "Lower coverage levels may result in insufficient uncertainty quantification."
    end

    # Validate score
    score == :partialling_out || throw(
        ArgumentError(
            "Only score=:partialling_out is supported. Got: $score"
        )
    )

    # Validate UT parameters
    n_gh in (3, 5) || throw(ArgumentError("n_gh must be 3 or 5. Got: $n_gh"))
    ut_alpha > 0 || throw(ArgumentError("ut_alpha must be positive. Got: $ut_alpha"))

    # Create score object
    score_obj = PartiallingOutScore()

    # Storage allocation
    n_obs = data.n_obs
    l_intervals = zeros(T, n_obs, 2)
    m_intervals = zeros(T, n_obs, 2)

    return DoubleMLPLRConformalUT{T, L, M}(
        data, ml_l, ml_m,
        conformal_method_l, T(coverage),
        score_obj,
        l_intervals, m_intervals,
        T(NaN), T(NaN),  # lm_correlation, lm_corr_std (estimated in fit!)
        T(NaN), T(NaN),  # coef, se (computed in fit!)
        T(NaN), T(NaN),  # standard_dml_coef, standard_dml_se (stored in fit!)
        T(NaN), T(NaN),  # ut_mean, ut_var (computed in fit!)
        n_folds, n_rep,
        T(ut_alpha), T(ut_beta), T(ut_kappa), n_gh,
        zeros(T, 2), zeros(T, 2, 2)  # score_mean, score_cov diagnostics
    )
end

# ============================================================================
# Core moment propagation: O(n) closed-form score moments
# ============================================================================

"""
    _score_moments(obj::DoubleMLPLRConformalUT, rho) -> (μA, μB, ΣA, ΣB, ΣAB)

Compute the mean and covariance of the aggregated score statistics
(Ā = mean(ψ_a), B̄ = mean(ψ_b)) from conformal prediction intervals,
assuming a Gaussian copula with correlation `rho` between l̂ and m̂ prediction
errors.

Uses exact Beta(2,2) marginal moments and Gaussian (Isserlis) cross-moment
approximation. Cost: O(n) closed form.
"""
function _score_moments(
        obj::DoubleMLPLRConformalUT{T},
        rho::T,
    ) where {T}

    n_obs = obj.data.n_obs
    Y = obj.data.y
    D = obj.data.d

    # Interval bounds
    @views l_lowers = obj.l_intervals[:, 1]
    @views l_uppers = obj.l_intervals[:, 2]
    @views m_lowers = obj.m_intervals[:, 1]
    @views m_uppers = obj.m_intervals[:, 2]

    # Accumulators
    mu_a = zero(T)
    mu_b = zero(T)
    var_a = zero(T)
    var_b = zero(T)
    cov_ab = zero(T)

    # Precompute Beta(2,2) variance constant on [0,1]: 1/20
    beta_var = T(1) / T(20)
    # 4th central moment coefficient for Beta(2,2): 15/7 (relative to σ⁴)
    # We use exact: Var(ψ_a) = 4 r_m² s_m + (8/7) s_m²
    coeff_4th = T(8) / T(7)

    @inbounds for i in 1:n_obs
        # Midpoints and widths
        l_mid = (l_lowers[i] + l_uppers[i]) / 2
        m_mid = (m_lowers[i] + m_uppers[i]) / 2
        l_width = l_uppers[i] - l_lowers[i]
        m_width = m_uppers[i] - m_lowers[i]

        # Variances of scaled Beta(2,2)
        s_l = l_width * l_width * beta_var
        s_m = m_width * m_width * beta_var

        # Nominal residuals (score evaluated at interval midpoints)
        r_l = Y[i] - l_mid
        r_m = D[i] - m_mid

        # Gaussian copula covariance between prediction errors
        c = rho * sqrt(s_l * s_m)

        # --- Mean of ψ_a = -m² ---
        mu_a += -(r_m * r_m + s_m)

        # --- Mean of ψ_b = g·m ---
        mu_b += r_l * r_m + c

        # --- Variance of ψ_a ---
        var_a += 4 * r_m * r_m * s_m + coeff_4th * s_m * s_m

        # --- Variance of ψ_b ---
        # Var = r_l² s_m + s_l r_m² + s_l s_m + 2 r_l r_m c + c²
        var_b += r_l^2 * s_m + r_m^2 * s_l + s_l * s_m +
            2 * r_l * r_m * c + c^2

        # --- Covariance(ψ_a, ψ_b) ---
        # Cov = -2[r_l r_m s_m + c(r_m² + s_m)]
        cov_ab += -2 * (r_l * r_m * s_m + c * (r_m^2 + s_m))
    end

    # Scale by 1/n and 1/n² for aggregation
    inv_n = T(1) / T(n_obs)
    mu_a *= inv_n
    mu_b *= inv_n
    var_a *= inv_n * inv_n
    var_b *= inv_n * inv_n
    cov_ab *= inv_n * inv_n

    return mu_a, mu_b, var_a, var_b, cov_ab
end

# ============================================================================
# 2D Unscented Transform for the ratio θ = -B̄/Ā
# ============================================================================

"""
    _make_psd_cholesky(M::AbstractMatrix{T}) -> Cholesky

Ensure a 2×2 matrix is positive definite by directly calculating its minimum eigenvalue
and applying an exact shift if needed, returning its Cholesky factorization.
"""
function _make_psd_cholesky(M::AbstractMatrix{T}) where {T}
    a = M[1, 1]
    c = M[2, 2]
    b = (M[1, 2] + M[2, 1]) / 2

    # Calculate the minimum eigenvalue analytically
    # hypot(x, y) is a safe way to compute sqrt(x^2 + y^2) preventing overflow
    min_eig = (a + c - hypot(a - c, 2 * b)) / 2
    
    jitter = cbrt(eps(T))^2

    # If the matrix isn't sufficiently positive definite, apply the exact shift needed
    if min_eig < jitter
        shift = jitter - min_eig
        a += shift
        c += shift
    end

    return cholesky(Symmetric([a b; b c]))
end

"""
    _ut_propagate(μA, μB, ΣA, ΣB, ΣAB, alpha, beta, kappa) -> (theta, var)

Apply the 2D unscented transform to propagate uncertainty through the
nonlinear map θ = -B̄/Ā.

Sigma point parameters (standard): α=1, β=2, κ=3-n_dim=1 for a 2D state.
This gives 5 sigma points. A numerical guard prevents division by near-zero
in the denominator.
"""
function _ut_propagate(
        mu_a::T, mu_b::T, var_a::T, var_b::T, cov_ab::T,
        alpha::T, beta::T, kappa::T,
    ) where {T}

    n_dim = 2
    lam = alpha * alpha * (n_dim + kappa) - n_dim

    chol = _make_psd_cholesky([var_a cov_ab; cov_ab var_b])
    L = chol.L

    # Scale for sigma point spread
    scale = sqrt(n_dim + lam)

    # Sigma point offsets from Cholesky columns
    (dx1, dy1, dx2, dy2) = scale * L

    # 5 sigma points: mean ± each Cholesky column
    chi_a = (
        mu_a,
        mu_a + dx1,
        mu_a - dx1,
        mu_a + dx2,
        mu_a - dx2,
    )
    chi_b = (
        mu_b,
        mu_b + dy1,
        mu_b - dy1,
        mu_b + dy2,
        mu_b - dy2,
    )

    # Weights
    w0_m = lam / (n_dim + lam)
    wi_m = T(1) / (2 * (n_dim + lam))
    wm = (w0_m, wi_m, wi_m, wi_m, wi_m)

    w0_c = lam / (n_dim + lam) + (1 - alpha * alpha + beta)
    wi_c = T(1) / (2 * (n_dim + lam))
    wc = (w0_c, wi_c, wi_c, wi_c, wi_c)

    # Propagate through θ = -B/A with near-zero guard
    gammas = Vector{T}(undef, 5)
    denom_threshold = sqrt(eps(T))
    for j in 1:5
        a = chi_a[j]
        if abs(a) < denom_threshold
            # Fallback to mean ratio when denominator is near zero
            gammas[j] = -mu_b / mu_a
        else
            gammas[j] = -chi_b[j] / a
        end
    end

    # Recombine to output mean and variance
    theta_est = zero(T)
    for j in 1:5
        theta_est += wm[j] * gammas[j]
    end

    var_theta = zero(T)
    for j in 1:5
        var_theta += wc[j] * (gammas[j] - theta_est)^2
    end

    return theta_est, var_theta
end

# ============================================================================
# Gauss-Hermite quadrature for correlation uncertainty
# ============================================================================

"""
    _gh_nodes_weights(n_gh::Int) -> (nodes, weights)

Return Gauss-Hermite quadrature nodes (in standard-normal space) and weights
for integrating over the Fisher z-transform correlation uncertainty.

Supports n_gh = 3 (default, fast) and n_gh = 5 (more accurate).
"""
function _gh_nodes_weights(n_gh::Int)
    if n_gh == 3
        nodes = (-sqrt(3.0), 0.0, sqrt(3.0))
        weights = (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0)
        return nodes, weights
    elseif n_gh == 5
        nodes = (
            -2.020182870456086,
            -0.9585724646138195,
            0.0,
            0.9585724646138195,
            2.020182870456086,
        )
        weights = (
            0.01125741132784671,
            0.22207592200561266,
            0.5333333333333333,
            0.22207592200561266,
            0.01125741132784671,
        )
        return nodes, weights
    else
        throw(ArgumentError("Only n_gh = 3 or 5 are supported. Got: $n_gh"))
    end
end

# ============================================================================
# Main UT uncertainty propagation (deterministic)
# ============================================================================

"""
    _propagate_ut!(obj::DoubleMLPLRConformalUT, verbose::Int)

Propagate conformal prediction uncertainty via the unscented transform with
Gauss-Hermite quadrature over the Fisher z-transform correlation distribution.

This function is fully deterministic (no RNG).
"""
function _propagate_ut!(
        obj::DoubleMLPLRConformalUT{T},
        verbose::Int,
    ) where {T}

    n_obs = obj.data.n_obs

    if verbose > 0
        @info "Propagating conformal uncertainty via Unscented Transform..."
        @info "  UT parameters: α=$(obj.ut_alpha), β=$(obj.ut_beta), κ=$(obj.ut_kappa)"
        @info "  Gauss-Hermite points: $(obj.n_gh)"
        @info "  Estimated l-m correlation: $(round(obj.lm_correlation, digits = 3))"
    end

    # Fisher z-transform parameters for correlation uncertainty
    z_est = atanh(clamp(obj.lm_correlation, T(-0.99), T(0.99)))
    z_std = obj.lm_corr_std

    # Gauss-Hermite quadrature over z ~ N(z_est, z_std²)
    gh_nodes, gh_weights = _gh_nodes_weights(obj.n_gh)

    # Storage for mixture components
    thetas = Vector{T}(undef, obj.n_gh)
    vars = Vector{T}(undef, obj.n_gh)

    for k in 1:obj.n_gh
        # Sample correlation from its uncertainty distribution (via GH node)
        z_k = z_est + sqrt(T(2)) * z_std * T(gh_nodes[k])
        rho_k = tanh(z_k)
        rho_k = clamp(rho_k, T(-0.99), T(0.99))

        # Compute O(n) score moments for this correlation value
        mu_a, mu_b, var_a, var_b, cov_ab = _score_moments(obj, rho_k)

        # Apply 2D UT to the ratio θ = -B/A
        theta_k, var_k = _ut_propagate(
            mu_a, mu_b, var_a, var_b, cov_ab,
            obj.ut_alpha, obj.ut_beta, obj.ut_kappa,
        )

        thetas[k] = theta_k
        vars[k] = var_k

        if verbose > 1
            @info "  GH node $k: ρ=$(round(rho_k, digits = 3)), θ=$(round(theta_k, digits = 4)), var=$(round(var_k, digits = 6))"
        end
    end

    # Mixture via law of total variance
    theta_bar = zero(T)
    for k in 1:obj.n_gh
        theta_bar += T(gh_weights[k]) * thetas[k]
    end

    # Within-rho variance + between-rho variance
    var_total = zero(T)
    for k in 1:obj.n_gh
        var_total += T(gh_weights[k]) * vars[k]
    end
    for k in 1:obj.n_gh
        diff = thetas[k] - theta_bar
        var_total += T(gh_weights[k]) * diff * diff
    end

    # Store diagnostics
    # Use the center (mean-rho) moments for diagnostics
    mu_a0, mu_b0, var_a0, var_b0, cov_ab0 = _score_moments(obj, obj.lm_correlation)
    obj.score_mean[1] = mu_a0
    obj.score_mean[2] = mu_b0
    obj.score_cov[1, 1] = var_a0
    obj.score_cov[2, 2] = var_b0
    obj.score_cov[1, 2] = cov_ab0
    obj.score_cov[2, 1] = cov_ab0

    # Store results
    obj.ut_mean = theta_bar
    obj.ut_var = var_total
    obj.coef = theta_bar                              # UT mean (Jensen-corrected)
    obj.se = sqrt(obj.standard_dml_se^2 + var_total)  # combined SE

    if verbose > 0
        ci = confint(obj; level = obj.coverage)
        @info "UT propagation complete!"
        @info "  Standard DML: θ=$(round(obj.standard_dml_coef, digits = 4)), SE=$(round(obj.standard_dml_se, digits = 4))"
        @info "  UT mean:        θ=$(round(obj.ut_mean, digits = 4)), SE=$(round(sqrt(obj.ut_var), digits = 4)) (conformal-only)"
        @info "  Combined:       θ=$(round(obj.coef, digits = 4)), SE=$(round(obj.se, digits = 4))"
        @info "  Difference:     $(round(obj.coef - obj.standard_dml_coef, digits = 4)) (UT - standard DML)"
        @info "  $(obj.coverage * 100)% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]"
    end

    return obj
end

# ============================================================================
# Fitting
# ============================================================================

"""
    fit!(obj::DoubleMLPLRConformalUT; verbose=0, rng=Random.default_rng())

Fit the conformal PLR model using the unscented transform for uncertainty
propagation.

Trains nuisance models, estimates the residual correlation, then propagates
uncertainty deterministically via closed-form moments + 2D UT.

# Arguments
- `obj`: The DoubleMLPLRConformalUT model to fit
- `verbose::Int=0`: Verbosity level (0=silent, 1=info, 2=debug)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator (used only
  for cross-fitting sample splitting; the UT propagation itself is deterministic)
"""
function MLJ.fit!(
        obj::DoubleMLPLRConformalUT{T};
        verbose::Int = 0,
        rng::AbstractRNG = Random.default_rng(),
    ) where {T}

    if obj.n_folds == 1
        return _fit_full_sample_ut!(obj, rng, verbose)
    else
        return _fit_cross_fitting_ut!(obj, rng, verbose)
    end
end

"""
    _fit_full_sample_ut!(obj, rng, verbose)

Internal function: Fit the conformal PLR model without cross-fitting,
using the unscented transform for uncertainty propagation.
"""
function _fit_full_sample_ut!(
        obj::DoubleMLPLRConformalUT{T},
        rng::AbstractRNG,
        verbose::Int,
    ) where {T}

    n_obs = obj.data.n_obs

    X = DataFrames.DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = obj.data.d
    D_coerced = DoubleML.coerce_target(D, obj.ml_m)

    if verbose > 0
        @info "Fitting DoubleMLPLRConformalUT"
        @info "  Conformal method: $(obj.conformal_method)"
        @info "  Coverage: $(obj.coverage)"
        @info "  UT: α=$(obj.ut_alpha), β=$(obj.ut_beta), κ=$(obj.ut_kappa), GH=$(obj.n_gh)"
    end

    # Fit ml_l and get prediction intervals
    if verbose > 0
        @info "Fitting ml_l (E[Y|X])..."
    end
    mach_l = MLJ.machine(obj.ml_l, X, Y)
    MLJ.fit!(mach_l, verbosity = verbose)
    l_intervals = MLJ.predict(mach_l, X)

    # Store intervals and compute midpoints
    l_midpoints = Vector{T}(undef, n_obs)
    for (i, (lower, upper)) in enumerate(l_intervals)
        obj.l_intervals[i, 1] = lower
        obj.l_intervals[i, 2] = upper
        l_midpoints[i] = (lower + upper) / 2
    end

    # Fit ml_m and get prediction intervals
    if verbose > 0
        @info "Fitting ml_m (E[D|X])..."
    end
    mach_m = MLJ.machine(obj.ml_m, X, D_coerced)
    MLJ.fit!(mach_m, verbosity = verbose)
    m_intervals = MLJ.predict(mach_m, X)

    m_midpoints = Vector{T}(undef, n_obs)
    for (i, (lower, upper)) in enumerate(m_intervals)
        obj.m_intervals[i, 1] = lower
        obj.m_intervals[i, 2] = upper
        m_midpoints[i] = (lower + upper) / 2
    end

    # Compute standard DML score using midpoints
    psi_a, psi_b = DoubleML.compute_score(
        obj.score_obj, Y, D, l_midpoints, m_midpoints,
    )

    # Standard coefficient and SE computation (for checking)
    obj.standard_dml_coef = DoubleML.dml2_solve(psi_a, psi_b)
    psi = @. (psi_a * obj.standard_dml_coef) + psi_b
    J = mean(psi_a)
    gamma_hat = mean(psi .^ 2)
    sigma2_hat = gamma_hat / (n_obs * J^2)
    obj.standard_dml_se = sqrt(sigma2_hat)

    # Estimate correlation between l and m nuisance uncertainties
    l_residuals = Y .- l_midpoints
    m_residuals = D .- m_midpoints
    rho_est = cor(l_residuals, m_residuals)

    # Fisher z-transform for better normal approximation
    z_est = atanh(clamp(rho_est, T(-0.99), T(0.99)))
    z_std = T(1.0) / sqrt(max(1, n_obs - 3))

    obj.lm_correlation = T(rho_est)
    obj.lm_corr_std = T(z_std)

    if verbose > 0
        @info "Estimated l-m correlation: $(round(obj.lm_correlation, digits = 3)) (±$(round(obj.lm_corr_std, digits = 3)) in z-space)"
    end

    # Propagate uncertainty via Unscented Transform (deterministic)
    _propagate_ut!(obj, verbose)

    return obj
end

"""
    _fit_cross_fitting_ut!(obj, rng, verbose)

Internal function: Fit the conformal PLR model using K-fold cross-fitting
with unscented transform uncertainty propagation.

Trains conformal models on (K-1) folds and predicts intervals on the held-out
fold. Intervals are aggregated across folds and repetitions (taking median
bounds). Correlation is estimated from full residuals, then uncertainty is
propagated via the deterministic UT.
"""
function _fit_cross_fitting_ut!(
        obj::DoubleMLPLRConformalUT{T},
        rng::AbstractRNG,
        verbose::Int,
    ) where {T}

    n_obs = obj.data.n_obs
    n_folds = obj.n_folds
    n_rep = obj.n_rep

    if verbose > 0
        @info "Fitting DoubleMLPLRConformalUT with cross-fitting"
        @info "  Conformal method: $(obj.conformal_method)"
        @info "  Coverage: $(obj.coverage)"
        @info "  UT: α=$(obj.ut_alpha), β=$(obj.ut_beta), κ=$(obj.ut_kappa), GH=$(obj.n_gh)"
        @info "  Cross-fitting: $(n_folds)-fold, $(n_rep) repetition(s)"
        @info "  Total observations: $n_obs"
    end

    # Generate sample splitting indices
    all_smpls = DoubleML.draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng)

    # Storage for intervals across all repetitions
    l_intervals_all = Array{T, 3}(undef, n_obs, 2, n_rep)
    m_intervals_all = Array{T, 3}(undef, n_obs, 2, n_rep)

    X = DataFrames.DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = obj.data.d
    D_coerced = DoubleML.coerce_target(D, obj.ml_m)

    # Fit models and collect intervals across folds and repetitions
    for r in 1:n_rep
        if verbose > 0 && n_rep > 1
            @info "Processing repetition $r/$n_rep..."
        end

        smpls = all_smpls[r]

        for (fold_idx, (train_idx, test_idx)) in enumerate(smpls)
            if verbose > 0 && n_rep == 1
                @info "  Processing fold $fold_idx/$n_folds ($(length(test_idx)) test observations)..."
            end

            # Get training and test data for this fold
            X_train = X[train_idx, :]
            X_test = X[test_idx, :]
            Y_train = Y[train_idx]
            D_train = D_coerced[train_idx]

            # Fit ml_l on training fold and predict on test fold
            mach_l = MLJ.machine(obj.ml_l, X_train, Y_train)
            MLJ.fit!(mach_l, verbosity = verbose > 1 ? verbose : 0)
            l_intervals_test = MLJ.predict(mach_l, X_test)

            for (i, (lower, upper)) in enumerate(l_intervals_test)
                l_intervals_all[test_idx[i], 1, r] = lower
                l_intervals_all[test_idx[i], 2, r] = upper
            end

            # Fit ml_m on training fold and predict on test fold
            mach_m = MLJ.machine(obj.ml_m, X_train, D_train)
            MLJ.fit!(mach_m, verbosity = verbose > 1 ? verbose : 0)
            m_intervals_test = MLJ.predict(mach_m, X_test)

            for (i, (lower, upper)) in enumerate(m_intervals_test)
                m_intervals_all[test_idx[i], 1, r] = lower
                m_intervals_all[test_idx[i], 2, r] = upper
            end
        end
    end

    # Aggregate intervals across repetitions using median
    if n_rep == 1
        obj.l_intervals .= l_intervals_all[:, :, 1]
        obj.m_intervals .= m_intervals_all[:, :, 1]
    else
        for i in 1:n_obs
            obj.l_intervals[i, 1] = median(l_intervals_all[i, 1, :])
            obj.l_intervals[i, 2] = median(l_intervals_all[i, 2, :])
            obj.m_intervals[i, 1] = median(m_intervals_all[i, 1, :])
            obj.m_intervals[i, 2] = median(m_intervals_all[i, 2, :])
        end
    end

    # Compute midpoints and standard DML results
    @views l_midpoints = (obj.l_intervals[:, 1] .+ obj.l_intervals[:, 2]) ./ 2
    @views m_midpoints = (obj.m_intervals[:, 1] .+ obj.m_intervals[:, 2]) ./ 2

    psi_a, psi_b = DoubleML.compute_score(
        obj.score_obj, Y, D, l_midpoints, m_midpoints,
    )
    obj.standard_dml_coef = DoubleML.dml2_solve(psi_a, psi_b)
    psi = @. (psi_a * obj.standard_dml_coef) + psi_b
    J = mean(psi_a)
    gamma_hat = mean(psi .^ 2)
    sigma2_hat = gamma_hat / (n_obs * J^2)
    obj.standard_dml_se = sqrt(sigma2_hat)

    # Compute residuals and correlation
    l_residuals = Y .- l_midpoints
    m_residuals = D .- m_midpoints
    rho_est = cor(l_residuals, m_residuals)

    z_est = atanh(clamp(rho_est, T(-0.99), T(0.99)))
    z_std = T(1.0) / sqrt(max(1, n_obs - 3))

    obj.lm_correlation = T(rho_est)
    obj.lm_corr_std = T(z_std)

    if verbose > 0
        @info "Cross-fitting complete. Estimated l-m correlation: $(round(obj.lm_correlation, digits = 3))"
    end

    # Propagate uncertainty via Unscented Transform (deterministic)
    _propagate_ut!(obj, verbose)

    if verbose > 0
        ci = confint(obj; level = obj.coverage)
        @info "Fit complete!"
        @info "  Standard DML: θ=$(round(obj.standard_dml_coef, digits = 4)), SE=$(round(obj.standard_dml_se, digits = 4))"
        @info "  UT mean:      θ=$(round(obj.coef, digits = 4)), SE=$(round(obj.se, digits = 4))"
        @info "  $(obj.coverage * 100)% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]"
    end

    return obj
end

# ============================================================================
# StatsAPI Interface
# ============================================================================

"""
    isfitted(obj::DoubleMLPLRConformalUT) -> Bool

Check if the model has been fitted.
"""
StatsAPI.isfitted(obj::DoubleMLPLRConformalUT) = !isnan(obj.coef)

"""
    coef(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the estimated coefficient from the fitted model.
Returns the unscented-transform mean (includes the Jensen correction from
correlated nuisance uncertainty).
"""
function StatsAPI.coef(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.coef]
end

"""
    stderror(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the standard error of the estimated coefficient.
This is the **combined** SE: sqrt(SE_DML² + SE_UT²), where SE_DML is the
standard DML influence-function SE and SE_UT is the conformal-only SE
from the unscented transform.
"""
function StatsAPI.stderror(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.se]
end

"""
    standard_dml_coef(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the standard DML coefficient (point estimate from interval midpoints).
"""
function standard_dml_coef(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.standard_dml_coef]
end

"""
    standard_dml_se(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the standard DML standard error (baseline, without conformal adjustment).
"""
function standard_dml_se(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.standard_dml_se]
end

"""
    ut_mean(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the unscented-transform mean (conformal-only point estimate).
This is the same as `coef(obj)` — included for explicitness.
"""
function ut_mean(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.ut_mean]
end

"""
    ut_se(obj::DoubleMLPLRConformalUT) -> Vector{Float64}

Return the conformal-only standard error from the unscented transform.
This is the additional uncertainty component, before combining with the
standard DML SE. The total reported SE is sqrt(ut_se² + standard_dml_se²).
"""
function ut_se(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [sqrt(obj.ut_var)]
end

"""
    confint(obj::DoubleMLPLRConformalUT; level=0.95)

Compute confidence intervals using normal approximation from the unscented
transform moments.

Note: the UT yields mean and variance, not a full empirical distribution.
Confidence intervals use the normal approximation: coef ± z_(1-α/2) * SE.

# Arguments
- `obj`: Fitted DoubleMLPLRConformalUT model
- `level::Real=0.95`: Confidence level (default 95%)

# Returns
Matrix with two columns: [lower_bound, upper_bound]
"""
function StatsAPI.confint(obj::DoubleMLPLRConformalUT; level::Real = 0.95)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")

    if !(0 < level < 1)
        throw(DomainError(level, "level must be in (0, 1)"))
    end

    alpha = 1 - level
    z = quantile(Normal(), 1 - alpha / 2)
    lower = obj.coef - z * obj.se
    upper = obj.coef + z * obj.se

    return hcat(lower, upper)
end

"""
    nobs(obj::DoubleMLPLRConformalUT) -> Int

Return the number of observations in the data.
"""
StatsAPI.nobs(obj::DoubleMLPLRConformalUT) = obj.data.n_obs

"""
    vcov(obj::DoubleMLPLRConformalUT) -> Matrix{Float64}

Return the variance-covariance matrix (scalar for single coefficient).
"""
function StatsAPI.vcov(obj::DoubleMLPLRConformalUT)
    !isfitted(obj) && error("Model not fitted")
    return fill(obj.se^2, 1, 1)
end

"""
    coeftable(obj::DoubleMLPLRConformalUT; level=0.95)

Return a coefficient table with UT-based statistics.
"""
function StatsAPI.coeftable(obj::DoubleMLPLRConformalUT; level::Real = 0.95)
    cc = coef(obj)[1]
    se = stderror(obj)[1]
    z = cc / se
    p = 2 * ccdf(Normal(), abs(z))
    ci = confint(obj, level = level)

    return StatsBase.CoefTable(
        hcat(cc, se, z, p, ci[1], ci[2]),
        ["Estimate", "Std. Error", "z value", "Pr(>|z|)", "Lower $(level * 100)%", "Upper $(level * 100)%"],
        [string(obj.data.d_col)],
        4,
        3,
    )
end

"""
    conformal_intervals(obj::DoubleMLPLRConformalUT; nuisance::Symbol=:l)

Get the stored conformal prediction intervals for a nuisance model.

# Arguments
- `obj`: Fitted DoubleMLPLRConformalUT model
- `nuisance::Symbol`: Which nuisance model (`:l` for E[Y|X], `:m` for E[D|X])

# Returns
- `Matrix{T}`: Prediction intervals with shape (n_obs, 2)
"""
function conformal_intervals(obj::DoubleMLPLRConformalUT; nuisance::Symbol = :l)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")

    if nuisance == :l
        return obj.l_intervals
    elseif nuisance == :m
        return obj.m_intervals
    else
        throw(ArgumentError("nuisance must be :l or :m. Got: $nuisance"))
    end
end

# ============================================================================
# Display
# ============================================================================

"""
    Base.show(io::IO, obj::DoubleMLPLRConformalUT)

Custom display for DoubleMLPLRConformalUT.
"""
function Base.show(io::IO, obj::DoubleMLPLRConformalUT)
    println(io, "DoubleMLPLRConformalUT (Unscented Transform uncertainty propagation)")
    println(io, "===========================================================")
    println(io, "Conformal method: $(obj.conformal_method)")
    println(io, "Coverage: $(obj.coverage)")
    println(io, "UT parameters: α=$(obj.ut_alpha), β=$(obj.ut_beta), κ=$(obj.ut_kappa)")
    println(io, "GH quadrature: $(obj.n_gh) points")
    if obj.n_folds == 1
        println(io, "Training: Full dataset (no cross-fitting)")
    else
        println(io, "Training: $(obj.n_folds)-fold cross-fitting, $(obj.n_rep) repetition(s)")
    end
    println(io, "Sampling: Deterministic (no MC)")
    println(io, "")

    return if !isfitted(obj)
        println(io, "Status: Not fitted")
    else
        ci = confint(obj; level = obj.coverage)
        println(io, "Results:")
        println(io, "  Coefficient: $(round(obj.coef, digits = 4))")
        println(io, "  Std. Error:  $(round(obj.se, digits = 4)) (combined)")
        println(io, "  $(obj.coverage * 100)% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
        if !isnan(obj.lm_correlation)
            println(io, "  l-m correlation: $(round(obj.lm_correlation, digits = 3)) (±$(round(obj.lm_corr_std, digits = 3)) in z-space)")
        end
        println(io, "")
        println(io, "  Variance decomposition:")
        println(io, "    Standard DML:   θ=$(round(obj.standard_dml_coef, digits = 4)), SE=$(round(obj.standard_dml_se, digits = 4))")
        println(io, "    Conformal (UT): θ=$(round(obj.ut_mean, digits = 4)), SE=$(round(sqrt(obj.ut_var), digits = 4))")
        println(io, "    Difference:     $(round(obj.coef - obj.standard_dml_coef, digits = 4)) (UT - standard DML)")
    end
end

# ============================================================================
# Summary support
# ============================================================================

"""
    _print_learners_table(obj::DoubleMLPLRConformalUT)

Print learners table for DoubleMLPLRConformalUT.
"""
function DoubleML._print_learners_table(obj::DoubleMLPLRConformalUT)
    learners = Pair{String, String}[]
    push!(learners, "Learner ml_l (conformal)" => string(obj.ml_l))
    push!(learners, "Learner ml_m (conformal)" => string(obj.ml_m))
    return DoubleML._print_kv_table(learners)
end

"""
    Base.summary(obj::DoubleMLPLRConformalUT; level=0.95, show_standard_dml=false)

Summary display for DoubleMLPLRConformalUT model.

# Arguments
- `level::Real=0.95`: Confidence level for intervals
- `show_standard_dml::Bool=false`: If true, also show standard DML results for comparison
"""
function Base.summary(obj::DoubleMLPLRConformalUT; level::Real = 0.95, show_standard_dml::Bool = false)
    model_type = typeof(obj).name.name

    println()
    printstyled("═"^20 * " $model_type " * "═"^20; color = :white, bold = true)
    println()

    DoubleML._print_section_header("Data Summary", :blue)
    DoubleML._print_kv_table(
        [
            "Outcome variable" => string(obj.data.y_col),
            "Treatment variable(s)" => string(obj.data.d_col),
            "Covariates" => join(obj.data.x_cols, ", "),
            "No. Observations" => string(obj.data.n_obs),
        ],
    )

    DoubleML._print_section_header("Score & Algorithm", :green)
    DoubleML._print_kv_table(
        [
            "Score function" => string(DoubleML.get_score_name(obj.score_obj)),
            "Conformal method" => string(obj.conformal_method),
            "Coverage" => "$(obj.coverage * 100)%",
            "UT α" => string(obj.ut_alpha),
            "UT β" => string(obj.ut_beta),
            "UT κ" => string(obj.ut_kappa),
            "GH points" => string(obj.n_gh),
        ],
    )

    DoubleML._print_section_header("Machine Learner", :magenta)
    DoubleML._print_learners_table(obj)

    DoubleML._print_section_header("Resampling", :yellow)
    training_mode = if obj.n_folds == 1
        "No cross-fitting"
    else
        "$(obj.n_folds)-fold cross-fitting"
    end
    n_folds_display = obj.n_folds == 1 ? "N/A" : string(obj.n_folds)
    n_rep_display = obj.n_folds == 1 ? "N/A" : string(obj.n_rep)
    DoubleML._print_kv_table(
        [
            "Training mode" => training_mode,
            "No. folds" => n_folds_display,
            "No. repeated sample splits" => n_rep_display,
        ],
    )

    DoubleML._print_section_header("Fit Summary", :cyan)
    if isfitted(obj)
        # Show combined results (coef = UT mean, se = combined)
        println("  Combined Results (UT mean + combined SE):")
        DoubleML._print_coef_table(obj, level)

        if show_standard_dml
            # Show variance decomposition
            println("\n  Variance Decomposition:")
            dml_ci_lower = obj.standard_dml_coef - obj.standard_dml_se * 1.96
            dml_ci_upper = obj.standard_dml_coef + obj.standard_dml_se * 1.96
            println("    Standard DML:   θ=$(round(obj.standard_dml_coef, digits = 4)), SE=$(round(obj.standard_dml_se, digits = 4))")
            println("    95% CI:         [$(round(dml_ci_lower, digits = 4)), $(round(dml_ci_upper, digits = 4))]")
            println("    Conformal (UT): θ=$(round(obj.ut_mean, digits = 4)), SE=$(round(sqrt(obj.ut_var), digits = 4))")
            println("    Difference:     $(round(obj.coef - obj.standard_dml_coef, digits = 4)) (UT mean - standard DML)")
        end
    else
        println("  Model not fitted")
    end

    return println()
end
