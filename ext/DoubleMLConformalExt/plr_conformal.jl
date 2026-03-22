"""
    DoubleMLPLRConformal{T, L, M}

Experimental: Double Machine Learning for Partially Linear Regression 
with Conformal Prediction for robust uncertainty quantification.

This variant trains without cross-fitting and uses Monte Carlo sampling 
from conformal prediction intervals to propagate uncertainty. 
Uses Beta(2,2) marginals with Gaussian copula to account
for correlation between l(X) and m(X) uncertainties.

Note: Conformal predictions rely on sample-splitting internally for 
validity; however, no further sample-splitting is implemented here.

# Type Parameters
- `T<:AbstractFloat`: Numeric type
- `L<:Supervised`: Type of ml_l learner (must be a conformal-wrapped model)
- `M<:Supervised`: Type of ml_m learner (must be a conformal-wrapped model)

# Constructor Options
- `n_mc_samples::Int=100`: Monte Carlo samples for uncertainty propagation
- `score::Symbol=:partialling_out`: Score type (`:partialling_out` only for now)

    # Example
    ```julia
    using DoubleML, ConformalPrediction, MLJ

    # User wraps their own models with conformal prediction
    base_l = @load LinearRegressor pkg=MLJLinearModels
    base_m = @load LinearRegressor pkg=MLJLinearModels

    ml_l = conformal_model(base_l; method=:simple_inductive, coverage=0.95)
    ml_m = conformal_model(base_m; method=:simple_inductive, coverage=0.95)

    # Create conformal PLR model (full data training)
    data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
    model = DoubleMLPLRConformal(data, ml_l, ml_m; n_mc_samples=100)
    fit!(model; rng=StableRNG(42))

    # Results
    coef(model)           # Point estimate (MC median)
    stderror(model)       # Conformal-based SE
    confint(model)        # Conformal prediction interval
    ```
    """
mutable struct DoubleMLPLRConformal{
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
    n_mc_samples::Int

    # Score object
    score_obj::AbstractScore

    # Prediction intervals storage: (n_obs, 2 for lower/upper)
    l_intervals::Matrix{T}
    m_intervals::Matrix{T}

    # Monte Carlo samples for theta: (n_mc_samples,)
    theta_samples::Vector{T}

    # Correlation between l and m nuisance function uncertainties
    lm_correlation::T
    lm_corr_std::T

    # Results
    coef::T
    se::T

    # Standard DML results (for comparison)
    standard_dml_coef::T
    standard_dml_se::T
end

"""
    DoubleMLPLRConformal(data, ml_l, ml_m; kwargs...)

Create a DoubleML PLR model with conformal prediction for uncertainty quantification.
Trains on the FULL dataset (no sample splitting).

Users must wrap their base MLJ models with `conformal_model()` from 
ConformalPrediction.jl before passing them to this constructor.

# Keyword Arguments
- `n_mc_samples::Int=100`: Monte Carlo samples for uncertainty propagation
- `score::Symbol=:partialling_out`: Score type (`:partialling_out` only for now)

# Examples
```julia
using DoubleML, DoubleML.Experimental, ConformalPrediction, MLJ

# Generate data
data = make_plr_CCDDHNR2018(500)

# Wrap models with conformal prediction
base_l = @load LinearRegressor pkg=MLJLinearModels
base_m = @load LinearRegressor pkg=MLJLinearModels

ml_l = conformal_model(base_l; method=:simple_inductive, coverage=0.95)
ml_m = conformal_model(base_m; method=:jackknife_plus, coverage=0.95)

# Create and fit model
model = DoubleMLPLRConformal(data, ml_l, ml_m; n_mc_samples=100)
fit!(model)

# Get results with conformal uncertainty
coef(model)
confint(model)
```
"""
function DoubleMLPLRConformal(
        data::DoubleMLData{T},
        ml_l::L,
        ml_m::M;
        n_mc_samples::Int = 100,
        score::Symbol = :partialling_out
    ) where {T <: AbstractFloat, L <: ConformalPrediction.ConformalModel, M <: ConformalPrediction.ConformalModel}

    # Determine conformal method and coverage from both conformal models
    conformal_method, coverage_l = _get_conformal_info(ml_l)
    _, coverage_m = _get_conformal_info(ml_m)

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

    n_mc_samples > 0 || throw(ArgumentError("n_mc_samples must be positive"))

    # Create score object
    score_obj = if score == :partialling_out
        PartiallingOutScore()
    else
        throw(ArgumentError("Only score=:partialling_out is supported. Got: $score"))
    end

    # Storage allocation
    n_obs = data.n_obs
    l_intervals = zeros(T, n_obs, 2)
    m_intervals = zeros(T, n_obs, 2)
    theta_samples = zeros(T, n_mc_samples)

    return DoubleMLPLRConformal{T, L, M}(
        data, ml_l, ml_m,
        conformal_method, T(coverage), n_mc_samples,
        score_obj,
        l_intervals, m_intervals, theta_samples,
        T(NaN), T(NaN),  # lm_correlation, lm_corr_std (estimated in fit!)
        T(NaN), T(NaN),  # coef, se (computed in fit!)
        T(NaN), T(NaN)  # standard_dml_coef, standard_dml_se (stored in fit!)
    )
end

"""
    _get_conformal_info(model::ConformalPrediction.ConformalInterval)

Extract conformal method and coverage from a conformal model.
Returns (method::Symbol, coverage::Float64).
"""
function _get_conformal_info(model::ConformalPrediction.ConformalInterval)
    type_name = string(typeof(model))

    # Extract method from type name
    method = if occursin("SimpleInductive", type_name)
        :simple_inductive
    elseif occursin("JackknifePlusAbMinMax", type_name)
        :jackknife_plus_ab_minmax
    elseif occursin("JackknifePlusAb", type_name)
        :jackknife_plus_ab
    elseif occursin("JackknifeMinMax", type_name)
        :jackknife_minmax
    elseif occursin("JackknifePlus", type_name)
        :jackknife_plus
    elseif occursin("Jackknife", type_name)
        :jackknife
    elseif occursin("CVMinMax", type_name)
        :cv_minmax
    elseif occursin("CVPlus", type_name)
        :cv_plus
    elseif occursin("CV", type_name)
        :cv
    elseif occursin("Quantile", type_name)
        :quantile
    elseif occursin("Bayes", type_name)
        :bayes
    elseif occursin("Naive", type_name)
        :naive
    else
        :inductive
    end

    # Extract coverage from model field if available
    coverage = if hasfield(typeof(model), :coverage)
        model.coverage
    elseif hasfield(typeof(model), :model) && hasfield(typeof(model.model), :coverage)
        model.model.coverage
    else
        0.95  # default
    end

    return (method, coverage)
end

"""
    fit!(obj::DoubleMLPLRConformal; verbose=0, rng=Random.default_rng())

Fit the conformal PLR model using full dataset (no cross fitting!), with conformal prediction intervals.

Trains nuisance models on the full dataset, then performs uncertainty 
propagation by sampling from conformal prediction intervals.

# Arguments
- `obj`: The DoubleMLPLRConformal model to fit
- `verbose::Int=0`: Verbosity level (0=silent, 1=info, 2=debug)
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for MC sampling
"""
function MLJ.fit!(
        obj::DoubleMLPLRConformal{T};
        verbose::Int = 0,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}

    n_obs = obj.data.n_obs

    X = DataFrames.DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = obj.data.d
    D_coerced = DoubleML.coerce_target(D, obj.ml_m)

    if verbose > 0
        @info "Fitting DoubleMLPLRConformal"
        @info "  Conformal method: $(obj.conformal_method)"
        @info "  Coverage: $(obj.coverage)"
        @info "  MC samples: $(obj.n_mc_samples)"
        @info "  Training on full dataset (n=$n_obs)"
    end

    # Fit ml_l and get prediction intervals
    if verbose > 0
        @info "Fitting ml_l (E[Y|X]) on full dataset..."
    end
    mach_l = MLJ.machine(obj.ml_l, X, Y)
    MLJ.fit!(mach_l, verbosity = verbose)
    l_intervals = MLJ.predict(mach_l, X)  # Vector of (lower, upper) tuples for ALL observations

    # Store intervals
    for (i, (lower, upper)) in enumerate(l_intervals)
        obj.l_intervals[i, 1] = lower
        obj.l_intervals[i, 2] = upper
    end

    # Midpoints for standard DML score
    l_hat = [(lower + upper) / 2 for (lower, upper) in l_intervals]

    # Fit ml_m on FULL dataset and get prediction intervals
    if verbose > 0
        @info "Fitting ml_m (E[D|X]) on full dataset..."
    end
    mach_m = MLJ.machine(obj.ml_m, X, D_coerced)
    MLJ.fit!(mach_m, verbosity = verbose)
    m_intervals = MLJ.predict(mach_m, X)

    for (i, (lower, upper)) in enumerate(m_intervals)
        obj.m_intervals[i, 1] = lower
        obj.m_intervals[i, 2] = upper
    end

    m_hat = [(lower + upper) / 2 for (lower, upper) in m_intervals]

    # Compute standard DML score using midpoints
    psi_a, psi_b = DoubleML.compute_score(
        obj.score_obj, Y, D, l_hat, m_hat
    )

    # Standard coefficient and SE computation (for checking)
    obj.standard_dml_coef = DoubleML.dml2_solve(psi_a, psi_b)
    psi = @. (psi_a * obj.standard_dml_coef) + psi_b
    J = mean(psi_a)
    gamma_hat = mean(psi .^ 2)
    sigma2_hat = gamma_hat / (n_obs * J^2)
    obj.standard_dml_se = sqrt(sigma2_hat)


    # Estimate correlation between l and m nuisance uncertainties
    # Use midpoints of conformal intervals as point estimates
    @views l_midpoints = (obj.l_intervals[:, 1] .+ obj.l_intervals[:, 2]) ./ 2
    @views m_midpoints = (obj.m_intervals[:, 1] .+ obj.m_intervals[:, 2]) ./ 2

    # Compute residuals
    l_residuals = Y .- l_midpoints
    m_residuals = D .- m_midpoints

    # Compute correlation and its standard error (Fisher z-transform)
    rho_est = cor(l_residuals, m_residuals)

    # Fisher z-transform for better normal approximation
    # z = atanh(rho), SE(z) = 1/sqrt(n-3)
    z_est = atanh(clamp(rho_est, -0.99, 0.99))
    z_std = 1.0 / sqrt(max(1, n_obs - 3))

    obj.lm_correlation = T(rho_est)
    obj.lm_corr_std = T(z_std)

    if verbose > 0
        @info "Estimated l-m correlation: $(round(obj.lm_correlation, digits = 3)) (±$(round(obj.lm_corr_std, digits = 3)) in z-space)"
    end

    # Propagate uncerainty via Monte Carlo
    _propagate_conformal_uncertainty!(obj, rng, verbose)

    if verbose > 0
        ci = confint(obj; level = obj.coverage)
        @info "Fit complete!"
        @info "  Standard DML: θ=$(round(obj.standard_dml_coef, digits = 4)), SE=$(round(obj.standard_dml_se, digits = 4))"
        @info "  MC Median:    θ=$(round(obj.coef, digits = 4)), SE=$(round(obj.se, digits = 4))"
        @info "  $(obj.coverage * 100)% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]"
    end

    return obj
end

"""
    _propagate_conformal_uncertainty!(obj, rng, verbose)

Internal function: Perform Monte Carlo uncertainty propagation from conformal intervals.

Uses Beta(2,2) distribution via Gaussian copula to sample from prediction intervals,
accounting for correlation between l(X) and m(X) uncertainties. The correlation is
estimated from residuals and its uncertainty is propagated.

This is a MULTI-THREADED implementation for performance.

# Methodology
1. Sample correlation ρ from its uncertainty distribution
2. Use Gaussian copula with correlation ρ to generate correlated uniforms
3. Transform to Beta(2,2) marginals (more mass near center than uniform)
4. Scale to conformal intervals [lower, upper]

This accounts for the fact that l̂(X) and m̂(X) uncertainties are typically correlated
(e.g., if both overestimate due to unobserved confounders).
"""
function _propagate_conformal_uncertainty!(
        obj::DoubleMLPLRConformal{T},
        rng::AbstractRNG,
        verbose::Int
    ) where {T}

    n_obs = obj.data.n_obs
    Y = obj.data.y
    D = obj.data.d
    n_mc = obj.n_mc_samples

    if verbose > 0
        @info "Propagating conformal uncertainty via Monte Carlo ($(n_mc) samples, multi-threaded)..."
        @info "  Using Beta(2,2) marginals with Gaussian copula correlation"
        @info "  Estimated l-m correlation: $(round(obj.lm_correlation, digits = 3))"
        @info "  Using $(Threads.nthreads()) threads"
    end

    # Precompute Beta(2,2) distribution
    beta_dist = Beta(T(2), T(2))

    # Cache interval widths and bounds as vectors for fast access
    l_lowers = @view obj.l_intervals[:, 1]
    l_uppers = @view obj.l_intervals[:, 2]
    m_lowers = @view obj.m_intervals[:, 1]
    m_uppers = @view obj.m_intervals[:, 2]
    l_widths = l_uppers .- l_lowers
    m_widths = m_uppers .- m_lowers

    # Pre-compute Fisher z-transform values for correlation sampling
    z_est = atanh(clamp(obj.lm_correlation, T(-0.99), T(0.99)))
    z_std = obj.lm_corr_std

    # Progress tracking
    progress_lock = Threads.SpinLock()
    progress_counter = Threads.Atomic{Int}(0)
    progress_interval = max(1, div(n_mc, 10))

    # Thread-local storage to avoid allocations in loop
    rng_threaded_seed = rand(rng, 1:Int(1.0e9))
    rngs = [StableRNG(rng_threaded_seed + mc) for mc in 1:n_mc] # Use unique seeds

    # Parallel MC sampling
    Threads.@threads for mc in 1:n_mc
        # Get unique seed for each mc (guarantees thread safety)
        thread_rng = rngs[mc]

        # Sample correlation from its distribution (Fisher z-transform)
        z_sample = z_est + randn(thread_rng) * z_std
        rho_sample = tanh(z_sample)
        rho_sample = clamp(rho_sample, T(-0.99), T(0.99))

        # Pre-allocate thread-local sample vectors
        l_sample = Vector{T}(undef, n_obs)
        m_sample = Vector{T}(undef, n_obs)

        # Gaussian copula parameters
        rho = rho_sample
        rho_comp = sqrt(max(T(0), T(1) - rho^2))

        # Vectorized sampling from Gaussian copula
        @inbounds for i in 1:n_obs
            # Sample from standard bivariate normal with correlation ρ
            z1 = randn(thread_rng)
            z2 = rho * z1 + rho_comp * randn(thread_rng)

            # Transform to uniform via Φ (CDF of standard normal)
            u1 = cdf(Normal(T(0), T(1)), z1)
            u2 = cdf(Normal(T(0), T(1)), z2)

            # Transform to Beta(2,2) via quantile function
            v1 = quantile(beta_dist, u1)
            v2 = quantile(beta_dist, u2)

            # Scale to conformal intervals
            l_sample[i] = l_lowers[i] + v1 * l_widths[i]
            m_sample[i] = m_lowers[i] + v2 * m_widths[i]
        end

        # Compute scores
        psi_a_mc, psi_b_mc = DoubleML.compute_score(
            obj.score_obj, Y, D, l_sample, m_sample
        )

        # Solve for theta using all observations
        obj.theta_samples[mc] = DoubleML.dml2_solve(psi_a_mc, psi_b_mc)

        # Progress reporting (thread-safe)
        if verbose > 0
            Threads.atomic_add!(progress_counter, 1)
            current = progress_counter[]
            if current % progress_interval == 0
                lock(progress_lock) do
                    @info "  MC Progress: $current/$n_mc"
                end
            end
        end
    end

    # Statistics about theta samples
    theta_mean = mean(obj.theta_samples)
    theta_median = median(obj.theta_samples)
    theta_std = std(obj.theta_samples)

    if verbose > 0
        @info "Theta samples statistics:"
        @info "  Standard DML coef: $(round(obj.standard_dml_coef, digits = 4))"
        @info "  MC mean of theta:  $(round(theta_mean, digits = 4))"
        @info "  MC median of theta: $(round(theta_median, digits = 4))"
        @info "  MC std of theta:   $(round(theta_std, digits = 4))"
        @info "  Difference (MC median - standard DML): $(round(theta_mean - obj.standard_dml_coef, digits = 4))"
    end

    # Store SE
    obj.se = theta_std

    # Use MC median as the point estimate
    obj.coef = theta_median

    return obj
end

# ============================================================================
# StatsAPI Interface
# ============================================================================

"""
    isfitted(obj::DoubleMLPLRConformal) -> Bool

Check if the model has been fitted.
"""
StatsAPI.isfitted(obj::DoubleMLPLRConformal) = !isnan(obj.coef)

"""
    coef(obj::DoubleMLPLRConformal) -> Vector{Float64}

Return the estimated coefficient from the fitted model.
Returns the median of Monte Carlo samples (robust to outliers).
"""
function StatsAPI.coef(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.coef]
end

"""
    stderror(obj::DoubleMLPLRConformal) -> Vector{Float64}

Return the standard error of the estimated coefficient.
This is the standard deviation of Monte Carlo samples (conformal-based SE).
"""
function StatsAPI.stderror(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.se]
end

"""
    standard_dml_coef(obj::DoubleMLPLRConformal) -> Vector{Float64}

Return the standard DML coefficient (point estimate from midpoints).
This is the coefficient computed using the midpoint of conformal intervals,
without Monte Carlo uncertainty propagation.

For the conformal-adjusted coefficient (MC median), use `coef(obj)`.
"""
function standard_dml_coef(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.standard_dml_coef]
end

"""
    standard_dml_se(obj::DoubleMLPLRConformal) -> Vector{Float64}

Return the standard DML standard error.
This is the standard error computed using the midpoint of conformal intervals,
without Monte Carlo uncertainty propagation.

For the conformal-adjusted SE, use `stderror(obj)`.
"""
function standard_dml_se(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return [obj.standard_dml_se]
end

"""
    confint(obj::DoubleMLPLRConformal; level=0.95)

Compute confidence intervals using conformal uncertainty quantification.

Uses quantiles of the Monte Carlo theta samples to construct the interval,
respecting the actual empirical distribution of the uncertainty propagation.

# Arguments
- `obj`: Fitted DoubleMLPLRConformal model
- `level::Real=0.95`: Confidence level (default 95%)

# Returns
Matrix with two columns: [lower_bound, upper_bound]
"""
function StatsAPI.confint(obj::DoubleMLPLRConformal; level::Real = 0.95)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")

    if !(0 < level < 1)
        throw(DomainError(level, "level must be in (0, 1)"))
    end

    # Use quantiles of theta_samples for proper conformal prediction interval
    # This respects the actual distribution from MC sampling
    alpha = 1 - level
    lower = quantile(obj.theta_samples, alpha / 2)
    upper = quantile(obj.theta_samples, 1 - alpha / 2)

    return hcat(lower, upper)
end

"""
    nobs(obj::DoubleMLPLRConformal) -> Int

Return the number of observations in the data.
"""
StatsAPI.nobs(obj::DoubleMLPLRConformal) = obj.data.n_obs

"""
    vcov(obj::DoubleMLPLRConformal) -> Matrix{Float64}

Return the variance-covariance matrix (scalar for single coefficient).
"""
function StatsAPI.vcov(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted")
    return fill(obj.se^2, 1, 1)
end

"""
    coeftable(obj::DoubleMLPLRConformal; level=0.95)

Return a coefficient table with conformal-based statistics.
"""
function StatsAPI.coeftable(obj::DoubleMLPLRConformal; level::Real = 0.95)
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
        3
    )
end

"""
    theta_samples(obj::DoubleMLPLRConformal)

Get the stored Monte Carlo samples of theta.

# Returns
- `Vector{T}`: MC samples of theta (length = n_mc_samples)
"""
function theta_samples(obj::DoubleMLPLRConformal)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")
    return obj.theta_samples
end

"""
    conformal_intervals(obj::DoubleMLPLRConformal; nuisance::Symbol=:l)

Get the stored conformal prediction intervals for a nuisance model.

# Arguments
- `obj`: Fitted DoubleMLPLRConformal model
- `nuisance::Symbol`: Which nuisance model (`:l` for E[Y|X], `:m` for E[D|X])

# Returns
- `Matrix{T}`: Prediction intervals with shape (n_obs, 2)
  where column 1 is lower bound and column 2 is upper bound

# Example
```julia
# Get intervals for l(X) = E[Y|X]
l_intervals = conformal_intervals(model; nuisance=:l)

# Access lower and upper bounds for first observation
lower = l_intervals[1, 1]
upper = l_intervals[1, 2]
```
"""
function conformal_intervals(obj::DoubleMLPLRConformal; nuisance::Symbol = :l)
    !isfitted(obj) && error("Model not fitted. Run fit!(model) first.")

    if nuisance == :l
        return obj.l_intervals
    elseif nuisance == :m
        return obj.m_intervals
    else
        throw(ArgumentError("nuisance must be :l or :m. Got: $nuisance"))
    end
end

"""
    Base.show(io::IO, obj::DoubleMLPLRConformal)

Custom display for DoubleMLPLRConformal.
"""
function Base.show(io::IO, obj::DoubleMLPLRConformal)
    println(io, "DoubleMLPLRConformal (Experimental)")
    println(io, "===========================================================")
    println(io, "Conformal method: $(obj.conformal_method)")
    println(io, "Coverage: $(obj.coverage)")
    println(io, "MC samples: $(obj.n_mc_samples)")
    println(io, "Training: (no cross-fitting)")
    println(io, "Sampling: Beta(2,2) marginals with Gaussian copula")
    println(io, "")

    return if !isfitted(obj)
        println(io, "Status: Not fitted")
    else
        ci = confint(obj; level = obj.coverage)
        println(io, "Results:")
        println(io, "  Coefficient: $(round(obj.coef, digits = 4))")
        println(io, "  Std. Error: $(round(obj.se, digits = 4))")
        println(io, "  $(obj.coverage * 100)% CI: [$(round(ci[1], digits = 4)), $(round(ci[2], digits = 4))]")
        if !isnan(obj.lm_correlation)
            println(io, "  l-m correlation: $(round(obj.lm_correlation, digits = 3)) (±$(round(obj.lm_corr_std, digits = 3)) in z-space)")
        end
    end
end

# ============================================================================
# Summary support
# ============================================================================

"""
    _print_learners_table(obj::DoubleMLPLRConformal)

Print learners table for DoubleMLPLRConformal.
"""
function DoubleML._print_learners_table(obj::DoubleMLPLRConformal)
    learners = Pair{String, String}[]
    push!(learners, "Learner ml_l (conformal)" => string(obj.ml_l))
    push!(learners, "Learner ml_m (conformal)" => string(obj.ml_m))
    return DoubleML._print_kv_table(learners)
end

"""
    Base.summary(obj::DoubleMLPLRConformal; level=0.95, show_standard_dml=false)

Summary display for DoubleMLPLRConformal model.

# Arguments
- `level::Real=0.95`: Confidence level for intervals
- `show_standard_dml::Bool=false`: If true, also show standard DML results for comparison
"""
function Base.summary(obj::DoubleMLPLRConformal; level::Real = 0.95, show_standard_dml::Bool = false)
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
        ]
    )

    DoubleML._print_section_header("Score & Algorithm", :green)
    DoubleML._print_kv_table(
        [
            "Score function" => string(DoubleML.get_score_name(obj.score_obj)),
            "Conformal method" => string(obj.conformal_method),
            "Coverage" => "$(obj.coverage * 100)%",
            "MC samples" => string(obj.n_mc_samples),
        ]
    )

    DoubleML._print_section_header("Machine Learner", :magenta)
    DoubleML._print_learners_table(obj)

    DoubleML._print_section_header("Resampling", :yellow)
    DoubleML._print_kv_table(
        [
            "Training mode" => "No cross-fitting",
            "No. folds" => "N/A",
            "No. repeated sample splits" => "N/A",
        ]
    )

    DoubleML._print_section_header("Fit Summary", :cyan)
    if isfitted(obj)
        # Show conformal results (MC median)
        println("  Conformal Results (MC median with conformal SE):")
        DoubleML._print_coef_table(obj, level)

        if show_standard_dml
            # Show standard DML comparison
            println("\n  Standard DML Results (for comparison):")
            dml_ci_lower = obj.standard_dml_coef - obj.standard_dml_se * 1.96
            dml_ci_upper = obj.standard_dml_coef + obj.standard_dml_se * 1.96
            println("    Coefficient: $(round(obj.standard_dml_coef, digits = 4))")
            println("    Std. Error:  $(round(obj.standard_dml_se, digits = 4))")
            println("    95% CI:      [$(round(dml_ci_lower, digits = 4)), $(round(dml_ci_upper, digits = 4))]")
            println("    Difference:  $(round(obj.coef - obj.standard_dml_coef, digits = 4)) (MC median - DML)")
        end
    else
        println("  Model not fitted")
    end

    return println()
end
