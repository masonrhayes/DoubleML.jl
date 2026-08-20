"""
DoubleMLLPLR: Logistic Partially Linear Regression model.

Implements Double/Debiased Machine Learning for binary outcomes:
``E[Y | D, X] = expit{\\beta_{0} D + r_{0}(X)}``
"""

"""
    DoubleMLLPLR(data, ml_M, ml_t, ml_m; ml_a=nothing, score=:nuisance_space,
                 n_folds=5, n_folds_inner=5, n_rep=1, n_folds_tune=0)

Create a DoubleML LPLR model.

# Arguments
- `data::DoubleMLData{T}`: Data container (outcome must be binary 0/1)
- `ml_M`: Probabilistic classifier for M(D,X) = P(Y=1 | D, X)
- `ml_t`: Regressor for t(X) = E[logit(M(D,X)) | X]
- `ml_m`: Model for m(X) = E[D | X, Y=0] (nuisance_space) or E[D | X] (instrument)
- `ml_a=nothing`: Optional model for a(X) = E[D | X] (defaults to ml_m)
- `score::Symbol=:nuisance_space`: Score type (:nuisance_space or :instrument)
- `n_folds::Int=5`: Number of outer cross-fitting folds
- `n_folds_inner::Int=5`: Number of inner folds used to construct leakage-free generated targets
- `n_rep::Int=1`: Number of sample splitting repetitions
- `n_folds_tune::Int=0`: Folds for tuning (0 = full sample)

# Returns
- `DoubleMLLPLR{T, typeof(ml_M), typeof(ml_t), typeof(ml_m), typeof(ml_a)}`

# Examples
```julia
data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
ml_M = @load RandomForestClassifier pkg=DecisionTree
ml_t = @load RandomForestRegressor pkg=DecisionTree
ml_m = @load RandomForestRegressor pkg=DecisionTree

model = DoubleMLLPLR(data, ml_M, ml_t, ml_m)
fit!(model)
```

# References
- Liu et al. (2021): https://doi.org/10.1093/ectj/utab019
"""
function DoubleMLLPLR(
        data::DoubleMLData{T}, ml_M::M, ml_t::Tt, ml_m::Mm;
        ml_a::Ma = nothing,
        n_folds::Int = 5,
        n_folds_inner::Int = 5,
        n_rep::Int = 1,
        score::Symbol = :nuisance_space,
        n_folds_tune::Int = 0
    ) where {
        T <: AbstractFloat, M <: Supervised, Tt <: Supervised,
        Mm <: Supervised, Ma,
    }

    # Validate data - Y must be binary {0, 1}
    _validate_binary_outcome(data.y)

    binary_treatment = _is_binary_treatment(data.d)
    _validate_lplr_treatment_learner(ml_m, "ml_m", binary_treatment)
    ml_a !== nothing && _validate_lplr_treatment_learner(
        ml_a, "ml_a", binary_treatment
    )

    # Set up score object
    score_obj = if score == :nuisance_space
        NuisanceSpaceScore()
    elseif score == :instrument
        InstrumentScore()
    else
        throw(ArgumentError("Score must be :nuisance_space or :instrument, got: $score"))
    end

    # Validate learner types
    _validate_lplr_learners(ml_M, ml_t, ml_m, ml_a, score)

    # Use ml_m as default for ml_a if not provided
    ml_a_actual = if ml_a === nothing
        ml_m
    else
        ml_a
    end

    # Get concrete type for ml_a_actual
    Ma_actual = typeof(ml_a_actual)

    _validate_fold_args(n_folds, n_rep, n_folds_tune)
    n_folds >= 2 || throw(ArgumentError("n_folds must be >= 2 for LPLR cross-fitting"))
    n_folds_inner >= 2 || throw(ArgumentError("n_folds_inner must be >= 2"))

    n_obs = data.n_obs

    return DoubleMLLPLR{T, M, Tt, Mm, Ma_actual}(
        data, ml_M, ml_t, ml_m, ml_a_actual, n_folds, n_folds_inner, n_rep,
        score_obj, n_folds_tune, T(NaN), T(NaN), zeros(T, n_rep), zeros(T, n_rep), T(NaN),
        zeros(T, n_obs, n_rep), zeros(T, n_obs, n_rep), zeros(T, n_obs, n_rep),
        false, zeros(T, 0, 0, 0), nothing, 0,
        MLJ.Machine[], MLJ.Machine[], MLJ.Machine[], MLJ.Machine[],
        (;)
    )
end

"""
    _is_binary_treatment(d::AbstractVector) -> Bool

Check if treatment variable is binary with values 0 and 1.
"""
function _is_binary_treatment(d::AbstractVector)
    unique_d = sort!(unique(d))
    return length(unique_d) == 2 && unique_d[1] == 0 && unique_d[2] == 1
end

"""
    _validate_binary_outcome(y)

Validate that outcome is binary with values 0 and 1.
"""
function _validate_binary_outcome(y::AbstractVector)
    unique_y = sort!(unique(y))
    if !(length(unique_y) == 2 && unique_y[1] == 0 && unique_y[2] == 1)
        throw(
            ArgumentError(
                "Outcome variable must be binary with values 0 and 1. " *
                    "Got: $unique_y"
            )
        )
    end
    return nothing
end

function _validate_lplr_treatment_learner(learner, name::String, binary::Bool)
    if binary
        if !(learner isa MLJBase.Probabilistic)
            @warn "Treatment is binary {0,1} but $name is Deterministic " *
                "($(typeof(learner))). Consider using a probabilistic classifier."
        end
    elseif !(learner isa MLJBase.Deterministic)
        throw(
            ArgumentError(
                "Treatment is continuous but $name is not Deterministic. Got: " *
                    "$(typeof(learner)). Use a regressor for continuous treatment."
            )
        )
    end
    return nothing
end

"""
    _validate_lplr_learners(ml_M, ml_t, ml_m, ml_a, score)

Validate learner types for LPLR model.
"""
function _validate_lplr_learners(
        ml_M::Supervised, ml_t::Supervised, ml_m::Supervised,
        ml_a::Union{Supervised, Nothing}, score::Symbol
    )
    # ml_M must be Probabilistic (classifier)
    if !(ml_M isa MLJBase.Probabilistic)
        throw(
            ArgumentError(
                "ml_M must be a probabilistic classifier for P(Y=1 | D, X). " *
                    "Got: $(typeof(ml_M))"
            )
        )
    end

    # ml_t must be Deterministic (regressor)
    if !(ml_t isa MLJBase.Deterministic)
        throw(
            ArgumentError(
                "ml_t must be a deterministic regressor for E[logit(M) | X]. " *
                    "Got: $(typeof(ml_t))"
            )
        )
    end

    # ml_m can be either (depends on treatment type)

    # MLJ does not expose weighted fitting for several otherwise useful
    # regressors. Preserve the existing unweighted fallback, but make the
    # resulting approximation explicit. ml_a is not weighted in this estimator.
    if score == :instrument && !MLJBase.supports_weights(ml_m)
        @warn "ml_m ($(typeof(ml_m))) does not support sample weights. " *
            "The instrument nuisance regression will be fitted without weights."
    end

    return nothing
end

"""
    fit!(obj::DoubleMLLPLR; verbose=0, force=false, rng=Random.default_rng())

Fit the DoubleML LPLR model using double cross-fitting with bracket-based root finding.

# Arguments
- `verbose::Int=0`: Verbosity level
- `force::Bool=false`: Force refit if already fitted
- `rng::AbstractRNG=Random.default_rng()`: Random number generator for sample splitting
"""
function MLJ.fit!(
        obj::DoubleMLLPLR{T}; verbose::Int = 0, force::Bool = false,
        rng::AbstractRNG = Random.default_rng()
    ) where {T}
    if isfitted(obj)
        !force && (@warn "Model already fitted. Use force=true to refit."; return obj)
        @warn "Forcing refit."
    end

    _reset_fit_state!(obj)
    obj.fitted_learners_M = MLJ.Machine[]
    obj.fitted_learners_t = MLJ.Machine[]
    obj.fitted_learners_m = MLJ.Machine[]
    obj.fitted_learners_a = MLJ.Machine[]

    if verbose > 0
        score_name = obj.score_obj isa NuisanceSpaceScore ? "nuisance_space" : "instrument"
        n_folds_tune = obj.n_folds_tune
        any_tuned = _is_tuned_model(obj.ml_M) || _is_tuned_model(obj.ml_t) ||
            _is_tuned_model(obj.ml_m) || _is_tuned_model(obj.ml_a)
        @info "Fitting DoubleMLLPLR with $(obj.n_folds) outer folds, " *
            "$(obj.n_folds_inner) inner folds, $(obj.n_rep) repetition(s)..."
        @info "Score function: $score_name"
        if any_tuned
            tune_info = n_folds_tune > 0 ? "$(n_folds_tune) tuning folds" : "full sample tuning"
            @info "Tuning: $tune_info"
        end
    end

    n_obs = obj.data.n_obs

    # Every outer training sample receives its own inner partition. This is
    # essential because the inner predictions become training targets for ml_t.
    all_smpls_outer = draw_sample_splitting(n_obs, obj.n_folds, obj.n_rep; rng = rng)
    all_smpls_inner = draw_double_sample_splitting(
        all_smpls_outer, obj.n_folds_inner; rng = rng
    )

    X = DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D_coerced_m = coerce_target(obj.data.d, obj.ml_m)
    D_coerced_a = coerce_target(obj.data.d, obj.ml_a)
    D_numeric = T.(to_numeric(obj.data.d))
    X_with_D = hcat(DataFrame(d = D_numeric), X; makeunique = true)

    # Handle tuning - full sample tuning (n_folds_tune == 0)
    if obj.n_folds_tune == 0
        any_tuned_full = _is_tuned_model(obj.ml_M) || _is_tuned_model(obj.ml_t) ||
            _is_tuned_model(obj.ml_m) || _is_tuned_model(obj.ml_a)
        if verbose > 0 && any_tuned_full
            @info "Tuning learners on full sample..."
        end
        ml_M_best, _ = _get_best_model(
            obj.ml_M, X_with_D, Y, verbose; model_name = "ml_M"
        )
        t_target = if _is_tuned_model(obj.ml_t)
            tuning_smpls = first(
                draw_sample_splitting(n_obs, obj.n_folds_inner, 1; rng = rng)
            )
            M_tune = _cross_fitted_M_predictions(
                ml_M_best, X_with_D, Y, tuning_smpls
            )
            _logit_targets(M_tune)
        else
            Y
        end
        ml_t_best, _ = _get_best_model(
            obj.ml_t, X, t_target, verbose; model_name = "ml_t"
        )
        m_train = obj.score_obj isa NuisanceSpaceScore ? findall(iszero, Y) : eachindex(Y)
        ml_m_best, _ = _get_best_model(
            obj.ml_m, X[m_train, :], D_coerced_m[m_train], verbose; model_name = "ml_m"
        )
        ml_a_best, _ = _get_best_model(
            obj.ml_a, X, D_coerced_a, verbose; model_name = "ml_a"
        )
    else
        ml_M_best = obj.ml_M
        ml_t_best = obj.ml_t
        ml_m_best = obj.ml_m
        ml_a_best = obj.ml_a
    end

    # Storage for per-repetition results
    all_psi = Vector{Vector{T}}(undef, obj.n_rep)
    all_psi_a = Vector{Vector{T}}(undef, obj.n_rep)

    for r in 1:obj.n_rep
        smpls_outer = all_smpls_outer[r]
        smpls_inner = all_smpls_inner[r]

        # Handle per-repetition tuning (n_folds_tune > 0)
        if obj.n_folds_tune > 0
            any_tuned_rep = _is_tuned_model(obj.ml_M) || _is_tuned_model(obj.ml_t) ||
                _is_tuned_model(obj.ml_m) || _is_tuned_model(obj.ml_a)
            if verbose > 0 && any_tuned_rep
                @info "Tuning learners for repetition $r/$(obj.n_rep)..."
            end
            train_idx_tune = smpls_outer[1][1]
            ml_M_rep, _ = _get_best_model(
                obj.ml_M, X_with_D[train_idx_tune, :], Y[train_idx_tune], verbose;
                model_name = "ml_M", context = "for repetition $r"
            )
            t_target = if _is_tuned_model(obj.ml_t)
                tuning_smpls_local = first(
                    draw_sample_splitting(
                        length(train_idx_tune), obj.n_folds_inner, 1; rng = rng
                    )
                )
                M_tune = _cross_fitted_M_predictions(
                    ml_M_rep, X_with_D[train_idx_tune, :], Y[train_idx_tune],
                    tuning_smpls_local
                )
                _logit_targets(M_tune)
            else
                Y[train_idx_tune]
            end
            ml_t_rep, _ = _get_best_model(
                obj.ml_t, X[train_idx_tune, :], t_target, verbose;
                model_name = "ml_t", context = "for repetition $r"
            )
            m_idx_tune = if obj.score_obj isa NuisanceSpaceScore
                train_idx_tune[Y[train_idx_tune] .== 0]
            else
                train_idx_tune
            end
            ml_m_rep, _ = _get_best_model(
                obj.ml_m, X[m_idx_tune, :], D_coerced_m[m_idx_tune], verbose;
                model_name = "ml_m", context = "for repetition $r"
            )
            ml_a_rep, _ = _get_best_model(
                obj.ml_a, X[train_idx_tune, :], D_coerced_a[train_idx_tune], verbose;
                model_name = "ml_a", context = "for repetition $r"
            )
        else
            ml_M_rep = ml_M_best
            ml_t_rep = ml_t_best
            ml_m_rep = ml_m_best
            ml_a_rep = ml_a_best
        end

        if verbose > 1
            @info "Processing repetition $r/$(obj.n_rep)..."
        end

        # Stage 1: Generate fold-specific M predictions inside each outer
        # training sample. Outer-test outcomes are never visible here.
        M_hat_inner = _fit_M_inner(
            obj, ml_M_rep, X_with_D, Y, smpls_outer, smpls_inner, verbose
        )

        # Stage 2: a(X) is only outer-cross-fitted. Nested a fits in the Python
        # estimator are needed for its preliminary beta, which is intentionally
        # absent from this reparameterized estimator.
        a_hat_outer = _fit_a_outer(
            obj, ml_a_rep, X, D_coerced_a, smpls_outer, verbose
        )

        # Stage 3: Fit t(X) against the fold-specific generated log-odds.
        t_hat_outer = _fit_t_outer(
            obj, ml_t_rep, X, M_hat_inner, smpls_outer, verbose
        )

        # Stage 4: For the instrument score, use the same nested M predictions
        # to construct weights on each outer training sample.
        m_hat_outer = _fit_m_outer(
            obj, ml_m_rep, X, Y, D_coerced_m, M_hat_inner, smpls_outer, verbose
        )

        # Stage 5: Compute score elements for each outer fold (dynamic r_hat computation)
        score_elements_rep = _compute_all_score_elements(
            obj.score_obj, Y, D_numeric, smpls_outer, t_hat_outer, a_hat_outer, m_hat_outer
        )

        # Stage 6: Root-finding for this repetition's coefficient (bracket-based)
        obj.all_coef[r] = _solve_score_equation_bracket(obj.score_obj, score_elements_rep)

        # Compute score and derivative at this repetition's coefficient
        psi_rep = Vector{T}(undef, n_obs)
        psi_a_rep = Vector{T}(undef, n_obs)

        for (elements, (_, test_idx)) in zip(score_elements_rep, smpls_outer)
            psi_rep[test_idx] .= compute_score(
                obj.score_obj, obj.all_coef[r], elements
            )
            psi_a_rep[test_idx] .= compute_score_deriv(
                obj.score_obj, obj.all_coef[r], elements
            )
        end

        all_psi[r] = psi_rep
        all_psi_a[r] = psi_a_rep

        # Compute SE for this repetition
        obj.all_se[r] = _compute_se(psi_rep, psi_a_rep)
    end

    # Aggregate across repetitions using median-based aggregation
    obj.coef, obj.se = _aggregate_coefs_and_ses(obj.all_coef, obj.all_se)

    # Store all psi components as matrices (n_obs × n_rep)
    # For LPLR with nonlinear scores:
    # - all_psi stores ψ(θ_r) (score at per-rep coefficient)
    # - all_psi_a stores dψ/dθ|_θ_r (derivative at per-rep coefficient)
    # - all_psi_b is computed as ψ(θ_r) - dψ/dθ * θ_r (offset for linearization)
    obj.all_psi = hcat(all_psi...)
    obj.all_psi_a = hcat(all_psi_a...)

    # Compute all_psi_b for bootstrap linearization
    for r in 1:obj.n_rep
        obj.all_psi_b[:, r] = @. obj.all_psi[:, r] - obj.all_psi_a[:, r] * obj.all_coef[r]
    end

    if verbose > 0
        @info "Done! Coefficient: $(round(obj.coef, digits = 4)), SE: $(round(obj.se, digits = 4))"
    end

    return obj
end

"""
    _fit_M_inner(obj, ml_M, X_with_D, Y, smpls_outer, smpls_inner, verbose)

Cross-fit `M(D,X)` separately within each outer training sample. Returns one
full-length prediction vector per outer fold; only that fold's outer-training
positions are populated.
"""
function _fit_M_inner(
        obj::DoubleMLLPLR{T}, ml_M, X_with_D::DataFrame, Y::Vector{T},
        smpls_outer::Vector, smpls_inner::Vector, verbose::Int
    ) where {T}
    n_obs = length(Y)
    n_outer = length(smpls_outer)
    length(smpls_inner) == n_outer || throw(
        DimensionMismatch("one inner partition is required per outer fold")
    )
    M_hat_inner = [fill(T(NaN), n_obs) for _ in 1:n_outer]
    Y_cat = coerce_target(Y, ml_M)

    for (outer_fold, ((outer_train, _), inner_folds)) in enumerate(
            zip(smpls_outer, smpls_inner)
        )
        for (inner_fold, (train_idx, test_idx)) in enumerate(inner_folds)
            if verbose > 2
                @info "  Fitting ml_M on outer fold $outer_fold/$n_outer, " *
                    "inner fold $inner_fold/$(length(inner_folds))..."
            end
            mach_M = machine(ml_M, X_with_D[train_idx, :], Y_cat[train_idx])
            MLJ.fit!(mach_M, verbosity = 0)
            push!(obj.fitted_learners_M, mach_M)

            predictions, _ = predict_nuisance(
                mach_M, X_with_D[test_idx, :], "ml_M inner"
            )
            _check_lplr_predictions(predictions, "ml_M inner")
            M_hat_inner[outer_fold][test_idx] .= predictions
        end
        all(isfinite, @view(M_hat_inner[outer_fold][outer_train])) || error(
            "Nested ml_M predictions do not cover outer training fold $outer_fold"
        )
    end

    return M_hat_inner
end

function _cross_fitted_M_predictions(ml_M, X_with_D, Y::AbstractVector{T}, smpls) where {T}
    predictions = fill(T(NaN), length(Y))
    Y_cat = coerce_target(Y, ml_M)
    for (train_idx, test_idx) in smpls
        mach = machine(ml_M, X_with_D[train_idx, :], Y_cat[train_idx])
        MLJ.fit!(mach, verbosity = 0)
        fold_predictions, _ = predict_nuisance(
            mach, X_with_D[test_idx, :], "ml_M tuning"
        )
        _check_lplr_predictions(fold_predictions, "ml_M tuning")
        predictions[test_idx] .= fold_predictions
    end
    _check_lplr_predictions(predictions, "ml_M tuning")
    return predictions
end

function _logit_targets(predictions::AbstractVector{T}) where {T <: AbstractFloat}
    bound = max(T(1.0e-8), eps(T))
    return logit.(clamp.(predictions, bound, one(T) - bound))
end

function _check_lplr_predictions(predictions, name::String)
    all(isfinite, predictions) || throw(
        ArgumentError("Predictions from $name must be finite")
    )
    return nothing
end

function _check_lplr_probability_predictions(predictions, name::String)
    _check_lplr_predictions(predictions, name)
    all(p -> 0 <= p <= 1, predictions) || throw(
        ArgumentError("Probability predictions from $name must lie in [0, 1]")
    )
    all(p -> iszero(p) || p == 1, predictions) && throw(
        ArgumentError(
            "Predictions from probabilistic learner $name are all hard labels; " *
                "probabilities are required"
        )
    )
    return nothing
end

"""
    _fit_a_outer(obj, ml_a, X, D, smpls_outer, verbose)

Fit ``a(X) = E[D | X]`` on outer cross-fitting folds.

Returns a_hat_outer for dynamic r_hat computation in root-finding.
"""
function _fit_a_outer(
        obj::DoubleMLLPLR{T}, ml_a, X::DataFrame, D::AbstractVector,
        smpls_outer::Vector, verbose::Int
    ) where {T}
    n_obs = length(D)
    n_outer = length(smpls_outer)
    a_hat_outer = zeros(T, n_obs)

    for (i, (train_idx, test_idx)) in enumerate(smpls_outer)
        if verbose > 2
            @info "  Fitting ml_a on outer fold $i/$n_outer..."
        end

        # Fit on outer training set
        X_train = X[train_idx, :]
        D_train = D[train_idx]

        mach_a = machine(ml_a, X_train, D_train)
        MLJ.fit!(mach_a, verbosity = 0)
        push!(obj.fitted_learners_a, mach_a)

        # Predict on outer test set. Probabilistic treatment learners must
        # contribute probabilities rather than distribution objects.
        predictions, _ = predict_nuisance(mach_a, X[test_idx, :], "ml_a")
        if ml_a isa MLJBase.Probabilistic
            _check_lplr_probability_predictions(predictions, "ml_a")
        else
            _check_lplr_predictions(predictions, "ml_a")
        end
        a_hat_outer[test_idx] .= predictions
    end

    return a_hat_outer
end

"""
    _fit_t_outer(obj, ml_t, X, M_hat_inner, smpls_outer, verbose)

Fit `t(X) = E[logit(M) | X]` on each outer training sample using its own
nested, out-of-inner-fold `M` predictions as targets.
"""
function _fit_t_outer(
        obj::DoubleMLLPLR{T}, ml_t, X::DataFrame, M_hat_inner::Vector{Vector{T}},
        smpls_outer::Vector, verbose::Int
    ) where {T}
    n_obs = size(X, 1)
    n_outer = length(smpls_outer)
    t_hat_outer = zeros(T, n_obs)

    for (i, (train_idx, test_idx)) in enumerate(smpls_outer)
        verbose > 2 && @info "  Fitting ml_t on outer fold $i/$n_outer..."
        W_train = _logit_targets(M_hat_inner[i][train_idx])
        mach_t = machine(ml_t, X[train_idx, :], W_train)
        MLJ.fit!(mach_t, verbosity = 0)
        push!(obj.fitted_learners_t, mach_t)

        predictions = MLJ.predict(mach_t, X[test_idx, :])
        _check_lplr_predictions(predictions, "ml_t")
        t_hat_outer[test_idx] .= predictions
    end

    return t_hat_outer
end

"""
    _fit_m_outer(obj, ml_m, X, Y, D, M_hat_inner, smpls_outer, verbose)

Fit ``m(X)`` on outer cross-fitting folds using tuned model ml_m.

For nuisance_space: fit only on Y=0 observations
For instrument: fit on all observations with sample weights ``M*(1-M)`` if supported
"""
function _fit_m_outer(
        obj::DoubleMLLPLR{T}, ml_m, X::DataFrame, Y::Vector{T}, D::AbstractVector,
        M_hat_inner::Vector{Vector{T}}, smpls_outer::Vector, verbose::Int
    ) where {T}
    n_obs = length(Y)
    n_outer = length(smpls_outer)
    m_hat_outer = zeros(T, n_obs)

    for (i, (train_idx, test_idx)) in enumerate(smpls_outer)
        if verbose > 2
            @info "  Fitting ml_m on outer fold $i/$n_outer..."
        end

        # Filter training data based on score type
        if obj.score_obj isa NuisanceSpaceScore
            # Use only Y=0 observations
            train_filtered = train_idx[Y[train_idx] .== 0]
            isempty(train_filtered) && throw(
                ArgumentError("No Y=0 observations in outer fold $i training sample")
            )
            X_train = X[train_filtered, :]
            D_train = D[train_filtered]
            mach_m = machine(ml_m, X_train, D_train)
        else
            # Instrument score: use all observations with sample weights
            X_train = X[train_idx, :]
            D_train = D[train_idx]

            # Check if model supports weights
            if MLJBase.supports_weights(ml_m)
                # Get M predictions for training indices and clamp to avoid 0 weights
                M_train = M_hat_inner[i][train_idx]
                bound = max(T(1.0e-8), eps(T))
                M_clamped = clamp.(M_train, bound, one(T) - bound)
                weights = M_clamped .* (one(T) .- M_clamped)

                mach_m = machine(ml_m, X_train, D_train, weights)
            else
                # Model doesn't support weights - fit without weights
                # This is suboptimal but allows flexibility in model choice
                mach_m = machine(ml_m, X_train, D_train)
            end
        end

        MLJ.fit!(mach_m, verbosity = 0)
        push!(obj.fitted_learners_m, mach_m)

        predictions, _ = predict_nuisance(mach_m, X[test_idx, :], "ml_m")
        if ml_m isa MLJBase.Probabilistic
            _check_lplr_probability_predictions(predictions, "ml_m")
        else
            _check_lplr_predictions(predictions, "ml_m")
        end
        m_hat_outer[test_idx] .= predictions
    end

    return m_hat_outer
end

"""
    _compute_all_score_elements(score_obj, Y, D, smpls_outer, t_hat, a_hat, m_hat)

Compute score elements for each outer fold using dynamic r_hat computation.
"""
function _compute_all_score_elements(
        score_obj::AbstractScore, Y::Vector{T}, D::AbstractVector,
        smpls_outer::Vector, t_hat::Vector{T}, a_hat::Vector{T}, m_hat::Vector{T}
    ) where {T}
    score_elements = NamedTuple[]

    for (_, test_idx) in smpls_outer
        Y_test = Y[test_idx]
        D_test = D[test_idx]
        t_hat_test = t_hat[test_idx]
        a_hat_test = a_hat[test_idx]
        m_hat_test = m_hat[test_idx]

        elements = compute_score_elements(score_obj, Y_test, D_test, t_hat_test, a_hat_test, m_hat_test)
        push!(score_elements, elements)
    end

    return score_elements
end

"""
    _find_bracket(objective, center; initial_step=0.5, max_attempts=30,
                  max_abs=1.0e6)

Find a finite sign-changing bracket by geometric expansion around `center`.
A degenerate `(x, x)` bracket denotes an exact endpoint root. Throws rather
than returning an interval that has not been verified.
"""
function _find_bracket(
        objective, center; initial_step = 0.5, max_attempts = 30,
        max_abs = 1.0e6
    )
    initial_step > 0 || throw(ArgumentError("initial_step must be positive"))
    f_center = objective(center)
    isfinite(f_center) || throw(
        ArgumentError("Score objective is not finite at beta=$center")
    )
    iszero(f_center) && return (center, center)

    last_interval = (center, center)
    last_values = (f_center, f_center)
    for attempt in 0:(max_attempts - 1)
        step = initial_step * 2.0^attempt
        lower = max(center - step, -max_abs)
        upper = min(center + step, max_abs)
        f_lower = objective(lower)
        f_upper = objective(upper)
        last_interval = (lower, upper)
        last_values = (f_lower, f_upper)

        isfinite(f_lower) && iszero(f_lower) && return (lower, lower)
        isfinite(f_upper) && iszero(f_upper) && return (upper, upper)
        if isfinite(f_lower) && sign(f_lower) != sign(f_center)
            return (lower, center)
        elseif isfinite(f_upper) && sign(f_upper) != sign(f_center)
            return (center, upper)
        elseif isfinite(f_lower) && isfinite(f_upper) &&
                sign(f_lower) != sign(f_upper)
            return (lower, upper)
        end
        lower == -max_abs && upper == max_abs && break
    end

    throw(
        ErrorException(
            "Could not find a finite sign-changing score bracket. Last interval " *
                "was $(last_interval) with objective values $(last_values)."
        )
    )
end

"""
    _solve_score_equation_bracket(score_obj, all_elements)

Solve E[``\\psi``(W; β, η)] = 0 for β using bracket-based root finding.

Uses AlefeldPotraShi method which doesn't require a good starting value.
"""
function _solve_score_equation_bracket(
        score_obj::AbstractScore, all_elements::Vector{<:NamedTuple}
    )
    # Define objective function: mean score as function of beta
    function objective(beta)
        return compute_mean_score(score_obj, beta, all_elements)
    end

    # Find bracket where objective changes sign
    bracket = _find_bracket(objective, 0.0)

    bracket[1] == bracket[2] && return bracket[1]

    # Use a robust bracket-based method after verifying the bracket.
    return find_zero(objective, bracket, AlefeldPotraShi())
end

"""
    learner_M(dml::DoubleMLLPLR)

Return the ml_M learner.
"""
learner_M(dml::DoubleMLLPLR) = dml.ml_M

"""
    learner_t(dml::DoubleMLLPLR)

Return the ml_t learner.
"""
learner_t(dml::DoubleMLLPLR) = dml.ml_t

"""
    learner_m(dml::DoubleMLLPLR)

Return the ml_m learner.
"""
learner_m(dml::DoubleMLLPLR) = dml.ml_m

"""
    learner_a(dml::DoubleMLLPLR)

Return the ml_a learner.
"""
learner_a(dml::DoubleMLLPLR) = dml.ml_a
