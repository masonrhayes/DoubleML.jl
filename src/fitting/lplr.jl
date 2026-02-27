"""
DoubleMLLPLR: Logistic Partially Linear Regression model.

Implements Double/Debiased Machine Learning for binary outcomes:
E[Y | D, X] = expit{β₀D + r₀(X)}
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
- `n_folds_inner::Int=5`: Number of inner folds for preliminary estimation
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
    n_folds_inner >= 2 || throw(ArgumentError("n_folds_inner must be >= 2"))

    n_obs = data.n_obs

    return DoubleMLLPLR{T, M, Tt, Mm, Ma_actual}(
        data, ml_M, ml_t, ml_m, ml_a_actual, n_folds, n_folds_inner, n_rep,
        score_obj, n_folds_tune, T(NaN), T(NaN), zeros(T, n_rep), zeros(T, n_rep), T(NaN),
        zeros(T, n_obs), zeros(T, n_obs), zeros(T, n_obs),
        false, zeros(T, 0, 1), nothing, 0,
        MLJ.Machine[], MLJ.Machine[], MLJ.Machine[], MLJ.Machine[],
        (;)
    )
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

    # For instrument score, ml_m should ideally support sample weights
    # for theoretically optimal estimation. However, some models (e.g.,
    # RandomForest from DecisionTree) do not support weights. In such cases,
    # we issue a warning but still allow the model to be used.
    if score == :instrument && !MLJBase.supports_weights(ml_m)
        @warn "ml_m ($(typeof(ml_m))) does not support sample weights. " *
            "For optimal estimation with instrument score, consider using a model " *
            "that supports weights (e.g., XGBoostRegressor, EvoTreeRegressor)."
    end

    # For instrument score with custom ml_a, check if it supports weights
    if score == :instrument && ml_a !== nothing && !MLJBase.supports_weights(ml_a)
        @warn "ml_a ($(typeof(ml_a))) does not support sample weights. " *
            "For optimal estimation with instrument score, consider using a model " *
            "that supports weights."
    end

    return nothing
end

"""
    fit!(obj::DoubleMLLPLR; verbose=0, force=false)

Fit the DoubleML LPLR model using double cross-fitting with bracket-based root finding.

# Arguments
- `verbose::Int=0`: Verbosity level
- `force::Bool=false`: Force refit if already fitted
"""
function MLJ.fit!(
        obj::DoubleMLLPLR{T}; verbose::Int = 0, force::Bool = false
    ) where {T}
    if isfitted(obj)
        !force && (@warn "Model already fitted. Use force=true to refit."; return obj)
        @warn "Forcing refit."
    end

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

    # Create double sample splitting (outer and inner folds)
    all_smpls_outer = draw_sample_splitting(n_obs, obj.n_folds, obj.n_rep)
    all_smpls_inner = draw_sample_splitting(n_obs, obj.n_folds_inner, obj.n_rep)

    X = DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = T.(to_numeric(obj.data.d))  # Ensure same type as Y

    obj.fitted_learners_M = MLJ.Machine[]
    obj.fitted_learners_t = MLJ.Machine[]
    obj.fitted_learners_m = MLJ.Machine[]
    obj.fitted_learners_a = MLJ.Machine[]

    # Handle tuning - full sample tuning (n_folds_tune == 0)
    if obj.n_folds_tune == 0
        any_tuned_full = _is_tuned_model(obj.ml_M) || _is_tuned_model(obj.ml_t) ||
            _is_tuned_model(obj.ml_m) || _is_tuned_model(obj.ml_a)
        if verbose > 0 && any_tuned_full
            @info "Tuning learners on full sample..."
        end
        ml_M_best, _ = _get_best_model(obj.ml_M, X, Y, verbose; model_name = "ml_M")
        ml_t_best, _ = _get_best_model(obj.ml_t, X, Y, verbose; model_name = "ml_t")
        ml_m_best, _ = _get_best_model(obj.ml_m, X, Y, verbose; model_name = "ml_m")
        ml_a_best, _ = _get_best_model(obj.ml_a, X, D, verbose; model_name = "ml_a")
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
            train_idx_tune = smpls_outer[1][1]  # Use first fold's training data for tuning
            ml_M_rep, _ = _get_best_model(
                obj.ml_M, X[train_idx_tune, :], Y[train_idx_tune], verbose;
                model_name = "ml_M", context = "for repetition $r"
            )
            ml_t_rep, _ = _get_best_model(
                obj.ml_t, X[train_idx_tune, :], Y[train_idx_tune], verbose;
                model_name = "ml_t", context = "for repetition $r"
            )
            ml_m_rep, _ = _get_best_model(
                obj.ml_m, X[train_idx_tune, :], Y[train_idx_tune], verbose;
                model_name = "ml_m", context = "for repetition $r"
            )
            ml_a_rep, _ = _get_best_model(
                obj.ml_a, X[train_idx_tune, :], D[train_idx_tune], verbose;
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

        # Stage 1: Fit M(D,X) = P(Y=1 | D, X) on inner folds
        M_hat_inner, M_hat_full = _fit_M_inner(obj, ml_M_rep, X, Y, D, smpls_inner, verbose)

        # Stage 2: Fit a(X) = E[D | X] on outer folds
        a_hat_outer = _fit_a_outer(obj, ml_a_rep, X, D, smpls_outer, verbose)

        # Stage 3: Fit t(X) = E[logit(M) | X] on outer folds
        t_hat_outer = _fit_t_outer(obj, ml_t_rep, X, M_hat_inner, smpls_outer, smpls_inner, verbose)

        # Stage 4: Fit m(X) on outer folds (nuisance_space: Y=0 only; instrument: weighted)
        m_hat_outer = _fit_m_outer(obj, ml_m_rep, X, Y, D, M_hat_full, smpls_outer, verbose)

        # Stage 5: Compute score elements for each outer fold (dynamic r_hat computation)
        score_elements_rep = _compute_all_score_elements(
            obj.score_obj, Y, D, smpls_outer, t_hat_outer, a_hat_outer, m_hat_outer
        )

        # Stage 6: Root-finding for this repetition's coefficient (bracket-based)
        obj.all_coef[r] = _solve_score_equation_bracket(obj.score_obj, score_elements_rep)

        # Compute score and derivative at this repetition's coefficient
        psi_rep = Vector{T}(undef, n_obs)
        psi_a_rep = Vector{T}(undef, n_obs)

        idx = 1
        for elements in score_elements_rep
            n_fold = length(elements.y)
            psi_fold = compute_score(obj.score_obj, obj.all_coef[r], elements)
            psi_a_fold = compute_score_deriv(obj.score_obj, obj.all_coef[r], elements)

            psi_rep[idx:(idx + n_fold - 1)] .= psi_fold
            psi_a_rep[idx:(idx + n_fold - 1)] .= psi_a_fold
            idx += n_fold
        end

        all_psi[r] = psi_rep
        all_psi_a[r] = psi_a_rep

        # Compute SE for this repetition
        J_rep = mean(psi_a_rep)
        gamma_hat_rep = mean(psi_rep .^ 2)
        sigma2_hat_rep = gamma_hat_rep / (n_obs * J_rep^2)
        obj.all_se[r] = sqrt(sigma2_hat_rep)
    end

    # Aggregate across repetitions using median-based aggregation
    obj.coef, obj.se = _aggregate_coefs_and_ses(obj.all_coef, obj.all_se)

    # Store final psi from last repetition (for bootstrap compatibility)
    obj.psi .= @. (all_psi_a[end] * obj.coef) + (all_psi[end] - all_psi_a[end] * obj.all_coef[end])
    obj.psi_a .= all_psi_a[end]
    obj.psi_b .= all_psi[end]

    if verbose > 0
        @info "Done! Coefficient: $(round(obj.coef, digits = 4)), SE: $(round(obj.se, digits = 4))"
    end

    return obj
end

"""
    _fit_M_inner(obj, ml_M, X, Y, D, smpls_inner, verbose)

Fit M(D,X) = P(Y=1 | D, X) on inner cross-fitting folds using tuned model ml_M.

Returns (M_hat_inner, M_hat_full) where:
- M_hat_inner: Vector of predictions for each inner fold's test set
- M_hat_full: Full-sample predictions (for creating targets for ml_t)
"""
function _fit_M_inner(
        obj::DoubleMLLPLR{T}, ml_M, X::DataFrame, Y::Vector{T}, D::Vector{T},
        smpls_inner::Vector, verbose::Int
    ) where {T}
    n_obs = length(Y)
    n_inner = length(smpls_inner)
    M_hat_inner = Vector{Vector{T}}(undef, n_inner)
    M_hat_full = zeros(T, n_obs)

    X_with_D = hcat(DataFrame(d = D), X; makeunique = true)

    # Coerce Y to categorical for classification
    Y_cat = coerce(Y, Multiclass)

    for (i, (train_idx, test_idx)) in enumerate(smpls_inner)
        X_train = X_with_D[train_idx, :]
        Y_train = Y_cat[train_idx]

        if verbose > 2
            @info "  Fitting ml_M on inner fold $i/$n_inner..."
        end

        mach_M = machine(ml_M, X_train, Y_train)
        MLJ.fit!(mach_M, verbosity = 0)
        push!(obj.fitted_learners_M, mach_M)

        # Get probability predictions for test set
        M_hat_inner[i], _ = predict_nuisance(mach_M, X_with_D[test_idx, :], "ml_M inner")
        M_hat_full[test_idx] .= M_hat_inner[i]
    end

    return M_hat_inner, M_hat_full
end

"""
    _fit_a_outer(obj, ml_a, X, D, smpls_outer, verbose)

Fit a(X) = E[D | X] on outer cross-fitting folds.

Returns a_hat_outer for dynamic r_hat computation in root-finding.
"""
function _fit_a_outer(
        obj::DoubleMLLPLR{T}, ml_a, X::DataFrame, D::Vector{T},
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

        # Predict on outer test set
        a_hat_outer[test_idx] .= MLJ.predict(mach_a, X[test_idx, :])
    end

    return a_hat_outer
end

"""
    _fit_t_outer(obj, ml_t, X, M_hat_inner, smpls_outer, smpls_inner, verbose)

Fit t(X) = E[logit(M) | X] on outer cross-fitting folds using tuned model ml_t.

Uses inner fold M predictions as targets.
"""
function _fit_t_outer(
        obj::DoubleMLLPLR{T}, ml_t, X::DataFrame, M_hat_inner::Vector{Vector{T}},
        smpls_outer::Vector, smpls_inner::Vector, verbose::Int
    ) where {T}
    n_obs = size(X, 1)
    n_outer = length(smpls_outer)
    t_hat_outer = zeros(T, n_obs)

    # Create targets: W = logit(M) from inner fold predictions
    W_targets = zeros(T, n_obs)
    for (i, (_, test_idx)) in enumerate(smpls_inner)
        W_targets[test_idx] .= logit.(M_hat_inner[i])
    end

    for (i, (train_idx, test_idx)) in enumerate(smpls_outer)
        if verbose > 2
            @info "  Fitting ml_t on outer fold $i/$n_outer..."
        end

        X_train = X[train_idx, :]
        W_train = W_targets[train_idx]

        mach_t = machine(ml_t, X_train, W_train)
        MLJ.fit!(mach_t, verbosity = 0)
        push!(obj.fitted_learners_t, mach_t)

        t_hat_outer[test_idx] .= MLJ.predict(mach_t, X[test_idx, :])
    end

    return t_hat_outer
end

"""
    _fit_m_outer(obj, ml_m, X, Y, D, M_hat_full, smpls_outer, verbose)

Fit m(X) on outer cross-fitting folds using tuned model ml_m.

For nuisance_space: fit only on Y=0 observations
For instrument: fit on all observations with sample weights M*(1-M) if supported
"""
function _fit_m_outer(
        obj::DoubleMLLPLR{T}, ml_m, X::DataFrame, Y::Vector{T}, D::Vector{T},
        M_hat_full::Vector{T}, smpls_outer::Vector, verbose::Int
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
            if isempty(train_filtered)
                @warn "No Y=0 observations in outer fold $i training set"
                # Fall back to all observations
                train_filtered = train_idx
            end
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
                M_train = M_hat_full[train_idx]
                M_clamped = clamp.(M_train, T(1.0e-8), T(1 - 1.0e-8))
                weights = M_clamped .* (T(1) .- M_clamped)

                mach_m = machine(ml_m, X_train, D_train, weights)
            else
                # Model doesn't support weights - fit without weights
                # This is suboptimal but allows flexibility in model choice
                mach_m = machine(ml_m, X_train, D_train)
            end
        end

        MLJ.fit!(mach_m, verbosity = 0)
        push!(obj.fitted_learners_m, mach_m)

        m_hat_outer[test_idx] .= MLJ.predict(mach_m, X[test_idx, :])
    end

    return m_hat_outer
end

"""
    _compute_all_score_elements(score_obj, Y, D, smpls_outer, t_hat, a_hat, m_hat)

Compute score elements for each outer fold using dynamic r_hat computation.
"""
function _compute_all_score_elements(
        score_obj::AbstractScore, Y::Vector{T}, D::Vector{T},
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
    _find_bracket(objective, start; step=0.5, max_attempts=20)

Find bracket (lower, upper) where objective changes sign for root finding.
"""
function _find_bracket(objective, start; step = 0.5, max_attempts = 20)
    f_start = objective(start)
    sign_start = sign(f_start)

    for i in 1:max_attempts
        lower = start - i * step
        upper = start + i * step

        f_lower = objective(lower)
        f_upper = objective(upper)

        if sign(f_lower) != sign_start
            return (lower, start)
        elseif sign(f_upper) != sign_start
            return (start, upper)
        end
    end

    # Fallback bracket - wider search
    return (start - 5.0, start + 5.0)
end

"""
    _solve_score_equation_bracket(score_obj, all_elements; tol=1e-8)

Solve E[ψ(W; β, η)] = 0 for β using bracket-based root finding.

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

    # Use robust bracket-based method
    result = find_zero(objective, bracket, AlefeldPotraShi())
    return result
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
