"""
DoubleMLIRM: Interactive Regression Model.

Implements Double/Debiased Machine Learning for IRM:
Y = g_0(D, X) + ζ, where D is binary (0/1)
"""

"""
    _normalize_ipw(propensity_score, treatment_indicator)

Hajek normalization of propensity scores for IPW.
"""
function _normalize_ipw(
        propensity_score::AbstractVector{T},
        treatment_indicator::AbstractVector
    ) where {T <: AbstractFloat}
    one_t = one(T)
    mean_treat1 = mean(treatment_indicator ./ propensity_score)
    mean_treat0 = mean((one_t .- treatment_indicator) ./ (one_t .- propensity_score))

    normalized = (
        treatment_indicator .* propensity_score .* mean_treat1 .+
            (one_t .- treatment_indicator) .* (one_t .- (one_t .- propensity_score) .* mean_treat0)
    )

    return normalized
end

"""
    _propensity_score_adjustment(propensity_score, treatment_indicator, normalize_ipw; clipping_threshold=0.01)

Adjust propensity scores with clipping and optional Hajek normalization.
"""
function _propensity_score_adjustment(
        propensity_score::AbstractVector{T},
        treatment_indicator::AbstractVector,
        normalize_ipw::Bool;
        clipping_threshold::Real = 0.01
    ) where {T <: AbstractFloat}
    ct = T(clipping_threshold)
    m_hat_clipped = clamp.(propensity_score, ct, one(T) - ct)

    if normalize_ipw
        m_hat_adj = _normalize_ipw(m_hat_clipped, treatment_indicator)
    else
        m_hat_adj = m_hat_clipped
    end

    return m_hat_adj
end

"""
    _validate_propensity_scores(m_hat, fold_idx, clipping_threshold)

Validate propensity scores and warn about potential issues.
"""
function _validate_propensity_scores(
        m_hat::AbstractVector,
        fold_idx::Int,
        clipping_threshold::Float64
    )
    n = length(m_hat)

    if all(m_hat .== m_hat[1])
        @warn "Propensity scores in fold $fold_idx are all identical ($(m_hat[1])). " *
            "The classifier may not be learning from covariates."
        return
    end

    n_clipped_low = sum(m_hat .<= clipping_threshold)
    n_clipped_high = sum(m_hat .>= (1.0 - clipping_threshold))

    if n_clipped_low == n || n_clipped_high == n || n_clipped_low + n_clipped_high == n
        @warn "All propensity scores in fold $fold_idx are at clipping boundaries. " *
            "Standard errors may be unreliable."
    end

    return nothing
end

"""
    DoubleMLIRM(data::DoubleMLData{T}, ml_g, ml_m; n_folds=5, n_rep=1, score=:ATE,
                normalize_ipw=false, clipping_threshold=0.01, n_folds_tune=0) where {T}

Create a DoubleML IRM model.

# Arguments
- `data::DoubleMLData{T}`: Data container (T inferred from data)
- `ml_g`: Model for g(X, D) = E[Y|X, D]
- `ml_m`: Model for m(X) = E[D|X] (propensity score, must be classifier)
- `n_folds::Int=5`: Number of cross-fitting folds
- `n_rep::Int=1`: Number of sample splitting repetitions
- `score::Symbol=:ATE`: Score type (:ATE or :ATTE)
- `normalize_ipw::Bool=false`: Whether to normalize IPW weights
- `clipping_threshold::AbstractFloat=0.01`: Threshold for propensity score clipping
- `n_folds_tune::Int=0`: Folds for tuning (0 = full sample)

# Returns
- `DoubleMLIRM{T, typeof(ml_g), typeof(ml_m)}`

# Examples
```julia
data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
ml_g = @load RandomForestRegressor pkg=DecisionTree
ml_m = @load RandomForestClassifier pkg=DecisionTree

model = DoubleMLIRM(data, ml_g, ml_m, score=:ATE)
fit!(model)
```
"""
function DoubleMLIRM(
        data::DoubleMLData{T}, ml_g::G, ml_m::M;
        n_folds::Int = 5,
        n_rep::Int = 1,
        score::Symbol = :ATE,
        normalize_ipw::Bool = false,
        clipping_threshold::AbstractFloat = 0.01,
        n_folds_tune::Int = 0
    ) where {T <: AbstractFloat, G <: Supervised, M <: Supervised}

    score_obj = if score == :ATE
        ATEScore()
    elseif score == :ATTE
        ATTEScore()
    else
        throw(ArgumentError("Score must be :ATE or :ATTE, got: $score"))
    end

    _validate_fold_args(n_folds, n_rep, n_folds_tune)

    if clipping_threshold <= 0 || clipping_threshold >= 0.5
        throw(DomainError(clipping_threshold, "clipping_threshold must be in (0, 0.5)"))
    end

    if !check_binary(data.d)
        d_unique = unique(data.d)
        throw(ArgumentError("Treatment must be binary (0 or 1) for IRM. Got: $d_unique"))
    end

    _warn_iterated_models((ml_g = ml_g, ml_m = ml_m))

    n_obs = data.n_obs
    ct = T(clipping_threshold)

    return DoubleMLIRM{T, G, M}(
        data, ml_g, ml_m, n_folds, n_rep, score_obj, normalize_ipw, ct, n_folds_tune,
        T(NaN), T(NaN), zeros(T, n_rep), zeros(T, n_rep), zeros(T, n_obs), zeros(T, n_obs), zeros(T, n_obs),
        false, zeros(T, 0, 1), nothing, 0,
        MLJ.Machine[], MLJ.Machine[], MLJ.Machine[],
        (;)
    )
end

"""
    fit!(obj::DoubleMLIRM; verbose=0, force=false)

Fit the DoubleML IRM model using cross-fitting.
"""
function MLJ.fit!(obj::DoubleMLIRM{T}; verbose::Int = 0, force::Bool = false) where {T}
    if isfitted(obj)
        !force && (@warn "Model already fitted. Use force=true to refit."; return obj)
        @warn "Forcing refit."
    end

    if verbose > 0
        @info "Fitting DoubleMLIRM with $(obj.n_folds)-fold cross-fitting, " *
            "$(obj.n_rep) repetition(s), score=$(get_score_name(obj.score_obj))..."
    end

    n_obs = obj.data.n_obs

    X = DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = obj.data.d

    all_cond_smpls = get_conditional_sample_splitting(n_obs, obj.n_folds, obj.n_rep, D)
    all_smpls = draw_sample_splitting(n_obs, obj.n_folds, obj.n_rep)

    obj.fitted_learners_g0 = MLJ.Machine[]
    obj.fitted_learners_g1 = MLJ.Machine[]
    obj.fitted_learners_m = MLJ.Machine[]

    ml_g = obj.ml_g
    ml_m = obj.ml_m

    if obj.n_folds_tune == 0
        any_tuned = _is_tuned_model(ml_g) || _is_tuned_model(ml_m)
        if verbose > 0 && any_tuned
            @info "Tuning ml_g and ml_m on full sample..."
        end
        ml_g, _ = _get_best_model(ml_g, X, Y, verbose; model_name = "ml_g")
        ml_m, _ = _get_best_model(ml_m, X, D, verbose; model_name = "ml_m")
    end

    D_num = to_numeric(D)
    E_D_global = mean(D_num)

    eval_g0_folds = NamedTuple[]
    eval_g1_folds = NamedTuple[]
    eval_m_folds = NamedTuple[]

    # Storage for per-repetition psi components
    all_psi_a = Vector{Vector{T}}(undef, obj.n_rep)
    all_psi_b = Vector{Vector{T}}(undef, obj.n_rep)

    for r in 1:obj.n_rep
        smpls = all_smpls[r]
        smpls_d0, smpls_d1 = all_cond_smpls[r]

        rep_ml_g = ml_g
        rep_ml_m = ml_m

        if obj.n_folds_tune > 0
            train_idx_tune, _ = smpls[1]
            any_tuned_rep = _is_tuned_model(obj.ml_g) || _is_tuned_model(obj.ml_m)
            if verbose > 0 && any_tuned_rep
                @info "Tuning for repetition $r..."
            end
            rep_ml_g, _ = _get_best_model(obj.ml_g, X[train_idx_tune, :], Y[train_idx_tune], verbose; model_name = "ml_g", context = "for repetition $r")
            rep_ml_m, _ = _get_best_model(obj.ml_m, X[train_idx_tune, :], D[train_idx_tune], verbose; model_name = "ml_m", context = "for repetition $r")
        end

        psi_a_rep = zeros(T, n_obs)
        psi_b_rep = zeros(T, n_obs)

        for (fold_idx, (train_idx, test_idx)) in enumerate(smpls)
            train_idx_d0, test_idx_d0 = smpls_d0[fold_idx]
            train_idx_d1, test_idx_d1 = smpls_d1[fold_idx]

            X_test = X[test_idx, :]
            Y_test = Y[test_idx]
            D_test = D[test_idx]
            D_test_numeric = D_num[test_idx]

            if length(train_idx_d0) > 0
                X_train_control = X[train_idx_d0, :]
                Y_train_control = Y[train_idx_d0]
                Y_train_control_coerced = coerce_target(Y_train_control, obj.ml_g)
                mach_g0 = machine(rep_ml_g, X_train_control, Y_train_control_coerced)
                MLJ.fit!(mach_g0, verbosity = verbose)
                push!(obj.fitted_learners_g0, mach_g0)
                g_hat0, g0_pred_raw = predict_nuisance(mach_g0, X_test, "g_0(X)")
            else
                error("No control observations in training fold $fold_idx.")
            end

            if obj.score_obj isa ATEScore
                if length(train_idx_d1) > 0
                    X_train_treated = X[train_idx_d1, :]
                    Y_train_treated = Y[train_idx_d1]
                    Y_train_treated_coerced = coerce_target(Y_train_treated, obj.ml_g)
                    mach_g1 = machine(rep_ml_g, X_train_treated, Y_train_treated_coerced)
                    MLJ.fit!(mach_g1, verbosity = verbose)
                    push!(obj.fitted_learners_g1, mach_g1)
                    g_hat1, g1_pred_raw = predict_nuisance(mach_g1, X_test, "g_1(X)")
                else
                    error("No treated observations in training fold $fold_idx.")
                end
            else
                g_hat1 = zeros(T, length(g_hat0))
                g1_pred_raw = g_hat1
            end

            X_train = X[train_idx, :]
            D_train = D[train_idx]
            D_train_coerced = coerce_target(D_train, obj.ml_m)
            mach_m = machine(rep_ml_m, X_train, D_train_coerced)
            MLJ.fit!(mach_m, verbosity = verbose)
            push!(obj.fitted_learners_m, mach_m)
            m_hat, m_pred_raw = predict_nuisance(mach_m, X_test, "m(X)")

            _validate_propensity_scores(m_hat, fold_idx, Float64(obj.clipping_threshold))

            m_hat_adj = _propensity_score_adjustment(
                m_hat, D_test_numeric, obj.normalize_ipw;
                clipping_threshold = obj.clipping_threshold
            )

            idx_control_test = findall(D_test .== 0)
            idx_treated_test = findall(D_test .== 1)

            if !isempty(idx_control_test)
                push!(eval_g0_folds, _evaluate_learner(mach_g0, Y_test[idx_control_test], g0_pred_raw[idx_control_test]))
            end

            if obj.score_obj isa ATEScore && !isempty(idx_treated_test)
                push!(eval_g1_folds, _evaluate_learner(mach_g1, Y_test[idx_treated_test], g1_pred_raw[idx_treated_test]))
            end

            push!(eval_m_folds, _evaluate_learner(mach_m, D_test, m_pred_raw))

            if obj.score_obj isa ATEScore
                psi_a, psi_b = compute_score(obj.score_obj, Y_test, D_test, g_hat0, g_hat1, m_hat_adj)
            else
                if E_D_global == 0
                    error("No treated observations. Cannot compute ATTE.")
                end
                psi_a, psi_b = compute_score(obj.score_obj, Y_test, D_test, g_hat0, g_hat1, m_hat_adj, E_D_global)
            end

            psi_a_rep[test_idx] .= psi_a
            psi_b_rep[test_idx] .= psi_b
        end

        all_psi_a[r] = psi_a_rep
        all_psi_b[r] = psi_b_rep

        # Solve DML2 for this repetition
        obj.all_coef[r] = dml2_solve(psi_a_rep, psi_b_rep)

        # Compute SE for this repetition
        psi_rep = @. (psi_a_rep * obj.all_coef[r]) + psi_b_rep
        J_rep = mean(psi_a_rep)
        gamma_hat_rep = mean(psi_rep .^ 2)
        sigma2_hat_rep = gamma_hat_rep / (n_obs * J_rep^2)
        obj.all_se[r] = sqrt(sigma2_hat_rep)
    end

    # Aggregate across repetitions using median-based aggregation
    obj.coef, obj.se = _aggregate_coefs_and_ses(obj.all_coef, obj.all_se)

    # Store final psi from last repetition (for bootstrap compatibility)
    obj.psi .= @. (all_psi_a[end] * obj.coef) + all_psi_b[end]
    obj.psi_a .= all_psi_a[end]
    obj.psi_b .= all_psi_b[end]

    if obj.score_obj isa ATEScore
        obj.learner_performance = (
            ml_g0 = _aggregate_performance(eval_g0_folds),
            ml_g1 = _aggregate_performance(eval_g1_folds),
            ml_m = _aggregate_performance(eval_m_folds),
        )
    else
        obj.learner_performance = (
            ml_g0 = _aggregate_performance(eval_g0_folds),
            ml_m = _aggregate_performance(eval_m_folds),
        )
    end

    if verbose > 0
        @info "Done! Coefficient: $(round(obj.coef, digits = 4)), SE: $(round(obj.se, digits = 4))"
    end

    return obj
end

"""
    learner_g(dml::DoubleMLIRM)

Return the ml_g learner.
"""
learner_g(dml::DoubleMLIRM) = dml.ml_g

"""
    learner_m(dml::DoubleMLIRM)

Return the ml_m learner.
"""
learner_m(dml::DoubleMLIRM) = dml.ml_m
