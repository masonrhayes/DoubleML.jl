"""
DoubleMLPLR: Partially Linear Regression model.

Implements Double/Debiased Machine Learning for the partially linear model:
Y = θ·D + g(X) + ε
"""

"""
    DoubleMLPLR(data::DoubleMLData{T}, ml_l, ml_m; ml_g=nothing, n_folds=5, n_rep=1, 
                score=:partialling_out, n_folds_tune=0) where {T}

Create a DoubleML PLR model.

# Arguments
- `data::DoubleMLData{T}`: Data container (T inferred from data)
- `ml_l`: Model for l(X) = E[Y|X] (must be Deterministic regressor)
- `ml_m`: Model for m(X) = E[D|X] (regressor or classifier)
- `ml_g=nothing`: Model for g(X) = E[Y - D·θ|X] (required for IV-type)
- `n_folds::Int=5`: Number of cross-fitting folds
- `n_rep::Int=1`: Number of sample splitting repetitions
- `score::Symbol=:partialling_out`: Score type (:partialling_out or :IV_type)
- `n_folds_tune::Int=0`: Folds for tuning (0 = full sample)

# Returns
- `DoubleMLPLR{T, typeof(ml_l), typeof(ml_m), typeof(ml_g)}`

# Examples
```julia
data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])
ml_l = @load RandomForestRegressor pkg=DecisionTree
ml_m = @load RandomForestRegressor pkg=DecisionTree

model = DoubleMLPLR(data, ml_l, ml_m)
fit!(model)
```
"""
function DoubleMLPLR(
        data::DoubleMLData{T}, ml_l::L, ml_m::M;
        ml_g::G = nothing,
        n_folds::Int = 5,
        n_rep::Int = 1,
        score::Symbol = :partialling_out,
        n_folds_tune::Int = 0
    ) where {T <: AbstractFloat, L <: Supervised, M <: Supervised, G}

    score_obj = if score == :partialling_out
        PartiallingOutScore()
    elseif score == :IV_type
        IVTypeScore()
    else
        throw(ArgumentError("Score must be :partialling_out or :IV_type, got: $score"))
    end

    if score == :IV_type && ml_g === nothing
        throw(
            ArgumentError(
                "score=:IV_type requires ml_g learner. " *
                    "Provide ml_g for estimating g(X) = E[Y - D·θ|X]"
            )
        )
    end

    if score == :IV_type && ml_g !== nothing && ml_g isa MLJBase.Probabilistic
        throw(
            ArgumentError(
                "ml_g for IV-type score must be a regressor (Deterministic), " *
                    "not a classifier (Probabilistic). " *
                    "Got $(typeof(ml_g)) which is a Probabilistic model. " *
                    "Use a model that predicts continuous values."
            )
        )
    end

    if ml_l isa MLJBase.Probabilistic
        throw(
            ArgumentError(
                "ml_l must be a regressor (Deterministic), not a classifier (Probabilistic)."
            )
        )
    end

    if score == :partialling_out && ml_g !== nothing
        @warn "ml_g was provided but will not be used with score=:partialling_out. " *
            "The partialling out score only requires ml_l and ml_m."
    end

    _validate_fold_args(n_folds, n_rep, n_folds_tune)

    _warn_iterated_models((ml_l = ml_l, ml_m = ml_m, ml_g = ml_g))

    n_obs = data.n_obs

    return DoubleMLPLR{T, L, M, G}(
        data, ml_l, ml_m, ml_g, n_folds, n_rep, score_obj, n_folds_tune,
        T(NaN), T(NaN), zeros(T, n_rep), zeros(T, n_rep), zeros(T, n_obs), zeros(T, n_obs), zeros(T, n_obs),
        false, zeros(T, 0, 1), nothing, 0,
        MLJ.Machine[], MLJ.Machine[], MLJ.Machine[],
        (;)
    )
end

"""
    fit!(obj::DoubleMLPLR; verbose=0, max_iter=1, tol=1e-4, force=false)

Fit the DoubleML PLR model using cross-fitting.
"""
function MLJ.fit!(
        obj::DoubleMLPLR{T}; verbose::Int = 0, max_iter::Int = 1,
        tol::Real = 1.0e-4, force::Bool = false
    ) where {T}
    if isfitted(obj)
        !force && (@warn "Model already fitted. Use force=true to refit."; return obj)
        @warn "Forcing refit."
    end

    if verbose > 0
        score_name = obj.score_obj isa PartiallingOutScore ? "partialling out" : "IV-type"
        @info "Fitting DoubleMLPLR with $(obj.n_folds)-fold cross-fitting, $(obj.n_rep) repetition(s)..."
        @info "Score function: $score_name"
    end

    n_obs = obj.data.n_obs
    all_smpls = draw_sample_splitting(n_obs, obj.n_folds, obj.n_rep)

    X = DataFrame(obj.data.x, obj.data.x_cols)
    Y = obj.data.y
    D = obj.data.d

    obj.fitted_learners_l = MLJ.Machine[]
    obj.fitted_learners_m = MLJ.Machine[]
    obj.fitted_learners_g = MLJ.Machine[]

    if obj.score_obj isa PartiallingOutScore
        _fit_partialling_out!(obj, X, Y, D, all_smpls, n_obs, verbose)
    else
        _fit_iv_type!(obj, X, Y, D, all_smpls, n_obs, verbose, max_iter, tol)
    end

    return obj
end

function _fit_partialling_out!(
        obj::DoubleMLPLR{T}, X, Y, D, all_smpls, n_obs, verbose
    ) where {T}
    ml_l = obj.ml_l
    ml_m = obj.ml_m

    if obj.n_folds_tune == 0
        ml_l, _ = _get_best_model(ml_l, X, Y, verbose; model_name = "ml_l")
        ml_m, _ = _get_best_model(ml_m, X, D, verbose; model_name = "ml_m")
    end

    eval_l_folds = NamedTuple[]
    eval_m_folds = NamedTuple[]

    # Storage for per-repetition results
    all_psi_a = Vector{Vector{T}}(undef, obj.n_rep)
    all_psi_b = Vector{Vector{T}}(undef, obj.n_rep)

    for r in 1:obj.n_rep
        smpls = all_smpls[r]
        rep_ml_l = ml_l
        rep_ml_m = ml_m

        if obj.n_folds_tune > 0
            train_idx_tune, _ = smpls[1]
            rep_ml_l, _ = _get_best_model(
                obj.ml_l, X[train_idx_tune, :], Y[train_idx_tune], verbose;
                model_name = "ml_l", context = "for repetition $r"
            )
            rep_ml_m, _ = _get_best_model(
                obj.ml_m, X[train_idx_tune, :], D[train_idx_tune], verbose;
                model_name = "ml_m", context = "for repetition $r"
            )
        end

        psi_a_rep = zeros(T, n_obs)
        psi_b_rep = zeros(T, n_obs)

        for (train_idx, test_idx) in smpls
            X_train = X[train_idx, :]
            X_test = X[test_idx, :]
            Y_train = Y[train_idx]
            D_train = D[train_idx]

            mach_l = machine(rep_ml_l, X_train, Y_train)
            MLJ.fit!(mach_l, verbosity = verbose)
            push!(obj.fitted_learners_l, mach_l)
            l_hat, l_pred_raw = predict_nuisance(mach_l, X_test, "l(X)")

            D_train_coerced = coerce_target(D_train, rep_ml_m)
            mach_m = machine(rep_ml_m, X_train, D_train_coerced)
            MLJ.fit!(mach_m, verbosity = verbose)
            push!(obj.fitted_learners_m, mach_m)
            m_hat, m_pred_raw = predict_nuisance(mach_m, X_test, "m(X)")

            D_test = D[test_idx]
            Y_test = Y[test_idx]

            eval_l = _evaluate_learner(mach_l, Y_test, l_pred_raw)
            eval_m = _evaluate_learner(mach_m, D_test, m_pred_raw)
            push!(eval_l_folds, eval_l)
            push!(eval_m_folds, eval_m)

            psi_a, psi_b = compute_score(obj.score_obj, Y_test, D_test, l_hat, m_hat)

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

    obj.learner_performance = (
        ml_l = _aggregate_performance(eval_l_folds),
        ml_m = _aggregate_performance(eval_m_folds),
    )

    return if verbose > 0
        @info "Done! Coefficient: $(round(obj.coef, digits = 4)), SE: $(round(obj.se, digits = 4))"
    end
end

function _fit_iv_type!(
        obj::DoubleMLPLR{T}, X, Y, D, all_smpls, n_obs, verbose, max_iter, tol
    ) where {T}
    ml_l = obj.ml_l
    ml_m = obj.ml_m
    ml_g = obj.ml_g

    if obj.n_folds_tune == 0
        ml_l, _ = _get_best_model(ml_l, X, Y, verbose; model_name = "ml_l")
        ml_m, _ = _get_best_model(ml_m, X, D, verbose; model_name = "ml_m")
        ml_g, _ = _get_best_model(ml_g, X, Y, verbose; model_name = "ml_g")
    end

    eval_l_folds = NamedTuple[]
    eval_m_folds = NamedTuple[]
    eval_g_folds = NamedTuple[]

    # Storage for per-repetition psi components
    all_psi_a = Vector{Vector{T}}(undef, obj.n_rep)
    all_psi_b = Vector{Vector{T}}(undef, obj.n_rep)

    for r in 1:obj.n_rep
        smpls = all_smpls[r]
        rep_ml_l = ml_l
        rep_ml_m = ml_m
        rep_ml_g = ml_g

        if obj.n_folds_tune > 0
            train_idx_tune, _ = smpls[1]
            rep_ml_l, _ = _get_best_model(obj.ml_l, X[train_idx_tune, :], Y[train_idx_tune], verbose; model_name = "ml_l", context = "for repetition $r")
            rep_ml_m, _ = _get_best_model(obj.ml_m, X[train_idx_tune, :], D[train_idx_tune], verbose; model_name = "ml_m", context = "for repetition $r")
            rep_ml_g, _ = _get_best_model(obj.ml_g, X[train_idx_tune, :], Y[train_idx_tune], verbose; model_name = "ml_g", context = "for repetition $r")
        end

        # Initial partialling out for starting value
        psi_a_temp = zeros(T, n_obs)
        psi_b_temp = zeros(T, n_obs)

        for (train_idx, test_idx) in smpls
            X_train = X[train_idx, :]
            X_test = X[test_idx, :]
            Y_train = Y[train_idx]
            D_train = D[train_idx]

            mach_l = machine(rep_ml_l, X_train, Y_train)
            MLJ.fit!(mach_l, verbosity = verbose)
            l_hat, l_pred_raw = predict_nuisance(mach_l, X_test, "l(X)")

            D_train_coerced = coerce_target(D_train, rep_ml_m)
            mach_m = machine(rep_ml_m, X_train, D_train_coerced)
            MLJ.fit!(mach_m, verbosity = verbose)
            m_hat, m_pred_raw = predict_nuisance(mach_m, X_test, "m(X)")

            D_test = D[test_idx]
            Y_test = Y[test_idx]

            psi_a, psi_b = compute_score(PartiallingOutScore(), Y_test, D_test, l_hat, m_hat)
            psi_a_temp[test_idx] .= psi_a
            psi_b_temp[test_idx] .= psi_b
        end

        theta_current = dml2_solve(psi_a_temp, psi_b_temp)

        # Iterative refinement for this repetition
        psi_a_rep = zeros(T, n_obs)
        psi_b_rep = zeros(T, n_obs)

        for iter in 1:max_iter
            psi_a_fold = zeros(T, n_obs)
            psi_b_fold = zeros(T, n_obs)

            for (train_idx, test_idx) in smpls
                X_train = X[train_idx, :]
                X_test = X[test_idx, :]
                Y_train = Y[train_idx]
                D_train = D[train_idx]

                D_train_coerced = coerce_target(D_train, rep_ml_m)
                mach_m = machine(rep_ml_m, X_train, D_train_coerced)
                MLJ.fit!(mach_m, verbosity = verbose)
                m_hat, m_pred_raw = predict_nuisance(mach_m, X_test, "m(X)")

                Y_target = Y_train .- D_train .* theta_current
                mach_g = machine(rep_ml_g, X_train, Y_target)
                MLJ.fit!(mach_g, verbosity = verbose)
                g_hat, g_pred_raw = predict_nuisance(mach_g, X_test, "g(X)")

                if iter == 1
                    push!(obj.fitted_learners_m, mach_m)
                    push!(obj.fitted_learners_g, mach_g)
                    mach_l = machine(rep_ml_l, X_train, Y_train)
                    MLJ.fit!(mach_l, verbosity = verbose)
                    push!(obj.fitted_learners_l, mach_l)
                    _, l_pred_raw = predict_nuisance(mach_l, X_test, "l(X)")

                    D_test = D[test_idx]
                    Y_test = Y[test_idx]

                    push!(eval_l_folds, _evaluate_learner(mach_l, Y_test, l_pred_raw))
                    push!(eval_m_folds, _evaluate_learner(mach_m, D_test, m_pred_raw))
                    push!(eval_g_folds, _evaluate_learner(mach_g, Y_test, g_pred_raw))
                end

                D_test = D[test_idx]
                Y_test = Y[test_idx]

                psi_a, psi_b = compute_score(IVTypeScore(), Y_test, D_test, g_hat, m_hat)
                psi_a_fold[test_idx] .= psi_a
                psi_b_fold[test_idx] .= psi_b
            end

            theta_new = dml2_solve(psi_a_fold, psi_b_fold)

            if abs(theta_new - theta_current) < tol
                theta_current = theta_new
                psi_a_rep .= psi_a_fold
                psi_b_rep .= psi_b_fold
                break
            elseif iter == max_iter
                theta_current = theta_new
                psi_a_rep .= psi_a_fold
                psi_b_rep .= psi_b_fold
            else
                theta_current = theta_new
            end
        end

        all_psi_a[r] = psi_a_rep
        all_psi_b[r] = psi_b_rep

        # Store coefficient and SE for this repetition
        obj.all_coef[r] = theta_current

        psi_rep = @. (psi_a_rep * theta_current) + psi_b_rep
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

    obj.learner_performance = (
        ml_l = _aggregate_performance(eval_l_folds),
        ml_m = _aggregate_performance(eval_m_folds),
        ml_g = _aggregate_performance(eval_g_folds),
    )

    return if verbose > 0
        @info "Done! Coefficient: $(round(obj.coef, digits = 4)), SE: $(round(obj.se, digits = 4))"
    end
end

"""
    learner_l(dml::DoubleMLPLR)

Return the ml_l learner.
"""
learner_l(dml::DoubleMLPLR) = dml.ml_l

"""
    learner_m(dml::DoubleMLPLR)

Return the ml_m learner.
"""
learner_m(dml::DoubleMLPLR) = dml.ml_m

"""
    learner_g(dml::DoubleMLPLR)

Return the ml_g learner (or nothing for partialling out).
"""
learner_g(dml::DoubleMLPLR) = dml.ml_g
