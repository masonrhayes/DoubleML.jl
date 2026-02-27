"""
Evaluation and reporting functions for DoubleML models.
"""

"""
    fitted_params(dml::DoubleMLPLR) -> NamedTuple

Return fitted parameters from nuisance models.

For partialling out score: returns (ml_l, ml_m)
For IV-type score: returns (ml_l, ml_m, ml_g)
"""
function MLJ.fitted_params(dml::DoubleMLPLR)
    !isfitted(dml) && error("Model must be fitted first")

    result = (
        ml_l = [m.fitresult for m in dml.fitted_learners_l],
        ml_m = [m.fitresult for m in dml.fitted_learners_m],
    )

    if !isempty(dml.fitted_learners_g)
        result = merge(result, (ml_g = [m.fitresult for m in dml.fitted_learners_g],))
    end

    return result
end

"""
    fitted_params(dml::DoubleMLIRM) -> NamedTuple

Return fitted parameters from nuisance models.

Returns (ml_g0, ml_g1, ml_m) where ml_g1 is empty for ATTE score.
"""
function MLJ.fitted_params(dml::DoubleMLIRM)
    !isfitted(dml) && error("Model must be fitted first")

    return (
        ml_g0 = [m.fitresult for m in dml.fitted_learners_g0],
        ml_g1 = [m.fitresult for m in dml.fitted_learners_g1],
        ml_m = [m.fitresult for m in dml.fitted_learners_m],
    )
end

"""
    MLJ.report(dml::DoubleMLPLR) -> NamedTuple

Return report with learner reports and DML summary.

For partialling out: returns NamedTuple with (ml_l, ml_m) learner_reports
For IV-type: returns NamedTuple with (ml_l, ml_m, ml_g) learner_reports
"""
function MLJ.report(dml::DoubleMLPLR)
    !isfitted(dml) && error("Model must be fitted before calling report")

    dml_summary = (
        coef = dml.coef,
        se = dml.se,
        n_folds = dml.n_folds,
        n_rep = dml.n_rep,
        n_obs = dml.data.n_obs,
        score = get_score_name(dml.score_obj),
    )

    if isempty(dml.fitted_learners_g)
        return (
            learner_reports = (
                ml_l = [MLJ.report(m) for m in dml.fitted_learners_l],
                ml_m = [MLJ.report(m) for m in dml.fitted_learners_m],
            ),
            dml_summary = dml_summary,
        )
    else
        return (
            learner_reports = (
                ml_l = [MLJ.report(m) for m in dml.fitted_learners_l],
                ml_m = [MLJ.report(m) for m in dml.fitted_learners_m],
                ml_g = [MLJ.report(m) for m in dml.fitted_learners_g],
            ),
            dml_summary = dml_summary,
        )
    end
end

"""
    MLJ.report(dml::DoubleMLIRM) -> NamedTuple

Return report with learner reports and DML summary.
"""
function MLJ.report(dml::DoubleMLIRM)
    !isfitted(dml) && error("Model must be fitted before calling report")

    return (
        learner_reports = (
            ml_g0 = [MLJ.report(m) for m in dml.fitted_learners_g0],
            ml_g1 = [MLJ.report(m) for m in dml.fitted_learners_g1],
            ml_m = [MLJ.report(m) for m in dml.fitted_learners_m],
        ),
        dml_summary = (
            coef = dml.coef,
            se = dml.se,
            n_folds = dml.n_folds,
            n_rep = dml.n_rep,
            n_obs = dml.data.n_obs,
            score = get_score_name(dml.score_obj),
            normalize_ipw = dml.normalize_ipw,
            clipping_threshold = dml.clipping_threshold,
        ),
    )
end

"""
    MLJ.evaluate!(dml::DoubleMLPLR; resampling=CV(), measure=nothing, learners=:all, 
                  operation=predict, verbosity=1)

Evaluate nuisance model quality using MLJ resampling.

# Returns
NamedTuple mapping learner symbols to PerformanceEvaluation objects.
"""
function MLJ.evaluate!(
        dml::DoubleMLPLR;
        resampling = CV(),
        measure = nothing,
        learners = :all,
        operation = MLJ.predict,
        verbosity = 1
    )
    X = DataFrame(dml.data.x, dml.data.x_cols)
    Y = dml.data.y
    D = dml.data.d

    results = NamedTuple()

    evaluate_l = learners == :all || learners == :l || (learners isa Vector && :l in learners)
    evaluate_m = learners == :all || learners == :m || (learners isa Vector && :m in learners)
    evaluate_g = learners == :all || learners == :g || (learners isa Vector && :g in learners)

    get_measures = (learner_key, default) -> measure isa Dict ? get(measure, learner_key, default) : measure

    if evaluate_l && dml.ml_l !== nothing
        verbosity > 0 && @info "Evaluating ml_l..."
        mach_l = machine(dml.ml_l, X, Y)
        results = merge(results, (ml_l = MLJ.evaluate!(mach_l; resampling, measure = get_measures(:ml_l, nothing), operation, verbosity),))
    end

    if evaluate_m && dml.ml_m !== nothing
        verbosity > 0 && @info "Evaluating ml_m..."
        D_coerced = coerce_target(D, dml.ml_m)
        mach_m = machine(dml.ml_m, X, D_coerced)
        results = merge(results, (ml_m = MLJ.evaluate!(mach_m; resampling, measure = get_measures(:ml_m, nothing), operation, verbosity),))
    end

    if evaluate_g && dml.ml_g !== nothing
        verbosity > 0 && @info "Evaluating ml_g..."
        mach_g = machine(dml.ml_g, X, Y)
        results = merge(results, (ml_g = MLJ.evaluate!(mach_g; resampling, measure = get_measures(:ml_g, nothing), operation, verbosity),))
    end

    return results
end

"""
    MLJ.evaluate!(dml::DoubleMLIRM; resampling=CV(), measure=nothing, learners=:all,
                  operation=predict, verbosity=1)

Evaluate nuisance model quality for IRM.

Evaluates ml_g separately on control (D=0) and treated (D=1) observations.

# Returns
NamedTuple mapping learner symbols to PerformanceEvaluation objects.
"""
function MLJ.evaluate!(
        dml::DoubleMLIRM;
        resampling = CV(),
        measure = nothing,
        learners = :all,
        operation = MLJ.predict,
        verbosity = 1
    )
    X = DataFrame(dml.data.x, dml.data.x_cols)
    Y = dml.data.y
    D = dml.data.d

    idx_d0 = findall(D .== 0)
    idx_d1 = findall(D .== 1)

    results = NamedTuple()

    evaluate_g = learners == :all || learners == :g || (learners isa Vector && (:g in learners || :g0 in learners || :g1 in learners))
    evaluate_g0 = learners == :all || learners == :g0 || (learners isa Vector && :g0 in learners)
    evaluate_g1 = learners == :all || learners == :g1 || (learners isa Vector && :g1 in learners)
    evaluate_m = learners == :all || learners == :m || (learners isa Vector && :m in learners)

    get_measures = (learner_key, default) -> measure isa Dict ? get(measure, learner_key, default) : measure

    if (evaluate_g || evaluate_g0) && dml.ml_g !== nothing && length(idx_d0) > 0
        verbosity > 0 && @info "Evaluating ml_g on control observations (D=0)..."
        X_d0 = X[idx_d0, :]
        Y_d0 = Y[idx_d0]
        mach_g0 = machine(dml.ml_g, X_d0, Y_d0)
        results = merge(results, (ml_g0 = MLJ.evaluate!(mach_g0; resampling, measure = get_measures(:ml_g0, nothing), operation, verbosity),))
    end

    if (evaluate_g || evaluate_g1) && dml.ml_g !== nothing && length(idx_d1) > 0
        verbosity > 0 && @info "Evaluating ml_g on treated observations (D=1)..."
        X_d1 = X[idx_d1, :]
        Y_d1 = Y[idx_d1]
        mach_g1 = machine(dml.ml_g, X_d1, Y_d1)
        results = merge(results, (ml_g1 = MLJ.evaluate!(mach_g1; resampling, measure = get_measures(:ml_g1, nothing), operation, verbosity),))
    end

    if evaluate_m && dml.ml_m !== nothing
        verbosity > 0 && @info "Evaluating ml_m (propensity score model)..."
        D_coerced = coerce_target(D, dml.ml_m)
        mach_m = machine(dml.ml_m, X, D_coerced)
        results = merge(results, (ml_m = MLJ.evaluate!(mach_m; resampling, measure = get_measures(:ml_m, nothing), operation, verbosity),))
    end

    return results
end
