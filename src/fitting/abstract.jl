"""
Shared fitting interface and StatsAPI implementations for DoubleML models.
"""

"""
    isfitted(obj::AbstractDoubleML) -> Bool

Check if the model has been fitted.
"""
StatsAPI.isfitted(obj::AbstractDoubleML) = !isnan(obj.coef)

"""
    coef(obj::AbstractDoubleML) -> Vector{Float64}

Return the estimated coefficient(s) from the fitted model.

Returns a vector with the treatment effect estimate.
"""
function StatsAPI.coef(obj::AbstractDoubleML)
    !isfitted(obj) && error("Model not fitted")
    return [obj.coef]
end

"""
    stderror(obj::AbstractDoubleML) -> Vector{Float64}

Return the standard error(s) of the estimated coefficient(s).
"""
function StatsAPI.stderror(obj::AbstractDoubleML)
    !isfitted(obj) && error("Model not fitted")
    return [obj.se]
end

"""
    vcov(obj::AbstractDoubleML) -> Matrix{Float64}

Return the variance-covariance matrix of the estimated coefficient(s).
"""
function StatsAPI.vcov(obj::AbstractDoubleML)
    !isfitted(obj) && error("Model not fitted")
    return fill(obj.se^2, 1, 1)
end

"""
    confint(obj::AbstractDoubleML; joint::Bool=false, level::Real=0.95)

Compute confidence intervals for the estimated coefficient(s).

# Arguments
- `obj`: Fitted DoubleML model
- `joint::Bool=false`: If true, compute joint confidence intervals (requires bootstrap)
- `level::Real=0.95`: Confidence level (default 95%)

# Returns
A matrix with two columns: lower and upper bounds of the confidence interval.
"""
function StatsAPI.confint(obj::AbstractDoubleML; joint::Bool = false, level::Real = 0.95)
    !isfitted(obj) && error("Model not fitted")

    if joint
        !obj.has_bootstrapped && error(
            "joint=true requires bootstrap! to be called first. Run: bootstrap!(model, n_rep_boot=1000)"
        )
        return _confint_joint(obj, level)
    else
        return _confint_pointwise(obj, level)
    end
end

StatsAPI.confint(obj::AbstractDoubleML, level::Real) = confint(obj; level = level)

function _confint_pointwise(obj::AbstractDoubleML, level::Real)
    alpha = 1.0 - level
    z = quantile(Normal(), 1.0 - alpha / 2)
    lower = obj.coef - z * obj.se
    upper = obj.coef + z * obj.se
    return hcat(lower, upper)
end

function _confint_joint(obj::AbstractDoubleML, level::Real)
    boot_t_stat = obj.boot_t_stat
    max_abs_t = vec(maximum(abs.(boot_t_stat), dims = 2))
    alpha = 1.0 - level
    critical_value = quantile(max_abs_t, 1.0 - alpha)
    lower = obj.coef - critical_value * obj.se
    upper = obj.coef + critical_value * obj.se
    return hcat(lower, upper)
end

"""
    nobs(obj::AbstractDoubleML) -> Int

Return the number of observations in the data.
"""
StatsAPI.nobs(obj::AbstractDoubleML) = obj.data.n_obs

StatsAPI.dof(obj::AbstractDoubleML) = 1

StatsAPI.dof_residual(obj::AbstractDoubleML) = nobs(obj) - dof(obj)

StatsAPI.islinear(obj::AbstractDoubleML) = false

StatsAPI.responsename(obj::AbstractDoubleML) = string(obj.data.y_col)

StatsAPI.coefnames(obj::AbstractDoubleML) = [string(obj.data.d_col)]

function StatsAPI.coeftable(obj::AbstractDoubleML; level::Real = 0.95)
    cc = coef(obj)[1]
    se = stderror(obj)[1]
    z = cc / se
    p = 2 * ccdf(Normal(), abs(z))
    ci = confint(obj, level = level)

    return StatsBase.CoefTable(
        hcat(cc, se, z, p, ci[1], ci[2]),
        ["Estimate", "Std. Error", "z value", "Pr(>|z|)", "Lower $(level * 100)%", "Upper $(level * 100)%"],
        coefnames(obj),
        4,
        3
    )
end

function Base.show(io::IO, obj::AbstractDoubleML)
    println(io, typeof(obj))
    println(io, "==========================")
    return if !isfitted(obj)
        println(io, "Status: Not fitted")
    else
        show(io, coeftable(obj))
    end
end

"""
    _get_best_model(model, X, y, verbose; model_name="", context="")

Get the best model from tuning or return unchanged for regular models.
"""
function _get_best_model(
        model::Supervised, X, y, verbose;
        model_name::String = "", context::String = ""
    )
    return model, nothing
end

function _is_tuned_model(model)
    return model isa Union{MLJ.MLJTuning.DeterministicTunedModel, MLJ.MLJTuning.ProbabilisticTunedModel}
end

function _get_best_model(
        model::Union{MLJIteration.DeterministicIteratedModel, MLJIteration.ProbabilisticIteratedModel},
        X, y, verbose;
        model_name::String = "", context::String = ""
    )
    return model, nothing
end

function _get_best_model(
        model::Union{MLJ.MLJTuning.DeterministicTunedModel, MLJ.MLJTuning.ProbabilisticTunedModel},
        X, y, verbose;
        model_name::String = "", context::String = ""
    )
    if verbose > 0
        msg = isempty(context) ? "Tuning $model_name..." : "Tuning $model_name $context..."
        @info msg
    end
    y_coerced = coerce_target(y, model.model)
    mach = machine(model, X, y_coerced)
    MLJ.fit!(mach, verbosity = verbose)
    best_model = MLJ.fitted_params(mach).best_model
    return best_model, mach
end

"""
    predict_nuisance(mach, X, context) -> (y_pred, y_pred_raw)

Return processed predictions and raw predictions from a fitted machine.

For Deterministic models: both values are identical (the predictions)
For Probabilistic models: returns (pdf values, distributions)

# Returns
- `y_pred`: Processed predictions for score computation
- `y_pred_raw`: Raw predictions for evaluation (distributions for Probabilistic)
"""
function predict_nuisance(mach::Machine{<:MLJBase.Deterministic}, X, context::String)
    y_pred = MLJ.predict(mach, X)
    return y_pred, y_pred
end

function predict_nuisance(mach::Machine{<:MLJBase.Probabilistic}, X, context::String)
    y_pred_dist = MLJ.predict(mach, X)
    y_pred = @. pdf(y_pred_dist, true)
    return y_pred, y_pred_dist
end
