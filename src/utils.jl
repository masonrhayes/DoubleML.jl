"""
Utility functions for DoubleML.

Includes:
- Sample splitting for cross-fitting
- Target coercion for MLJ models
"""

"""
    coerce_target(y::AbstractVector, model)

Coerce target variable to appropriate type for MLJ model.
"""
function coerce_target(y::AbstractVector, model)
    expected = MLJ.target_scitype(model)
    return _coerce_by_scitype(y, expected)
end

function _coerce_by_scitype(y, ::Type{<:AbstractVector{<:ScientificTypes.Continuous}})
    return ScientificTypes.coerce(y, ScientificTypes.Continuous)
end

function _coerce_by_scitype(y, ::Type{<:AbstractVector{<:ScientificTypes.Multiclass{N}}}) where {N}
    return ScientificTypes.coerce(y, ScientificTypes.Multiclass{N})
end

function _coerce_by_scitype(y, ::Type{<:AbstractVector{<:ScientificTypes.Multiclass}})
    return ScientificTypes.coerce(y, ScientificTypes.Multiclass)
end

function _coerce_by_scitype(y, ::Type{<:AbstractVector{<:ScientificTypes.Finite}})
    return ScientificTypes.coerce(y, ScientificTypes.Multiclass)
end

function _coerce_by_scitype(y, expected)
    @warn "Unknown target_scitype: $expected. Passing data through unchanged."
    return y
end

"""
    draw_sample_splitting(n_obs, n_folds, n_rep; shuffle=true, rng)

Generate sample splitting indices for cross-fitting.

# Returns
Vector of length `n_rep`, each containing `n_folds` tuples of (train_idx, test_idx).
"""
function draw_sample_splitting(
        n_obs::Int, n_folds::Int, n_rep::Int;
        shuffle::Bool = true, rng::AbstractRNG = Random.default_rng()
    )
    all_smpls = Vector{Vector{Tuple{Vector{Int}, Vector{Int}}}}(undef, n_rep)
    seeds = rand(rng, UInt32, n_rep)

    for r in 1:n_rep
        cv = CV(nfolds = n_folds, shuffle = shuffle, rng = Xoshiro(seeds[r]))
        pairs = collect(MLJBase.train_test_pairs(cv, 1:n_obs))
        folds = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n_folds)
        for (k, (train_idx, test_idx)) in enumerate(pairs)
            folds[k] = (collect(train_idx), collect(test_idx))
        end
        all_smpls[r] = folds
    end

    return all_smpls
end

"""
    get_conditional_sample_splitting(n_obs, n_folds, n_rep, d; shuffle=true, rng)

Create conditional sample splits for control (D=0) and treated (D=1) groups.

Used for IRM where separate models are fit for each treatment group.

# Returns
Vector of length `n_rep`, each a tuple (smpls_d0, smpls_d1).
"""
function get_conditional_sample_splitting(
        n_obs::Int, n_folds::Int, n_rep::Int,
        d::AbstractVector;
        shuffle::Bool = true, rng::AbstractRNG = Random.default_rng()
    )
    all_smpls = draw_sample_splitting(n_obs, n_folds, n_rep; shuffle = shuffle, rng = rng)

    all_cond_smpls = Vector{
        Tuple{
            Vector{Tuple{Vector{Int}, Vector{Int}}},
            Vector{Tuple{Vector{Int}, Vector{Int}}},
        },
    }(undef, n_rep)

    for r in 1:n_rep
        smpls = all_smpls[r]
        n_folds_actual = length(smpls)
        smpls_d0 = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n_folds_actual)
        smpls_d1 = Vector{Tuple{Vector{Int}, Vector{Int}}}(undef, n_folds_actual)

        for (k, (train_idx, test_idx)) in enumerate(smpls)
            train_idx_d0 = train_idx[d[train_idx] .== 0]
            test_idx_d0 = test_idx[d[test_idx] .== 0]
            smpls_d0[k] = (train_idx_d0, test_idx_d0)

            train_idx_d1 = train_idx[d[train_idx] .== 1]
            test_idx_d1 = test_idx[d[test_idx] .== 1]
            smpls_d1[k] = (train_idx_d1, test_idx_d1)
        end

        all_cond_smpls[r] = (smpls_d0, smpls_d1)
    end

    return all_cond_smpls
end

"""
    _validate_fold_args(n_folds, n_rep, n_folds_tune)

Validate cross-fitting arguments.
"""
function _validate_fold_args(n_folds::Int, n_rep::Int, n_folds_tune::Int)
    n_folds < 1 && throw(DomainError(n_folds, "n_folds must be >= 1"))
    n_rep < 1 && throw(DomainError(n_rep, "n_rep must be >= 1"))
    n_folds_tune > n_folds && throw(
        DomainError(n_folds_tune, "n_folds_tune must be <= n_folds (got n_folds=$n_folds)")
    )
    return nothing
end

"""
    _warn_iterated_models(learners::NamedTuple)

Warn if any learners are IteratedModels (may be slow).
"""
function _warn_iterated_models(learners::NamedTuple)
    iterated_learners = String[]
    for (name, model) in pairs(learners)
        if model !== nothing &&
                model isa Union{MLJIteration.DeterministicIteratedModel, MLJIteration.ProbabilisticIteratedModel}
            push!(iterated_learners, string(name))
        end
    end

    if !isempty(iterated_learners)
        @warn "The following learner(s) are IteratedModels: $(join(iterated_learners, ", ")). " *
            "Fitting may be slow and iteration control is the user's responsibility."
    end

    return nothing
end

"""
    _evaluate_learner(mach, y_true, y_pred) -> NamedTuple

Compute out-of-sample performance using MLJ measures.

Uses multiple dispatch to select appropriate metric:
- Deterministic models → `rmse` (root mean squared error)
- Probabilistic models → `log_loss` (logarithmic loss)

# Arguments
- `mach`: Fitted MLJ machine
- `y_true`: True target values
- `y_pred`: Predictions (raw for Probabilistic, processed for Deterministic)

# Returns
- NamedTuple with `value` (metric value) and `measure` (MLJ measure object)
"""
function _evaluate_learner(
        mach::Machine{<:MLJBase.Deterministic},
        y_true::AbstractVector,
        y_pred::AbstractVector
    )
    val = MLJ.rmse(y_pred, y_true)
    return (value = val, measure = MLJ.rmse)
end

function _evaluate_learner(
        mach::Machine{<:MLJBase.Probabilistic},
        y_true::AbstractVector,
        y_pred_dist::AbstractVector
    )
    val = MLJ.log_loss(y_pred_dist, y_true)
    return (value = val, measure = MLJ.log_loss)
end

"""
    _aggregate_performance(eval_results::Vector{<:NamedTuple}) -> NamedTuple

Aggregate performance metrics across folds.

# Arguments
- `eval_results`: Vector of NamedTuples with `value` and `measure` fields

# Returns
- NamedTuple with mean `value` and the `measure`
"""
function _aggregate_performance(eval_results::Vector{<:NamedTuple})
    values = [e.value for e in eval_results]
    measure = eval_results[1].measure
    return (value = mean(values), measure = measure)
end

"""
    _aggregate_coefs_and_ses(all_coefs::Vector{T}, all_ses::Vector{T}) where T -> (coef, se)

Aggregate coefficient and standard error estimates across sample splitting repetitions.

Uses median-based aggregation following the Python DoubleML implementation.
The aggregation formula is:
- coef = median(all_coefs)
- agg_upper = median(all_coefs + 1.96 * all_ses)
- se = (agg_upper - coef) / 1.96

# Arguments
- `all_coefs::Vector{T}`: Coefficient estimates from each repetition
- `all_ses::Vector{T}`: Standard errors from each repetition

# Returns
- `coef::T`: Median-aggregated coefficient
- `se::T`: Aggregated standard error
"""
function _aggregate_coefs_and_ses(all_coefs::Vector{T}, all_ses::Vector{T}) where {T}
    n_rep = length(all_coefs)
    n_rep == 0 && return T(NaN), T(NaN)
    n_rep == 1 && return all_coefs[1], all_ses[1]

    # Compute median coefficient
    coef = median(all_coefs)

    # Compute critical value (using 1.96 for 95% CI)
    critical_value = T(1.96)

    # Compute upper bounds for each repetition
    all_upper_bounds = all_coefs .+ critical_value .* all_ses

    # Median aggregate the upper bounds
    agg_upper = median(all_upper_bounds)

    # Reverse to get aggregated SE
    se = (agg_upper - coef) / critical_value

    return coef, se
end
