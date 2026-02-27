"""
Bootstrap inference for DoubleML models.
"""

"""
    _draw_weights(method, n_rep_boot, n_obs, ::Type{T}; rng)

Draw bootstrap weights for multiplier bootstrap.

# Methods
- `:normal`: Standard normal N(0,1) weights
- `:wild`: Mammen's wild bootstrap
- `:bayes`: Exponential(1) - 1 weights (Bayesian bootstrap)
"""
function _draw_weights(
        method::Symbol, n_rep_boot::Int, n_obs::Int, ::Type{T};
        rng::AbstractRNG = Random.default_rng()
    ) where {T <: AbstractFloat}
    if method == :normal
        return randn(rng, T, n_rep_boot, n_obs)
    elseif method == :wild
        xx = randn(rng, T, n_rep_boot, n_obs)
        yy = randn(rng, T, n_rep_boot, n_obs)
        return @. xx / sqrt(T(2)) + (yy^2 - 1) / 2
    elseif method == :bayes
        weights = rand(rng, T, n_rep_boot, n_obs)
        return @. -log(weights) - 1
    else
        throw(ArgumentError("Bootstrap method $method not supported. Use :normal, :wild, or :bayes."))
    end
end

"""
    multiplier_bootstrap(psi, psi_a, n_rep_boot=1000; method=:normal, rng)

Perform multiplier bootstrap for inference.

# Returns
Vector of bootstrap t-statistics.
"""
function multiplier_bootstrap(
        psi::AbstractVector{T}, psi_a::AbstractVector{T}, n_rep_boot::Int = 1000;
        method::Symbol = :normal, rng::AbstractRNG = Random.default_rng()
    ) where {T <: AbstractFloat}
    n_obs = length(psi)

    length(psi_a) != n_obs && throw(DimensionMismatch("psi and psi_a must have same length"))
    n_rep_boot < 1 && throw(ArgumentError("n_rep_boot must be >= 1"))

    mean_psi_a = mean(psi_a)
    abs(mean_psi_a) < eps(T) && throw(ArgumentError("mean(psi_a) is numerically zero."))

    weights = _draw_weights(method, n_rep_boot, n_obs, T; rng = rng)
    psi_scaled = psi ./ mean_psi_a

    boot_draws = Vector{T}(undef, n_rep_boot)

    @inbounds for b in eachindex(boot_draws)
        s = zero(T)
        @simd for i in eachindex(psi_scaled)
            s += weights[b, i] * psi_scaled[i]
        end
        boot_draws[b] = s / n_obs
    end

    return boot_draws
end

"""
    has_bootstrapped(obj) -> Bool

Check if bootstrap has been performed.
"""
has_bootstrapped(obj::AbstractDoubleML) = obj.has_bootstrapped

"""
    bootstrap!(obj; n_rep_boot=1000, method=:normal, rng)

Perform multiplier bootstrap on a fitted DoubleML model.

# Arguments
- `obj`: Fitted DoubleML model
- `n_rep_boot::Int=1000`: Number of bootstrap replications
- `method::Symbol=:normal`: Bootstrap method (:normal, :wild, or :bayes)
- `rng::AbstractRNG`: Random number generator
"""
function bootstrap!(
        obj::AbstractDoubleML{T};
        n_rep_boot::Int = 1000,
        method::Symbol = :normal,
        rng::AbstractRNG = Random.default_rng()
    ) where {T <: AbstractFloat}
    isnan(obj.coef) && throw(ArgumentError("Model must be fitted before calling bootstrap!"))

    boot_draws = multiplier_bootstrap(obj.psi, obj.psi_a, n_rep_boot; method = method, rng = rng)
    boot_t_stats = boot_draws ./ obj.se

    obj.boot_t_stat = reshape(boot_t_stats, n_rep_boot, 1)
    obj.has_bootstrapped = true
    obj.boot_method = method
    obj.n_rep_boot = n_rep_boot

    return obj
end

"""
    summary_stats(obj::AbstractDoubleML; level=0.95)

Compute summary statistics as a NamedTuple.
"""
function summary_stats(obj::AbstractDoubleML; level::Real = 0.95)
    ct = coeftable(obj; level = level)
    return (
        coef = ct.cols[1][1],
        se = ct.cols[2][1],
        t = ct.cols[3][1],
        p = ct.cols[4][1],
        ci_lower = ct.cols[5][1],
        ci_upper = ct.cols[6][1],
        level = level,
    )
end

"""
    summarize(obj::AbstractDoubleML; level=0.95)

Generate formatted summary table. Alias for coeftable.
"""
summarize(obj::AbstractDoubleML; level::Real = 0.95) = coeftable(obj; level = level)

const _HORIZONTAL_TABLE_FORMAT = TextTableFormat(
    horizontal_line_at_beginning = true,
    horizontal_line_after_column_labels = false,
    horizontal_line_after_data_rows = true,
    vertical_line_at_beginning = false,
    vertical_lines_at_data_columns = :none,
    vertical_line_after_data_columns = false
)

"""
    summary(obj::AbstractDoubleML; level=0.95)

Print a comprehensive summary of the DoubleML model.

Displays data summary, score function, learners with out-of-sample performance,
resampling details, and coefficient estimates with colored output.

# Arguments
- `obj`: Fitted DoubleML model
- `level::Real=0.95`: Confidence level for intervals

# Returns
- `nothing` (prints to stdout)
"""
function Base.summary(obj::AbstractDoubleML; level::Real = 0.95)
    model_type = typeof(obj).name.name

    println()
    printstyled("═"^20 * " $model_type " * "═"^20; color = :white, bold = true)
    println()

    _print_section_header("Data Summary", :blue)
    _print_kv_table(
        [
            "Outcome variable" => string(obj.data.y_col),
            "Treatment variable(s)" => string(obj.data.d_col),
            "Covariates" => join(obj.data.x_cols, ", "),
            "No. Observations" => string(obj.data.n_obs),
        ]
    )

    _print_section_header("Score & Algorithm", :green)
    _print_kv_table(
        [
            "Score function" => string(get_score_name(obj.score_obj)),
        ]
    )

    _print_section_header("Machine Learner", :magenta)
    _print_learners_table(obj)

    _print_section_header("Resampling", :yellow)
    _print_kv_table(
        [
            "No. folds" => string(obj.n_folds),
            "No. repeated sample splits" => string(obj.n_rep),
        ]
    )

    _print_section_header("Fit Summary", :cyan)
    if isfitted(obj)
        _print_coef_table(obj, level)
    else
        println("  Model not fitted")
    end

    println()
    return nothing
end

"""
    summary(obj::DoubleMLLPLR; level=0.95)

Print a comprehensive summary of the DoubleMLLPLR model.

Specialized version that displays n_folds_inner for the double cross-fitting
used in LPLR models.

# Arguments
- `obj`: Fitted DoubleMLLPLR model
- `level::Real=0.95`: Confidence level for intervals

# Returns
- `nothing` (prints to stdout)
"""
function Base.summary(obj::DoubleMLLPLR; level::Real = 0.95)
    println()
    printstyled("═"^20 * " DoubleMLLPLR " * "═"^20; color = :white, bold = true)
    println()

    _print_section_header("Data Summary", :blue)
    _print_kv_table(
        [
            "Outcome variable" => string(obj.data.y_col),
            "Treatment variable(s)" => string(obj.data.d_col),
            "Covariates" => join(obj.data.x_cols, ", "),
            "No. Observations" => string(obj.data.n_obs),
        ]
    )

    _print_section_header("Score & Algorithm", :green)
    _print_kv_table(
        [
            "Score function" => string(get_score_name(obj.score_obj)),
        ]
    )

    _print_section_header("Machine Learner", :magenta)
    _print_learners_table(obj)

    _print_section_header("Resampling", :yellow)
    _print_kv_table(
        [
            "No. outer folds" => string(obj.n_folds),
            "No. inner folds" => string(obj.n_folds_inner),
            "No. repeated sample splits" => string(obj.n_rep),
        ]
    )

    _print_section_header("Fit Summary", :cyan)
    if isfitted(obj)
        _print_coef_table(obj, level)
    else
        println("  Model not fitted")
    end

    println()
    return nothing
end

function _print_section_header(title::String, color::Symbol)
    println()
    printstyled("─"^18 * " $title " * "─"^18; color = color, bold = true)
    return println()
end

function _print_kv_table(pairs::Vector{<:Pair})
    n = length(pairs)
    data = Matrix{String}(undef, n, 2)
    for (i, p) in enumerate(pairs)
        data[i, 1] = p.first
        data[i, 2] = p.second
    end
    return pretty_table(
        data;
        show_column_labels = false,
        alignment = [:l, :l],
        table_format = _HORIZONTAL_TABLE_FORMAT
    )
end

function _print_learners_table(obj::DoubleMLPLR)
    learners = Pair{String, String}[]
    push!(learners, "Learner ml_l" => string(obj.ml_l))
    push!(learners, "Learner ml_m" => string(obj.ml_m))
    if obj.ml_g !== nothing
        push!(learners, "Learner ml_g" => string(obj.ml_g))
    end
    _print_kv_table(learners)

    return if !isempty(obj.learner_performance)
        println()
        printstyled("  Out-of-sample Performance:"; color = :white, bold = true)
        println()
        _print_performance_table(obj.learner_performance)
    end
end

function _print_learners_table(obj::DoubleMLIRM)
    learners = Pair{String, String}[]
    push!(learners, "Learner ml_g" => string(obj.ml_g))
    push!(learners, "Learner ml_m" => string(obj.ml_m))
    _print_kv_table(learners)

    return if !isempty(obj.learner_performance)
        println()
        printstyled("  Out-of-sample Performance:"; color = :white, bold = true)
        println()
        _print_performance_table(obj.learner_performance)
    end
end

function _print_learners_table(obj::DoubleMLLPLR)
    learners = Pair{String, String}[]
    push!(learners, "Learner ml_M" => string(obj.ml_M))
    push!(learners, "Learner ml_t" => string(obj.ml_t))
    push!(learners, "Learner ml_m" => string(obj.ml_m))
    push!(learners, "Learner ml_a" => string(obj.ml_a))
    _print_kv_table(learners)

    return if !isempty(obj.learner_performance)
        println()
        printstyled("  Out-of-sample Performance:"; color = :white, bold = true)
        println()
        _print_performance_table(obj.learner_performance)
    end
end

function _print_performance_table(perf::NamedTuple)
    n = length(perf)
    data = Matrix{String}(undef, n, 3)
    for (i, (name, p)) in enumerate(pairs(perf))
        data[i, 1] = string(name)
        data[i, 2] = p isa NamedTuple && haskey(p, :value) ?
            @sprintf("%.4f", p.value) : "N/A"
        data[i, 3] = p isa NamedTuple && haskey(p, :measure) ?
            string(p.measure) : "N/A"
    end
    return pretty_table(
        data;
        column_labels = ["Learner", "Value", "Measure"],
        alignment = [:l, :r, :l],
        show_column_labels = true,
        table_format = _HORIZONTAL_TABLE_FORMAT
    )
end

function _print_coef_table(obj::AbstractDoubleML, level::Real)
    ct = coeftable(obj; level = level)
    coef_val = ct.cols[1][1]
    se_val = ct.cols[2][1]
    t_val = ct.cols[3][1]
    p_val = ct.cols[4][1]
    ci_lower = ct.cols[5][1]
    ci_upper = ct.cols[6][1]

    alpha = 1.0 - level
    lower_pct = @sprintf("%.1f%%", alpha / 2 * 100)
    upper_pct = @sprintf("%.1f%%", (1 - alpha / 2) * 100)

    data = Matrix{String}(undef, 1, 7)
    data[1, 1] = ct.rownms[1]
    data[1, 2] = @sprintf("%.4f", coef_val)
    data[1, 3] = @sprintf("%.4f", se_val)
    data[1, 4] = @sprintf("%.2f", t_val)
    data[1, 5] = @sprintf("%.4f", p_val)
    data[1, 6] = @sprintf("%.4f", ci_lower)
    data[1, 7] = @sprintf("%.4f", ci_upper)

    column_labels = ["", "coef", "std err", "t", "P>|t|", lower_pct, upper_pct]

    return pretty_table(
        data;
        column_labels = column_labels,
        alignment = [:l, :r, :r, :r, :r, :r, :r],
        show_column_labels = true,
        table_format = _HORIZONTAL_TABLE_FORMAT
    )
end
