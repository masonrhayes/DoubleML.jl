"""
Score functions for Double Machine Learning.

This module provides score functions for different DML model types:
- `PartiallingOutScore`: For PLR with partialling out
- `IVTypeScore`: For PLR with IV-type estimation
- `ATEScore`: For IRM Average Treatment Effect
- `ATTEScore`: For IRM Average Treatment Effect on the Treated
- `NuisanceSpaceScore`: For LPLR using nuisance space approach
- `InstrumentScore`: For LPLR using instrument approach
"""

"""
    to_numeric(d::AbstractVector)

Convert treatment vector to numeric for arithmetic operations.
Type-stable via dispatch.
"""
to_numeric(d::AbstractVector{T}) where {T <: Number} = d

to_numeric(d::CategoricalVector{T}) where {T} = unwrap.(d)

to_numeric(d::AbstractVector) = float.(d)

"""
    get_score_name(score::AbstractScore) -> Symbol

Return the symbol name of the score type.
"""
get_score_name(::PartiallingOutScore) = :partialling_out
get_score_name(::IVTypeScore) = :IV_type
get_score_name(::ATEScore) = :ATE
get_score_name(::ATTEScore) = :ATTE

"""
    compute_score(score::AbstractScore, args...) -> Tuple{Vector{T}, Vector{T}}

Compute score components (psi_a, psi_b) for DML2 estimation.

The score function is linear in θ: ψ(W; θ, η) = ψ_a · θ + ψ_b

The DML2 estimator solves: θ̂ = -E[ψ_b] / E[ψ_a]
"""
function compute_score end

"""
    compute_score(::PartiallingOutScore, Y_test, D_test, l_hat, m_hat)

Compute partialling out score components.

# Returns
- `psi_a = -(D - m̂)^2`
- `psi_b = (Y - l̂) · (D - m̂)`
"""
function compute_score(
        ::PartiallingOutScore,
        Y_test::AbstractVector,
        D_test::AbstractVector,
        l_hat::AbstractVector,
        m_hat::AbstractVector
    )
    D_num = to_numeric(D_test)
    n = length(Y_test)
    T = eltype(Y_test)

    psi_a = Vector{T}(undef, n)
    psi_b = Vector{T}(undef, n)

    m_res = @. D_num - m_hat
    g_res = @. Y_test - l_hat
    @. psi_a = -m_res^2
    @. psi_b = g_res * m_res

    return psi_a, psi_b
end

"""
    compute_score(::IVTypeScore, Y_test, D_test, g_hat, m_hat)

Compute IV-type score components.

# Returns
- `psi_a = -(D - m̂) · D`
- `psi_b = (Y - ĝ) · (D - m̂)`
"""
function compute_score(
        ::IVTypeScore,
        Y_test::AbstractVector,
        D_test::AbstractVector,
        g_hat::AbstractVector,
        m_hat::AbstractVector
    )
    D_num = to_numeric(D_test)
    n = length(Y_test)
    T = eltype(Y_test)

    psi_a = Vector{T}(undef, n)
    psi_b = Vector{T}(undef, n)

    m_res = @. D_num - m_hat
    g_res = @. Y_test - g_hat
    @. psi_a = -m_res * D_num
    @. psi_b = g_res * m_res

    return psi_a, psi_b
end

"""
    compute_score(::ATEScore, Y_test, D_test, g_hat0, g_hat1, m_hat_adj)

Compute ATE score components using doubly robust AIPW estimator.

# Returns
- `psi_a = -1` (constant)
- `psi_b = τ̂(X) + IPW_correction`
"""
function compute_score(
        ::ATEScore,
        Y_test::AbstractVector,
        D_test::AbstractVector,
        g_hat0::AbstractVector,
        g_hat1::AbstractVector,
        m_hat_adj::AbstractVector
    )
    n = length(Y_test)
    T = eltype(Y_test)
    D_num = to_numeric(D_test)

    psi_a = Vector{T}(undef, n)
    psi_b = Vector{T}(undef, n)

    @. psi_a = -one(T)

    u_hat0 = @. Y_test - g_hat0
    u_hat1 = @. Y_test - g_hat1
    tau_hat = @. g_hat1 - g_hat0
    ipw_correction = @. D_num * u_hat1 / m_hat_adj - (one(T) - D_num) * u_hat0 / (one(T) - m_hat_adj)
    @. psi_b = tau_hat + ipw_correction

    return psi_a, psi_b
end

"""
    compute_score(::ATTEScore, Y_test, D_test, g_hat0, g_hat1, m_hat_adj, E_D_global)

Compute ATTE score components focusing on treated population.

# Returns
- `psi_a = -D / E[D]`
- `psi_b = (D/E[D])·τ̂ + (m̂/E[D])·IPW_correction`
"""
function compute_score(
        ::ATTEScore,
        Y_test::AbstractVector,
        D_test::AbstractVector,
        g_hat0::AbstractVector,
        g_hat1::AbstractVector,
        m_hat_adj::AbstractVector,
        E_D_global::Real
    )
    n = length(Y_test)
    T = eltype(Y_test)
    D_num = to_numeric(D_test)

    psi_a = Vector{T}(undef, n)
    psi_b = Vector{T}(undef, n)

    inv_E_D = T(1.0 / E_D_global)

    weights = @. D_num * inv_E_D
    weights_bar = @. m_hat_adj * inv_E_D

    u_hat0 = @. Y_test - g_hat0
    u_hat1 = @. Y_test - g_hat1
    tau_hat = @. g_hat1 - g_hat0

    ipw_correction = @. D_num * u_hat1 / m_hat_adj - (one(T) - D_num) * u_hat0 / (one(T) - m_hat_adj)

    @. psi_b = weights * tau_hat + weights_bar * ipw_correction
    @. psi_a = -weights

    return psi_a, psi_b
end

"""
    dml2_solve(psi_a::AbstractVector, psi_b::AbstractVector)

Solve for θ using DML2 estimator: θ̂ = -mean(psi_b) / mean(psi_a)
"""
function dml2_solve(psi_a::AbstractVector, psi_b::AbstractVector)
    return -mean(psi_b) / mean(psi_a)
end

"""
    NuisanceSpaceScore <: AbstractScore

Score function for LPLR using nuisance space estimation.

Computes: ψ(W, β, η) = ψ(X){Ye^(βD) - (1-Y)e^(r₀(X))}{D - m₀(X)}
where ψ(X) = expit(-r₀(X))

# References
- Liu et al. (2021): Double/debiased machine learning for logistic partially linear models
"""
struct NuisanceSpaceScore <: AbstractScore end

"""
    InstrumentScore <: AbstractScore

Score function for LPLR using instrument approach.

Computes: ψ(W; β, η) = {Y - expit(β₀D + r₀(X))}(D - m(X))

# References
- Liu et al. (2021): Double/debiased machine learning for logistic partially linear models
"""
struct InstrumentScore <: AbstractScore end

get_score_name(::NuisanceSpaceScore) = :nuisance_space
get_score_name(::InstrumentScore) = :instrument

"""
    compute_score_elements(::NuisanceSpaceScore, Y, D, t_hat, a_hat, m_hat)

Compute score elements for nuisance_space score with dynamic r_hat computation.

Returns NamedTuple with fields needed for score computation and root-finding:
- y: outcome vector
- d: treatment vector
- d_tilde: D - m_hat (residualized treatment)
- t_hat: estimated t(X) = E[logit(M) | X]
- a_hat: estimated a(X) = E[D | X]
- m_hat: estimated m(X) = E[D | X, Y=0]

Note: r_hat = t_hat - coef * a_hat is computed dynamically in compute_score
"""
function compute_score_elements(
        ::NuisanceSpaceScore,
        Y::AbstractVector{T},
        D::AbstractVector,
        t_hat::AbstractVector{T},
        a_hat::AbstractVector{T},
        m_hat::AbstractVector{T}
    ) where {T}
    d_tilde = D .- m_hat

    return (
        y = Y,
        d = D,
        d_tilde = d_tilde,
        t_hat = t_hat,
        a_hat = a_hat,
        m_hat = m_hat,
    )
end

"""
    compute_score_elements(::InstrumentScore, Y, D, t_hat, a_hat, m_hat)

Compute score elements for instrument score with dynamic r_hat computation.

Returns NamedTuple with fields needed for score computation and root-finding:
- y: outcome vector
- d: treatment vector
- d_tilde: D - m_hat (residualized treatment)
- t_hat: estimated t(X) = E[logit(M) | X]
- a_hat: estimated a(X) = E[D | X]
- m_hat: estimated m(X) = E[D | X]

Note: r_hat = t_hat - coef * a_hat is computed dynamically in compute_score
"""
function compute_score_elements(
        ::InstrumentScore,
        Y::AbstractVector{T},
        D::AbstractVector,
        t_hat::AbstractVector{T},
        a_hat::AbstractVector{T},
        m_hat::AbstractVector{T}
    ) where {T}
    d_tilde = D .- m_hat

    return (
        y = Y,
        d = D,
        d_tilde = d_tilde,
        t_hat = t_hat,
        a_hat = a_hat,
        m_hat = m_hat,
    )
end

"""
    compute_score(::NuisanceSpaceScore, coef, score_elements)

Compute score value for given coefficient using dynamic r_hat computation.

Score: ψ = ψ_hat * (Y * exp(-coef * D) * d_tilde - score_const)
where r_hat = t_hat - coef * a_hat (computed dynamically)
"""
function compute_score(
        ::NuisanceSpaceScore,
        coef::Real,
        score_elements::NamedTuple
    )
    # Compute r_hat dynamically: r_hat = t_hat - coef * a_hat
    r_hat = @. score_elements.t_hat - coef * score_elements.a_hat

    # Compute psi_hat and score_const from dynamic r_hat
    psi_hat = logistic.(-r_hat)  # expit(-r_hat)
    score_const = @. score_elements.d_tilde * (1 - score_elements.y) * exp(r_hat)

    # Compute score
    score_1 = @. score_elements.y * exp(-coef * score_elements.d) * score_elements.d_tilde
    return @. psi_hat * (score_1 - score_const)
end

"""
    compute_score(::InstrumentScore, coef, score_elements)

Compute score value for given coefficient using dynamic r_hat computation.

Score: ψ = (Y - expit(coef * D + r_hat)) * d_tilde
where r_hat = t_hat - coef * a_hat (computed dynamically)
"""
function compute_score(
        ::InstrumentScore,
        coef::Real,
        score_elements::NamedTuple
    )
    # Compute r_hat dynamically: r_hat = t_hat - coef * a_hat
    r_hat = @. score_elements.t_hat - coef * score_elements.a_hat

    # Compute expit(coef * D + r_hat)
    expit_val = logistic.(coef .* score_elements.d .+ r_hat)
    return @. (score_elements.y - expit_val) * score_elements.d_tilde
end

"""
    compute_score_deriv(::NuisanceSpaceScore, coef, score_elements)

Compute derivative of score with respect to coefficient using dynamic r_hat.

The derivative accounts for both explicit coef dependence and implicit through r_hat.
"""
function compute_score_deriv(
        ::NuisanceSpaceScore,
        coef::Real,
        score_elements::NamedTuple
    )
    # Compute r_hat dynamically
    r_hat = @. score_elements.t_hat - coef * score_elements.a_hat

    # psi_hat = expit(-r_hat)
    psi_hat = logistic.(-r_hat)

    # Derivative with respect to coef (accounting for r_hat dependence)
    # d(psi_hat)/d(coef) = psi_hat * (1-psi_hat) * a_hat  (chain rule through r_hat)
    # Full derivative is complex; for Newton method we use implicit derivative

    deriv_1 = @. -score_elements.y * score_elements.d * exp(-coef * score_elements.d)
    return @. psi_hat * score_elements.d_tilde * deriv_1
end

"""
    compute_score_deriv(::InstrumentScore, coef, score_elements)

Compute derivative of score with respect to coefficient using dynamic r_hat.
"""
function compute_score_deriv(
        ::InstrumentScore,
        coef::Real,
        score_elements::NamedTuple
    )
    # Compute r_hat dynamically
    r_hat = @. score_elements.t_hat - coef * score_elements.a_hat

    expit_val = logistic.(coef .* score_elements.d .+ r_hat)
    return @. -score_elements.d * expit_val * (1 - expit_val) * score_elements.d_tilde
end

"""
    compute_mean_score(score_obj, coef, all_score_elements)

Compute mean score across all observations.
"""
function compute_mean_score(
        score_obj::AbstractScore,
        coef::Real,
        all_score_elements::Vector{<:NamedTuple}
    )
    total_score = 0.0
    n_total = 0

    for elements in all_score_elements
        score = compute_score(score_obj, coef, elements)
        total_score += sum(score)
        n_total += length(score)
    end

    return total_score / n_total
end
