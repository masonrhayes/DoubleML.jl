"""
    DoubleMLData{T<:AbstractFloat, D<:AbstractVector}

Container for data used in Double Machine Learning models.

Stores outcome variable, treatment variable, and covariates. The outcome (y) and 
covariates (x) are stored as Float32 by default for ML performance.

# Type Parameters
- `T<:AbstractFloat`: Numeric type for y and x (Float32 or Float64)
- `D<:AbstractVector`: Type for treatment d (preserves CategoricalVector if applicable)

# Fields
- `y::Vector{T}`: Outcome variable
- `d::D`: Treatment variable (preserves original type for MLJ compatibility)
- `x::Matrix{T}`: Covariate matrix (n_obs × dim_x)
- `n_obs::Int`: Number of observations
- `dim_x::Int`: Number of covariates
- `y_col::Symbol`: Name of outcome variable
- `d_col::Symbol`: Name of treatment variable
- `x_cols::Vector{Symbol}`: Names of covariate columns
"""
struct DoubleMLData{T <: AbstractFloat, D <: AbstractVector}
    y::Vector{T}
    d::D
    x::Matrix{T}
    n_obs::Int
    dim_x::Int
    y_col::Symbol
    d_col::Symbol
    x_cols::Vector{Symbol}
end

"""
    AbstractScore

Abstract base type for score functions in DoubleML models.

Score functions define how the DML estimator computes the estimating equations
for different model types and estimands.
"""
abstract type AbstractScore end

"""
    AbstractBootstrapMethod

Abstract supertype for bootstrap methods in DoubleML inference.

Supported concrete subtypes:
- `NormalBootstrap`: Standard Gaussian N(0,1) bootstrap
- `WildBootstrap`: Mammen's wild bootstrap for heteroskedastic data
- `BayesBootstrap`: Bayesian bootstrap with Exponential(1) - 1 weights

Use these types with `bootstrap!` for type-stable dispatch, or pass a Symbol
(`:normal`, `:wild`, `:Bayes`) for backward compatibility.
"""
abstract type AbstractBootstrapMethod end

"""
    NormalBootstrap <: AbstractBootstrapMethod

Standard Gaussian multiplier bootstrap.

Draws bootstrap weights from N(0, 1), suitable for homoskedastic settings.
"""
struct NormalBootstrap <: AbstractBootstrapMethod end

"""
    WildBootstrap <: AbstractBootstrapMethod

Mammen's wild bootstrap for heteroskedastic data.

Uses a two-point distribution: ξ ∼ X/√2 + (X² - 1)/2 where X ∼ N(0,1).
Provides better coverage under heteroskedasticity.
"""
struct WildBootstrap <: AbstractBootstrapMethod end

"""
    BayesBootstrap <: AbstractBootstrapMethod

Bayesian bootstrap with Exponential weights.

Draws bootstrap weights as Exp(1) - 1, which corresponds to a Dirichlet(1,...,1)
prior on the probability weights.
"""
struct BayesBootstrap <: AbstractBootstrapMethod end

"""
    AbstractDoubleML{T<:AbstractFloat}

Abstract base type for Double Machine Learning models.

# Type Parameter
- `T<:AbstractFloat`: Numeric type for all computations (inferred from data)
"""
abstract type AbstractDoubleML{T <: AbstractFloat} <: StatsAPI.StatisticalModel end

"""
    PartiallingOutScore <: AbstractScore

Score function for partialling out in Partially Linear Regression.

The score is: ``\\psi(W; \\theta, \\eta) = (Y - l(X) - \\theta(D - m(X))) \\cdot (D - m(X))``
"""
struct PartiallingOutScore <: AbstractScore end

"""
    IVTypeScore <: AbstractScore

IV-type score for Partially Linear Regression.

The score is: ``\\psi(W; \\theta, \\eta) = (Y - g(X) - \\theta \\cdot D) \\cdot (D - m(X))``
"""
struct IVTypeScore <: AbstractScore end

"""
    ATEScore <: AbstractScore

Score function for Average Treatment Effect in Interactive Regression Models.

Uses doubly robust AIPW estimator.
"""
struct ATEScore <: AbstractScore end

"""
    ATTEScore <: AbstractScore

Score function for Average Treatment Effect on the Treated.

Focuses on treated population only.
"""
struct ATTEScore <: AbstractScore end

"""
    DoubleMLPLR{T<:AbstractFloat, L<:Supervised, M<:Supervised, G} <: AbstractDoubleML{T}

Double Machine Learning for Partially Linear Regression models.

Implements: Y = ``\\theta``·D + g(X) + ε

# Type Parameters
- `T<:AbstractFloat`: Numeric type (inferred from data)
- `L<:Supervised`: Type of ml_l learner
- `M<:Supervised`: Type of ml_m learner  
- `G`: Type of ml_g learner (Union{Supervised, Nothing})

# Fields
- `data::DoubleMLData{T}`: Data container
- `ml_l::L`: Model for l(X) = E[Y|X]
- `ml_m::M`: Model for m(X) = E[D|X]
- `ml_g::G`: Model for g(X) = E[Y - D·``\\theta``|X] (IV-type only)
- `n_folds::Int`: Number of cross-fitting folds
- `n_rep::Int`: Number of sample splitting repetitions
- `score_obj::AbstractScore`: Score function type
- `n_folds_tune::Int`: Folds for tuning (0 = full sample)
- `coef::T`: Estimated treatment effect
- `se::T`: Standard error
- `all_coef::Vector{T}`: Coefficient estimates for each repetition
- `all_se::Vector{T}`: Standard errors for each repetition
- `all_psi::Matrix{T}`: Influence function values (n_obs × n_rep)
- `all_psi_a::Matrix{T}`: Score coefficient component (n_obs × n_rep)
- `all_psi_b::Matrix{T}`: Score constant component (n_obs × n_rep)
- `has_bootstrapped::Bool`: Whether bootstrap has been performed
- `boot_t_stat::Array{T, 3}`: Bootstrap t-statistics (n_rep_boot × n_coefs × n_rep)
- `boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}`: Bootstrap method used
- `n_rep_boot::Int`: Number of bootstrap replications
- `fitted_learners_l::Vector{MLJ.Machine}`: Fitted machines for ml_l
- `fitted_learners_m::Vector{MLJ.Machine}`: Fitted machines for ml_m
- `fitted_learners_g::Vector{MLJ.Machine}`: Fitted machines for ml_g
- `learner_performance::NamedTuple`: Performance metrics
"""
mutable struct DoubleMLPLR{T <: AbstractFloat, L <: Supervised, M <: Supervised, G} <:
    AbstractDoubleML{T}
    data::DoubleMLData{T}
    ml_l::L
    ml_m::M
    ml_g::G
    n_folds::Int
    n_rep::Int
    score_obj::AbstractScore
    n_folds_tune::Int

    coef::T
    se::T
    all_coef::Vector{T}
    all_se::Vector{T}
    all_psi::Matrix{T}
    all_psi_a::Matrix{T}
    all_psi_b::Matrix{T}

    has_bootstrapped::Bool
    boot_t_stat::Array{T, 3}
    boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}
    n_rep_boot::Int

    fitted_learners_l::Vector{MLJ.Machine}
    fitted_learners_m::Vector{MLJ.Machine}
    fitted_learners_g::Vector{MLJ.Machine}

    learner_performance::NamedTuple
end

"""
    DoubleMLIRM{T<:AbstractFloat, G<:Supervised, M<:Supervised} <: AbstractDoubleML{T}

Double Machine Learning for Interactive Regression Models.

Implements: Y = g_0(D, X) + ``\\zeta``, where D is binary.

# Type Parameters
- `T<:AbstractFloat`: Numeric type (inferred from data)
- `G<:Supervised`: Type of ml_g learner
- `M<:Supervised`: Type of ml_m learner

# Fields
- `data::DoubleMLData{T}`: Data container
- `ml_g::G`: Model for g(X, D) = E[Y|X, D]
- `ml_m::M`: Model for m(X) = E[D|X] (propensity score)
- `n_folds::Int`: Number of cross-fitting folds
- `n_rep::Int`: Number of sample splitting repetitions
- `score_obj::AbstractScore`: Score function type (ATEScore or ATTEScore)
- `normalize_ipw::Bool`: Whether to normalize IPW weights
- `clipping_threshold::T`: Threshold for propensity score clipping
- `n_folds_tune::Int`: Folds for tuning (0 = full sample)
- `coef::T`: Estimated treatment effect
- `se::T`: Standard error
- `all_coef::Vector{T}`: Coefficient estimates for each repetition
- `all_se::Vector{T}`: Standard errors for each repetition
- `all_psi::Matrix{T}`: Influence function values (n_obs × n_rep)
- `all_psi_a::Matrix{T}`: Score coefficient component (n_obs × n_rep)
- `all_psi_b::Matrix{T}`: Score constant component (n_obs × n_rep)
- `has_bootstrapped::Bool`: Whether bootstrap has been performed
- `boot_t_stat::Array{T, 3}`: Bootstrap t-statistics (n_rep_boot × n_coefs × n_rep)
- `boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}`: Bootstrap method used
- `n_rep_boot::Int`: Number of bootstrap replications
- `fitted_learners_g0::Vector{MLJ.Machine}`: Fitted machines for control group (D=0)
- `fitted_learners_g1::Vector{MLJ.Machine}`: Fitted machines for treated group (D=1)
- `fitted_learners_m::Vector{MLJ.Machine}`: Fitted machines for propensity score
- `learner_performance::NamedTuple`: Performance metrics
"""
mutable struct DoubleMLIRM{T <: AbstractFloat, G <: Supervised, M <: Supervised} <:
    AbstractDoubleML{T}
    data::DoubleMLData{T}
    ml_g::G
    ml_m::M
    n_folds::Int
    n_rep::Int
    score_obj::AbstractScore
    normalize_ipw::Bool
    clipping_threshold::T
    n_folds_tune::Int

    coef::T
    se::T
    all_coef::Vector{T}
    all_se::Vector{T}
    all_psi::Matrix{T}
    all_psi_a::Matrix{T}
    all_psi_b::Matrix{T}

    has_bootstrapped::Bool
    boot_t_stat::Array{T, 3}
    boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}
    n_rep_boot::Int

    fitted_learners_g0::Vector{MLJ.Machine}
    fitted_learners_g1::Vector{MLJ.Machine}
    fitted_learners_m::Vector{MLJ.Machine}

    learner_performance::NamedTuple
end

"""
    DoubleMLLPLR{T<:AbstractFloat, M<:Supervised, Tt<:Supervised,
                 Mm<:Supervised, Ma<:Supervised} <: AbstractDoubleML{T}

Double Machine Learning for Logistic Partially Linear Regression.

Implements: E[Y | D, X] = expit{β_0·D + r_0(X)} where Y ∈ {0, 1}

# Type Parameters
- `T<:AbstractFloat`: Numeric type (inferred from data)
- `M<:Supervised`: Type of ml_M learner (probabilistic classifier)
- `Tt<:Supervised`: Type of ml_t learner (regressor)
- `Mm<:Supervised`: Type of ml_m learner
- `Ma<:Supervised`: Type of ml_a learner

# Fields
- `data::DoubleMLData{T}`: Data container
- `ml_M::M`: Model for M(D,X) = P(Y=1 | D, X) - probabilistic classifier
- `ml_t::Tt`: Model for t(X) = E[logit(M(D,X)) | X]
- `ml_m::Mm`: Model for m(X) = E[D | X, Y=0] (nuisance_space) or E[D | X] (instrument)
- `ml_a::Ma`: Model for a(X) = E[D | X]
- `n_folds::Int`: Number of outer cross-fitting folds
- `n_folds_inner::Int`: Number of inner folds for preliminary estimation
- `n_rep::Int`: Number of sample splitting repetitions
- `score_obj::AbstractScore`: Score function type (NuisanceSpaceScore or InstrumentScore)
- `n_folds_tune::Int`: Folds for tuning (0 = full sample)
- `coef::T`: Estimated treatment effect (log-odds ratio)
- `se::T`: Standard error
- `all_coef::Vector{T}`: Coefficient estimates for each repetition
- `all_se::Vector{T}`: Standard errors for each repetition
- `coef_start_val::T`: Preliminary estimate used as starting value
- `all_psi::Matrix{T}`: Influence function values (n_obs × n_rep)
- `all_psi_a::Matrix{T}`: Derivative of score (n_obs × n_rep)
- `all_psi_b::Matrix{T}`: Score values at estimated coefficient (n_obs × n_rep)
- `has_bootstrapped::Bool`: Whether bootstrap has been performed
- `boot_t_stat::Array{T, 3}`: Bootstrap t-statistics (n_rep_boot × n_coefs × n_rep)
- `boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}`: Bootstrap method used
- `n_rep_boot::Int`: Number of bootstrap replications
- `fitted_learners_M::Vector{MLJ.Machine}`: Fitted machines for ml_M
- `fitted_learners_t::Vector{MLJ.Machine}`: Fitted machines for ml_t
- `fitted_learners_m::Vector{MLJ.Machine}`: Fitted machines for ml_m
- `fitted_learners_a::Vector{MLJ.Machine}`: Fitted machines for ml_a
- `learner_performance::NamedTuple`: Performance metrics

# References
- Liu et al. (2021): Double/debiased machine learning for logistic partially linear models
  https://doi.org/10.1093/ectj/utab019
"""
mutable struct DoubleMLLPLR{
        T <: AbstractFloat, M <: Supervised, Tt <: Supervised,
        Mm <: Supervised, Ma <: Supervised,
    } <: AbstractDoubleML{T}
    data::DoubleMLData{T}
    ml_M::M
    ml_t::Tt
    ml_m::Mm
    ml_a::Ma
    n_folds::Int
    n_folds_inner::Int
    n_rep::Int
    score_obj::AbstractScore
    n_folds_tune::Int

    coef::T
    se::T
    all_coef::Vector{T}
    all_se::Vector{T}
    coef_start_val::T
    all_psi::Matrix{T}
    all_psi_a::Matrix{T}
    all_psi_b::Matrix{T}

    has_bootstrapped::Bool
    boot_t_stat::Array{T, 3}
    boot_method::Union{AbstractBootstrapMethod, Symbol, Nothing}
    n_rep_boot::Int

    fitted_learners_M::Vector{MLJ.Machine}
    fitted_learners_t::Vector{MLJ.Machine}
    fitted_learners_m::Vector{MLJ.Machine}
    fitted_learners_a::Vector{MLJ.Machine}

    learner_performance::NamedTuple
end
