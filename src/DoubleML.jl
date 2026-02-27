"""
    DoubleML

Double/Debiased Machine Learning for Julia.

Provides estimators for causal parameters in models with high-dimensional nuisance functions.

# Main Types
- `DoubleMLData`: Data container
- `DoubleMLPLR`: Partially Linear Regression model
- `DoubleMLIRM`: Interactive Regression Model

# Example
```julia
using DoubleML, MLJ, DataFrames

# Create data
data = DoubleMLData(df; y_col=:y, d_col=:d, x_cols=[:x1, :x2])

# Load ML models
ml_l = @load RandomForestRegressor pkg=DecisionTree
ml_m = @load RandomForestRegressor pkg=DecisionTree

# Create and fit model
model = DoubleMLPLR(data, ml_l, ml_m)
fit!(model)

# Get results
coef(model)
stderror(model)
confint(model)
```
"""
module DoubleML

using Reexport
using MLJ
using MLJBase
using StatsAPI
using StatsBase
using Distributions
using DataFrames
using CategoricalArrays
using ScientificTypes
using Random
using LinearAlgebra
using Statistics
using Printf
using PrettyTables
using StatsFuns: logistic, logit
using Roots: find_zero, Order0, Newton, AlefeldPotraShi

import Base: summary

# Types and structures
include("types.jl")

# Core functionality
include("data.jl")
include("scores.jl")
include("utils.jl")

# Fitting interface
include("fitting/abstract.jl")
include("fitting/plr.jl")
include("fitting/irm.jl")
include("fitting/lplr.jl")

# Inference
include("inference.jl")

# Evaluation and reporting
include("evaluation.jl")

# Data generators
include("datasets/dgp/plr_CCDDHNR2018.jl")
include("datasets/dgp/irm_data.jl")
include("datasets/dgp/lplr_LZZ2020.jl")

# ============================================================================
# Exports
# ============================================================================

# Data types
export DoubleMLData, check_binary, dtype

# Model types
export DoubleMLPLR, DoubleMLIRM, DoubleMLLPLR
export AbstractDoubleML, AbstractScore

# Score types
export PartiallingOutScore, IVTypeScore
export ATEScore, ATTEScore
export NuisanceSpaceScore, InstrumentScore

# Core functions
export fit!, bootstrap!, has_bootstrapped
export compute_score, compute_score_elements, compute_score_deriv, get_score_name, dml2_solve, to_numeric
export draw_sample_splitting, get_conditional_sample_splitting

# Learner accessors
export learner_l, learner_m, learner_g, learner_M, learner_t, learner_a

# StatsAPI methods
export coef, stderror, vcov, confint, nobs, dof, dof_residual, islinear, isfitted
export responsename, coefnames, coeftable

# Inference
export summary_stats, multiplier_bootstrap

# Evaluation
export fitted_params, report, evaluate!

# Data generators
export make_plr_CCDDHNR2018, make_irm_data, make_lplr_LZZ2020

end # module
