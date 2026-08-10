module DoubleMLConformalExt

using DoubleML
using MLJ
using MLJBase
using StatsAPI
using Statistics
using Random
using DataFrames
using ConformalPrediction
using Distributions
using StatsBase
using StableRNGs
using LinearAlgebra

import DoubleML: AbstractDoubleML, DoubleMLData, PartiallingOutScore
import DoubleML: compute_score, dml2_solve, coerce_target
import MLJ: Supervised

export DoubleMLPLRConformal, DoubleMLPLRConformalUT
export standard_dml_coef, standard_dml_se
export theta_samples, conformal_intervals

# Include the conformal model implementations
include("plr_conformal.jl")
include("plr_conformal_ut.jl")

end # module
