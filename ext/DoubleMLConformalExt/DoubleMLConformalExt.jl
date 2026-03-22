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

import DoubleML: AbstractDoubleML, DoubleMLData, PartiallingOutScore
import DoubleML: compute_score, dml2_solve, coerce_target
import MLJ: Supervised

export DoubleMLPLRConformal
export standard_dml_coef, standard_dml_se
export theta_samples, conformal_intervals

# Include the conformal model implementation
include("plr_conformal.jl")

end # module
