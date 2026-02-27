### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 66862f5f-b882-4344-96fd-56c6b5373e68
import Pkg; Pkg.develop(path=joinpath(@__DIR__, "../.."))

# ╔═╡ aedf55f4-51b3-4b85-b169-4524b241086d
Pkg.activate(joinpath(@__DIR__, "../../examples"))

# ╔═╡ 563e2e19-68b7-432f-93f4-b0ad015f8b33
using DoubleML; using StableRNGs; using MLJ; using TreeParzen; using MLJDecisionTreeInterface; using EvoTrees

# ╔═╡ d4e5f6a7-b8c9-0123-defa-456789012345
begin
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
end

# ╔═╡ e5f6a7b8-c9d0-1234-efab-567890123456
# PLR Data
data_plr = DoubleML.make_plr_CCDDHNR2018(5000, alpha = 0.5, dim_x = 20, rng = StableRNG(42))

# ╔═╡ f6a7b8c9-d0e1-2345-fabc-678901234567
# Find matching models
models() do model
    matching(model, data_plr.x, data_plr.y)
end

# ╔═╡ a7b8c9d0-e1f2-3456-abcd-789012345678
begin
    # Simple PLR with RandomForest
    ml_m = RandomForestRegressor()
    ml_g = RandomForestRegressor()

    dml_plr_simple = DoubleML.DoubleMLPLR(data_plr, ml_g, ml_m, n_folds = 4, n_rep = 1)

    fit!(dml_plr_simple)

    coeftable(dml_plr_simple)
end

# ╔═╡ 1089a55b-b9ec-40dc-9297-252527fd1c07
md"""
# Self-tuning models 
"""

# ╔═╡ b8c9d0e1-f2a3-4567-bcde-890123456789
begin
    # PLR with TreeParzen hyperparameter tuning

    # Set up the hyperparameter space
    space = Dict(
        :n_trees => HP.Choice(:n_trees, Float64.(10:700)),
        :max_depth => HP.Choice(:max_depth, Float64.(1:10)),
        :min_samples_leaf => HP.Choice(:min_samples_leaf, Float64.(1:15)),
        :min_purity_increase => HP.Choice(:min_purity_increase, Float64.(0:3)),
        :sampling_fraction => HP.Choice(:sampling_fraction, Float64.(0.6:0.99)),
        :feature_importance => HP.Choice(:feature_importance, [:impurity, :split]),
    )

    # Set up the self-tuning models
    tuned_ml_m = TunedModel(
        model = RandomForestRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 100, max_simultaneous_draws = 5, linear_forgetting = 50),
        resampling = CV(nfolds = 3),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_g = TunedModel(
        model = RandomForestRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 100, max_simultaneous_draws = 5, linear_forgetting = 50),
        resampling = CV(nfolds = 3),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    # Pass the self-tuning models as learners to the DoubleMLPLR constructor
    dml_plr = DoubleML.DoubleMLPLR(data_plr, tuned_ml_g, tuned_ml_m, n_folds = 4, n_rep = 1)

    # Fit it
    fit!(dml_plr, verbose = 0)
end

# ╔═╡ fb3c97b4-cc43-4606-8778-a8b15970bc6a
coeftable(dml_plr)

# ╔═╡ ab1bc710-c605-4132-9c23-3d82769975d6
md"""
# Iterated models
"""

# ╔═╡ 65f19e6b-cc02-445b-8df6-14685e06bb0c
# A simple example
# EvoTrees have in-built early stopping; the below is just for demonstration purposes.

begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(10),
        NumberLimit(30),
    ]

    # Set up learners with iteration control and early stopping
    ml_l_iterated = IteratedModel(
        EvoTreeRegressor(),
        resampling = Holdout(),
        measure = rmse,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_m_iterated = IteratedModel(
        EvoTreeRegressor(),
        resampling = Holdout(),
        measure = rmse,
        iteration_parameter = :nrounds,
        controls = controls
    )

    # Pass the learners to the DoulbleMLPLR contructor
    dml_plr_iterated = DoubleML.DoubleMLPLR(data_plr, ml_l_iterated, ml_m_iterated, n_folds = 4, n_rep = 1)

    # Fit it
    fit!(dml_plr_iterated, verbose = 1)
end


# ╔═╡ 7a164eca-64ca-407e-b412-a46119fe2c03
coeftable(dml_plr_iterated)

# ╔═╡ 86205d4c-2f1d-4daf-99a5-50c5e8b62f56
summary(dml_plr_iterated)

# ╔═╡ Cell order:
# ╠═66862f5f-b882-4344-96fd-56c6b5373e68
# ╠═aedf55f4-51b3-4b85-b169-4524b241086d
# ╠═563e2e19-68b7-432f-93f4-b0ad015f8b33
# ╠═d4e5f6a7-b8c9-0123-defa-456789012345
# ╠═e5f6a7b8-c9d0-1234-efab-567890123456
# ╠═f6a7b8c9-d0e1-2345-fabc-678901234567
# ╠═a7b8c9d0-e1f2-3456-abcd-789012345678
# ╠═1089a55b-b9ec-40dc-9297-252527fd1c07
# ╠═b8c9d0e1-f2a3-4567-bcde-890123456789
# ╠═fb3c97b4-cc43-4606-8778-a8b15970bc6a
# ╟─ab1bc710-c605-4132-9c23-3d82769975d6
# ╠═65f19e6b-cc02-445b-8df6-14685e06bb0c
# ╠═7a164eca-64ca-407e-b412-a46119fe2c03
# ╠═86205d4c-2f1d-4daf-99a5-50c5e8b62f56
