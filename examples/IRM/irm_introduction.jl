### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-e5f6-7890-abcd-ef1234567890
begin
    import Pkg; Pkg.develop(path=joinpath(@__DIR__, "../.."))
end

# ╔═╡ c3d4e5f6-a7b8-9012-cdef-345678901234
Pkg.activate(joinpath(@__DIR__, "../../examples"))

# ╔═╡ 0a9939fa-ff7a-466d-b1a0-9077bdfe41b2
begin
    using DoubleML
    using StableRNGs
    using MLJ
    using TreeParzen
    using EvoTrees
end

# ╔═╡ d4e5f6a7-b8c9-0123-defa-456789012345
begin
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees verbosity = 0
    RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree verbosity = 0
end

# ╔═╡ e5f6a7b8-c9d0-1234-efab-567890123456
# IRM Data
data_irm = DoubleML.make_irm_data(10000, theta = 0.5, dim_x = 100, rng = StableRNG(42))

# ╔═╡ f6a7b8c9-d0e1-2345-fabc-678901234567
begin
    # Find matching models for y
    models() do model
        matching(model, data_irm.x, data_irm.y)
    end
end

# ╔═╡ a7b8c9d0-e1f2-3456-abcd-789012345678
begin
    # Find matching models for d
    models() do model
        matching(model, data_irm.x, data_irm.d)
    end
end

# ╔═╡ b8c9d0e1-f2a3-4567-bcde-890123456789
begin
    # Simple IRM with RandomForest
    ml_g = RandomForestRegressor()
    ml_m = RandomForestClassifier()

    dml_irm_simple = DoubleML.DoubleMLIRM(data_irm, ml_g, ml_m, score = :ATE)

    fit!(dml_irm_simple)

    coeftable(dml_irm_simple)
end

# ╔═╡ c9d0e1f2-a3b4-5678-cdef-901234567890
begin
    # IRM with TreeParzen hyperparameter tuning

    space = Dict(
        :max_depth => HP.QuantUniform(:max_depth, 2.0, 12.0, 1.0)
    )

    tuned_ml_g = TunedModel(
        model = EvoTreeRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 75, draws = 50),
        resampling = Holdout(fraction_train = 0.8),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_m = TunedModel(
        model = EvoTreeClassifier(),
        tuning = MLJTreeParzenTuning(random_trials = 75, draws = 50),
        resampling = Holdout(fraction_train = 0.8),
        range = space,
        measure = MLJ.brier_score,
        acceleration = CPUProcesses(),
    )


    dml_irm = DoubleML.DoubleMLIRM(data_irm, tuned_ml_g, tuned_ml_m)

    fit!(dml_irm, verbose = 1)

end

# ╔═╡ c8a0594d-c954-4106-bd78-41fe358d4535
coeftable(dml_irm)

# ╔═╡ Cell order:
# ╠═a1b2c3d4-e5f6-7890-abcd-ef1234567890
# ╠═c3d4e5f6-a7b8-9012-cdef-345678901234
# ╠═0a9939fa-ff7a-466d-b1a0-9077bdfe41b2
# ╠═d4e5f6a7-b8c9-0123-defa-456789012345
# ╠═e5f6a7b8-c9d0-1234-efab-567890123456
# ╠═f6a7b8c9-d0e1-2345-fabc-678901234567
# ╠═a7b8c9d0-e1f2-3456-abcd-789012345678
# ╠═b8c9d0e1-f2a3-4567-bcde-890123456789
# ╠═c9d0e1f2-a3b4-5678-cdef-901234567890
# ╠═c8a0594d-c954-4106-bd78-41fe358d4535
