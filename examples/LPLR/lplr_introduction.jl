### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ 8ddab706-13ec-11f1-86d9-cf1e4a214f61
begin
    using Pkg
    Pkg.activate("..")
    Pkg.resolve()
    Pkg.instantiate()
end

# ╔═╡ 5a072569-e9a6-4634-bec5-00b751791c7d
begin
    using DoubleML
    using StableRNGs
    using MLJ
    using TreeParzen
    using EvoTrees
end

# ╔═╡ dd0872f0-affc-4718-b5ab-7f8387f237c2
begin
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees verbosity = 0
    RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree verbosity = 0
end

# ╔═╡ 2256cb84-ad2b-48af-beea-a50dce7fdcf4
md"""
## Generate LPLR data
"""

# ╔═╡ 53c84214-cc5d-4894-a612-e6894dca814c

data_lplr = make_lplr_LZZ2020(1000, alpha = 0.5, rng = StableRNG(42))

# ╔═╡ 16ff5227-f59d-4c78-ba0e-80186bf19c06
md"""
Find models that match the data we have:
"""

# ╔═╡ 61655243-5256-4021-9931-a638c30203d8
begin
    # Find matching models for y
    models() do model
        matching(model, data_lplr.x, data_lplr.y)
    end
end

# ╔═╡ cfb01698-a12d-4ff8-808e-7e04d65882b7
begin
    # Find matching models for y
    models() do model
        matching(model, data_lplr.x, data_lplr.d)
    end
end

# ╔═╡ 90943f72-2c68-493c-8903-21e2caf07b79
md"""
## Estimate a simple model
"""

# ╔═╡ c6e4e115-c2db-4bfe-b8fc-c6cb73e60d0d
begin
    # Simple IRM with RandomForest
    ml_M = RandomForestClassifier()
    ml_t = RandomForestRegressor()
    ml_m = RandomForestRegressor()

    dml_lplr_simple = DoubleML.DoubleMLLPLR(data_lplr, ml_M, ml_t, ml_m, score = :nuisance_space)

    fit!(dml_lplr_simple)

end

# ╔═╡ 3a228ad5-78a0-4c7e-9d85-3b37ce0ae646
coeftable(dml_lplr_simple)

# ╔═╡ d1ccecf6-c53c-4da7-8d3b-f6392a8ff6eb
md"""
## Estimate a more complex model with iteration control
"""

# ╔═╡ 584302b8-1735-412a-a1a3-6bd180310efa
begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(10),
        NumberLimit(20),
    ]

    ml_M_iterated = IteratedModel(
        EvoTreeClassifier(max_depth = 4, eta = 0.05),
        resampling = Holdout(),
        measure = cross_entropy,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_t_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.05),
        resampling = Holdout(),
        measure = mav,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_m_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.05),
        resampling = Holdout(),
        measure = mae,
        iteration_parameter = :nrounds,
        controls = controls
    )

    # Set up the model
    dml_lplr_iterated = DoubleML.DoubleMLLPLR(data_lplr, ml_M_iterated, ml_t_iterated, ml_m_iterated, score = :nuisance_space)

    # Fit the model
    fit!(dml_lplr_iterated)


end

# ╔═╡ 47f51991-0a11-4920-a4fb-aca0a55b5ce9
coeftable(dml_lplr_iterated)

# ╔═╡ Cell order:
# ╠═8ddab706-13ec-11f1-86d9-cf1e4a214f61
# ╠═5a072569-e9a6-4634-bec5-00b751791c7d
# ╠═dd0872f0-affc-4718-b5ab-7f8387f237c2
# ╟─2256cb84-ad2b-48af-beea-a50dce7fdcf4
# ╠═53c84214-cc5d-4894-a612-e6894dca814c
# ╟─16ff5227-f59d-4c78-ba0e-80186bf19c06
# ╠═61655243-5256-4021-9931-a638c30203d8
# ╠═cfb01698-a12d-4ff8-808e-7e04d65882b7
# ╟─90943f72-2c68-493c-8903-21e2caf07b79
# ╠═c6e4e115-c2db-4bfe-b8fc-c6cb73e60d0d
# ╠═3a228ad5-78a0-4c7e-9d85-3b37ce0ae646
# ╠═d1ccecf6-c53c-4da7-8d3b-f6392a8ff6eb
# ╠═584302b8-1735-412a-a1a3-6bd180310efa
# ╠═47f51991-0a11-4920-a4fb-aca0a55b5ce9
