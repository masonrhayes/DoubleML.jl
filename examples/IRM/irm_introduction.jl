### A Pluto.jl notebook ###
# v0.20.23

using Markdown
using InteractiveUtils

# ╔═╡ a1b2c3d4-e5f6-7890-abcd-ef1234567890
# ╠═╡ show_logs = false
begin
    import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))
end

# ╔═╡ c3d4e5f6-a7b8-9012-cdef-345678901234
# ╠═╡ show_logs = false
Pkg.activate(joinpath(@__DIR__, "../../examples"))

# ╔═╡ 0a9939fa-ff7a-466d-b1a0-9077bdfe41b2
begin
    using DoubleML
    using StableRNGs
    using MLJ
    using TreeParzen
    using EvoTrees
end

# ╔═╡ 291fb6c3-883b-441b-a34c-061fb1dfb3fa
md"""
# Interactive Regression Model (IRM) Tutorial

This tutorial demonstrates how to use the `DoubleMLIRM` model for estimating treatment effects with binary treatments.

## Overview

The Interactive Regression Model assumes:

```math
Y = g_0(D, X) + \zeta, \quad \text{where } D \in \{0, 1\}
```

Where:

-  $Y$ is the outcome variable
-  $D$ is a **binary** treatment variable (0 or 1)
-  $X$ are control variables (covariates)
-  $g_0(D, X)$ is the conditional mean function

IRM allows for heterogeneous treatment effects and uses doubly robust estimation.
"""

# ╔═╡ 783af00f-2364-4475-836e-3b460476e0a7
md"""
## Load packages and import ML models 
"""

# ╔═╡ d4e5f6a7-b8c9-0123-defa-456789012345
begin
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees verbosity = 0
    RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree verbosity = 0
end

# ╔═╡ 2348d23e-46be-4003-9c87-790485df3c4e
md"""
## Generate IRM data
"""

# ╔═╡ e5f6a7b8-c9d0-1234-efab-567890123456
# IRM Data
data_irm = DoubleML.make_irm_data(1000, theta = 0.5, dim_x = 100, rng = StableRNG(42))

# ╔═╡ b28a5fc0-0bf7-41d0-9552-a7cd7bcb047e
md"""
View what models are available for our data
"""

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

# ╔═╡ e111bc12-a461-4509-86c9-c093fa5e7593
md"""
## Run a simple model
"""

# ╔═╡ b8c9d0e1-f2a3-4567-bcde-890123456789
begin
    # Simple IRM with RandomForest
    ml_g = RandomForestRegressor(rng = StableRNG(42))
    ml_m = RandomForestClassifier(rng = StableRNG(42))

    dml_irm_simple = DoubleML.DoubleMLIRM(data_irm, ml_g, ml_m, score = :ATE)

    fit!(dml_irm_simple)
end

# ╔═╡ c82c9d93-0086-4770-89a2-8fb0952e517c
coeftable(dml_irm_simple)

# ╔═╡ 0300a3a6-d009-41d3-a5c9-4700c1941ea2
md"""
## Advanced example: self-tuning models
"""

# ╔═╡ c9d0e1f2-a3b4-5678-cdef-901234567890
begin
    # IRM with TreeParzen hyperparameter tuning

    space = Dict(
        :max_depth => HP.QuantUniform(:max_depth, 2.0, 8.0, 1.0)
    )

    tuned_ml_g = TunedModel(
        model = EvoTreeRegressor(seed = 42),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_m = TunedModel(
        model = EvoTreeClassifier(seed = 42),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.cross_entropy,
        acceleration = CPUProcesses(),
    )


    dml_irm = DoubleML.DoubleMLIRM(data_irm, tuned_ml_g, tuned_ml_m)

    fit!(dml_irm, verbose = 0)

end

# ╔═╡ c8a0594d-c954-4106-bd78-41fe358d4535
coeftable(dml_irm)

# ╔═╡ Cell order:
# ╟─a1b2c3d4-e5f6-7890-abcd-ef1234567890
# ╟─c3d4e5f6-a7b8-9012-cdef-345678901234
# ╟─291fb6c3-883b-441b-a34c-061fb1dfb3fa
# ╟─783af00f-2364-4475-836e-3b460476e0a7
# ╠═0a9939fa-ff7a-466d-b1a0-9077bdfe41b2
# ╠═d4e5f6a7-b8c9-0123-defa-456789012345
# ╟─2348d23e-46be-4003-9c87-790485df3c4e
# ╠═e5f6a7b8-c9d0-1234-efab-567890123456
# ╟─b28a5fc0-0bf7-41d0-9552-a7cd7bcb047e
# ╠═f6a7b8c9-d0e1-2345-fabc-678901234567
# ╠═a7b8c9d0-e1f2-3456-abcd-789012345678
# ╟─e111bc12-a461-4509-86c9-c093fa5e7593
# ╠═b8c9d0e1-f2a3-4567-bcde-890123456789
# ╠═c82c9d93-0086-4770-89a2-8fb0952e517c
# ╟─0300a3a6-d009-41d3-a5c9-4700c1941ea2
# ╠═c9d0e1f2-a3b4-5678-cdef-901234567890
# ╠═c8a0594d-c954-4106-bd78-41fe358d4535
