# DoubleML.jl

[![SciML Code Style](https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826)](https://github.com/SciML/SciMLStyle)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue)](https://julialang.org)

**Double Machine Learning for Causal Inference in Julia**

DoubleML.jl implements double/de-biased machine learning methods for causal inference, following [Chernozhukov et al. (2018)](https://arxiv.org/abs/1608.00060).

This package is inspired by, and aims to closely follow, the [DoubleML](https://docs.doubleml.org/stable/index.html) Python package, but is unaffiiliated with it.

## Features

Why DoubleML.jl?

- Leverage Julia's speed, with up to 10x faster model fitting compared to Python (based on early benchmarks).
- **[MLJ](https://juliaml.ai/) Integration**: Use any MLJ-compatible model for nuisance estimation, with the flexibility to control model iteration and model tuning (see examples)
- **[StatsAPI](https://github.com/JuliaStats/StatsAPI.jl) Compliance**: `coef()`, `stderror()`, `confint()`, `coeftable()`
- **Cross-fitting**: K-fold sample splitting with multiple repetitions
- **Bootstrap Inference**: Joint confidence intervals with bootstrapped standard errors

## Models currently implemented

This package remains in early development and testing stages. The following models are currently implemented:

| Model            | Use Case                    | Learners                                    | Status            |
| ---------------- | --------------------------- | ------------------------------------------- | ----------------- |
| `DoubleMLPLR`  | Continuous/binary treatment | `ml_l`, `ml_m` (+ `ml_g` for IV-type) | Implemented       |
| `DoubleMLIRM`  | Binary treatment only       | `ml_g`, `ml_m` (classifier)             | Implemented       |
| `DoubleMLLPLR` | Binary outcome (Y ∈ {0,1}) | `ml_M`, `ml_t`, `ml_m` (+ `ml_a`)   | ⚠️ Experimental |

## Quick Example

```julia
using DoubleML, MLJ, DataFrames, StableRNGs

data = make_plr_CCDDHNR2018(500, alpha=0.5, rng=StableRNG(42))

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0
ml_l = RandomForestRegressor()
ml_m = RandomForestRegressor()

model = DoubleMLPLR(data, ml_l, ml_m, n_folds=5)
fit!(model)

summary(model)

println("Treatment effect: ", coef(model)[1])
println("95% CI: ", confint(model))
```

## Documentation

- [User Guide](https://masonrhayes.github.io/DoubleML.jl/stable/user-guide/) - Installation, concepts, and workflow
- [Tutorials](https://masonrhayes.github.io/DoubleML.jl/stable/tutorials/) - Step-by-step examples
- [API Reference](https://masonrhayes.github.io/DoubleML.jl/stable/api/) - Complete API documentation
- [Examples](https://masonrhayes.github.io/DoubleML.jl/stable/examples/) - Pluto notebooks

## Roadmap

There are many features and models still not yet implemented in this package. The broad roadmap is to achieve feature parity with the [DoubleML](https://docs.doubleml.org/stable/index.html) package in Python.

Currently, a variety of tests against the Python package are implemented to ensure similar functionality of the DoubleMLPLR, DoubleMLIRM, and DoubleMLLPLR models.

In early benchmarks, the Julia implementation performs well

## Other packages

Other similar Julia packages include [CausalELM](https://github.com/dscolby/CausalELM.jl), which offers a very lightweight approach to causal machine learning, where all the machine learners take the form of extreme learning machines. In comparison, this package aims to offer more similar features to those of the DoubleML Python package and allow flexibility of the model choice.

## Index

```@index
Pages = ["api.md"]
```
