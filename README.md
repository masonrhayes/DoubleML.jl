<picture>
  <source srcset="material/logo/doubleml-logo-dark.png" media="(prefers-color-scheme: dark)">
  <img src="material/logo/doubleml-logo.png" width="400">
</picture>

[![SciML Code Style](<https://img.shields.io/static/v1?label=code%20style&message=SciML&color=9558b2&labelColor=389826>)](https://github.com/SciML/SciMLStyle)
[![Julia](https://img.shields.io/badge/Julia-1.10%2B-blue)](https://julialang.org)

**Double Machine Learning for Causal Inference in Julia**

`DoubleML.jl` implements double/de-biased machine learning methods for causal inference, following [Chernozhukov et al. (2018)](https://arxiv.org/abs/1608.00060).

This package is inspired by, and aims to closely follow, the [DoubleML](https://docs.doubleml.org/stable/index.html) Python package, but is unaffiliated with it.

## Features

Why DoubleML.jl?

- Leverage Julia's speed, with up to 10x faster model fitting compared to Python (based on early benchmarks).
- **[MLJ](https://juliaml.ai/) Integration**: Use any MLJ-compatible model for nuisance estimation, with the flexibility to control model iteration, tuning, stacking, etc. (see examples)
- **[StatsAPI](https://github.com/JuliaStats/StatsAPI.jl) Compliance**: `coef()`, `stderror()`, `confint()`, `coeftable()`

## Models currently implemented

This package remains in early development and testing stages. The following models are currently implemented:


| Model                                            | Use Case                                                                                              | Learners                              | Status                              |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------- | ------------------------------------- |
| `DoubleMLPLR`                                    | Continuous/binary treatment                                                                           | `ml_l`, `ml_m` (+ `ml_g` for IV-type) | Implemented                         |
| `DoubleMLIRM`                                    | Binary treatment only                                                                                 | `ml_g`, `ml_m` (classifier)           | Implemented                         |
| `DoubleMLLPLR`                                   | Binary outcome (Y ∈ {0,1})                                                                           | `ml_M`, `ml_t`, `ml_m` (+ `ml_a`)     | ⚠️ Experimental                   |
| `DoubleMLPLRConformal`, `DoubleMLPLRConformalUT` | Conformal predictions for better uncertainty quantification in the causal inference stage (research) | `ml_l`, `ml_m` (conformal-wrapped)    | 🔬 Experimental, research prototype |

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

 Documentation

- [User Guide](https://masonrhayes.github.io/DoubleML.jl/dev/user-guide/) - Installation, concepts, and workflow
- [Tutorials](https://masonrhayes.github.io/DoubleML.jl/dev/tutorials/) - Step-by-step examples
- [API Reference](https://masonrhayes.github.io/DoubleML.jl/dev/api/) - Complete API documentation
- [Examples](https://masonrhayes.github.io/DoubleML.jl/dev/examples/) - Pluto notebooks

## Roadmap

There are many features and models still not yet implemented in this package. The broad roadmap is to achieve feature parity with the [DoubleML](https://docs.doubleml.org/stable/index.html) package in Python, and to continue research on more experimental features (e.g., conformal predictions).

Currently, a variety of tests against the Python package are implemented to ensure similar functionality of the DoubleMLPLR, DoubleMLIRM, and DoubleMLLPLR models.

In early benchmarks, the Julia implementation performs well and up to 10x faster than the Python package (see the [benchmark](test/test_python/benchmarks/benchmarks.md))

## Other packages

Other similar Julia packages include [CausalELM](https://github.com/dscolby/CausalELM.jl), which offers a very lightweight approach to causal machine learning, where all the machine learners take the form of extreme learning machines. In comparison, this package aims to offer more similar features to those of the DoubleML Python package and allow flexibility of the model choice.

## Experimental Features

### Conformal Double Machine Learning (CDML)

This package also implements, as a package extension, a proof-of-concept implementation of Conformal Double Machine Learning (CDML).

CDML uses **conformal prediction** (from [ConformalPrediction.jl](https://github.com/JuliaTrustworthyAI/ConformalPrediction.jl)) for uncertainty quantification. Conformal prediction is a distribution-free approach that provides prediction intervals for the nuisance models with finite-sample *marginal* coverage guaranteed under data exchangeability, without further assumptions on the data distribution or model specification ([Angelopoulos and Bates (2022)](https://doi.org/10.48550/arXiv.2107.07511)).

In the standard DML framework, the estimated nuisance parameters enter the causal inference stage only as fixed point estimates. The benefit of this is that it allows the use of any supervised learning models, is computationally convenient, and works well in large samples; however, the limitation is that it discards information about uncertainty in the predictions which leads to poor empirical coverage in practice, most especially in small samples.

In Bayesian Double Machine Learning ([DiTraglia and Liu (2025)](https://doi.org/10.48550/arXiv.2508.12688)), one approaches this problem by relaxing that assumption and integrating over the nuisance space. The downside of this approach is that is often requires greater computational resources and is less flexible, since one cannot simply plug-in fixed point estimates; however, the benefit is significantly better inference for the causal parameter, especially in small sample sizes.

> "*In practice, we find that recognition of a relevant, but unknown and uninteresting, parameter by including it in the model and then integrating it out again as a nuisance parameter, can greatly improve our ability to extract the information we want from our data – often by orders of magnitude*" - E.T. Jaynes (Probability theory: the logic of science, ch. 7)

Using conformal predictions in the frequentist DML framework can be viewed as a step towards the Bayesian DML framework, as it attempts to propagate uncertainty in the nuisance parameters into the causal inference stage. :warning: However, the implementation remains very experimental.

**Key characteristics:**

- By default, the DoubleMLConformal implementation does *not* use cross-fitting. The calibration split required for conformal validity is handled internally by `ConformalPrediction.jl`; no cross-fitting is required, though optional K-fold cross-fitting (`n_folds > 1`) is supported.
- Without cross-fitting, over-fitting bias is not controlled in the usual DML sense; whether sampling from conformal intervals avoids over-fitting bias is an open research hypothesis.
- Two uncertainty-propagation implementations:
  - `DoubleMLPLRConformalUT` — an unscented transform propagates nuisance prediction uncertainty into the causal inference stage (deterministic)
  - `DoubleMLPLRConformal` — Monte Carlo sampling from conformal intervals propagates uncertainty to the causal inference stage (a third variant, `DoubleMLPLRConformalSimple`, uses interval midpoints without propagation)
- Models dependence between nuisance uncertainties via a Gaussian copula with Beta(2,2) marginals. The copula correlation is estimated from residuals, so this is an assumed parametric model of the uncertainty.
- Requires conformal-wrapped MLJ models from `ConformalPrediction.jl`

**Status:** Experimental - API may change. Not recommended to use except for research/testing purposes.

See documentation for usage details.
