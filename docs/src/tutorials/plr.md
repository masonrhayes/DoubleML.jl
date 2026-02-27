# Partially Linear Regression (PLR) Tutorial

This tutorial demonstrates how to use the `DoubleMLPLR` model for estimating treatment effects in a partially linear regression framework.

## Overview

The Partially Linear Regression model assumes:

```math
Y = \theta \cdot D + g_0(X) + \epsilon \\
D = m_0(X) + v
```

Where:

- `Y` is the outcome variable
- `D` is the treatment variable (can be continuous or binary)
- `X` are control variables (covariates)
- θ is the treatment effect we want to estimate
- g_0(X) and m_0(X) are nuisance functions estimated via ML

## Basic Usage

### Step 1: Load Packages and Data

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(12345)
data = make_plr_CCDDHNR2018(500, alpha=0.5, rng=rng)
```

### Step 2: Set Up MLJ Models

```julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0

ml_l = RandomForestRegressor()  # E[Y|X]
ml_m = RandomForestRegressor()  # E[D|X]
```

### Step 3: Create and Fit DoubleML Model

```julia
model = DoubleMLPLR(data, ml_l, ml_m, n_folds=5, score=:partialling_out)
fit!(model)
```

### Step 4: Extract Results

```julia
println("Treatment effect: ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))
```

## Score Functions

The PLR model supports two score functions:

### Partialling Out Score (default)

```julia
model = DoubleMLPLR(data, ml_l, ml_m)
```

### IV-Type Score

Requires an additional `ml_g` learner for endogenous treatment:

```julia
ml_g = RandomForestRegressor()  # E[Y - D·θ|X]
model = DoubleMLPLR(data, ml_l, ml_m, ml_g=ml_g, score=:IV_type)
```

## Hyperparameter Tuning

Use MLJ's `TunedModel` for automatic hyperparameter tuning:

```julia
using MLJTuning

rf = RandomForestRegressor()
range_depth = range(rf, :max_depth, lower=3, upper=10)

tuned_l = TunedModel(
    model=rf,
    tuning=Grid(resolution=3),
    resampling=CV(nfolds=3),
    measure=rmse,
    range=range_depth
)

model = DoubleMLPLR(data, tuned_l, ml_m, n_folds=5)
fit!(model)
```

## Bootstrap Inference

For joint confidence intervals that control family-wise error rate:

```julia
bootstrap!(model, n_rep_boot=1000, method=:normal)

println("Joint 95% CI: ", confint(model, joint=true))
```

Note: Joint CIs are wider than pointwise CIs because they control the family-wise error rate.

## Complete Example

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(42)
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

data = make_plr_CCDDHNR2018(1000, alpha=0.5, rng=rng)

ml_l = LinearRegressor()
ml_m = LinearRegressor()

model = DoubleMLPLR(data, ml_l, ml_m, n_folds=5)
fit!(model)

summary(model))

println("Estimated treatment effect: ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))

bootstrap!(model, n_rep_boot=500)
println("Joint 95% CI: ", confint(model, joint=true))
```
