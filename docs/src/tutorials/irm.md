# Interactive Regression Model (IRM) Tutorial

This tutorial demonstrates how to use the `DoubleMLIRM` model for estimating treatment effects with binary treatments.

## Overview

The Interactive Regression Model assumes:

```math
Y = g_0(D, X) + \zeta \quad \text{where } D \in \{0, 1\}
```

Where:

- `Y` is the outcome variable
- `D` is a **binary** treatment variable (0 or 1)
- `X` are control variables (covariates)
- g_0(D, X) is the conditional mean function

IRM allows for heterogeneous treatment effects and uses doubly robust estimation.

## Basic Usage

### Step 1: Load Packages and Data

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(12345)
data = make_irm_data(500, theta=0.5, rng=rng)
```

### Step 2: Set Up MLJ Models

IRM requires:

- `ml_g`: Regressor for E[Y|X,D]
- `ml_m`: **Classifier** for P(D=1|X) (propensity score)

```julia
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

ml_g = RandomForestRegressor()     # E[Y|X,D]
ml_m = LogisticClassifier()        # P(D=1|X)
```

### Step 3: Create and Fit DoubleML Model

```julia
model = DoubleMLIRM(data, ml_g, ml_m, n_folds=5, score=:ATE)
fit!(model)
```

### Step 4: Extract Results

```julia
println("ATE: ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))
```

## Estimands

IRM supports two estimands:

### Average Treatment Effect (ATE)

The effect of treatment on the entire population:

```julia
model = DoubleMLIRM(data, ml_g, ml_m, score=:ATE)
fit!(model)
```

### Average Treatment Effect on the Treated (ATTE)

The effect of treatment on those who received it:

```julia
model = DoubleMLIRM(data, ml_g, ml_m, score=:ATTE)
fit!(model)
```

## Propensity Score Options

### Clipping

Propensity scores are clipped to avoid extreme values (default: 0.01):

```julia
model = DoubleMLIRM(data, ml_g, ml_m, clipping_threshold=0.05)
```

### Hajek Normalization

Enable inverse probability weight normalization:

```julia
model = DoubleMLIRM(data, ml_g, ml_m, normalize_ipw=true)
```

## Bootstrap Inference

For joint confidence intervals:

```julia
bootstrap!(model, n_rep_boot=1000, method=:normal)
println("Joint 95% CI: ", confint(model, joint=true))
```

## Complete Example

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(42)

RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0

data = make_irm_data(1000, theta=0.5, rng=rng)

ml_g = RandomForestRegressor()
ml_m = LogisticClassifier()

model = DoubleMLIRM(data, ml_g, ml_m, n_folds=5, score=:ATE)
fit!(model)

println("ATE: ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))

bootstrap!(model, n_rep_boot=500)
println("Joint 95% CI: ", confint(model, joint=true))
```

## Best Practices

1. **Use a classifier for ml_m**: The propensity score model must predict probabilities, so use `LogisticClassifier`, `RandomForestClassifier`, etc.
2. **Check propensity scores**: Watch for warnings about extreme propensity scores - this indicates poor overlap between treatment groups.
3. **Clipping**: Always use propensity score clipping (default is 0.01) to avoid division by near-zero values.
4. **Choose appropriate estimand**:

   - ATE for population-level effects
   - ATTE when interested specifically in the treated population
5. **Doubly robust**: IRM is consistent if either the outcome model OR the propensity score is correctly specified.
