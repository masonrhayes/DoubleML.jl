# Logistic Partially Linear Regression (LPLR) Tutorial

⚠️ **Experimental Model**: This model is still under development.

## Overview

The LPLR model estimates treatment effects with **binary outcomes** (Y ∈ {0,1}):

```math
E[Y|D,X] = \text{expit}(\beta_0 D + r_0(X))
```

Where:

- ``Y \in \{0, 1\}`` is the binary outcome
- ``D`` is the treatment (continuous or binary)
- ``X`` are control variables (covariates)
- ``\beta_0`` is the treatment effect on the log-odds scale
- ``r_0(X)`` is the nuisance function (conditional log-odds)

The treatment effect ``\beta_0`` represents the change in log-odds of the outcome per unit change in treatment.

## Basic Usage

### Step 1: Load Packages and Data

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(12345)
data = make_lplr_LZZ2020(n_obs=500, alpha=0.5, rng=rng)
```

### Step 2: Set Up MLJ Models

LPLR requires 3-4 learners:

- `ml_M`: **Classifier** for ``P(Y=1|D,X)``
- `ml_t`: Regressor for ``E[\text{logit}(M)|X]``
- `ml_m`: Regressor for ``E[D|X]`` (nuisance_space) or sample weights (instrument)
- `ml_a`: Optional regressor for ``E[D|X]`` (defaults to `ml_m`)

```julia
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=0
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0

ml_M = RandomForestClassifier()  # P(Y=1|D,X)
ml_t = RandomForestRegressor()   # E[logit(M)|X]
ml_m = RandomForestRegressor()   # E[D|X]
```

### Step 3: Create and Fit Model

```julia
model = DoubleMLLPLR(data, ml_M, ml_t, ml_m, 
                     n_folds=5, score=:nuisance_space)
fit!(model)


```

### Step 4: Extract Results

```julia
summary(model)

println("Treatment effect: ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))
```

## Score Functions

LPLR supports two score functions for different estimation strategies:

Both score functions are based on those [described](https://docs.doubleml.org/stable/guide/scores.html#logistic-partial-linear-regression-lplr) in the Python package. Note, however, that unlike the Python package, this package currenly estimates the 'preliminary beta' in a different manner. This implementation remains experimental.

### Nuisance Space Score (default)

```julia
model = DoubleMLLPLR(data, ml_M, ml_t, ml_m, score=:nuisance_space)
```

Instrument Score

```julia
model = DoubleMLLPLR(data, ml_M, ml_t, ml_m, score=:instrument)
```

## Complete Example

```julia
using DoubleML
using MLJ
using DataFrames
using StableRNGs

rng = StableRNG(42)

# Load MLJ models
RandomForestClassifier = @load RandomForestClassifier pkg=DecisionTree verbosity=0
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0

# Generate binary outcome data
data = make_lplr_LZZ2020(1000, alpha=0.5, rng=rng)

# Set up learners
ml_M = RandomForestClassifier()  # Classifier for P(Y=1|D,X)
ml_t = RandomForestRegressor()   # Regressor for E[logit(M)|X]
ml_m = RandomForestRegressor()   # Regressor for E[D|X]

# Create and fit model with nuisance_space score
model = DoubleMLLPLR(data, ml_M, ml_t, ml_m, 
                     n_folds=5, score=:nuisance_space)
fit!(model)

# Results
println("Estimated treatment effect (log-odds): ", coef(model)[1])
println("Standard error: ", stderror(model)[1])
println("95% CI: ", confint(model))

# Compare with instrument score
model_instrument = DoubleMLLPLR(data, ml_M, ml_t, ml_m, 
                                n_folds=5, score=:instrument)
fit!(model_instrument)
println("Estimated treatment effect (log-odds)", coef(model_instrument)[1])
println("Standard error: ", stderror(model_instrument)[1])
```

## Key Considerations

1. **Binary Outcome Required**: LPLR requires Y to be binary (0/1). Check with `check_binary(data.y)`.
2. **Classifier for ml_M**: The outcome model must be a classifier (probabilistic), not a regressor.
3. **Score Selection**:

   - Use `:nuisance_space` for standard DML approach
   - Use `:instrument` when you need weighted estimation
4. **Nested Cross-Fitting**: LPLR uses both outer and inner folds for robust estimation.
5. **Log-Odds Interpretation**: The coefficient represents the change in log-odds, which can be converted to odds ratio via `exp(coef)`.
