# User Guide

## Installation

This package is still under active development. To install it, from the Julia REPL, run:

```julia
using Pkg
Pkg.add(url = "https://github.com/masonrhayes/DoubleML.jl")
```

### Required MLJ Models

DoubleML.jl uses MLJ.jl for machine learning. Install model packages:

```julia
using Pkg
Pkg.add(["MLJ", "MLJLinearModels", "DecisionTree", "DataFrames"])
```

## What is Double Machine Learning?

Double Machine Learning (DML) estimates causal effects under the presence of (potentially) high-dimensional control variables. It solves the "regularization bias" problem by using **two separate ML models** to orthogonalize the problem:

1. One model predicts outcome `Y` from controls: `E[Y|X]`
2. Another model predicts treatment `D` from controls: `E[D|X]`

By taking residuals from both, we can isolate the causal effect of the treatment on the outcome.

## Cross-Fitting

Cross-fitting prevents overfitting by ensuring predictions are made on data not used for training:

1. Split data into K folds
2. For each fold k: train on other folds, predict on fold k
3. Combine predictions across all folds

This is essential for valid inference - without it, regularization bias returns.

## Model Types

### Partially Linear Regression (PLR)

**When to use:** Continuous or binary treatment, constant treatment effect

```
Y = θ·D + g₀(X) + ε
D = m₀(X) + v
```

**Learners:** `ml_l` for E[Y|X], `ml_m` for E[D|X]

```julia
model = DoubleMLPLR(data, ml_l, ml_m, n_folds=5, score=:partialling_out)
fit!(model)
```

**Score functions:**

- `:partialling_out` (default) - standard orthogonalization
- `:IV_type` - requires additional `ml_g` learner for endogenous treatment

### Interactive Regression Model (IRM)

**When to use:** Binary treatment (D ∈ {0,1}), allows heterogeneous effects

```
Y = g₀(D, X) + ζ   where D ∈ {0, 1}
```

**Learners:** `ml_g` for E[Y|X,D], `ml_m` (classifier) for P(D=1|X)

```julia
model = DoubleMLIRM(data, ml_g, ml_m, n_folds=5, score=:ATE)
fit!(model)
```

**Estimands:**

- `:ATE` - Average Treatment Effect
- `:ATTE` - Average Treatment Effect on the Treated

### Logistic Partially Linear Regression (LPLR) ⚠️ Experimental

**When to use:** Binary outcome (Y ∈ {0,1}), treatment effect on log-odds scale

```
E[Y|D,X] = expit(β₀·D + r₀(X))   where Y ∈ {0, 1}
```

**Learners:** 
- `ml_M` (classifier) for P(Y=1|D,X)
- `ml_t` for E[logit(M)|X]
- `ml_m` for nuisance estimation
- `ml_a` (optional) for E[D|X]

```julia
model = DoubleMLLPLR(data, ml_M, ml_t, ml_m, n_folds=5, score=:nuisance_space)
fit!(model)
```

**Score functions:**

- `:nuisance_space` (default) - fits `ml_m` on Y=0 observations only
- `:instrument` - uses weighted estimation with M*(1-M) weights

**Note:** This model is experimental and may change in future versions.

## Learner Naming Convention

| Model          | Learner 1            | Learner 2            | Notes                            |
| -------------- | -------------------- | -------------------- | -------------------------------- |
| DoubleMLPLR    | `ml_l` (E[Y\|X])   | `ml_m` (E[D\|X])   | `ml_g` optional for IV-type    |
| DoubleMLIRM    | `ml_g` (E[Y\|X,D]) | `ml_m` (P(D=1\|X)) | Must use classifier for `ml_m` |
| DoubleMLLPLR   | `ml_M` (P(Y=1\|D,X)) | `ml_t` (E[logit(M)\|X]) | `ml_m` for nuisance; ⚠️ Experimental |

## Workflow

### 1. Prepare Data

```julia
using DoubleML, DataFrames

# From DataFrame
df = DataFrame(y=..., d=..., x1=..., x2=...)
data = DoubleMLData(
	df, 
	y_col=:y, 
	d_col=:d, 
	x_cols=[:x1, :x2]
)

# Or use built-in generators
data = make_plr_CCDDHNR2018(500, alpha=0.5)
```

### 2. Set Up Learners

```julia
using MLJ

# For PLR: both can be regressors
RandomForestRegressor = @load RandomForestRegressor pkg=DecisionTree verbosity=0
ml_l = RandomForestRegressor(max_depth=10)
ml_m = RandomForestRegressor(max_depth=5)

# For IRM: ml_m must be a classifier
LogisticClassifier = @load LogisticClassifier pkg=MLJLinearModels verbosity=0
ml_g = RandomForestRegressor()
ml_m = LogisticClassifier()
```

### 3. Fit Model

```julia
model = DoubleMLPLR(data, ml_l, ml_m, n_folds=5, n_rep=1)
fit!(model)
```

### 4. Extract Results

```julia
θ = coef(model)[1]           # Point estimate
se = stderror(model)[1]      # Standard error
ci = confint(model)          # 95% CI
ct = coeftable(model)        # Summary table
```

### 5. Bootstrap (Optional)

```julia
bootstrap!(model, n_rep_boot=1000, method=:normal)
joint_ci = confint(model, joint=true)  # Controls family-wise error rate
```

## StatsAPI Interface

All models implement StatsAPI:

```julia
coef(model)       # Treatment effect(s)
stderror(model)   # Standard error(s)
confint(model)    # Confidence intervals
vcov(model)       # Variance-covariance matrix
nobs(model)       # Number of observations
coeftable(model)  # Formatted table
```
