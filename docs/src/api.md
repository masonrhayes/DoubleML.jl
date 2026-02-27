# API Reference

## Models

### DoubleMLPLR

Partially Linear Regression model.

```@docs
DoubleMLPLR
```

**Constructor:**

```julia
DoubleMLPLR(data, ml_l, ml_m; ml_g=nothing, n_folds=5, n_rep=1, score=:partialling_out, n_folds_tune=0, T=Float64)
```

| Parameter | Description |
|-----------|-------------|
| `data` | DoubleMLData container |
| `ml_l` | MLJ regressor for E[Y\|X] |
| `ml_m` | MLJ model for E[D\|X] |
| `ml_g` | MLJ regressor for E[Y-D·θ\|X] (IV-type only) |
| `n_folds` | Cross-fitting folds (default: 5) |
| `n_rep` | Sample splitting repetitions (default: 1) |
| `score` | `:partialling_out` or `:IV_type` |

**Methods:**

```@docs
fit!(::DoubleMLPLR)
learner_l(::DoubleMLPLR)
learner_m(::DoubleMLPLR)
isfitted(::DoubleML.AbstractDoubleML)
```

### DoubleMLIRM

Interactive Regression Model for binary treatments.

```@docs
DoubleMLIRM
```

**Constructor:**

```julia
DoubleMLIRM(data, ml_g, ml_m; n_folds=5, n_rep=1, score=:ATE, normalize_ipw=false, clipping_threshold=0.01, n_folds_tune=0, T=Float64)
```

| Parameter | Description |
|-----------|-------------|
| `data` | DoubleMLData container (binary treatment) |
| `ml_g` | MLJ regressor for E[Y\|X,D] |
| `ml_m` | MLJ classifier for P(D=1\|X) |
| `n_folds` | Cross-fitting folds (default: 5) |
| `score` | `:ATE` or `:ATTE` |
| `normalize_ipw` | Hajek normalization (default: false) |
| `clipping_threshold` | Propensity clip threshold (default: 0.01) |

**Methods:**

```@docs
fit!(::DoubleMLIRM)
learner_g(::DoubleMLIRM)
learner_m(::DoubleMLIRM)
```

### DoubleMLLPLR

⚠️ **Experimental** - Logistic Partially Linear Regression for binary outcomes.

```@docs
DoubleMLLPLR
```

**Constructor:**

```julia
DoubleMLLPLR(data, ml_M, ml_t, ml_m; ml_a=nothing, score=:nuisance_space, n_folds=5, n_folds_inner=5, n_rep=1, n_folds_tune=0)
```

| Parameter | Description |
|-----------|-------------|
| `data` | DoubleMLData container (binary outcome required) |
| `ml_M` | MLJ **classifier** for P(Y=1\|D,X) |
| `ml_t` | MLJ regressor for E[logit(M)\|X] |
| `ml_m` | MLJ regressor for nuisance estimation |
| `ml_a` | Optional regressor for E[D\|X] (defaults to ml_m) |
| `score` | `:nuisance_space` or `:instrument` |

**Methods:**

```@docs
fit!(::DoubleMLLPLR)
learner_M(::DoubleMLLPLR)
learner_t(::DoubleMLLPLR)
learner_m(::DoubleMLLPLR)
learner_a(::DoubleMLLPLR)
```

## Data

```@docs
DoubleMLData
make_plr_CCDDHNR2018
make_irm_data
make_lplr_LZZ2020
```

**Data access:** `data.y`, `data.d`, `data.x`, `data.n_obs`, `data.dim_x`

## Inference

```@docs
coef(::DoubleML.AbstractDoubleML)
stderror(::DoubleML.AbstractDoubleML)
confint(::DoubleML.AbstractDoubleML)
vcov(::DoubleML.AbstractDoubleML)
nobs(::DoubleML.AbstractDoubleML)
bootstrap!
has_bootstrapped
```

## Score Functions

```@docs
AbstractScore
PartiallingOutScore
IVTypeScore
ATEScore
ATTEScore
NuisanceSpaceScore
InstrumentScore
dml2_solve
```

## Utilities

```@docs
draw_sample_splitting
get_conditional_sample_splitting
fitted_params(::DoubleMLPLR)
fitted_params(::DoubleMLIRM)
```

## Additional Functions

```@docs
multiplier_bootstrap
summary_stats
check_binary
dtype
to_numeric
compute_score
get_score_name
learner_g(::DoubleMLPLR)
```

## StatsAPI Methods

All models implement:

- `coef(model)` - Treatment effect(s)
- `stderror(model)` - Standard error(s)
- `confint(model)` - Confidence intervals
- `vcov(model)` - Variance-covariance matrix
- `nobs(model)` - Number of observations
- `coeftable(model)` - Formatted summary table
- `dof(model)` - Degrees of freedom
- `dof_residual(model)` - Residual degrees of freedom
