### A Pluto.jl notebook ###
# v0.20.24

using Markdown
using InteractiveUtils

# ╔═╡ 67af2dce-2534-11f1-8d5b-d7a2267620e9
# ╠═╡ show_logs = false
import Pkg; Pkg.develop(path = joinpath(@__DIR__, "../.."))

# ╔═╡ 13224815-ed19-458a-be1d-29e277cda921
# ╠═╡ show_logs = false
Pkg.activate(joinpath(@__DIR__, "../../examples"))

# ╔═╡ c7999558-7893-4667-ad2f-ebc632442825
begin
    using DoubleML
    using ConformalPrediction  # This triggers loading of DoubleMLConformalExt
    using MLJ
    using StableRNGs
    using Random
    using DataFrames
end

# ╔═╡ 4bd80a4b-b9d0-4d2a-a81e-8ad3fb5bbfa8
md"""
# An example of Double Machine Learning using Conformal Prediction

This notebook illustrates an early-stage demonstration of the potential for the use of conformal predictions in the double machine learning framework.

The main motivation for bringing conformal predictions into double machine learning framework is to:
- Propagate uncertainty in nuisance model predictions to causal inference
- Reduce computational burden by avoiding cross-fitting

The below introduces some of this motivation in more detail, and shows a particular example using simulated data (from `make_plr_CCDDHNR2018()`).
"""

# ╔═╡ 85a3cef4-22c5-4948-bca5-449a2e56625c
md"""
## What is Conformal Prediction?

> Conformal prediction (a.k.a. conformal inference) is a user-friendly paradigm for creating statistically rigorous uncertainty sets/intervals for the predictions of such models. Critically, the sets are valid in a distribution-free sense: they possess explicit, non-asymptotic guarantees even without distributional assumptions or model assumptions.
>
> — [Angelopoulos and Bates (2022)](http://arxiv.org/abs/2107.07511)

"""

# ╔═╡ 6770489c-da27-4da3-af10-085480d4d01d
md"""
## Why Conformal Double Machine Learning?
"""

# ╔═╡ 0102aaa6-e8c7-4d1f-991e-2e17574c9f6d
md"""
Sample splitting, typically in the form of cross-fitting, is one of the key features of standard Frequentist Double Machine Learning (FDML) which aims to solve the issue of *over-fitting bias*. 

Cross-fitting alleviates the "*potential dependence between nuisance estimates and parts of the data used for estimating the target parameter*" ([Ahrens et al (2025)](https://arxiv.org/pdf/2504.08324)).

As stated in Ahrens et al (2025), p 4:

"...Because $\hat{\eta}$ is an estimator, it is itself a random function of the data. $\hat{\eta}$ is thus generally correlated with the observations $\{W_i\}_{i=1}^n$ also used in the estimating equation $\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \hat{\eta})$. When this dependence is strong, for example due to "overfitting", it may generate large differences between $\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \hat{\eta})$ and $\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \eta_0)$, which results in poor performance of $\hat{\theta}$."


"""

# ╔═╡ c38bd709-1f0b-4188-9623-940387eccfc0
md"""

In practice, however, there are a few issues that cross-fitting does not resolve: 
- First, in the presence of large data, a **practical** issue is that cross-fitting can be computationally costly as it requires fitting a model at least 1 time for each fold of the cross-validation set. 
- Second, and more fundamentally, a **theoretical** issue is that cross-fitting does not account for *uncertainty* in the predictions $\hat{\eta}$, but rather treats them as point estimates. Any uncertainty in these point estimates is not propogated into the causal inference for $\theta$. Thus, FDML estimates of the causal parameter, $\hat{\theta}$, often do not have good *coverage* - e.g., using simulated data where the true causal effect is known, FDML often leads to confidence intervals which do not include the true effect.

As shown in this notebook, however, over-fitting bias can be alleviated without cross-fitting! If we instead think of our estimates for $\hat{\eta}$ as following some *joint probability distribution*, we can simply fit one time, using one holdout set for the conformal prediction calibration; we can then directly account for the uncertainty in our nuisance estimates, and propagate this uncertainty through to the final inference stage.

!!! hypothesis
    By *sampling* from the joint probability distribution for each prediction from the nuisance models, the hypothesis is that we can maintain Neyman orthogonality and avoid over-fitting bias, and make better inference decisions by improving uncertainty quantification.
"""

# ╔═╡ d3f4e9e5-3e9a-4b96-a1f6-a6daafbcd0a7
md"""
## Load MLJ models
Let's experiment with EvoTrees, RandomForest, and Symbolic Regression.
"""

# ╔═╡ dc46ab56-0a8f-44c9-9a71-47d983de4ad0
begin # loading MLJ models
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = BetaML verbosity = 0
    SRRegressor = @load SRRegressor pkg = SymbolicRegression verbosity = 0
end;

# ╔═╡ 04737055-53c5-4763-a7d1-265aa5f055ab
md"""
## Ensure the DoubleMLConformalExt is accessible
Below, we get the extension for estimating Conformal Double Machine Learning models.

This is implemented as a package extension as it remains experimental.
"""

# ╔═╡ 49a22224-7d0c-4d03-abc3-22bb2a15d9c2
const Ext = Base.get_extension(DoubleML, :DoubleMLConformalExt)

# ╔═╡ 53acf85c-0ca5-4e88-820d-03a5dbe98cd7
md"""
## Data generation 
"""

# ╔═╡ e42d5705-fab0-47eb-9219-bb88773d0ce5
md"""
!!! note 
    The below is specifically a counter-example to show where standard DML may fail in terms of coverage, where conformal DML may succeed. A large-scale assessment across multiple random seeds would be needed for a more comprehensive evaluation of the performance of the different methods.
"""

# ╔═╡ 167599c7-70ad-43b4-abb9-d68897286e94
begin
    seed = 33
    rng = StableRNG(seed)

    true_alpha = 0.5

    n_obs = 500
    dim_x = 250
    data = make_plr_CCDDHNR2018(n_obs; dim_x = dim_x, alpha = true_alpha, rng = rng)
end


# ╔═╡ a56b0f77-e2a2-4510-8e31-e6fc41be8721
md"""
## Estimating a Conformal Double Machine Learning (CDML) model
"""

# ╔═╡ 1f11818f-90ca-4bc2-95a7-824df68a6d32
begin
    Random.seed!(seed)

    # Set the coverage for the nuisance models.
    coverage = 0.999

    ml_l = conformal_model(
        RandomForestRegressor(rng = rng);
        method = :simple_inductive,
        coverage = coverage
    )
    ml_m = conformal_model(
        RandomForestRegressor(rng = rng);
        method = :simple_inductive,
        coverage = coverage
    )
    # Create and fit conformal model
    model_conformal = Ext.DoubleMLPLRConformal(data, ml_l, ml_m; n_mc_samples = 1_000)
    @time Ext.fit!(model_conformal, rng = rng, verbose = 0)

    coeftable(model_conformal)
end

# ╔═╡ b6c5aed9-1be4-4840-9561-fce18f74e58f
# Test against standard PLR model
begin
    Random.seed!(seed)

    model = DoubleMLPLR(
        data,
        RandomForestRegressor(rng = rng),
        RandomForestRegressor(rng = rng);
        n_folds = 5,
        n_rep = 1
    )

    @time DoubleML.fit!(model; verbose = 0)

    coeftable(model)
end

# ╔═╡ 31054492-0276-402f-95d1-674193c735d2
md"""
!!! results
    The above example shows that the conformal model ran in roughly 4-5x faster than the standard model that uses 5-fold cross-fitting. In addition, the conformal model includes the true causal effect, whereas the standard model does not. 

As noted above, this is a bit of a contrived example. In anecdotal testing, the conformal model appears to give better empirical coverage. However, more extensive testing is needed to properly evaluate the performance of the different methods.
"""

# ╔═╡ 58045322-9e8c-4ed8-9f7c-445a4b8b2a70
md"""
## How does Conformal Double Machine Learning work?

!!! warning
    This package, and the implementation of Conformal Double Machine Learning, remain experimental. 

Currently, the implementation of CDML works by:
- Training the conformal models without cross-fitting. Users may specify `train_ratio` for some conformal prediction methods, but *predictions* are made on the full dataset
- Obtaining conformal predictions (i.e, a tuple of a lower and upper bound for each prediction). These conformal predictions guaranteed a user-defined coverage level (e.g., 95%).
- Use Monte Carlo sampling from conformal prediction intervals to propagate uncertainty, using Beta(2,2) marginals with Gaussian copula to account for correlation between the uncertainties in predictions for the outcome $\hat{l}(x)$ and treatment $\hat{m}(x)$.

"""

# ╔═╡ Cell order:
# ╟─67af2dce-2534-11f1-8d5b-d7a2267620e9
# ╟─13224815-ed19-458a-be1d-29e277cda921
# ╟─4bd80a4b-b9d0-4d2a-a81e-8ad3fb5bbfa8
# ╟─85a3cef4-22c5-4948-bca5-449a2e56625c
# ╟─6770489c-da27-4da3-af10-085480d4d01d
# ╟─0102aaa6-e8c7-4d1f-991e-2e17574c9f6d
# ╟─c38bd709-1f0b-4188-9623-940387eccfc0
# ╠═c7999558-7893-4667-ad2f-ebc632442825
# ╟─d3f4e9e5-3e9a-4b96-a1f6-a6daafbcd0a7
# ╠═dc46ab56-0a8f-44c9-9a71-47d983de4ad0
# ╟─04737055-53c5-4763-a7d1-265aa5f055ab
# ╠═49a22224-7d0c-4d03-abc3-22bb2a15d9c2
# ╟─53acf85c-0ca5-4e88-820d-03a5dbe98cd7
# ╟─e42d5705-fab0-47eb-9219-bb88773d0ce5
# ╠═167599c7-70ad-43b4-abb9-d68897286e94
# ╟─a56b0f77-e2a2-4510-8e31-e6fc41be8721
# ╠═1f11818f-90ca-4bc2-95a7-824df68a6d32
# ╠═b6c5aed9-1be4-4840-9561-fce18f74e58f
# ╟─31054492-0276-402f-95d1-674193c735d2
# ╟─58045322-9e8c-4ed8-9f7c-445a4b8b2a70
