```@raw html
<style>
    #documenter-page table {
        display: table !important;
        margin: 2rem auto !important;
        border-top: 2pt solid rgba(0,0,0,0.2);
        border-bottom: 2pt solid rgba(0,0,0,0.2);
    }

    #documenter-page pre, #documenter-page div {
        margin-top: 1.4rem !important;
        margin-bottom: 1.4rem !important;
    }

    .code-output {
        padding: 0.7rem 0.5rem !important;
    }

    .admonition-body {
        padding: 0em 1.25em !important;
    }
</style>

<!-- PlutoStaticHTML.Begin -->
<!--
    # This information is used for caching.
    [PlutoStaticHTML.State]
    input_sha = "1d385d7ed87ecb790e69e6f7a363a01d64a86d1e4817c0f6af6200a7a3e76645"
    julia_version = "1.12.6"
-->







<div class="markdown"><h1 id="An-example-of-Double-Machine-Learning-using-Conformal-Prediction">An example of Double Machine Learning using Conformal Prediction</h1><p>This notebook illustrates an early-stage demonstration of the potential for the use of conformal predictions in the double machine learning framework.</p><p>The main motivation for bringing conformal predictions into double machine learning framework is to:</p><ul><li><p>Propagate uncertainty in nuisance model predictions to causal inference</p></li><li><p>Reduce computational burden by avoiding cross-fitting</p></li></ul><p>The below introduces some of this motivation in more detail, and shows a particular example using simulated data (from <code>make_plr_DTL2025()</code>).</p></div>


<div class="markdown"><h2 id="What-is-Conformal-Prediction?">What is Conformal Prediction?</h2><blockquote><p>Conformal prediction (a.k.a. conformal inference) is a user-friendly paradigm for creating statistically rigorous uncertainty sets/intervals for the predictions of such models. Critically, the sets are valid in a distribution-free sense: they possess explicit, non-asymptotic guarantees even without distributional assumptions or model assumptions.</p><p>— <a href="http://arxiv.org/abs/2107.07511">Angelopoulos and Bates (2022)</a></p></blockquote></div>


<div class="markdown"><h2 id="Why-Conformal-Double-Machine-Learning?">Why Conformal Double Machine Learning?</h2></div>


<div class="markdown"><p>Sample splitting, typically in the form of cross-fitting, is one of the key features of standard Frequentist Double Machine Learning (FDML) which aims to solve the issue of <em>over-fitting bias</em>. </p><p>Cross-fitting alleviates the "<em>potential dependence between nuisance estimates and parts of the data used for estimating the target parameter</em>" (<a href="https://arxiv.org/pdf/2504.08324">Ahrens et al (2025)</a>).</p><p>As stated in Ahrens et al (2025), p 4:</p><p>"...Because <span class="tex">\(\hat{\eta}\)</span> is an estimator, it is itself a random function of the data. <span class="tex">\(\hat{\eta}\)</span> is thus generally correlated with the observations <span class="tex">\(\{W_i\}_{i=1}^n\)</span> also used in the estimating equation <span class="tex">\(\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \hat{\eta})\)</span>. When this dependence is strong, for example due to "overfitting", it may generate large differences between <span class="tex">\(\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \hat{\eta})\)</span> and <span class="tex">\(\frac{1}{n}\sum_{i=1}^n m(W_i; \theta, \eta_0)\)</span>, which results in poor performance of <span class="tex">\(\hat{\theta}\)</span>."</p></div>


<div class="markdown"><p>In practice, however, there are a few issues that cross-fitting does not resolve: </p><ul><li><p>First, in the presence of large data, a <strong>practical</strong> issue is that cross-fitting can be computationally costly as it requires fitting a model at least 1 time for each fold of the cross-validation set. </p></li><li><p>Second, and more fundamentally, a <strong>theoretical</strong> issue is that cross-fitting does not account for <em>uncertainty</em> in the predictions <span class="tex">\(\hat{\eta}\)</span>, but rather treats them as point estimates. Any uncertainty in these point estimates is not propogated into the causal inference for <span class="tex">\(\theta\)</span>. Thus, FDML estimates of the causal parameter, <span class="tex">\(\hat{\theta}\)</span>, often do not have good <em>coverage</em> - e.g., using simulated data where the true causal effect is known, FDML often leads to confidence intervals which do not include the true effect.</p></li></ul><p>As shown in this notebook, however, over-fitting bias can be alleviated without cross-fitting! If we instead think of our estimates for <span class="tex">\(\hat{\eta}\)</span> as following some <em>joint probability distribution</em>, we can simply fit one time, using one holdout set for the conformal prediction calibration; we can then directly account for the uncertainty in our nuisance estimates, and propagate this uncertainty through to the final inference stage.</p><div class="admonition is-hypothesis"><header class="admonition-header">Hypothesis</header><div class="admonition-body"><p>By <em>sampling</em> from the joint probability distribution for each prediction from the nuisance models, the hypothesis is that we can maintain Neyman orthogonality and avoid over-fitting bias, and make better inference decisions by improving uncertainty quantification.</p></div></div></div>

<pre class='language-julia'><code class='language-julia'>begin
    using DoubleML
    using ConformalPrediction  # This triggers loading of DoubleMLConformalExt
    using MLJ
    using StableRNGs
    using Random
    using DataFrames
end</code></pre>



<div class="markdown"><h2 id="Load-MLJ-models">Load MLJ models</h2><p>Let's experiment with EvoTrees, RandomForest, and Symbolic Regression.</p></div>

<pre class='language-julia'><code class='language-julia'>begin # loading MLJ models
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = BetaML verbosity = 0
    SRRegressor = @load SRRegressor pkg = SymbolicRegression verbosity = 0
end;</code></pre>



<div class="markdown"><h2 id="Ensure-the-DoubleMLConformalExt-is-accessible">Ensure the DoubleMLConformalExt is accessible</h2><p>Below, we get the extension for estimating Conformal Double Machine Learning models.</p><p>This is implemented as a package extension as it remains experimental.</p></div>

<pre class='language-julia'><code class='language-julia'>const Ext = Base.get_extension(DoubleML, :DoubleMLConformalExt)</code></pre>
<pre class="code-output documenter-example-output" id="var-const Ext">DoubleMLConformalExt</pre>


<div class="markdown"><h2 id="Data-generation">Data generation</h2></div>


<div class="markdown"><div class="admonition is-note"><header class="admonition-header">Note</header><div class="admonition-body"><p>The below is specifically a counter-example to show where FDML may fail in terms of coverage. A more comprehensive assessment across multiple random seeds would be needed for more a proper evaluation.</p></div></div></div>


<div class="markdown"><p>Here, we use the data generation process from Section 6 of <a href="https://arxiv.org/abs/2508.12688">DiTraglia and Liu (2025)</a>, and use their default paramater choices of:</p><ul><li><p><span class="tex">\(\alpha =\)</span> 2.0 (the true causal effect)</p></li><li><p><span class="tex">\(n =\)</span> 200 (number of observations)</p></li><li><p><span class="tex">\(p =\)</span> 100, (number of covariates), and</p></li><li><p><span class="tex">\(\sigma_{\varepsilon} = 2\)</span></p></li></ul></div>


<div class="markdown"><p>Let's compare the results of fitting a Conformal Double Machine Learning model, vs the a standard DML model with 5-fold cross-fitting. </p><div class="admonition is-note"><header class="admonition-header">Note</header><div class="admonition-body"><p>Note that the illustrative example below is specifically a counter-example to illustrate where standard DML may fail, where conformal model may succeed. In reality, on this specific problem, both methods (conformal and non-conformal FDML) often fail to capture the true causal effect. </p></div></div><p>Larger-scale simulations across a wide range of random seeds are needed for more comprehensive evaluation of the empirical performance of the different methods to truly evaluate the performance of each method.</p><p>As set out in the paper, Bayesian Double Machine Learning (BDML) performs very well on this problem. Keep an eye out for a forthcoming BayesianDoubleML.jl package which implements the models set out in DiTraglia and Liu (2025)!</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    seed = 60
    rng = StableRNG(seed)

    true_alpha = 2.0

    n = 200
    p = 100
    sigma_epsilon = 2.0

    data = make_plr_DTL2025(n, p, sigma_epsilon; alpha = true_alpha, rng = rng)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-p">DoubleMLData{Float32, Vector{Float32}}(Float32[-8.547153, 2.537241, 2.5340595, -0.4896508, -4.6916947, 7.371019, 2.0968182, 0.22039314, -3.3462834, 4.2125087  …  -4.048314, 1.3922781, 0.025425058, 2.4426374, -1.2662034, -4.7019324, -0.8984496, 2.6580913, 1.6113335, -0.037568532], Float32[-3.8231306, 1.3155899, 0.91706735, 1.2340404, -1.9633923, 0.707324, 0.7664218, -1.1376064, -1.4266428, 1.698602  …  -0.90348893, -0.3089625, -0.57444024, -0.040365264, -0.7284262, -0.42811778, -2.2260504, 0.31699485, 0.84488803, -0.39746693], Float32[-0.64719266 0.3935131 … 0.43429512 -0.1457195; 0.06077252 -0.046264276 … 0.42417604 0.0051582777; … ; 0.68033195 -3.2544427 … 0.22436728 1.3658123; 1.3604089 1.5732734 … -0.3286756 0.60283846], 200, 100, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10  …  :X91, :X92, :X93, :X94, :X95, :X96, :X97, :X98, :X99, :X100])</pre>


<div class="markdown"><h2 id="Estimating-a-Conformal-Double-Machine-Learning-(CDML)-model">Estimating a Conformal Double Machine Learning (CDML) model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    Random.seed!(seed)

    # Set the coverage for the nuisance models.
    coverage = 0.999

    ml_l = conformal_model(
        EvoTreeRegressor(seed = seed);
        method = :simple_inductive,
        coverage = coverage
    )
    ml_m = conformal_model(
        EvoTreeRegressor(seed = seed);
        method = :simple_inductive,
        coverage = coverage
    )
    # Create and fit conformal model
    model_conformal = Ext.DoubleMLPLRConformal(
        data,
        ml_l,
        ml_m;
        n_mc_samples = 1_000
    )
    @time Ext.fit!(model_conformal, rng = rng, verbose = 0)

    coeftable(model_conformal)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-coverage">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d   1.78495    0.133532    13.37    &lt;1e-40      1.51971      2.04702
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'># Test against standard PLR model
begin
    Random.seed!(seed)

    model = DoubleMLPLR(
        data,
        EvoTreeRegressor(seed = seed),
        EvoTreeRegressor(seed = seed);
        n_folds = 5,
        n_rep = 1
    )

    @time DoubleML.fit!(model; verbose = 0)

    coeftable(model)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-model">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d   1.57854    0.110481    14.29    &lt;1e-45        1.362      1.79508
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><div class="admonition is-results"><header class="admonition-header">Results</header><div class="admonition-body"><p>The above example shows that the conformal model ran roughly 4-5x faster than the standard model that uses 5-fold cross-fitting. In addition, the conformal model includes the true causal effect, whereas the standard model does not. </p></div></div></div>


<div class="markdown"><h2 id="How-does-Conformal-Double-Machine-Learning-work?">How does Conformal Double Machine Learning work?</h2><div class="admonition is-warning"><header class="admonition-header">Warning</header><div class="admonition-body"><p>This package, and the implementation of Conformal Double Machine Learning, remain experimental. </p></div></div><p>Currently, the implementation of CDML works by:</p><ul><li><p>Training the conformal models without cross-fitting. Users may specify <code>train_ratio</code> for some conformal prediction methods, but <em>predictions</em> are made on the full dataset</p></li><li><p>Obtaining conformal predictions (i.e, a tuple of a lower and upper bound for each prediction). These conformal predictions guaranteed a user-defined coverage level (e.g., 95%).</p></li><li><p>Use Monte Carlo sampling from conformal prediction intervals to propagate uncertainty, using Beta(2,2) marginals with Gaussian copula to account for correlation between the uncertainties in predictions for the outcome <span class="tex">\(\hat{l}(x)\)</span> and treatment <span class="tex">\(\hat{m}(x)\)</span>.</p></li></ul></div>

<!-- PlutoStaticHTML.End -->
```

