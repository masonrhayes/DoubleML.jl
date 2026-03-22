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
    input_sha = "9e18930474650da55343ffa77c3243c48ce029d9acb0495a70189ad27d4de884"
    julia_version = "1.12.5"
-->







<div class="markdown"><h1 id="An-example-of-Double-Machine-Learning-using-Conformal-Prediction">An example of Double Machine Learning using Conformal Prediction</h1><p>This notebook illustrates an early-stage demonstration of the potential for the use of conformal predictions in the double machine learning framework.</p><p>The main motivation for bringing conformal predictions into double machine learning framework is to:</p><ul><li><p>Propagate uncertainty in nuisance model predictions to causal inference</p></li><li><p>Reduce computational burden by avoiding cross-fitting</p></li></ul><p>The below introduces some of this motivation in more detail, and shows a particular example using simulated data (from <code>make_plr_CCDDHNR2018()</code>).</p></div>


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


<div class="markdown"><div class="admonition is-note"><header class="admonition-header">Note</header><div class="admonition-body"><p>The below is specifically a counter-example to show where standard DML may fail in terms of coverage, where conformal DML may succeed. A large-scale assessment across multiple random seeds would be needed for a more comprehensive evaluation of the performance of the different methods.</p></div></div></div>

<pre class='language-julia'><code class='language-julia'>begin
    seed = 33
    rng = StableRNG(seed)

    true_alpha = 0.5

    n_obs = 500
    dim_x = 250
    data = make_plr_CCDDHNR2018(n_obs; dim_x = dim_x, alpha = true_alpha, rng = rng)
end
</code></pre>
<pre class="code-output documenter-example-output" id="var-rng">DoubleMLData{Float32, Vector{Float32}}(Float32[-0.90963966, 1.9374812, -0.25847855, 0.47272336, 0.64437914, -0.10265304, -0.9905324, 0.94382477, -0.72001314, 1.7434089  …  0.6319347, -2.9627357, 0.44794837, -0.48833418, -2.304316, -2.1842134, 2.6824763, -2.6089628, -0.81449336, 0.043069214], Float32[0.43613905, 0.9887245, -1.0767803, -0.10072567, 0.7472056, 1.2857444, -1.9633851, 0.62001693, -0.39885935, -0.22904095  …  0.075055815, -1.5850143, 1.1537043, 0.29475603, -2.7867234, -3.6445894, 1.4557726, -1.6433831, 0.831413, -0.31156892], Float32[0.08301931 -1.0187424 … 0.7081886 1.3663985; 0.6239522 0.4052753 … -1.0074859 -0.53348225; … ; 0.23453191 1.2632104 … -0.78497356 -1.0187758; 0.61482817 -0.8486912 … -1.1574534 -1.8353295], 500, 250, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10  …  :X241, :X242, :X243, :X244, :X245, :X246, :X247, :X248, :X249, :X250])</pre>


<div class="markdown"><h2 id="Estimating-a-Conformal-Double-Machine-Learning-(CDML)-model">Estimating a Conformal Double Machine Learning (CDML) model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
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
end</code></pre>
<pre class="code-output documenter-example-output" id="var-coverage">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.544354   0.0417292    13.04    &lt;1e-38     0.463793     0.625546
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'># Test against standard PLR model
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
end</code></pre>
<pre class="code-output documenter-example-output" id="var-model">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.593208   0.0381025    15.57    &lt;1e-53     0.518529     0.667888
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><div class="admonition is-results"><header class="admonition-header">Results</header><div class="admonition-body"><p>The above example shows that the conformal model ran in roughly 4-5x faster than the standard model that uses 5-fold cross-fitting. In addition, the conformal model includes the true causal effect, whereas the standard model does not. </p></div></div><p>As noted above, this is a bit of a contrived example. In anecdotal testing, the conformal model appears to give better empirical coverage. However, more extensive testing is needed to properly evaluate the performance of the different methods.</p></div>


<div class="markdown"><h2 id="How-does-Conformal-Double-Machine-Learning-work?">How does Conformal Double Machine Learning work?</h2><div class="admonition is-warning"><header class="admonition-header">Warning</header><div class="admonition-body"><p>This package, and the implementation of Conformal Double Machine Learning, remain experimental. </p></div></div><p>Currently, the implementation of CDML works by:</p><ul><li><p>Training the conformal models without cross-fitting. Users may specify <code>train_ratio</code> for some conformal prediction methods, but <em>predictions</em> are made on the full dataset</p></li><li><p>Obtaining conformal predictions (i.e, a tuple of a lower and upper bound for each prediction). These conformal predictions guaranteed a user-defined coverage level (e.g., 95%).</p></li><li><p>Use Monte Carlo sampling from conformal prediction intervals to propagate uncertainty, using Beta(2,2) marginals with Gaussian copula to account for correlation between the uncertainties in predictions for the outcome <span class="tex">\(\hat{l}(x)\)</span> and treatment <span class="tex">\(\hat{m}(x)\)</span>.</p></li></ul></div>

<!-- PlutoStaticHTML.End -->
```

