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
    input_sha = "ca9e4eff3637e7fcf6a64d3e280227ee6a8d5d05040871e85e09fb074de899e9"
    julia_version = "1.12.6"
-->







<div class="markdown"><h1 id="Logistic-Partially-Linear-Regression-(LPLR)-Tutorial">Logistic Partially Linear Regression (LPLR) Tutorial</h1><p>⚠️ <strong>Experimental Model</strong>: This model is still under development.</p><h2 id="Overview">Overview</h2><p>The LPLR model estimates treatment effects with <strong>binary outcomes</strong> (<span class="tex">\(Y \in \{0,1\}\)</span>):</p><p class="tex">$$E[Y|D,X] = \text{expit}(\beta_0 D + r_0(X))$$</p><p>Where:</p><ul><li><p><span class="tex">\(Y \in \{0, 1\}\)</span> is the binary outcome</p></li><li><p><span class="tex">\(D\)</span> is the treatment (continuous or binary)</p></li><li><p><span class="tex">\(X\)</span> are control variables (covariates)</p></li><li><p><span class="tex">\(\beta_0\)</span> is the treatment effect on the log-odds scale</p></li><li><p><span class="tex">\(r_0(X)\)</span> is the nuisance function (conditional log-odds)</p></li></ul><p>The treatment effect <span class="tex">\(\beta_0\)</span> represents the change in log-odds of the outcome per unit change in treatment.</p></div>


<div class="markdown"><h2 id="Load-packages-and-import-ML-models">Load packages and import ML models</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    using DoubleML
    using StableRNGs
    using MLJ
    using TreeParzen
    using EvoTrees
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    EvoTreeClassifier = @load EvoTreeClassifier pkg = EvoTrees verbosity = 0
    RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree verbosity = 0
end</code></pre>
<pre class="code-output documenter-example-output" id="var-#544#dic">MLJDecisionTreeInterface.RandomForestClassifier</pre>


<div class="markdown"><h2 id="Generate-LPLR-data">Generate LPLR data</h2></div>

<pre class='language-julia'><code class='language-julia'>
data_lplr = make_lplr_LZZ2020(1000, alpha = 0.5, rng = StableRNG(42))</code></pre>
<pre class="code-output documenter-example-output" id="var-data_lplr">DoubleMLData{Float32, Vector{Float32}}(Float32[1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0  …  0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0], Float32[1.137392, -1.0203186, -1.2559067, 0.35913706, -0.98865694, 2.5477915, 1.6435643, -0.2593958, 2.7788334, 0.5610351  …  1.3883054, 1.6827291, 1.2748861, 1.5950062, 0.8413675, 0.1603024, 0.6246069, 0.86354524, 1.0373387, -1.603643], Float32[-0.67025167 0.3040378 … 0.9723363 -0.6289253; 2.0 -1.3471706 … -0.49255872 1.9596015; … ; -0.5976384 0.034506414 … 0.3402903 -0.70474565; 0.32168102 0.52544063 … 0.5724332 0.40678954], 1000, 20, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10, :X11, :X12, :X13, :X14, :X15, :X16, :X17, :X18, :X19, :X20])</pre>


<div class="markdown"><p>Find models that match the data we have:</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Find matching models for y
    models() do model
        matching(model, data_lplr.x, data_lplr.y)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash135420">12-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = CatBoostRegressor, package_name = CatBoost, ... )
 (name = DecisionTreeRegressor, package_name = BetaML, ... )
 (name = EvoTreeGaussian, package_name = EvoTrees, ... )
 (name = EvoTreeMLE, package_name = EvoTrees, ... )
 (name = EvoTreeRegressor, package_name = EvoTrees, ... )
 (name = GaussianMixtureRegressor, package_name = BetaML, ... )
 (name = NeuralNetworkRegressor, package_name = BetaML, ... )
 (name = NeuralNetworkRegressor, package_name = MLJFlux, ... )
 (name = PartLS, package_name = PartitionedLS, ... )
 (name = RandomForestRegressor, package_name = BetaML, ... )
 (name = SRRegressor, package_name = SymbolicRegression, ... )
 (name = SRTestRegressor, package_name = SymbolicRegression, ... )</pre>

<pre class='language-julia'><code class='language-julia'>begin
    # Find matching models for d
    models() do model
        matching(model, data_lplr.x, data_lplr.d)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash120331">12-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = CatBoostRegressor, package_name = CatBoost, ... )
 (name = DecisionTreeRegressor, package_name = BetaML, ... )
 (name = EvoTreeGaussian, package_name = EvoTrees, ... )
 (name = EvoTreeMLE, package_name = EvoTrees, ... )
 (name = EvoTreeRegressor, package_name = EvoTrees, ... )
 (name = GaussianMixtureRegressor, package_name = BetaML, ... )
 (name = NeuralNetworkRegressor, package_name = BetaML, ... )
 (name = NeuralNetworkRegressor, package_name = MLJFlux, ... )
 (name = PartLS, package_name = PartitionedLS, ... )
 (name = RandomForestRegressor, package_name = BetaML, ... )
 (name = SRRegressor, package_name = SymbolicRegression, ... )
 (name = SRTestRegressor, package_name = SymbolicRegression, ... )</pre>


<div class="markdown"><h2 id="Estimate-a-simple-model">Estimate a simple model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Simple LPLR with RandomForest
    ml_M = RandomForestClassifier(rng = StableRNG(42))
    ml_t = RandomForestRegressor(rng = StableRNG(42))
    ml_m = RandomForestRegressor(rng = StableRNG(42))

    dml_lplr_simple = DoubleML.DoubleMLLPLR(data_lplr, ml_M, ml_t, ml_m, score = :nuisance_space)

    fit!(dml_lplr_simple)

end</code></pre>
<pre class="code-output documenter-example-output" id="var-ml_t">DoubleMLLPLR{Float32, MLJDecisionTreeInterface.RandomForestClassifier, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestRegressor}
==========================
StatsBase.CoefTable(Any[[0.5794071555137634], [0.07449068874120712], [7.778249263763428], [7.353507254078456e-15], [0.4334080883974139], [0.725406222630113]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_lplr_simple)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash360715">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.579407   0.0744907     7.78    &lt;1e-14     0.433408     0.725406
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h2 id="Estimate-a-more-complex-model-with-iteration-control">Estimate a more complex model with iteration control</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(6),
        NumberLimit(25),
    ]

    ml_M_iterated = IteratedModel(
        EvoTreeClassifier(max_depth = 4, eta = 0.01, seed = 42),
        resampling = Holdout(),
        measure = cross_entropy,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_t_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.01, seed = 42),
        resampling = Holdout(),
        measure = mav,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_m_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.01, seed = 42),
        resampling = Holdout(),
        measure = mae,
        iteration_parameter = :nrounds,
        controls = controls
    )

    # Set up the model
    dml_lplr_iterated = DoubleML.DoubleMLLPLR(data_lplr, ml_M_iterated, ml_t_iterated, ml_m_iterated, score = :nuisance_space)

    # Fit the model
    fit!(dml_lplr_iterated)


end</code></pre>
<pre class="code-output documenter-example-output" id="var-controls">DoubleMLLPLR{Float32, MLJIteration.ProbabilisticIteratedModel{EvoTrees.EvoTreeClassifier, Nothing}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor, Nothing}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor, Nothing}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor, Nothing}}
==========================
StatsBase.CoefTable(Any[[0.5143537521362305], [0.06825204938650131], [7.536092281341553], [4.8426394865056066e-14], [0.38058219346763855], [0.6481253108048224]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_lplr_iterated)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash950978">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.514354    0.068252     7.54    &lt;1e-13     0.380582     0.648125
────────────────────────────────────────────────────────────────────</pre>

<!-- PlutoStaticHTML.End -->
```

