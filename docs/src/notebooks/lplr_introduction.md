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
    input_sha = "608f5ab3ff7f9c3c1e0165b0fe879290984a7cff3fbc7bccfa761a3811b5dbb4"
    julia_version = "1.12.4"
-->
<pre class='language-julia'><code class='language-julia'>import Pkg; Pkg.develop(path=joinpath(@__DIR__, "../.."))</code></pre>


<pre class='language-julia'><code class='language-julia'>Pkg.activate(joinpath(@__DIR__, "../../examples"))</code></pre>


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
<pre class="code-output documenter-example-output" id="var-#540#dic">MLJDecisionTreeInterface.RandomForestClassifier</pre>


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
<pre class="code-output documenter-example-output" id="var-hash135420">11-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
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
 (name = SRRegressor, package_name = SymbolicRegression, ... )</pre>

<pre class='language-julia'><code class='language-julia'>begin
    # Find matching models for y
    models() do model
        matching(model, data_lplr.x, data_lplr.d)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash261476">11-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
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
 (name = SRRegressor, package_name = SymbolicRegression, ... )</pre>


<div class="markdown"><h2 id="Estimate-a-simple-model">Estimate a simple model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Simple IRM with RandomForest
    ml_M = RandomForestClassifier()
    ml_t = RandomForestRegressor()
    ml_m = RandomForestRegressor()

    dml_lplr_simple = DoubleML.DoubleMLLPLR(data_lplr, ml_M, ml_t, ml_m, score = :nuisance_space)

    fit!(dml_lplr_simple)

end</code></pre>
<pre class="code-output documenter-example-output" id="var-ml_t">DoubleMLLPLR{Float32, MLJDecisionTreeInterface.RandomForestClassifier, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestRegressor}
==========================
StatsBase.CoefTable(Any[[0.5567321181297302], [0.0747859850525856], [7.444337368011475], [9.743221856517205e-14], [0.41015428087831135], [0.703309955381149]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_lplr_simple)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash360715">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.556732    0.074786     7.44    &lt;1e-13     0.410154      0.70331
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h2 id="Estimate-a-more-complex-model-with-iteration-control">Estimate a more complex model with iteration control</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(10),
        NumberLimit(20),
    ]

    ml_M_iterated = IteratedModel(
        EvoTreeClassifier(max_depth = 4, eta = 0.05),
        resampling = Holdout(),
        measure = cross_entropy,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_t_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.05),
        resampling = Holdout(),
        measure = mav,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_m_iterated = IteratedModel(
        EvoTreeRegressor(max_depth = 4, eta = 0.05),
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
<pre class="code-output documenter-example-output" id="var-controls">DoubleMLLPLR{Float32, MLJIteration.ProbabilisticIteratedModel{EvoTrees.EvoTreeClassifier}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor}}
==========================
StatsBase.CoefTable(Any[[0.5537656545639038], [0.07333461195230484], [7.551218032836914], [4.3120625571213347e-14], [0.41003245631716545], [0.6974988528106422]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_lplr_iterated)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash950978">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.553766   0.0733346     7.55    &lt;1e-13     0.410032     0.697499
────────────────────────────────────────────────────────────────────</pre>

<!-- PlutoStaticHTML.End -->
```

