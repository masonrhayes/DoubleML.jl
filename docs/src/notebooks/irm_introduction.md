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
    input_sha = "6771e0a97536fd67dd4b1881f10505ed837d32ddd15b1d3a6c1751d68bb0cb59"
    julia_version = "1.12.5"
-->







<div class="markdown"><h1 id="Interactive-Regression-Model-(IRM)-Tutorial">Interactive Regression Model (IRM) Tutorial</h1><p>This tutorial demonstrates how to use the <code>DoubleMLIRM</code> model for estimating treatment effects with binary treatments.</p><h2 id="Overview">Overview</h2><p>The Interactive Regression Model assumes:</p><p class="tex">$$Y = g_0(D, X) + \zeta, \quad \text{where } D \in \{0, 1\}$$</p><p>Where:</p><ul><li><p><span class="tex">\(Y\)</span> is the outcome variable</p></li><li><p><span class="tex">\(D\)</span> is a <strong>binary</strong> treatment variable (0 or 1)</p></li><li><p><span class="tex">\(X\)</span> are control variables (covariates)</p></li><li><p><span class="tex">\(g_0(D, X)\)</span> is the conditional mean function</p></li></ul><p>IRM allows for heterogeneous treatment effects and uses doubly robust estimation.</p></div>


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
<pre class="code-output documenter-example-output" id="var-#540#dic">MLJDecisionTreeInterface.RandomForestClassifier</pre>


<div class="markdown"><h2 id="Generate-IRM-data">Generate IRM data</h2></div>

<pre class='language-julia'><code class='language-julia'># IRM Data
data_irm = DoubleML.make_irm_data(1000, theta = 0.5, dim_x = 100, rng = StableRNG(42))</code></pre>
<pre class="code-output documenter-example-output" id="var-data_irm">DoubleMLData{Float32, CategoricalArrays.CategoricalVector{Float32, UInt32, Float32, CategoricalArrays.CategoricalValue{Float32, UInt32}, Union{}}}(Float32[-0.6748344, 0.3039633, 0.6760439, 1.4083221, -0.8368299, -0.61595523, 3.0014334, -0.036194555, 1.864324, -0.28150964  …  0.8323356, -1.2143124, -1.2397234, -1.3907428, -0.14038181, 2.120363, -1.8365414, -0.623508, 1.898227, 1.4702643], CategoricalArrays.CategoricalValue{Float32, UInt32}[CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1)  …  CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 2), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1), CategoricalValue(CategoricalArrays.CategoricalPool{Float32, UInt32}([0.0f0, 1.0f0]), 1)], Float32[-0.8857411 -0.54276526 … -0.4588549 0.062866904; -0.9109771 -0.48209068 … -0.4358175 1.0236975; … ; -0.8479842 -1.4113976 … 0.59593326 0.031701036; -0.9461808 -1.1490784 … -1.3875467 -1.6245216], 1000, 100, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10  …  :X91, :X92, :X93, :X94, :X95, :X96, :X97, :X98, :X99, :X100])</pre>


<div class="markdown"><p>View what models are available for our data</p></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Find matching models for y
    models() do model
        matching(model, data_irm.x, data_irm.y)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash150203">11-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
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
    # Find matching models for d
    models() do model
        matching(model, data_irm.x, data_irm.d)
    end
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash806559">11-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
 (name = CatBoostClassifier, package_name = CatBoost, ... )
 (name = DecisionTreeClassifier, package_name = BetaML, ... )
 (name = EvoTreeClassifier, package_name = EvoTrees, ... )
 (name = GaussianNBClassifier, package_name = NaiveBayes, ... )
 (name = KernelPerceptronClassifier, package_name = BetaML, ... )
 (name = NeuralNetworkBinaryClassifier, package_name = MLJFlux, ... )
 (name = NeuralNetworkClassifier, package_name = BetaML, ... )
 (name = NeuralNetworkClassifier, package_name = MLJFlux, ... )
 (name = PegasosClassifier, package_name = BetaML, ... )
 (name = PerceptronClassifier, package_name = BetaML, ... )
 (name = RandomForestClassifier, package_name = BetaML, ... )</pre>


<div class="markdown"><h2 id="Run-a-simple-model">Run a simple model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Simple IRM with RandomForest
    ml_g = RandomForestRegressor(rng = StableRNG(42))
    ml_m = RandomForestClassifier(rng = StableRNG(42))

    dml_irm_simple = DoubleML.DoubleMLIRM(data_irm, ml_g, ml_m, score = :ATE)

    fit!(dml_irm_simple)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-dml_irm_simple">DoubleMLIRM{Float32, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestClassifier}
==========================
StatsBase.CoefTable(Any[[0.9093611836433411], [0.055683430284261703], [16.33091163635254], [5.948874387636241e-60], [0.8002236657505409], [1.018498701536141]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_irm_simple)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash845485">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.909361   0.0556834    16.33    &lt;1e-59     0.800224       1.0185
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h2 id="Advanced-example:-self-tuning-models">Advanced example: self-tuning models</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # IRM with TreeParzen hyperparameter tuning

    space = Dict(
        :max_depth =&gt; HP.QuantUniform(:max_depth, 2.0, 8.0, 1.0)
    )

    tuned_ml_g = TunedModel(
        model = EvoTreeRegressor(seed = 42),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_m = TunedModel(
        model = EvoTreeClassifier(seed = 42),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.cross_entropy,
        acceleration = CPUProcesses(),
    )


    dml_irm = DoubleML.DoubleMLIRM(data_irm, tuned_ml_g, tuned_ml_m)

    fit!(dml_irm, verbose = 0)

end</code></pre>
<pre class="code-output documenter-example-output" id="var-tuned_ml_m">DoubleMLIRM{Float32, MLJTuning.DeterministicTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeRegressor, Nothing}, MLJTuning.ProbabilisticTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeClassifier, Nothing}}
==========================
StatsBase.CoefTable(Any[[0.7141482830047607], [0.05785326659679413], [12.344130516052246], [5.239723911948737e-35], [0.60075796408705], [0.8275386019224715]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_irm)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash997185">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.714148   0.0578533    12.34    &lt;1e-34     0.600758     0.827539
────────────────────────────────────────────────────────────────────</pre>

<!-- PlutoStaticHTML.End -->
```

