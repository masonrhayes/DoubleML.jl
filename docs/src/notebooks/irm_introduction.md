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
    input_sha = "8c0d6dfd009aa865f39a494478d77a1d51690bd82f25ebf71b510a103e845c5a"
    julia_version = "1.12.4"
-->







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
<pre class="code-output documenter-example-output" id="var-data_irm">DoubleMLData{Float32, CategoricalArrays.CategoricalVector{Float32, UInt32, Float32, CategoricalArrays.CategoricalValue{Float32, UInt32}, Union{}}}(Float32[-0.6748344, 0.3039633, 0.6760439, 1.4083221, -0.8368299, -0.61595523, 3.0014334, -0.036194555, 1.864324, -0.28150964  …  0.8323356, -1.2143124, -1.2397234, -1.3907428, -0.14038181, 2.120363, -1.8365414, -0.623508, 1.898227, 1.4702643], CategoricalArrays.CategoricalValue{Float32, UInt32}[0.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 1.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0  …  1.0f0, 0.0f0, 0.0f0, 0.0f0, 1.0f0, 1.0f0, 1.0f0, 0.0f0, 0.0f0, 0.0f0], Float32[-0.8857411 -0.54276526 … -0.4588549 0.062866904; -0.9109771 -0.48209068 … -0.4358175 1.0236975; … ; -0.8479842 -1.4113976 … 0.59593326 0.031701036; -0.9461808 -1.1490784 … -1.3875467 -1.6245216], 1000, 100, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10  …  :X91, :X92, :X93, :X94, :X95, :X96, :X97, :X98, :X99, :X100])</pre>


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
    ml_g = RandomForestRegressor()
    ml_m = RandomForestClassifier()

    dml_irm_simple = DoubleML.DoubleMLIRM(data_irm, ml_g, ml_m, score = :ATE)

    fit!(dml_irm_simple)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-dml_irm_simple">DoubleMLIRM{Float32, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestClassifier}
==========================
StatsBase.CoefTable(Any[[0.9053707122802734], [0.05485549196600914], [16.504650115966797], [3.396961989328215e-61], [0.797855923672669], [1.0128855008878779]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_irm_simple)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash845485">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.905371   0.0548555    16.50    &lt;1e-60     0.797856      1.01289
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h2 id="Advanced-example:-self-tuning-models">Advanced example: self-tuning models</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # IRM with TreeParzen hyperparameter tuning

    space = Dict(
        :max_depth =&gt; HP.QuantUniform(:max_depth, 3.0, 8.0, 1.0)
    )

    tuned_ml_g = TunedModel(
        model = EvoTreeRegressor(),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_m = TunedModel(
        model = EvoTreeClassifier(),
        tuning = MLJTreeParzenTuning(),
        resampling = Holdout(),
        range = space,
        measure = MLJ.cross_entropy,
        acceleration = CPUProcesses(),
    )


    dml_irm = DoubleML.DoubleMLIRM(data_irm, tuned_ml_g, tuned_ml_m)

    fit!(dml_irm, verbose = 1)

end</code></pre>
<pre class="code-output documenter-example-output" id="var-tuned_ml_m">DoubleMLIRM{Float32, MLJTuning.DeterministicTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeRegressor, Nothing}, MLJTuning.ProbabilisticTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeClassifier, Nothing}}
==========================
StatsBase.CoefTable(Any[[0.8129682540893555], [0.10276787728071213], [7.9107232093811035], [2.5589791357466022e-15], [0.6115469158515272], [1.0143895923271837]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_irm)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash997185">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.812968    0.102768     7.91    &lt;1e-14     0.611547      1.01439
────────────────────────────────────────────────────────────────────</pre>

<!-- PlutoStaticHTML.End -->
```

