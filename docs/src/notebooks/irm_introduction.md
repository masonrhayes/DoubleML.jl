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
    input_sha = "7bd830a3473d17a6cf11e7e59a67a38ef8bc07443c8e514b1512939325a84087"
    julia_version = "1.12.4"
-->
<pre class='language-julia'><code class='language-julia'>begin
    using Pkg
    Pkg.activate("..")
    Pkg.instantiate()
end</code></pre>


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

<pre class='language-julia'><code class='language-julia'># IRM Data
data_irm = DoubleML.make_irm_data(10000, theta = 0.5, dim_x = 100, rng = StableRNG(42))</code></pre>
<pre class="code-output documenter-example-output" id="var-data_irm">DoubleMLData{Float32, CategoricalArrays.CategoricalVector{Float32, UInt32, Float32, CategoricalArrays.CategoricalValue{Float32, UInt32}, Union{}}}(Float32[0.37352446, 1.125557, -0.5569645, 0.17525621, -1.0872172, 0.32317752, 1.2394512, 0.6840453, 1.0432272, 2.5331562  …  -1.0704536, -0.2607919, 1.1385626, -1.582762, 1.5980701, 2.1474342, 2.6292956, 2.458277, 2.2120817, -0.028367579], CategoricalArrays.CategoricalValue{Float32, UInt32}[0.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0, 0.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0  …  0.0f0, 1.0f0, 0.0f0, 0.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 1.0f0, 0.0f0], Float32[-0.56396604 0.22146161 … -0.026084585 -0.55734754; 0.31028554 0.50824165 … 1.0575864 1.1089114; … ; 2.0035193 0.33156925 … -2.093451 -1.0798455; 0.2469462 -0.33030185 … -1.2763116 0.20448275], 10000, 100, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10  …  :X91, :X92, :X93, :X94, :X95, :X96, :X97, :X98, :X99, :X100])</pre>

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

<pre class='language-julia'><code class='language-julia'>begin
    # Simple IRM with RandomForest
    ml_g = RandomForestRegressor()
    ml_m = RandomForestClassifier()

    dml_irm_simple = DoubleML.DoubleMLIRM(data_irm, ml_g, ml_m, score = :ATE)

    fit!(dml_irm_simple)

    coeftable(dml_irm_simple)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-dml_irm_simple">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d   0.67957   0.0189615    35.84    &lt;1e-99     0.642406     0.716733
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'>begin
    # IRM with TreeParzen hyperparameter tuning

    space = Dict(
        :max_depth =&gt; HP.QuantUniform(:max_depth, 2.0, 12.0, 1.0)
    )
    
    tuned_ml_g = TunedModel(
        model = EvoTreeRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 75, draws = 50),
        resampling = Holdout(fraction_train = 0.8),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_m = TunedModel(
        model = EvoTreeClassifier(),
        tuning = MLJTreeParzenTuning(random_trials = 75, draws = 50),
        resampling = Holdout(fraction_train = 0.8),
        range = space,
        measure = MLJ.brier_score,
        acceleration = CPUProcesses(),
    )


    dml_irm = DoubleML.DoubleMLIRM(data_irm, tuned_ml_g, tuned_ml_m)

    fit!(dml_irm, verbose = 1)

end</code></pre>
<pre class="code-output documenter-example-output" id="var-tuned_ml_m">DoubleMLIRM{Float32, MLJTuning.DeterministicTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeRegressor, Nothing}, MLJTuning.ProbabilisticTunedModel{MLJTreeParzenTuning, EvoTrees.EvoTreeClassifier, Nothing}}
==========================
StatsBase.CoefTable(Any[[0.5067636370658875], [0.020287886261940002], [24.9786319732666], [1.0436473597475374e-137], [0.46700011067004005], [0.5465271634617349]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_irm)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash997185">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.506764   0.0202879    24.98    &lt;1e-99        0.467     0.546527
────────────────────────────────────────────────────────────────────</pre>

<!-- PlutoStaticHTML.End -->
```

