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
    input_sha = "c914bf5e5f1edfbbb3db6b95b854c73dd2b4917d9b59a15147d3e8a99b237087"
    julia_version = "1.12.4"
-->
<pre class='language-julia'><code class='language-julia'>begin
    using Pkg
    Pkg.activate("..")
    Pkg.resolve()
    Pkg.instantiate()
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    Pkg.add("TreeParzen")
    Pkg.add("EvoTrees")
    Pkg.add("StableRNGs")
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    using DoubleML
    using StableRNGs
    using MLJ
    using MLJDecisionTreeInterface
    using TreeParzen
    using EvoTrees
end</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
end</code></pre>
<pre class="code-output documenter-example-output" id="var-#449#handle">MLJDecisionTreeInterface.RandomForestRegressor</pre>

<pre class='language-julia'><code class='language-julia'># PLR Data
data_plr = DoubleML.make_plr_CCDDHNR2018(5000, alpha = 0.5, dim_x = 20, rng = StableRNG(42))</code></pre>
<pre class="code-output documenter-example-output" id="var-data_plr">DoubleMLData{Float32, Vector{Float32}}(Float32[1.2566956, 4.16209, -0.79488957, 0.67903817, 0.69652003, 2.2781403, 3.7445633, -0.9087717, 0.84654087, 1.3683066  …  0.9275026, -0.06812125, 0.9634995, -2.82726, 0.6035034, 0.45007327, 1.7233989, -0.08308343, 0.41544443, -1.7952783], Float32[1.3364831, 2.6378622, -0.07964723, 0.9288351, -2.2224646, 0.66748816, 1.9711579, -1.4561268, -1.0280817, 2.306414  …  1.0702964, 0.07992836, -0.26474428, -2.1283271, -1.5947412, 0.9675669, 0.48882946, -1.9179863, -1.6641521, -2.1840165], Float32[-0.67025167 -0.14986733 … 0.41640848 -0.30865937; 2.085484 0.17391905 … -0.9364084 0.844609; … ; -0.3613366 -0.132516 … -0.69747114 -0.92015976; -3.212914 -1.6065888 … -0.5180494 -1.0555202], 5000, 20, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10, :X11, :X12, :X13, :X14, :X15, :X16, :X17, :X18, :X19, :X20])</pre>

<pre class='language-julia'><code class='language-julia'># Find matching models
models() do model
    matching(model, data_plr.x, data_plr.y)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-hash994832">11-element Vector{NamedTuple{(:name, :package_name, :is_supervised, :abstract_type, :constructor, :deep_properties, :docstring, :fit_data_scitype, :human_name, :hyperparameter_ranges, :hyperparameter_types, :hyperparameters, :implemented_methods, :inverse_transform_scitype, :is_pure_julia, :is_wrapper, :iteration_parameter, :load_path, :package_license, :package_url, :package_uuid, :predict_scitype, :prediction_type, :reporting_operations, :reports_feature_importances, :supports_class_weights, :supports_online, :supports_training_losses, :supports_weights, :tags, :target_in_fit, :transform_scitype, :input_scitype, :target_scitype, :output_scitype)}}:
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
    # Simple PLR with RandomForest
    ml_m = RandomForestRegressor()
    ml_g = RandomForestRegressor()

    dml_plr_simple = DoubleML.DoubleMLPLR(data_plr, ml_g, ml_m, n_folds = 4, n_rep = 1)

    fit!(dml_plr_simple)

    coeftable(dml_plr_simple)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-ml_m">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.495325   0.0137186    36.11    &lt;1e-99     0.468437     0.522213
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h1 id="Self-tuning-models">Self-tuning models</h1></div>

<pre class='language-julia'><code class='language-julia'>begin
    # PLR with TreeParzen hyperparameter tuning

    # Set up the hyperparameter space
    space = Dict(
        :n_trees =&gt; HP.Choice(:n_trees, Float64.(10:700)),
        :max_depth =&gt; HP.Choice(:max_depth, Float64.(1:10)),
        :min_samples_leaf =&gt; HP.Choice(:min_samples_leaf, Float64.(1:15)),
        :min_purity_increase =&gt; HP.Choice(:min_purity_increase, Float64.(0:3)),
        :sampling_fraction =&gt; HP.Choice(:sampling_fraction, Float64.(0.6:0.99)),
        :feature_importance =&gt; HP.Choice(:feature_importance, [:impurity, :split]),
    )

    # Set up the self-tuning models
    tuned_ml_m = TunedModel(
        model = RandomForestRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 100, max_simultaneous_draws = 5, linear_forgetting = 50),
        resampling = CV(nfolds = 3),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    tuned_ml_g = TunedModel(
        model = RandomForestRegressor(),
        tuning = MLJTreeParzenTuning(random_trials = 100, max_simultaneous_draws = 5, linear_forgetting = 50),
        resampling = CV(nfolds = 3),
        range = space,
        measure = MLJ.rmse,
        acceleration = CPUProcesses(),
    )

    # Pass the self-tuning models as learners to the DoubleMLPLR constructor
    dml_plr = DoubleML.DoubleMLPLR(data_plr, tuned_ml_g, tuned_ml_m, n_folds = 4, n_rep = 1)

    # Fit it
    fit!(dml_plr, verbose = 0)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-tuned_ml_m">DoubleMLPLR{Float32, MLJTuning.DeterministicTunedModel{MLJTreeParzenTuning, MLJDecisionTreeInterface.RandomForestRegressor, Nothing}, MLJTuning.DeterministicTunedModel{MLJTreeParzenTuning, MLJDecisionTreeInterface.RandomForestRegressor, Nothing}, Nothing}
==========================
StatsBase.CoefTable(Any[[0.5058156847953796], [0.013562729582190514], [37.294532775878906], [2.0125951526746944e-304], [0.4792332232822302], [0.5323981463085291]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash110135">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.505816   0.0135627    37.29    &lt;1e-99     0.479233     0.532398
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h1 id="Iterated-models">Iterated models</h1></div>

<pre class='language-julia'><code class='language-julia'># A simple example
# EvoTrees have in-built early stopping; the below is just for demonstration purposes.

begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(10),
        NumberLimit(30)
    ]

    # Set up learners with iteration control and early stopping
    ml_l_iterated = IteratedModel(
        EvoTreeRegressor(),
        resampling = Holdout(),
        measure = rmse,
        iteration_parameter = :nrounds,
        controls = controls
    )

    ml_m_iterated = IteratedModel(
        EvoTreeRegressor(),
        resampling = Holdout(),
        measure = rmse,
        iteration_parameter = :nrounds,
        controls = controls
    )

    # Pass the learners to the DoulbleMLPLR contructor
    dml_plr_iterated = DoubleML.DoubleMLPLR(data_plr, ml_l_iterated, ml_m_iterated, n_folds = 4, n_rep = 1)

    # Fit it
    fit!(dml_plr_iterated, verbose = 1)
end
</code></pre>
<pre class="code-output documenter-example-output" id="var-controls">DoubleMLPLR{Float32, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor}, MLJIteration.DeterministicIteratedModel{EvoTrees.EvoTreeRegressor}, Nothing}
==========================
StatsBase.CoefTable(Any[[0.4820321500301361], [0.014262313023209572], [33.79761505126953], [2.1379053565212677e-250], [0.4540785301684087], [0.5099857698918635]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr_iterated)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash219819">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.482032   0.0142623    33.80    &lt;1e-99     0.454079     0.509986
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'>summary(dml_plr_iterated)</code></pre>


<!-- PlutoStaticHTML.End -->
```

