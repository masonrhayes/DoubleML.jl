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
    input_sha = "56f7f90396552bdae2611d7ae0e91475044eaba9ca9be6d2d760365aa4f580b6"
    julia_version = "1.12.4"
-->







<div class="markdown"><h2 id="Load-packages-and-set-up-ML-models">Load packages and set up ML models</h2></div>

<pre class='language-julia'><code class='language-julia'>using DoubleML; using StableRNGs; using MLJ; using TreeParzen; using MLJDecisionTreeInterface; using EvoTrees</code></pre>


<pre class='language-julia'><code class='language-julia'>begin
    EvoTreeRegressor = @load EvoTreeRegressor pkg = EvoTrees verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0
end</code></pre>
<pre class="code-output documenter-example-output" id="var-#449#handle">MLJDecisionTreeInterface.RandomForestRegressor</pre>


<div class="markdown"><h2 id="Generate-PLR-data">Generate PLR data</h2></div>

<pre class='language-julia'><code class='language-julia'># PLR Data
data_plr = DoubleML.make_plr_CCDDHNR2018(1000, alpha = 0.5, dim_x = 20, rng = StableRNG(42))</code></pre>
<pre class="code-output documenter-example-output" id="var-data_plr">DoubleMLData{Float32, Vector{Float32}}(Float32[0.21417502, 0.9692105, -0.30895242, -0.048992738, 2.6029072, 1.4883567, 3.2605982, 1.0506742, -1.6848451, -0.40627423  …  -0.27870387, 2.7265549, 1.6948698, 2.7781668, -0.66583717, 0.78912425, 1.5714866, 0.21167372, 0.35720244, 1.2874476], Float32[-1.2644778, 1.2479526, -0.07056438, -1.4938519, 0.06656515, 1.2609595, 1.7690849, -0.95087236, -2.1373367, 0.64379275  …  -0.8578431, 0.8335334, 0.060642224, 1.8599323, -1.3742881, -1.7428911, 0.71109384, -0.05331813, -0.5296481, 1.9346514], Float32[-0.67025167 -0.14986733 … 0.41640848 -0.30865937; 2.085484 0.17391905 … -0.9364084 0.844609; … ; -0.5976384 -0.30607623 … 0.2812654 -0.3663869; 0.32168102 0.5612614 … 0.33550787 0.44790605], 1000, 20, :y, :d, [:X1, :X2, :X3, :X4, :X5, :X6, :X7, :X8, :X9, :X10, :X11, :X12, :X13, :X14, :X15, :X16, :X17, :X18, :X19, :X20])</pre>


<div class="markdown"><p>We can check what models are available for predicting the outcome variable:</p></div>

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


<div class="markdown"><h2 id="Run-a-simple-model">Run a simple model</h2></div>

<pre class='language-julia'><code class='language-julia'>begin
    # Simple PLR with RandomForest
    ml_m = RandomForestRegressor()
    ml_g = RandomForestRegressor()

    dml_plr_simple = DoubleML.DoubleMLPLR(data_plr, ml_g, ml_m, n_folds = 4, n_rep = 1)

    fit!(dml_plr_simple)
end</code></pre>
<pre class="code-output documenter-example-output" id="var-ml_m">DoubleMLPLR{Float32, MLJDecisionTreeInterface.RandomForestRegressor, MLJDecisionTreeInterface.RandomForestRegressor, Nothing}
==========================
StatsBase.CoefTable(Any[[0.49683234095573425], [0.030681872740387917], [16.193025588989258], [5.648170330220501e-59], [0.43669697540633257], [0.5569677065051359]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr_simple)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash142443">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.496832   0.0306819    16.19    &lt;1e-58     0.436697     0.556968
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h1 id="Advanced-example:-self-tuning-models">Advanced example: self-tuning models</h1></div>

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
StatsBase.CoefTable(Any[[0.5114830136299133], [0.030144501477479935], [16.96770477294922], [1.4238954846254198e-64], [0.4524008764021381], [0.5705651508576886]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash110135">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.511483   0.0301445    16.97    &lt;1e-63     0.452401     0.570565
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h1 id="Advanced-example:-iterated-models">Advanced example: iterated models</h1></div>

<pre class='language-julia'><code class='language-julia'># A simple example
# EvoTrees have in-built early stopping as an option; the below is just for demonstration purposes.

begin
    # Set up iteration controls
    controls = [
        Step(1),
        Patience(10),
        NumberLimit(30),
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
StatsBase.CoefTable(Any[[0.4626036584377289], [0.03196043521165848], [14.474260330200195], [1.7621110128896377e-47], [0.3999623564926524], [0.5252449603828054]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr_iterated)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash219819">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.462604   0.0319604    14.47    &lt;1e-46     0.399962     0.525245
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'>summary(dml_plr_iterated)</code></pre>


<!-- PlutoStaticHTML.End -->
```

