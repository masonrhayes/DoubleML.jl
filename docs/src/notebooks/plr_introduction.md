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
    input_sha = "7931131912ed756759e167445cc37f5a60abce6a345ed5972c76539e26a2874a"
    julia_version = "1.12.4"
-->
<pre class='language-julia'><code class='language-julia'>import Pkg; Pkg.develop(path=joinpath(@__DIR__, "../.."))</code></pre>


<pre class='language-julia'><code class='language-julia'>Pkg.activate(joinpath(@__DIR__, "../../examples"))</code></pre>


<pre class='language-julia'><code class='language-julia'>using DoubleML; using StableRNGs; using MLJ; using TreeParzen; using MLJDecisionTreeInterface; using EvoTrees</code></pre>


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
d  0.493806   0.0136729    36.12    &lt;1e-99     0.467007     0.520604
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
StatsBase.CoefTable(Any[[0.5018609166145325], [0.013468807563185692], [37.2609748840332], [7.037737654769499e-304], [0.4754625388759878], [0.5282592943530772]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash110135">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.501861   0.0134688    37.26    &lt;1e-99     0.475463     0.528259
────────────────────────────────────────────────────────────────────</pre>


<div class="markdown"><h1 id="Iterated-models">Iterated models</h1></div>

<pre class='language-julia'><code class='language-julia'># A simple example
# EvoTrees have in-built early stopping; the below is just for demonstration purposes.

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
StatsBase.CoefTable(Any[[0.48700183629989624], [0.014193546026945114], [34.311500549316406], [5.287256313219979e-258], [0.45918299727417217], [0.5148206753256203]], ["Estimate", "Std. Error", "z value", "Pr(&gt;|z|)", "Lower 95.0%", "Upper 95.0%"], ["d"], 4, 3)</pre>

<pre class='language-julia'><code class='language-julia'>coeftable(dml_plr_iterated)</code></pre>
<pre class="code-output documenter-example-output" id="var-hash219819">────────────────────────────────────────────────────────────────────
   Estimate  Std. Error  z value  Pr(&gt;|z|)  Lower 95.0%  Upper 95.0%
────────────────────────────────────────────────────────────────────
d  0.487002   0.0141935    34.31    &lt;1e-99     0.459183     0.514821
────────────────────────────────────────────────────────────────────</pre>

<pre class='language-julia'><code class='language-julia'>summary(dml_plr_iterated)</code></pre>


<!-- PlutoStaticHTML.End -->
```

