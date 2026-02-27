using DoubleML
using Test
using MLJ
using MLJLinearModels
using StatsBase
using StatsAPI

# Load needed models
LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0

@testset "DoubleMLPLR Basic" begin
    # 1. Generate PLR data
    n_obs = 500
    alpha_true = 0.5
    data = make_plr_CCDDHNR2018(n_obs; alpha = alpha_true)

    # 2. Setup MLJ models
    ml_l = LinearRegressor()
    ml_m = LinearRegressor()

    # 3. Create and fit model
    dml = DoubleMLPLR(data, ml_l, ml_m; n_folds = 3)
    fit!(dml)

    # 4. Check results - coefficient estimate is reasonable
    @test dml.se > 0
    # Tolerance is loose because n_obs is small and models are simple
    @test isapprox(dml.coef, alpha_true; atol = 0.25)

    # 5. Check confidence interval contains estimate
    ci = confint(dml)
    @test ci[1] < dml.coef < ci[2]

    # StatsAPI consistency checks
    @test coef(dml)[1] == dml.coef
    @test stderror(dml)[1] == dml.se
    @test nobs(dml) == n_obs

    # Check coeftable format
    ct = coeftable(dml)
    @test ct isa StatsBase.CoefTable

    # Verify model was fitted and psi arrays are populated
    @test length(dml.psi) == n_obs
    @test length(dml.psi_a) == n_obs
    @test length(dml.psi_b) == n_obs
    @test !isempty(dml.fitted_learners_l)
    @test !isempty(dml.fitted_learners_m)
end

@testset "IV-type ml_g validation" begin
    data = make_plr_CCDDHNR2018(100; alpha = 0.5)
    ml_l = LinearRegressor()
    ml_m = LinearRegressor()

    # Test 1: ml_g must be provided for IV-type
    @test_throws ArgumentError DoubleMLPLR(data, ml_l, ml_m; score = :IV_type)

    # Test 2: ml_g must be regressor, not classifier
    LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0
    @test_throws ArgumentError DoubleMLPLR(
        data, ml_l, ml_m;
        ml_g = LogisticClassifier(), score = :IV_type
    )

    # Verify the error message mentions the specific type
    try
        DoubleMLPLR(data, ml_l, ml_m; ml_g = LogisticClassifier(), score = :IV_type)
    catch e
        @test e isa ArgumentError
        err_msg = sprint(showerror, e)
        @test occursin("LogisticClassifier", err_msg)
        @test occursin("Probabilistic", err_msg)
        @test occursin("Deterministic", err_msg)
    end

    # Test 3: IV-type works with regressor ml_g
    ml_g = LinearRegressor()
    model = DoubleMLPLR(data, ml_l, ml_m; ml_g = ml_g, score = :IV_type, n_folds = 3)
    fit!(model)
    @test isfitted(model)
    @test !isnan(model.coef)
    @test model.se > 0

    # Test 4: Check that learner_g returns the ml_g model
    @test learner_g(model) === ml_g

    # Test 5: IV-type produces different estimate than partialling out
    model_po = DoubleMLPLR(data, ml_l, ml_m; n_folds = 3)
    fit!(model_po)
    # Estimates should be similar but not identical due to different score functions
    @test abs(model.coef - model_po.coef) < 0.3
end

@testset "Partialling out with different learner types" begin
    using StableRNGs
    rng = StableRNG(42)

    # Test with standard regressors (both ml_l and ml_m as Deterministic)
    data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
    ml_l = LinearRegressor()
    ml_m = LinearRegressor()
    model = DoubleMLPLR(data, ml_l, ml_m; n_folds = 3)
    fit!(model)
    @test isfitted(model)
    @test !isnan(model.coef)
    @test model.se > 0

    # Test consistency: same data should give same results with same seed
    data2 = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = StableRNG(42))
    model2 = DoubleMLPLR(data2, LinearRegressor(), LinearRegressor(); n_folds = 3)
    fit!(model2)
    # Coefficients should be similar but not necessarily identical
    # due to different RNG states in sample splitting
    @test abs(model.coef - model2.coef) < 0.5

    # Verify that both models fitted correctly
    @test isfitted(model2)
    @test model2.se > 0
end

@testset "ml_g unused warning for partialling out" begin
    data = make_plr_CCDDHNR2018(100; alpha = 0.5)
    ml_l = LinearRegressor()
    ml_m = LinearRegressor()
    ml_g = LinearRegressor()

    # Should show warning when ml_g provided with partialling out
    @test_logs (:warn, r"ml_g was provided but will not be used") DoubleMLPLR(
        data, ml_l, ml_m; ml_g = ml_g, score = :partialling_out, n_folds = 3
    )

    # Should NOT warn when ml_g is nothing
    @test_logs DoubleMLPLR(
        data, ml_l, ml_m; ml_g = nothing, score = :partialling_out, n_folds = 3
    )

    # Should NOT warn when ml_g omitted entirely (defaults to nothing)
    @test_logs DoubleMLPLR(data, ml_l, ml_m; score = :partialling_out, n_folds = 3)
end

@testset "IteratedModel with XGBoost - Partialling Out" begin
    # Load XGBoost models
    XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0

    # Generate data
    data = make_plr_CCDDHNR2018(500; alpha = 0.5)

    # Create iterated models with early stopping
    # For ml_l (E[Y|X])
    ml_l_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(8), NumberLimit(25)]
    )

    # For ml_m (E[D|X])
    ml_m_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(8), NumberLimit(25)]
    )

    # Create and fit PLR model
    model = DoubleMLPLR(data, ml_l_iterated, ml_m_iterated; n_folds = 3, n_rep = 1)

    @test model isa DoubleMLPLR
    @test model.ml_l isa MLJ.MLJIteration.DeterministicIteratedModel
    @test model.ml_m isa MLJ.MLJIteration.DeterministicIteratedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check (true alpha = 0.5)
    @test abs(model.coef - 0.5) < 0.5
end

@testset "IteratedModel with XGBoost - IV-type" begin
    # Load XGBoost models
    XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0

    # Generate data
    data = make_plr_CCDDHNR2018(500; alpha = 0.5)

    # Create iterated models with early stopping
    # For ml_l (E[Y|X])
    ml_l_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(8), NumberLimit(25)]
    )

    # For ml_m (E[D|X])
    ml_m_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(8), NumberLimit(25)]
    )

    # For ml_g (E[Y - θD|X]) - used in IV-type score
    ml_g_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(8), NumberLimit(25)]
    )

    # Create and fit PLR model with IV-type score
    model = DoubleMLPLR(
        data, ml_l_iterated, ml_m_iterated;
        ml_g = ml_g_iterated, score = :IV_type, n_folds = 3, n_rep = 1
    )

    @test model isa DoubleMLPLR
    @test model.ml_l isa MLJ.MLJIteration.DeterministicIteratedModel
    @test model.ml_m isa MLJ.MLJIteration.DeterministicIteratedModel
    @test model.ml_g isa MLJ.MLJIteration.DeterministicIteratedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check (true alpha = 0.5)
    @test abs(model.coef - 0.5) < 0.5
end

@testset "TunedModel with DecisionTree - Partialling Out" begin
    # Load DecisionTree model
    Tree = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

    # Generate data
    rng = StableRNG(12345)
    data = make_plr_CCDDHNR2018(500; alpha = 0.5, rng = rng)

    # Create tuned models for ml_l and ml_m
    tree_model = Tree()
    tuned_l = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 5)
    )

    tuned_m = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 5)
    )

    # Use TunedModel with full sample tuning
    model = DoubleMLPLR(data, tuned_l, tuned_m; n_folds = 3, n_rep = 1, score = :partialling_out, n_folds_tune = 0)

    @test model isa DoubleMLPLR
    @test model.ml_l isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.ml_m isa MLJ.MLJTuning.DeterministicTunedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check (true alpha = 0.5)
    @test abs(model.coef - 0.5) < 0.5
end

@testset "TunedModel with DecisionTree - IV-type" begin
    # Load DecisionTree model
    Tree = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

    # Generate data
    rng = StableRNG(54321)
    data = make_plr_CCDDHNR2018(500; alpha = 0.5, rng = rng)

    # Create tuned models
    tree_model = Tree()
    tuned_l = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 5)
    )

    tuned_m = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 5)
    )

    tuned_g = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 5)
    )

    # Use TunedModel with IV-type score
    model = DoubleMLPLR(data, tuned_l, tuned_m; ml_g = tuned_g, score = :IV_type, n_folds = 3, n_rep = 1, n_folds_tune = 0)

    @test model isa DoubleMLPLR
    @test model.ml_l isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.ml_m isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.ml_g isa MLJ.MLJTuning.DeterministicTunedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check
    @test abs(model.coef - 0.5) < 0.5
end

@testset "TunedModel - Per-repetition tuning" begin
    # Load DecisionTree model
    Tree = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

    # Generate data
    rng = StableRNG(98765)
    data = make_plr_CCDDHNR2018(400; alpha = 0.5, rng = rng)

    # Create tuned model for ml_m only
    tree_model = Tree()
    tuned_m = MLJ.TunedModel(
        model = tree_model,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(tree_model, :max_depth, lower = 2, upper = 4)
    )

    # Use TunedModel with per-repetition tuning
    model = DoubleMLPLR(data, Tree(), tuned_m; n_folds = 3, n_rep = 2, score = :partialling_out, n_folds_tune = 2)

    @test model isa DoubleMLPLR
    @test model.ml_m isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.n_folds_tune == 2

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing

    # Check coefficient is reasonable
    @test abs(model.coef - 0.5) < 0.5
end
