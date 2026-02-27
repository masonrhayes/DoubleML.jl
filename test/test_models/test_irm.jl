using DoubleML
using Test
using DataFrames
using MLJ
using StableRNGs
using StatsBase
using Statistics

# Load models outside to avoid world age issues
Tree = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0

@testset "DoubleMLIRM" begin
    @testset "Basic ATE estimation" begin
        rng = StableRNG(12345)
        n_obs = 500

        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        ml_g = Tree()
        ml_m = LogisticClassifier()

        model = DoubleMLIRM(data, ml_g, ml_m; n_folds = 3, n_rep = 1, score = :ATE)

        @test model.score_obj isa DoubleML.ATEScore
        @test model.n_folds == 3
        @test model.n_rep == 1
        @test !isfitted(model)

        # Fit the model
        fit!(model)

        @test isfitted(model)

        # Check coefficient is reasonable (true theta = 0.5)
        @test abs(model.coef - 0.5) < 1.0

        # Check standard error is positive and reasonable
        @test 0 < model.se < 1.0

        # Test that we can get inference statistics
        ct = coeftable(model)
        @test ct isa StatsBase.CoefTable

        # Verify psi arrays are populated
        @test length(model.psi) == n_obs
        @test length(model.psi_a) == n_obs
        @test length(model.psi_b) == n_obs
        @test !isempty(model.fitted_learners_g0)
        @test !isempty(model.fitted_learners_g1)
        @test !isempty(model.fitted_learners_m)
    end

    @testset "ATTE estimation" begin
        rng = StableRNG(54321)
        n_obs = 500

        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        ml_g = Tree()
        ml_m = LogisticClassifier()

        model = DoubleMLIRM(data, ml_g, ml_m; n_folds = 3, n_rep = 1, score = :ATTE)

        @test model.score_obj isa DoubleML.ATTEScore

        # Fit the model
        fit!(model)

        @test isfitted(model)

        # Check coefficient is reasonable
        @test abs(model.coef - 0.5) < 1.0
        @test 0 < model.se < 1.0

        # For ATTE, ml_g1 should be empty (only control outcomes are modeled)
        @test isempty(model.fitted_learners_g1)
    end

    @testset "Parameter validation" begin
        data = make_irm_data(100; theta = 0.0)

        # Invalid score - throws ArgumentError
        @test_throws ArgumentError DoubleMLIRM(data, Tree(), LogisticClassifier(); score = :invalid)

        # Invalid n_folds - throws DomainError
        @test_throws DomainError DoubleMLIRM(data, Tree(), LogisticClassifier(); n_folds = 0)

        # Invalid n_rep - throws DomainError
        @test_throws DomainError DoubleMLIRM(data, Tree(), LogisticClassifier(); n_rep = 0)

        # Invalid clipping_threshold - throws DomainError
        @test_throws DomainError DoubleMLIRM(data, Tree(), LogisticClassifier(); clipping_threshold = 0.0)
        @test_throws DomainError DoubleMLIRM(data, Tree(), LogisticClassifier(); clipping_threshold = 0.5)
    end

    @testset "StatsAPI methods" begin
        rng = StableRNG(12345)
        data = make_irm_data(300; theta = 0.5, rng = rng)

        model = DoubleMLIRM(data, Tree(), LogisticClassifier(); n_folds = 3)

        fit!(model)

        @test length(coef(model)) == 1
        @test length(stderror(model)) == 1
        @test size(vcov(model)) == (1, 1)
        @test size(confint(model)) == (1, 2)
        @test nobs(model) == data.n_obs
        @test dof(model) == 1
        @test responsename(model) == "y"
        @test coefnames(model) == ["d"]

        fp = fitted_params(model)
        @test haskey(fp, :ml_g0)
        @test haskey(fp, :ml_m)
    end

    @testset "Multiple repetitions" begin
        rng = StableRNG(12345)
        data = make_irm_data(400; theta = 0.5, rng = rng)

        model = DoubleMLIRM(data, Tree(), LogisticClassifier(); n_folds = 3, n_rep = 2)
        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
    end

    @testset "TunedModel - Full sample tuning" begin
        rng = StableRNG(12345)
        n_obs = 500
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        # Create a TunedModel for ml_g with a simple tuning grid
        tree_model = Tree()
        tuned_g = MLJ.TunedModel(
            model = tree_model,
            tuning = Grid(resolution = 2),
            resampling = CV(nfolds = 3),
            measure = rms,
            range = [
                range(tree_model, :max_depth, lower = 2, upper = 5),
                range(tree_model, :min_samples_leaf, lower = 5, upper = 10),
            ]
        )

        # Use TunedModel with full sample tuning (n_folds_tune = 0)
        model = DoubleMLIRM(data, tuned_g, LogisticClassifier(); n_folds = 3, n_rep = 1, score = :ATE, n_folds_tune = 0)

        @test model isa DoubleMLIRM
        @test model.ml_g isa MLJ.MLJTuning.DeterministicTunedModel

        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
        @test model.se !== nothing

        # Check coefficient is reasonable (true theta = 0.5)
        @test abs(model.coef - 0.5) < 1.0
        @test 0 < model.se < 1.0
    end

    @testset "TunedModel - Per-repetition tuning" begin
        rng = StableRNG(54321)
        n_obs = 400
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        # Create a simple TunedModel for ml_m
        log_model = LogisticClassifier()
        tuned_m = MLJ.TunedModel(
            model = log_model,
            tuning = Grid(resolution = 2),
            resampling = CV(nfolds = 3),
            measure = log_loss,
            range = range(log_model, :lambda, lower = 0.001, upper = 0.1)
        )

        # Use TunedModel with per-repetition tuning (n_folds_tune > 0)
        model = DoubleMLIRM(data, Tree(), tuned_m; n_folds = 3, n_rep = 2, score = :ATE, n_folds_tune = 2)

        @test model isa DoubleMLIRM
        @test model.ml_m isa MLJ.MLJTuning.ProbabilisticTunedModel
        @test model.n_folds_tune == 2

        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
        @test model.se !== nothing

        # Check coefficient is reasonable
        @test abs(model.coef - 0.5) < 1.0
        @test 0 < model.se < 1.0
    end

    @testset "TunedModel - Both learners tuned" begin
        rng = StableRNG(98765)
        n_obs = 400
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        # Tune both ml_g and ml_m
        tree_model = Tree()
        tuned_g = MLJ.TunedModel(
            model = tree_model,
            tuning = Grid(resolution = 2),
            resampling = CV(nfolds = 3),
            measure = rms,
            range = range(tree_model, :max_depth, lower = 2, upper = 4)
        )

        log_model = LogisticClassifier()
        tuned_m = MLJ.TunedModel(
            model = log_model,
            tuning = Grid(resolution = 2),
            resampling = CV(nfolds = 3),
            measure = log_loss,
            range = range(log_model, :lambda, lower = 0.01, upper = 0.1)
        )

        # Test ATE with full sample tuning
        model = DoubleMLIRM(data, tuned_g, tuned_m; n_folds = 3, n_rep = 1, score = :ATE, n_folds_tune = 0)

        @test model.ml_g isa MLJ.MLJTuning.DeterministicTunedModel
        @test model.ml_m isa MLJ.MLJTuning.ProbabilisticTunedModel

        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
        @test model.se !== nothing
    end

    @testset "TunedModel - ATTE score" begin
        rng = StableRNG(11111)
        n_obs = 400
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)

        tree_model = Tree()
        tuned_g = MLJ.TunedModel(
            model = tree_model,
            tuning = Grid(resolution = 2),
            resampling = CV(nfolds = 3),
            measure = rms,
            range = range(tree_model, :max_depth, lower = 2, upper = 5)
        )

        model = DoubleMLIRM(data, tuned_g, LogisticClassifier(); n_folds = 3, n_rep = 1, score = :ATTE, n_folds_tune = 0)
        fit!(model)

        @test isfitted(model)
        @test model.score_obj isa DoubleML.ATTEScore
        @test model.coef !== nothing
    end

    @testset "Binary treatment validation" begin
        n = 100
        df = DataFrame(y = rand(n), d = rand(n), x1 = randn(n), x2 = randn(n))
        data = DoubleMLData(df; y_col = :y, d_col = :d, x_cols = [:x1, :x2])

        @test_throws ArgumentError DoubleMLIRM(data, Tree(), LogisticClassifier())
    end

    @testset "IteratedModel with XGBoost - ATE" begin
        # Load XGBoost models
        XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0
        XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost verbosity = 0

        # Generate data
        rng = StableRNG(12345)
        data = make_irm_data(500; theta = 0.5, rng = rng)

        # Create iterated models with early stopping
        # For ml_g (outcome model E[Y|X,D])
        ml_g_iterated = MLJ.IteratedModel(
            model = XGBoostRegressor(),
            resampling = Holdout(fraction_train = 0.8),
            measure = rmse,
            iteration_parameter = :num_round,
            controls = [Step(1), Patience(8), NumberLimit(25)]
        )

        # For ml_m (propensity score E[D|X])
        ml_m_iterated = MLJ.IteratedModel(
            model = XGBoostClassifier(),
            resampling = Holdout(fraction_train = 0.8),
            measure = log_loss,
            iteration_parameter = :num_round,
            controls = [Step(1), Patience(8), NumberLimit(25)]
        )

        # Create and fit IRM model with ATE score
        model = DoubleMLIRM(
            data, ml_g_iterated, ml_m_iterated;
            n_folds = 3, n_rep = 1, score = :ATE
        )

        @test model isa DoubleMLIRM
        @test model.ml_g isa MLJ.MLJIteration.DeterministicIteratedModel
        @test model.ml_m isa MLJ.MLJIteration.ProbabilisticIteratedModel

        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
        @test model.se !== nothing

        # Wide tolerance check (true theta = 0.5)
        # Note: IteratedModel with XGBoost can have high variance due to early stopping
        @test abs(model.coef - 0.5) < 0.75
    end

    @testset "IteratedModel with XGBoost - ATTE" begin
        # Load XGBoost models
        XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0
        XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost verbosity = 0

        # Generate data
        rng = StableRNG(54321)
        data = make_irm_data(500; theta = 0.5, rng = rng)

        # Create iterated models with early stopping
        # For ml_g (outcome model E[Y|X,D])
        ml_g_iterated = MLJ.IteratedModel(
            model = XGBoostRegressor(),
            resampling = Holdout(fraction_train = 0.8),
            measure = rmse,
            iteration_parameter = :num_round,
            controls = [Step(1), Patience(8), NumberLimit(25)]
        )

        # For ml_m (propensity score E[D|X])
        ml_m_iterated = MLJ.IteratedModel(
            model = XGBoostClassifier(),
            resampling = Holdout(fraction_train = 0.8),
            measure = log_loss,
            iteration_parameter = :num_round,
            controls = [Step(1), Patience(8), NumberLimit(25)]
        )

        # Create and fit IRM model with ATTE score
        model = DoubleMLIRM(
            data, ml_g_iterated, ml_m_iterated;
            n_folds = 3, n_rep = 1, score = :ATTE
        )

        @test model isa DoubleMLIRM
        @test model.ml_g isa MLJ.MLJIteration.DeterministicIteratedModel
        @test model.ml_m isa MLJ.MLJIteration.ProbabilisticIteratedModel
        @test model.score_obj isa DoubleML.ATTEScore

        fit!(model)

        @test isfitted(model)
        @test model.coef !== nothing
        @test model.se !== nothing

        # Wide tolerance check (true theta = 0.5)
        # ATTE is not very precise at low sample size...
        @test abs(model.coef - 0.5) < 1.5
    end
end
