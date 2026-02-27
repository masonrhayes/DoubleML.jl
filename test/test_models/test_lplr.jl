using DoubleML
using Test
using MLJ
using MLJLinearModels
using DataFrames
using StableRNGs
using StatsFuns: logistic

rng = StableRNG(42)

@testset "DoubleMLLPLR Model" begin

    @testset "Constructor and Validation" begin
        # Create binary outcome data - must be Float for DoubleMLData
        n = 200
        df = DataFrame(
            y = Float64.(rand(rng, [0, 1], n)),  # Convert to Float64
            d = rand(rng, n),
            x1 = randn(rng, n),
            x2 = randn(rng, n)
        )
        data = DoubleMLData(df; y_col = :y, d_col = :d, x_cols = [:x1, :x2])

        # Load learners
        LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        ml_M = LogisticClassifier()
        ml_t = LinearRegressor()
        ml_m = LinearRegressor()

        @testset "Basic construction" begin
            model = DoubleMLLPLR(data, ml_M, ml_t, ml_m)
            @test model isa DoubleMLLPLR
            @test model.n_folds == 5
            @test model.n_folds_inner == 5
            @test model.score_obj isa NuisanceSpaceScore
        end

        @testset "Score selection" begin
            model_nuisance = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :nuisance_space)
            @test model_nuisance.score_obj isa NuisanceSpaceScore

            model_instrument = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :instrument)
            @test model_instrument.score_obj isa InstrumentScore
        end

        @testset "Invalid score type" begin
            @test_throws ArgumentError DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :invalid)
        end

        @testset "Binary outcome validation" begin
            # Non-binary outcome should throw
            df_invalid = DataFrame(
                y = rand(rng, n),  # Continuous, not binary
                d = rand(rng, n),
                x1 = randn(rng, n)
            )
            data_invalid = DoubleMLData(df_invalid; y_col = :y, d_col = :d, x_cols = [:x1])
            @test_throws ArgumentError DoubleMLLPLR(data_invalid, ml_M, ml_t, ml_m)
        end

        @testset "Learner type validation" begin
            # ml_M must be probabilistic
            @test_throws ArgumentError DoubleMLLPLR(data, ml_t, ml_t, ml_m)  # ml_t is deterministic

            # ml_t must be deterministic
            @test_throws ArgumentError DoubleMLLPLR(data, ml_M, ml_M, ml_m)  # ml_M is probabilistic
        end
    end

    @testset "Fitting" begin
        n = 300
        # Generate data with known relationship
        X = randn(rng, n, 2)
        D = rand(rng, n)
        # Binary outcome with logistic relationship
        linear_pred = 0.5 .* D .+ 0.3 .* X[:, 1] .- 0.2 .* X[:, 2]
        prob = logistic.(linear_pred)
        Y = Float64.(rand(rng, n) .< prob)

        df = DataFrame(
            y = Y,
            d = D,
            x1 = X[:, 1],
            x2 = X[:, 2]
        )
        data = DoubleMLData(df; y_col = :y, d_col = :d, x_cols = [:x1, :x2])

        LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        ml_M = LogisticClassifier()
        ml_t = LinearRegressor()
        ml_m = LinearRegressor()

        @testset "nuisance_space score" begin
            model = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :nuisance_space, n_folds = 3, n_folds_inner = 3)
            fit!(model)

            @test isfitted(model)
            @test !isnan(model.coef)
            @test !isnan(model.se)
            @test model.se > 0

            # Check coefficient is in reasonable range (true value is 0.5)
            @test abs(model.coef - 0.5) < 1.0  # Very loose tolerance for small sample
        end

        @testset "instrument score" begin
            model = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :instrument, n_folds = 3, n_folds_inner = 3)
            fit!(model)

            @test isfitted(model)
            @test !isnan(model.coef)
            @test !isnan(model.se)
            @test model.se > 0
        end

        @testset "API methods" begin
            model = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :nuisance_space, n_folds = 3)
            fit!(model)

            # Test StatsAPI methods
            @test length(coef(model)) == 1
            @test length(stderror(model)) == 1
            @test size(vcov(model)) == (1, 1)
            @test nobs(model) == n

            # Test confint
            ci = confint(model)
            @test size(ci) == (1, 2)
            @test ci[1, 1] < ci[1, 2]  # Lower < Upper
            @test ci[1, 1] < model.coef < ci[1, 2]  # Coef in interval
        end

        @testset "Bootstrap" begin
            model = DoubleMLLPLR(data, ml_M, ml_t, ml_m; score = :nuisance_space, n_folds = 3)
            fit!(model)
            bootstrap!(model; n_rep_boot = 100)

            @test has_bootstrapped(model)
            @test size(model.boot_t_stat, 1) == 100

            # Test joint confidence intervals
            ci_joint = confint(model; joint = true)
            @test size(ci_joint) == (1, 2)
        end
    end

    @testset "Score Functions" begin
        using DoubleML: compute_score_elements, compute_score, compute_score_deriv
        using DoubleML: NuisanceSpaceScore, InstrumentScore

        n = 100
        Y = Float64.(rand(rng, [0, 1], n))  # Float64, not Int64
        D = rand(rng, n)
        t_hat = randn(rng, n)  # t(X) = E[logit(M) | X]
        a_hat = rand(rng, n)    # a(X) = E[D | X]
        m_hat = rand(rng, n)    # m(X) = E[D | X, Y=0]

        @testset "NuisanceSpaceScore" begin
            score_obj = NuisanceSpaceScore()
            elements = compute_score_elements(score_obj, Y, D, t_hat, a_hat, m_hat)

            @test haskey(elements, :y)
            @test haskey(elements, :d)
            @test haskey(elements, :d_tilde)
            @test haskey(elements, :t_hat)
            @test haskey(elements, :a_hat)

            # Test score computation
            coef = 0.5
            score = compute_score(score_obj, coef, elements)
            @test length(score) == n
            @test all(isfinite, score)

            # Test derivative computation
            deriv = compute_score_deriv(score_obj, coef, elements)
            @test length(deriv) == n
            @test all(isfinite, deriv)
        end

        @testset "InstrumentScore" begin
            score_obj = InstrumentScore()
            elements = compute_score_elements(score_obj, Y, D, t_hat, a_hat, m_hat)

            @test haskey(elements, :y)
            @test haskey(elements, :d)
            @test haskey(elements, :d_tilde)
            @test haskey(elements, :t_hat)
            @test haskey(elements, :a_hat)

            # Test score computation
            coef = 0.5
            score = compute_score(score_obj, coef, elements)
            @test length(score) == n
            @test all(isfinite, score)

            # Test derivative computation
            deriv = compute_score_deriv(score_obj, coef, elements)
            @test length(deriv) == n
            @test all(isfinite, deriv)
        end
    end

    @testset "Helper Functions" begin
        using DoubleML: _validate_binary_outcome

        @testset "binary validation" begin
            @test _validate_binary_outcome([0, 1, 0, 1]) === nothing
            @test _validate_binary_outcome([0.0, 1.0, 0.0]) === nothing
            @test_throws ArgumentError _validate_binary_outcome([0, 2, 1])
            @test_throws ArgumentError _validate_binary_outcome([1, 2, 3])
        end

    end
end

@testset "TunedModel with DecisionTree - Nuisance Space" begin
    # Load models
    RandomForestClassifier = @load RandomForestClassifier pkg = DecisionTree verbosity = 0
    RandomForestRegressor = @load RandomForestRegressor pkg = DecisionTree verbosity = 0

    # Generate LPLR data
    rng = StableRNG(12345)
    data = make_lplr_LZZ2020(500; alpha = 0.5, dim_x = 20, rng = rng, treatment = "continuous")

    # Create tuned models
    rf_M = RandomForestClassifier()
    tuned_M = MLJ.TunedModel(
        model = rf_M,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = log_loss,
        range = range(rf_M, :n_trees, lower = 10, upper = 50)
    )

    rf_t = RandomForestRegressor()
    tuned_t = MLJ.TunedModel(
        model = rf_t,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(rf_t, :n_trees, lower = 10, upper = 50)
    )

    rf_m = RandomForestRegressor()
    tuned_m = MLJ.TunedModel(
        model = rf_m,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(rf_m, :n_trees, lower = 10, upper = 50)
    )

    # Use TunedModel with full sample tuning
    model = DoubleMLLPLR(data, tuned_M, tuned_t, tuned_m; score = :nuisance_space, n_folds = 3, n_rep = 1, n_folds_tune = 0)

    @test model isa DoubleMLLPLR
    @test model.ml_M isa MLJ.MLJTuning.ProbabilisticTunedModel
    @test model.ml_t isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.ml_m isa MLJ.MLJTuning.DeterministicTunedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check
    @test abs(model.coef - 0.5) < 0.5
end

@testset "TunedModel with XGBoost - Instrument Score" begin
    # Load XGBoost models - more stable for instrument score
    XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost verbosity = 0
    XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0

    # Generate LPLR data
    rng = StableRNG(54321)
    data = make_lplr_LZZ2020(500; alpha = 0.5, dim_x = 20, rng = rng, treatment = "continuous")

    # Create tuned models with XGBoost (more stable than RandomForest)
    xgb_M = XGBoostClassifier()
    tuned_M = MLJ.TunedModel(
        model = xgb_M,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = log_loss,
        range = range(xgb_M, :max_depth, lower = 3, upper = 6)
    )

    xgb_t = XGBoostRegressor()
    tuned_t = MLJ.TunedModel(
        model = xgb_t,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(xgb_t, :max_depth, lower = 3, upper = 6)
    )

    xgb_m = XGBoostRegressor()
    tuned_m = MLJ.TunedModel(
        model = xgb_m,
        tuning = Grid(resolution = 2),
        resampling = CV(nfolds = 3),
        measure = rms,
        range = range(xgb_m, :max_depth, lower = 3, upper = 6)
    )

    # Use TunedModel with instrument score
    model = DoubleMLLPLR(data, tuned_M, tuned_t, tuned_m; score = :instrument, n_folds = 3, n_rep = 1, n_folds_tune = 0)

    @test model isa DoubleMLLPLR
    @test model.ml_M isa MLJ.MLJTuning.ProbabilisticTunedModel
    @test model.ml_t isa MLJ.MLJTuning.DeterministicTunedModel
    @test model.ml_m isa MLJ.MLJTuning.DeterministicTunedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check
    @test abs(model.coef - 0.5) < 0.5
end

@testset "IteratedModel with XGBoost - Nuisance Space" begin
    # Load XGBoost models
    XGBoostClassifier = @load XGBoostClassifier pkg = XGBoost verbosity = 0
    XGBoostRegressor = @load XGBoostRegressor pkg = XGBoost verbosity = 0

    # Generate LPLR data
    rng = StableRNG(98765)
    data = make_lplr_LZZ2020(400; alpha = 0.5, dim_x = 20, rng = rng, treatment = "continuous")

    # Create iterated models with early stopping
    ml_M_iterated = MLJ.IteratedModel(
        model = XGBoostClassifier(),
        resampling = Holdout(fraction_train = 0.8),
        measure = log_loss,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(5), NumberLimit(20)]
    )

    ml_t_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(5), NumberLimit(20)]
    )

    ml_m_iterated = MLJ.IteratedModel(
        model = XGBoostRegressor(),
        resampling = Holdout(fraction_train = 0.8),
        measure = rmse,
        iteration_parameter = :num_round,
        controls = [Step(1), Patience(5), NumberLimit(20)]
    )

    # Create LPLR model with IteratedModels
    model = DoubleMLLPLR(data, ml_M_iterated, ml_t_iterated, ml_m_iterated; score = :nuisance_space, n_folds = 3, n_rep = 1)

    @test model isa DoubleMLLPLR
    @test model.ml_M isa MLJ.MLJIteration.ProbabilisticIteratedModel
    @test model.ml_t isa MLJ.MLJIteration.DeterministicIteratedModel
    @test model.ml_m isa MLJ.MLJIteration.DeterministicIteratedModel

    fit!(model)

    @test isfitted(model)
    @test model.coef !== nothing
    @test model.se !== nothing
    @test !isnan(model.coef)

    # Wide tolerance check
    @test abs(model.coef - 0.5) < 0.5
end
