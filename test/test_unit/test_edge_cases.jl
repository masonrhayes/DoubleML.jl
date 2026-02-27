using DoubleML
using Test
using MLJ
using MLJLinearModels
using DataFrames
using StableRNGs
using Random
using Statistics

LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0

@testset "Edge Cases" begin
    rng = StableRNG(12345)

    @testset "Small sample sizes" begin
        @testset "n_obs equal to n_folds" begin
            # Minimum viable case: n_obs = n_folds
            data = make_plr_CCDDHNR2018(5; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 5, n_rep = 1)
            fit!(model)

            @test isfitted(model)
            @test !isnan(model.coef)
            @test model.se > 0
        end

        @testset "Very small sample" begin
            data = make_plr_CCDDHNR2018(20; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 2, n_rep = 1)
            fit!(model)

            @test isfitted(model)
        end
    end

    @testset "Float32 vs Float64 consistency" begin
        @testset "PLR with Float32 data" begin
            data32 = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            model32 = DoubleMLPLR(data32, LinearRegressor(), LinearRegressor(); n_folds = 3)
            fit!(model32)

            @test eltype(data32.y) == Float32
            @test isfitted(model32)
            @test model32.coef isa Float32
        end
    end

    @testset "Treatment and outcome variable handling" begin
        @testset "Binary treatment with different types" begin
            # Test with Float64 treatment
            n = 100
            df_f64 = DataFrames.DataFrame(
                y = randn(rng, n),
                d = Float64.(rand(rng, [0.0, 1.0], n)),
                x1 = randn(rng, n),
                x2 = randn(rng, n)
            )
            data_f64 = DoubleMLData(df_f64; y_col = :y, d_col = :d, x_cols = [:x1, :x2])
            model_f64 = DoubleMLIRM(data_f64, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATE)
            fit!(model_f64)
            @test isfitted(model_f64)

            # Test with Int treatment
            df_int = DataFrames.DataFrame(
                y = randn(rng, n),
                d = rand(rng, [0, 1], n),
                x1 = randn(rng, n),
                x2 = randn(rng, n)
            )
            data_int = DoubleMLData(df_int; y_col = :y, d_col = :d, x_cols = [:x1, :x2])
            model_int = DoubleMLIRM(data_int, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATE)
            fit!(model_int)
            @test isfitted(model_int)
        end
    end

    @testset "Error handling for invalid inputs" begin
        @testset "Invalid n_folds" begin
            data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            @test_throws DomainError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 0)
            @test_throws DomainError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = -1)
        end

        @testset "Invalid n_rep" begin
            data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            @test_throws DomainError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_rep = 0)
        end

        @testset "n_folds_tune > n_folds" begin
            data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            @test_throws DomainError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_folds_tune = 5)
        end

        @testset "Invalid score type" begin
            data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            @test_throws ArgumentError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); score = :invalid_score)

            data_irm = make_irm_data(100; theta = 0.5, rng = rng)
            @test_throws ArgumentError DoubleMLIRM(data_irm, LinearRegressor(), LogisticClassifier(); score = :invalid_score)
        end

        @testset "Non-binary treatment for IRM" begin
            n = 100
            df = DataFrames.DataFrame(
                y = randn(rng, n),
                d = rand(rng, n),  # Continuous treatment
                x1 = randn(rng, n),
                x2 = randn(rng, n)
            )
            data = DoubleMLData(df; y_col = :y, d_col = :d, x_cols = [:x1, :x2])
            @test_throws ArgumentError DoubleMLIRM(data, LinearRegressor(), LogisticClassifier())
        end

        @testset "IV-type without ml_g" begin
            data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
            @test_throws ArgumentError DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); score = :IV_type)
        end
    end

    @testset "Refitting with force parameter" begin
        data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

        fit!(model)
        first_coef = model.coef

        # Without force, should warn and return early
        @test_logs (:warn, r"already fitted") fit!(model)
        @test model.coef == first_coef

        # With force=true, should refit
        fit!(model; force = true)
        @test_logs (:warn, r"Forcing refit") fit!(model; force = true)
    end

    @testset "Sample splitting edge cases" begin
        @testset "shuffle=false produces deterministic splits" begin
            splits1 = draw_sample_splitting(100, 5, 1; shuffle = false, rng = rng)
            splits2 = draw_sample_splitting(100, 5, 1; shuffle = false, rng = rng)

            @test splits1 == splits2
        end

        @testset "Multiple repetitions are different" begin
            splits = draw_sample_splitting(100, 5, 3; shuffle = true, rng = rng)

            # At least one repetition should be different from another
            all_same = true
            for k in 1:5
                if splits[1][k] != splits[2][k]
                    all_same = false
                    break
                end
            end
            @test !all_same
        end
    end

    @testset "Propensity score clipping" begin
        rng_irm = StableRNG(42)
        data = make_irm_data(200; theta = 0.5, rng = rng_irm)

        # Test with different clipping thresholds
        model_001 = DoubleMLIRM(
            data, LinearRegressor(), LogisticClassifier();
            n_folds = 3, score = :ATE, clipping_threshold = 0.01
        )
        model_005 = DoubleMLIRM(
            data, LinearRegressor(), LogisticClassifier();
            n_folds = 3, score = :ATE, clipping_threshold = 0.05
        )

        fit!(model_001)
        fit!(model_005)

        @test isfitted(model_001)
        @test isfitted(model_005)

        # Estimates should be reasonable (true theta = 0.5)
        # Wide tolerance due to small sample size and simple models
        @test abs(model_001.coef - 0.5) < 3.0
        @test abs(model_005.coef - 0.5) < 3.0
    end

    @testset "IPW normalization" begin
        rng_irm = StableRNG(123)
        data = make_irm_data(200; theta = 0.5, rng = rng_irm)

        model_unnorm = DoubleMLIRM(
            data, LinearRegressor(), LogisticClassifier();
            n_folds = 3, score = :ATE, normalize_ipw = false
        )
        model_norm = DoubleMLIRM(
            data, LinearRegressor(), LogisticClassifier();
            n_folds = 3, score = :ATE, normalize_ipw = true
        )

        fit!(model_unnorm)
        fit!(model_norm)

        @test isfitted(model_unnorm)
        @test isfitted(model_norm)

        # Report should reflect normalization setting
        r_unnorm = report(model_unnorm)
        r_norm = report(model_norm)

        @test r_unnorm.dml_summary.normalize_ipw == false
        @test r_norm.dml_summary.normalize_ipw == true
    end
end
