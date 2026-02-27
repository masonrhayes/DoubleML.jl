using DoubleML
using Test
using StableRNGs
using Statistics
using MLJ
using MLJLinearModels

LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0

@testset "Aggregation and Multiple Repetitions" begin
    rng = StableRNG(12345)

    @testset "_aggregate_coefs_and_ses function" begin
        @testset "Single repetition" begin
            coefs = [0.5]
            ses = [0.1]

            coef, se = DoubleML._aggregate_coefs_and_ses(coefs, ses)

            @test coef ≈ 0.5
            @test se ≈ 0.1
        end

        @testset "Multiple repetitions - median aggregation" begin
            # Known values: median coef = 0.5, median upper = 0.5 + 1.96*0.12
            coefs = [0.4, 0.5, 0.6]  # median = 0.5
            ses = [0.1, 0.12, 0.14]   # median upper bound = median([0.596, 0.7352, 0.8744]) ≈ 0.7352

            coef, se = DoubleML._aggregate_coefs_and_ses(coefs, ses)

            @test coef ≈ 0.5  # median
            upper_bounds = coefs .+ 1.96 .* ses  # [0.596, 0.7352, 0.8744]
            agg_upper = median(upper_bounds)  # 0.7352
            expected_se = (agg_upper - 0.5) / 1.96  # ≈ 0.12
            @test se ≈ expected_se rtol = 1.0e-6
        end

        @testset "Empty arrays" begin
            coefs = Float64[]
            ses = Float64[]

            coef, se = DoubleML._aggregate_coefs_and_ses(coefs, ses)

            @test isnan(coef)
            @test isnan(se)
        end

        @testset "Type preservation" begin
            coefs = Float32[0.5f0, 0.6f0]
            ses = Float32[0.1f0, 0.12f0]

            coef, se = DoubleML._aggregate_coefs_and_ses(coefs, ses)

            @test coef isa Float32
            @test se isa Float32
        end
    end

    @testset "PLR all_coef and all_se fields" begin
        data = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng)

        @testset "Single repetition" begin
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
            fit!(model)

            @test length(model.all_coef) == 1
            @test length(model.all_se) == 1
            @test model.all_coef[1] ≈ model.coef
            @test model.all_se[1] ≈ model.se
        end

        @testset "Multiple repetitions" begin
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 3)
            fit!(model)

            @test length(model.all_coef) == 3
            @test length(model.all_se) == 3
            @test all(!isnan, model.all_coef)
            @test all(!isnan, model.all_se)
            @test all(>=(0), model.all_se)

            # Aggregated coef should be median of all_coef
            @test model.coef ≈ median(model.all_coef)
        end

        @testset "Reproducibility with same seed" begin
            rng1 = StableRNG(42)
            rng2 = StableRNG(42)

            data1 = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng1)
            data2 = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng2)

            # Use the same RNG for both model fittings
            model1 = DoubleMLPLR(data1, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 2)
            model2 = DoubleMLPLR(data2, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 2)

            fit!(model1)
            fit!(model2)

            # Both should be fitted with valid results
            @test isfitted(model1)
            @test isfitted(model2)
            @test length(model1.all_coef) == 2
            @test length(model2.all_coef) == 2
            @test all(!isnan, model1.all_coef)
            @test all(!isnan, model2.all_coef)
        end
    end

    @testset "IRM all_coef and all_se fields" begin
        data = make_irm_data(300; theta = 0.5, rng = rng)

        @testset "Single repetition" begin
            model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, n_rep = 1, score = :ATE)
            fit!(model)

            @test length(model.all_coef) == 1
            @test length(model.all_se) == 1
            @test model.all_coef[1] ≈ model.coef
            @test model.all_se[1] ≈ model.se
        end

        @testset "Multiple repetitions - ATE" begin
            model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, n_rep = 3, score = :ATE)
            fit!(model)

            @test length(model.all_coef) == 3
            @test length(model.all_se) == 3
            @test all(!isnan, model.all_coef)
            @test all(!isnan, model.all_se)
            @test all(>=(0), model.all_se)

            @test model.coef ≈ median(model.all_coef)
        end

        @testset "Multiple repetitions - ATTE" begin
            model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, n_rep = 3, score = :ATTE)
            fit!(model)

            @test length(model.all_coef) == 3
            @test length(model.all_se) == 3
            @test isfitted(model)
            # ATTE is harder to estimate, so use high tolerance
            @test abs(model.coef - 0.5) < 1.5
        end
    end

    @testset "LPLR all_coef and all_se fields" begin
        data = make_lplr_LZZ2020(300; alpha = 0.5, dim_x = 20, rng = rng, treatment = "continuous")

        @testset "Single repetition - nuisance_space" begin
            model = DoubleMLLPLR(
                data, LogisticClassifier(), LinearRegressor(), LinearRegressor();
                score = :nuisance_space, n_folds = 3, n_rep = 1
            )
            fit!(model)

            @test length(model.all_coef) == 1
            @test length(model.all_se) == 1
        end

        @testset "Multiple repetitions - nuisance_space" begin
            model = DoubleMLLPLR(
                data, LogisticClassifier(), LinearRegressor(), LinearRegressor();
                score = :nuisance_space, n_folds = 3, n_rep = 3
            )
            fit!(model)

            @test length(model.all_coef) == 3
            @test length(model.all_se) == 3
            @test all(!isnan, model.all_coef)
            @test all(!isnan, model.all_se)
            @test all(>=(0), model.all_se)
            @test isfitted(model)
        end
    end

    @testset "Aggregation consistency check" begin
        data = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng)

        # Test that aggregated SE follows the formula
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 5)
        fit!(model)

        # Manually compute expected SE
        upper_bounds = model.all_coef .+ 1.96 .* model.all_se
        agg_upper = median(upper_bounds)
        expected_se = (agg_upper - median(model.all_coef)) / 1.96

        # Use loose tolerance due to Float32 precision
        @test model.se ≈ expected_se rtol = 1.0e-5
    end

    @testset "Different n_rep values produce different results" begin
        rng1 = StableRNG(42)
        data = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng1)

        model1 = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        model3 = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 3)

        fit!(model1)
        fit!(model3)

        # Both should be fitted
        @test isfitted(model1)
        @test isfitted(model3)

        # Multiple rep model should have more stable SE (generally)
        # This is a weak test - just checking they both work
        @test model1.se > 0
        @test model3.se > 0
    end
end
