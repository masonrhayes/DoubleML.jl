using DoubleML
using Test
using MLJ
using MLJLinearModels
using StatsBase
using StatsAPI
using StableRNGs

LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0

@testset "StatsAPI Methods" begin
    rng = StableRNG(12345)
    n_obs = 200

    @testset "PLR StatsAPI" begin
        data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

        @testset "Unfitted model errors" begin
            @test !isfitted(model)
            @test_throws ErrorException coef(model)
            @test_throws ErrorException stderror(model)
            @test_throws ErrorException vcov(model)
            @test_throws ErrorException confint(model)
        end

        fit!(model)
        @test isfitted(model)

        @testset "coef" begin
            c = coef(model)
            @test c isa Vector
            @test length(c) == 1
            @test c[1] == model.coef
            @test !isnan(c[1])
        end

        @testset "stderror" begin
            se = stderror(model)
            @test se isa Vector
            @test length(se) == 1
            @test se[1] == model.se
            @test se[1] > 0
        end

        @testset "vcov" begin
            v = vcov(model)
            @test v isa Matrix
            @test size(v) == (1, 1)
            @test v[1, 1] == model.se^2
            @test v[1, 1] > 0
        end

        @testset "confint" begin
            ci = confint(model)
            @test ci isa Matrix
            @test size(ci) == (1, 2)
            @test ci[1] < model.coef < ci[2]  # CI should contain estimate
        end

        @testset "nobs" begin
            @test nobs(model) == n_obs
            @test nobs(model) == data.n_obs
        end

        @testset "dof and dof_residual" begin
            @test dof(model) == 1
            @test dof_residual(model) == n_obs - 1
        end

        @testset "islinear" begin
            @test islinear(model) == false
        end

        @testset "responsename" begin
            @test responsename(model) == "y"
        end

        @testset "coefnames" begin
            cn = coefnames(model)
            @test cn isa Vector{String}
            @test length(cn) == 1
            @test cn[1] == "d"
        end

        @testset "coeftable" begin
            ct = coeftable(model)
            @test ct isa StatsBase.CoefTable
            @test length(ct.rownms) == 1
            @test length(ct.colnms) == 6
            @test ct.rownms[1] == "d"

            # Check expected column names
            expected_cols = ["Estimate", "Std. Error", "z value", "Pr(>|z|)", "Lower 95.0%", "Upper 95.0%"]
            @test ct.colnms == expected_cols
        end

        @testset "Base.show" begin
            io = IOBuffer()
            show(io, model)
            output = String(take!(io))
            @test occursin("DoubleMLPLR", output)
            @test occursin("d", output)
        end
    end

    @testset "IRM StatsAPI" begin
        rng_irm = StableRNG(54321)
        data = make_irm_data(n_obs; theta = 0.5, rng = rng_irm)
        model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATE)

        fit!(model)
        @test isfitted(model)

        @testset "Basic StatsAPI methods" begin
            @test coef(model) isa Vector
            @test length(coef(model)) == 1
            @test stderror(model)[1] > 0
            @test nobs(model) == n_obs
            @test responsename(model) == "y"
            @test coefnames(model) == ["d"]
        end

        @testset "coeftable with ATE score" begin
            ct = coeftable(model)
            @test ct isa StatsBase.CoefTable
            @test occursin("95.0%", ct.colnms[5])
        end

        @testset "ATTE score StatsAPI" begin
            model_atte = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATTE)
            fit!(model_atte)

            # ATTE and ATE are different estimands, so coefficients can differ
            # Just verify the model fitted correctly
            @test isfitted(model_atte)
            @test !isnan(coef(model_atte)[1])
            @test stderror(model_atte)[1] > 0
            @test responsename(model_atte) == "y"
            @test coefnames(model_atte) == ["d"]
        end
    end

    @testset "Multiple repetitions consistency" begin
        data = make_plr_CCDDHNR2018(300; alpha = 0.5, rng = rng)

        model_single = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        model_multi = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 3)

        fit!(model_single)
        fit!(model_multi)

        # Both should be fitted and have valid outputs
        @test isfitted(model_single)
        @test isfitted(model_multi)
        @test coef(model_single)[1] ≈ coef(model_multi)[1] atol = 0.3
        @test stderror(model_single)[1] > 0
        @test stderror(model_multi)[1] > 0
    end
end

@testset "Learner Accessors" begin
    rng = StableRNG(12345)
    data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)

    @testset "PLR learner accessors" begin
        ml_l = LinearRegressor()
        ml_m = LinearRegressor()
        model = DoubleMLPLR(data, ml_l, ml_m; n_folds = 3)

        @test learner_l(model) === ml_l
        @test learner_m(model) === ml_m
        @test learner_g(model) === nothing

        # IV-type with ml_g
        ml_g = LinearRegressor()
        model_iv = DoubleMLPLR(data, ml_l, ml_m; ml_g = ml_g, score = :IV_type, n_folds = 3)
        @test learner_g(model_iv) === ml_g
    end

    @testset "IRM learner accessors" begin
        data_irm = make_irm_data(100; theta = 0.5, rng = rng)
        ml_g = LinearRegressor()
        ml_m = LogisticClassifier()
        model = DoubleMLIRM(data_irm, ml_g, ml_m; n_folds = 3, score = :ATE)

        @test learner_g(model) === ml_g
        @test learner_m(model) === ml_m
    end
end
