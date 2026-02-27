using DoubleML
using Test
using MLJ
using MLJLinearModels
using StableRNGs

LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0

@testset "Evaluation Functions" begin
    rng = StableRNG(12345)
    n_obs = 200

    @testset "fitted_params - PLR" begin
        @testset "Partialling out score" begin
            data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

            @test_throws ErrorException fitted_params(model)

            fit!(model)
            fp = fitted_params(model)

            @test fp isa NamedTuple
            @test haskey(fp, :ml_l)
            @test haskey(fp, :ml_m)
            @test !haskey(fp, :ml_g)  # Should not have ml_g for partialling out

            @test fp.ml_l isa Vector
            @test fp.ml_m isa Vector
            @test length(fp.ml_l) == model.n_folds * model.n_rep
            @test length(fp.ml_m) == model.n_folds * model.n_rep
        end

        @testset "IV-type score" begin
            data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(
                data, LinearRegressor(), LinearRegressor();
                ml_g = LinearRegressor(), score = :IV_type, n_folds = 3
            )

            fit!(model)
            fp = fitted_params(model)

            @test haskey(fp, :ml_l)
            @test haskey(fp, :ml_m)
            @test haskey(fp, :ml_g)

            @test fp.ml_g isa Vector
            @test length(fp.ml_g) == model.n_folds * model.n_rep
        end
    end

    @testset "fitted_params - IRM" begin
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)
        model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATE)

        @test_throws ErrorException fitted_params(model)

        fit!(model)
        fp = fitted_params(model)

        @test fp isa NamedTuple
        @test haskey(fp, :ml_g0)
        @test haskey(fp, :ml_g1)
        @test haskey(fp, :ml_m)

        @test fp.ml_g0 isa Vector
        @test fp.ml_g1 isa Vector
        @test fp.ml_m isa Vector
    end

    @testset "report - PLR" begin
        @testset "Partialling out" begin
            data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

            @test_throws ErrorException report(model)

            fit!(model)
            r = report(model)

            @test r isa NamedTuple
            @test haskey(r, :learner_reports)
            @test haskey(r, :dml_summary)

            @test r.dml_summary.coef == model.coef
            @test r.dml_summary.se == model.se
            @test r.dml_summary.n_folds == 3
            @test r.dml_summary.n_rep == 1
            @test r.dml_summary.n_obs == n_obs
            @test r.dml_summary.score == :partialling_out
        end

        @testset "IV-type" begin
            data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
            model = DoubleMLPLR(
                data, LinearRegressor(), LinearRegressor();
                ml_g = LinearRegressor(), score = :IV_type, n_folds = 3
            )

            fit!(model)
            r = report(model)

            @test haskey(r.learner_reports, :ml_g)
            @test r.dml_summary.score == :IV_type
        end
    end

    @testset "report - IRM" begin
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)
        model = DoubleMLIRM(
            data, LinearRegressor(), LogisticClassifier();
            n_folds = 3, score = :ATE, normalize_ipw = false, clipping_threshold = 0.01
        )

        fit!(model)
        r = report(model)

        @test haskey(r.learner_reports, :ml_g0)
        @test haskey(r.learner_reports, :ml_g1)
        @test haskey(r.learner_reports, :ml_m)

        @test r.dml_summary.score == :ATE
        @test r.dml_summary.normalize_ipw == false
        @test r.dml_summary.clipping_threshold ≈ 0.01 rtol = 0.01
    end

    @testset "evaluate! - PLR" begin
        data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

        fit!(model)

        @testset "Evaluate all learners" begin
            results = evaluate!(model; resampling = CV(nfolds = 3), verbosity = 0)

            @test results isa NamedTuple
            @test haskey(results, :ml_l)
            @test haskey(results, :ml_m)
        end

        @testset "Evaluate specific learners" begin
            results_l = evaluate!(model; learners = :l, resampling = CV(nfolds = 3), verbosity = 0)
            @test haskey(results_l, :ml_l)
            @test !haskey(results_l, :ml_m)

            results_m = evaluate!(model; learners = :m, resampling = CV(nfolds = 3), verbosity = 0)
            @test haskey(results_m, :ml_m)
            @test !haskey(results_m, :ml_l)

            results_both = evaluate!(model; learners = [:l, :m], resampling = CV(nfolds = 3), verbosity = 0)
            @test haskey(results_both, :ml_l)
            @test haskey(results_both, :ml_m)
        end
    end

    @testset "evaluate! - IRM" begin
        data = make_irm_data(n_obs; theta = 0.5, rng = rng)
        model = DoubleMLIRM(data, LinearRegressor(), LogisticClassifier(); n_folds = 3, score = :ATE)
        fit!(model)

        @testset "Evaluate all learners" begin
            results = evaluate!(model; resampling = CV(nfolds = 3), verbosity = 0)

            @test haskey(results, :ml_g0)
            @test haskey(results, :ml_g1)
            @test haskey(results, :ml_m)
        end

        @testset "Evaluate specific learners" begin
            results_g0 = evaluate!(model; learners = :g0, resampling = CV(nfolds = 3), verbosity = 0)
            @test haskey(results_g0, :ml_g0)

            results_m = evaluate!(model; learners = :m, resampling = CV(nfolds = 3), verbosity = 0)
            @test haskey(results_m, :ml_m)
        end
    end

    @testset "summary function" begin
        data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)

        # Test summary on unfitted model - should not throw
        @test summary(model) === nothing

        fit!(model)

        # Test summary on fitted model - should not throw and return nothing
        @test summary(model) === nothing

        # Verify model has the expected data
        @test isfitted(model)
        @test model.coef isa AbstractFloat
    end
end
