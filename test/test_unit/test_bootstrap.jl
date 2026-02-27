using DoubleML
using MLJ
using Test
using Statistics
using StableRNGs
using Distributions

# Load HypothesisTests only if available (test dependency)
const has_hypothesis_tests = try
    @eval using HypothesisTests
    true
catch
    false
end

@testset "Bootstrap Inference Tests" begin
    rng = StableRNG(12345)
    n_obs = 500

    @testset "Multiplier Bootstrap - Gaussian Method" begin
        # Create synthetic psi and psi_a
        psi = randn(rng, n_obs)
        psi_a = -ones(n_obs) .+ 0.1 .* randn(rng, n_obs)

        n_rep_boot = 1000
        boot_draws = multiplier_bootstrap(psi, psi_a, n_rep_boot, method = :normal)

        @test length(boot_draws) == n_rep_boot
        @test all(isfinite, boot_draws)

        # Check distribution properties
        # Bootstrap mean should be close to analytical mean
        analytical_mean = mean(psi) / abs(mean(psi_a))
        @test abs(mean(boot_draws) - analytical_mean) < 0.5

        # Check variance scales correctly
        theoretical_se = sqrt(var(psi) / (n_obs * mean(psi_a)^2))
        boot_se = std(boot_draws)
        @test 0.3 < boot_se / theoretical_se < 3.0
    end

    @testset "Multiplier Bootstrap - Wild Method" begin
        psi = randn(rng, n_obs)
        psi_a = -ones(n_obs)

        n_rep_boot = 1000
        boot_draws = multiplier_bootstrap(psi, psi_a, n_rep_boot, method = :wild)

        @test length(boot_draws) == n_rep_boot
        @test all(isfinite, boot_draws)

        # Wild bootstrap weights have E[w] = 0 and Var[w] = 1
        # So the bootstrap distribution should be centered around the true value
        analytical_mean = mean(psi) / abs(mean(psi_a))
        @test abs(mean(boot_draws) - analytical_mean) < 0.5

        # Check that wild bootstrap produces finite, valid results
        @test all(isfinite, boot_draws)
        @test std(boot_draws) > 0
    end

    @testset "Multiplier Bootstrap - Bayes Method" begin
        psi = randn(rng, n_obs)
        psi_a = -ones(n_obs)

        n_rep_boot = 1000
        boot_draws = multiplier_bootstrap(psi, psi_a, n_rep_boot, method = :bayes)

        @test length(boot_draws) == n_rep_boot
        @test all(isfinite, boot_draws)

        # Bayes bootstrap should be centered around the same value
        analytical_mean = mean(psi) / abs(mean(psi_a))
        @test abs(mean(boot_draws) - analytical_mean) < 0.5
    end

    @testset "Multiplier Bootstrap - Different n_rep values" begin
        psi = randn(rng, n_obs)
        psi_a = -ones(n_obs)

        # Small bootstrap
        boot_small = multiplier_bootstrap(psi, psi_a, 100, method = :normal)
        @test length(boot_small) == 100

        # Large bootstrap
        boot_large = multiplier_bootstrap(psi, psi_a, 2000, method = :normal)
        @test length(boot_large) == 2000
    end

    @testset "Bootstrap Error Handling" begin
        # Test that invalid methods throw ArgumentError
        @test_throws ArgumentError multiplier_bootstrap(randn(10), -ones(10), 100, method = :invalid)

        # Test zero psi_a throws error
        @test_throws ArgumentError multiplier_bootstrap(randn(10), zeros(10), 100)

        # Test n_rep_boot < 1 throws error
        @test_throws ArgumentError multiplier_bootstrap(randn(10), -ones(10), 0)

        # Test dimension mismatch throws error
        @test_throws DimensionMismatch multiplier_bootstrap(randn(10), -ones(5), 100)
    end

    @testset "Joint Confidence Intervals" begin
        # Generate synthetic fitted model data
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        data = make_plr_CCDDHNR2018(n_obs; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        fit!(model)

        # Bootstrap first
        bootstrap!(model; n_rep_boot = 500, method = :normal, rng = rng)
        @test has_bootstrapped(model)
        @test model.boot_method == :normal
        @test model.n_rep_boot == 500

        # Get joint confidence intervals
        ci_joint = confint(model; joint = true, level = 0.95)
        @test size(ci_joint) == (1, 2)

        # Joint CI should be wider than pointwise CI
        ci_pointwise = confint(model; joint = false, level = 0.95)
        @test ci_joint[1] <= ci_pointwise[1]  # Lower bound should be lower
        @test ci_joint[2] >= ci_pointwise[2]  # Upper bound should be higher

        # Test confint with level argument directly
        ci_90 = confint(model, 0.9)
        ci_95 = confint(model, 0.95)
        ci_99 = confint(model, 0.99)

        # Wider confidence level should give wider intervals
        @test ci_90[1] >= ci_95[1]  # 90% lower bound >= 95% lower bound
        @test ci_90[2] <= ci_95[2]  # 90% upper bound <= 95% upper bound
        @test ci_95[1] >= ci_99[1]  # 95% lower bound >= 99% lower bound
        @test ci_95[2] <= ci_99[2]  # 95% upper bound <= 99% upper bound
    end

    @testset "Joint CI without bootstrap throws error" begin
        data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        fit!(model)

        # Should throw error if bootstrap not called first
        @test_throws ErrorException confint(model; joint = true)
    end

    @testset "summary_stats function" begin
        data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        fit!(model)

        stats = DoubleML.summary_stats(model)
        @test stats.coef == model.coef
        @test stats.se == model.se
        @test haskey(stats, :t)
        @test haskey(stats, :p)
        @test haskey(stats, :ci_lower)
        @test haskey(stats, :ci_upper)
        @test haskey(stats, :level)
        @test stats.level == 0.95
    end
end
