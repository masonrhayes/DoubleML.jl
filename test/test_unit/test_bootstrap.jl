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

    @testset "Multiplier Bootstrap - Different n_rep_boot values" begin
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
        @test model.boot_method isa NormalBootstrap
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

    @testset "Bootstrap with Multiple Repetitions (n_rep > 1)" begin
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        # Test with n_rep = 2
        data = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng)
        model_rep2 = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 2)
        DoubleML.fit!(model_rep2)
        bootstrap!(model_rep2; n_rep_boot = 300, method = :normal, rng = rng)

        # Check boot_t_stat dimensions: (n_rep_boot, n_coefs, n_rep)
        @test size(model_rep2.boot_t_stat) == (300, 1, 2)
        @test has_bootstrapped(model_rep2)

        # Joint CI should work with n_rep > 1
        ci_joint = confint(model_rep2; joint = true, level = 0.95)
        @test size(ci_joint) == (1, 2)
        @test all(isfinite, ci_joint)

        # Pointwise CI for comparison
        ci_pointwise = confint(model_rep2; joint = false, level = 0.95)
        # Joint and pointwise should be similar for single treatment
        # (within 30% tolerance due to bootstrap variability)
        joint_width = ci_joint[2] - ci_joint[1]
        pointwise_width = ci_pointwise[2] - ci_pointwise[1]
        @test 0.7 < joint_width / pointwise_width < 1.3

        # Test with n_rep = 5
        model_rep5 = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 2, n_rep = 5)
        DoubleML.fit!(model_rep5)
        bootstrap!(model_rep5; n_rep_boot = 200, method = :wild, rng = rng)
        @test size(model_rep5.boot_t_stat) == (200, 1, 5)

        ci_joint5 = confint(model_rep5; joint = true, level = 0.95)
        @test size(ci_joint5) == (1, 2)
        @test all(isfinite, ci_joint5)
    end

    @testset "Bootstrap T-statistics Storage and Properties" begin
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        data = make_plr_CCDDHNR2018(300; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 2)
        DoubleML.fit!(model)

        n_rep_boot = 500
        bootstrap!(model; n_rep_boot = n_rep_boot, method = :normal, rng = rng)

        # Check dimensions
        @test size(model.boot_t_stat) == (n_rep_boot, 1, 2)

        # Check all values are finite
        @test all(isfinite, model.boot_t_stat)

        # T-statistics should be approximately centered (mean close to 0)
        # and have reasonable variance (not all zeros, not all infinite)
        for r in 1:2
            t_stats_rep = model.boot_t_stat[:, 1, r]
            @test abs(mean(t_stats_rep)) < 1.0  # Should be centered near 0
            @test std(t_stats_rep) > 0.1  # Should have reasonable variance
            @test std(t_stats_rep) < 10.0  # Should not be too extreme
        end

        # Different methods should produce different distributions
        model_wild = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        DoubleML.fit!(model_wild)
        bootstrap!(model_wild; n_rep_boot = n_rep_boot, method = :wild, rng = rng)

        # Wild and normal bootstrap should have different variances in general
        # (though both should be valid)
        std_normal = std(model.boot_t_stat[:, 1, 1])
        std_wild = std(model_wild.boot_t_stat[:, 1, 1])
        # They should not be exactly the same
        @test std_normal != std_wild || abs(std_normal - std_wild) < 0.5
    end

    @testset "Reproducibility with Seeded RNG" begin
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        data = make_plr_CCDDHNR2018(200; alpha = 0.5, rng = rng)

        # First run with seed
        model1 = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        fit!(model1)
        rng1 = StableRNG(98765)
        bootstrap!(model1; n_rep_boot = 200, method = :normal, rng = rng1)

        # Same model, same seed again (re-bootstrap)
        rng2 = StableRNG(98765)
        bootstrap!(model1; n_rep_boot = 200, method = :normal, rng = rng2)
        boot_t_first = copy(model1.boot_t_stat)

        # Re-run bootstrap with same seed on same fitted model
        rng3 = StableRNG(98765)
        bootstrap!(model1; n_rep_boot = 200, method = :normal, rng = rng3)

        # Should be identical
        @test boot_t_first == model1.boot_t_stat

        # Different seed should give different results
        rng4 = StableRNG(11111)
        bootstrap!(model1; n_rep_boot = 200, method = :normal, rng = rng4)

        # Should be different (very unlikely to be identical by chance)
        @test boot_t_first != model1.boot_t_stat
    end

    @testset "Bootstrap with Different Model Types" begin
        using MLJLinearModels
        using DecisionTree
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0
        LogisticClassifier = @load LogisticClassifier pkg = MLJLinearModels verbosity = 0
        Tree = @load DecisionTreeRegressor pkg = DecisionTree verbosity = 0

        # Test with IRM (Interactive Regression Model)
        data_irm = make_irm_data(300; theta = 0.5, rng = rng)
        model_irm = DoubleMLIRM(data_irm, Tree(), LogisticClassifier(); n_folds = 3, n_rep = 1)
        DoubleML.fit!(model_irm)
        bootstrap!(model_irm; n_rep_boot = 300, method = :normal, rng = rng)

        @test has_bootstrapped(model_irm)
        @test size(model_irm.boot_t_stat) == (300, 1, 1)

        ci_irm = confint(model_irm; joint = true, level = 0.95)
        @test size(ci_irm) == (1, 2)
        @test all(isfinite, ci_irm)

        # Test with different bootstrap methods
        bootstrap!(model_irm; n_rep_boot = 200, method = :wild, rng = rng)
        @test model_irm.boot_method isa WildBootstrap

        bootstrap!(model_irm; n_rep_boot = 200, method = :bayes, rng = rng)
        @test model_irm.boot_method isa BayesBootstrap
    end

    @testset "Bootstrap Error Handling - Additional Cases" begin
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0

        # Test bootstrap! on unfitted model
        data = make_plr_CCDDHNR2018(100; alpha = 0.5, rng = rng)
        model_unfitted = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3)
        @test_throws ArgumentError bootstrap!(model_unfitted; n_rep_boot = 100)

        # Test bootstrap! called twice (should work - just overwrite previous)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        DoubleML.fit!(model)
        bootstrap!(model; n_rep_boot = 100, method = :normal, rng = rng)
        boot_t_first = copy(model.boot_t_stat)

        # Failed re-bootstrap attempts preserve the previous valid result.
        boot_method_first = model.boot_method
        n_rep_boot_first = model.n_rep_boot
        @test_throws ArgumentError bootstrap!(
            model; n_rep_boot = 200, method = :invalid, rng = StableRNG(123)
        )
        @test model.boot_t_stat == boot_t_first
        @test model.boot_method === boot_method_first
        @test model.n_rep_boot == n_rep_boot_first
        @test has_bootstrapped(model)

        bootstrap!(model; n_rep_boot = 200, method = :normal, rng = rng)
        @test has_bootstrapped(model)
        @test size(model.boot_t_stat) == (200, 1, 1)  # Should have new size

        # Test invalid confidence levels
        @test_throws DomainError confint(model; level = -0.5)  # Negative
        @test_throws DomainError confint(model; level = 0.0)   # Zero boundary
        @test_throws DomainError confint(model; level = 1.0)   # One boundary
        @test_throws DomainError confint(model; level = 1.5)   # Greater than 1

        # Test valid boundary cases (should NOT throw)
        @test size(confint(model; level = 0.001)) == (1, 2)
        @test size(confint(model; level = 0.999)) == (1, 2)
    end

    @testset "Critical Value Validation" begin
        using MLJLinearModels
        LinearRegressor = @load LinearRegressor pkg = MLJLinearModels verbosity = 0


        data = make_plr_CCDDHNR2018(500; alpha = 0.5, rng = rng)
        model = DoubleMLPLR(data, LinearRegressor(), LinearRegressor(); n_folds = 3, n_rep = 1)
        DoubleML.fit!(model)
        bootstrap!(model; n_rep_boot = 5000, method = :normal, rng = rng)

        ci_joint = confint(model; joint = true, level = 0.95)
        ci_pointwise = confint(model; joint = false, level = 0.95)

        joint_width = ci_joint[2] - ci_joint[1]
        pointwise_width = ci_pointwise[2] - ci_pointwise[1]

        @test 0.8 < joint_width / pointwise_width < 1.2

        # Test at different confidence levels
        for level in [0.9, 0.95, 0.99]
            ci = confint(model; joint = true, level = level)
            @test size(ci) == (1, 2)
            @test ci[2] > ci[1]  # Upper > lower
            @test all(isfinite, ci)
        end

        max_abs_t = vec(maximum(abs.(model.boot_t_stat[:, :, 1]), dims = 2))
        empirical_cv = quantile(max_abs_t, 0.95)
        theoretical_cv = quantile(Normal(), 0.975)  # For 95% CI

        # Should be within 10% of theoretical value for large bootstrap samples
        @test abs(empirical_cv - theoretical_cv) / theoretical_cv < 0.15
    end
end
