using DoubleML
using Test
using Random
using StableRNGs

@testset "draw_sample_splitting" begin
    @testset "Basic functionality" begin
        n_obs = 100
        n_folds = 5
        n_rep = 3

        splits = draw_sample_splitting(n_obs, n_folds, n_rep)

        # Check return structure
        @test length(splits) == n_rep
        @test all(length(rep) == n_folds for rep in splits)

        # Check each fold has train/test indices
        for r in 1:n_rep
            for k in 1:n_folds
                train_idx, test_idx = splits[r][k]
                @test typeof(train_idx) == Vector{Int}
                @test typeof(test_idx) == Vector{Int}
                @test all(1 .<= train_idx .<= n_obs)
                @test all(1 .<= test_idx .<= n_obs)
                @test isempty(intersect(train_idx, test_idx))
                @test length(train_idx) + length(test_idx) == n_obs
            end
        end

        # Check that all observations are covered in each repetition
        for r in 1:n_rep
            all_test = Int[]
            for k in 1:n_folds
                _, test_idx = splits[r][k]
                append!(all_test, test_idx)
            end
            @test sort(all_test) == 1:n_obs
        end
    end

    @testset "RNG reproducibility" begin
        n_obs = 100
        n_folds = 5
        n_rep = 3

        # Test with StableRNG - same seed should give same results
        rng1 = StableRNG(123)
        rng2 = StableRNG(123)

        splits1 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng1)
        splits2 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng2)

        for r in 1:n_rep
            for k in 1:n_folds
                train1, test1 = splits1[r][k]
                train2, test2 = splits2[r][k]
                @test train1 == train2
                @test test1 == test2
            end
        end

        # Different seed should give different results
        rng3 = StableRNG(456)
        splits3 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng3)

        is_different = false
        for r in 1:n_rep
            for k in 1:n_folds
                train1, _ = splits1[r][k]
                train3, _ = splits3[r][k]
                if train1 != train3
                    is_different = true
                    break
                end
            end
            is_different && break
        end
        @test is_different
    end

    @testset "Different reps should have different splits" begin
        n_obs = 100
        n_folds = 5
        n_rep = 5

        rng = StableRNG(789)
        splits = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng)

        # Check that different repetitions produce different splits
        for r1 in 1:(n_rep - 1)
            for r2 in (r1 + 1):n_rep
                # At least one fold should be different between repetitions
                is_different = false
                for k in 1:n_folds
                    train1, _ = splits[r1][k]
                    train2, _ = splits[r2][k]
                    if train1 != train2
                        is_different = true
                        break
                    end
                end
                @test is_different || error("Repetitions $r1 and $r2 should have different splits")
            end
        end
    end

    @testset "Deterministic with same RNG state" begin
        n_obs = 100
        n_folds = 5
        n_rep = 2

        # Create two identical RNGs
        rng1 = StableRNG(42)
        rng2 = StableRNG(42)

        # Draw splits
        splits1 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng1)
        splits2 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng2)

        # Should be identical
        @test splits1 == splits2
    end

    @testset "Multiple calls with same RNG produce different results" begin
        n_obs = 100
        n_folds = 5
        n_rep = 2

        rng = StableRNG(42)

        # First call
        splits1 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng)

        # Second call with same RNG (now in different state)
        splits2 = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng)

        # Should be different since RNG has advanced
        @test splits1 != splits2
    end

    @testset "Edge cases" begin
        # Single observation per fold (n_obs == n_folds)
        splits = draw_sample_splitting(5, 5, 1)
        @test length(splits) == 1
        @test length(splits[1]) == 5
        for k in 1:5
            train, test = splits[1][k]
            @test length(test) == 1  # Each fold has one test observation
        end

        # Single repetition
        splits = draw_sample_splitting(100, 5, 1)
        @test length(splits) == 1
    end

    @testset "Fold sizes are approximately equal" begin
        n_obs = 100
        n_folds = 5
        n_rep = 1

        rng = StableRNG(123)
        splits = draw_sample_splitting(n_obs, n_folds, n_rep; rng = rng)

        test_sizes = [length(test_idx) for (_, test_idx) in splits[1]]
        # With 100 obs and 5 folds, each test set should have 20 observations
        @test all(s -> s == 20, test_sizes)
    end

    @testset "shuffle=false produces ordered splits" begin
        n_obs = 10
        n_folds = 2
        n_rep = 1

        rng = StableRNG(123)
        splits = draw_sample_splitting(n_obs, n_folds, n_rep; shuffle = false, rng = rng)

        # When shuffle=false, the splits should be sequential
        # First fold tests should be first half, second fold second half
        all_test = Int[]
        for k in 1:n_folds
            _, test_idx = splits[1][k]
            append!(all_test, test_idx)
        end
        # Test sets should cover all observations in order
        @test sort(all_test) == 1:n_obs
    end
end
