using DoubleML
using Test
using DataFrames

@testset "make_irm_data" begin
    @testset "Basic functionality" begin
        n_obs = 500
        data = make_irm_data(n_obs)

        @test data isa DoubleMLData
        @test data.n_obs == n_obs
        @test data.dim_x == 20  # default
        @test length(data.y) == n_obs
        @test length(data.d) == n_obs

        # Treatment should be binary (0 or 1)
        @test all(d -> d == 0 || d == 1, data.d)
    end

    @testset "Custom parameters" begin
        n_obs = 300
        dim_x = 10
        theta = 2.0
        R2_d = 0.6
        R2_y = 0.4

        data = make_irm_data(
            n_obs;
            dim_x = dim_x,
            theta = theta,
            R2_d = R2_d,
            R2_y = R2_y
        )

        @test data.n_obs == n_obs
        @test data.dim_x == dim_x
    end

    @testset "DataFrame return type" begin
        n_obs = 200
        df = make_irm_data(n_obs; return_type = :DataFrame)

        @test df isa DataFrame
        @test size(df, 1) == n_obs
        @test size(df, 2) == 22  # 20 covariates + y + d
        @test "y" in names(df)
        @test "d" in names(df)
        @test "X1" in names(df)
        @test "X20" in names(df)
    end

    @testset "Reproducibility with RNG" begin
        using StableRNGs

        rng1 = StableRNG(123)
        rng2 = StableRNG(123)

        data1 = make_irm_data(100; rng = rng1)
        data2 = make_irm_data(100; rng = rng2)

        @test data1.y == data2.y
        @test data1.d == data2.d
        @test data1.x == data2.x
    end

    @testset "Different RNG seeds produce different data" begin
        using StableRNGs

        rng1 = StableRNG(123)
        rng2 = StableRNG(456)

        data1 = make_irm_data(100; rng = rng1)
        data2 = make_irm_data(100; rng = rng2)

        # At least one should be different
        @test data1.y != data2.y || data1.d != data2.d || data1.x != data2.x
    end
end
