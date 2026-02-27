using DoubleML
using Test
using DataFrames

@testset "check_binary function" begin
    using DoubleML: check_binary

    # Binary variables (should return true)
    @test check_binary([0, 1, 0, 1]) == true
    @test check_binary([0.0, 1.0, 0.0]) == true
    @test check_binary([1, 1, 1]) == true  # Only 1s
    @test check_binary([0, 0, 0]) == true  # Only 0s
    @test check_binary([0, 1]) == true     # Minimum binary set

    # Non-binary variables (should return false)
    @test check_binary([1, 2, 3]) == false
    @test check_binary([0.5, 0.3]) == false
    @test check_binary([0, 1, 2]) == false
    @test check_binary([0, 1, 0.5]) == false
    @test check_binary([-1, 0, 1]) == false

    # With missing values (should still work if non-missing are binary)
    @test check_binary([0, 1, missing, 0]) == true
    @test check_binary([missing, 0, 1]) == true

    # Empty or all missing (should return false)
    @test check_binary([]) == false
    @test check_binary([missing, missing]) == false
    @test check_binary([]) == false
end

@testset "DoubleMLData shorthand constructor" begin
    # Create test data
    df = DataFrame(
        y = randn(100),
        d = rand(100),
        x1 = randn(100),
        x2 = randn(100),
        x3 = randn(100)
    )

    # Test 1: Basic shorthand with Float32 (default)
    data = DoubleMLData(df, :y, :d)
    @test data isa DoubleMLData
    @test data.y_col == :y
    @test data.d_col == :d
    @test data.x_cols == [:x1, :x2, :x3]
    @test data.dtype == Float32
    @test data.n_obs == 100
    @test data.dim_x == 3

    # Test 2: Shorthand with explicit Float32
    data32 = DoubleMLData(df, :y, :d, Float32)
    @test data32 isa DoubleMLData
    @test data32.dtype == Float32
    @test data32.x_cols == [:x1, :x2, :x3]

    # Test 3: Error when y_col not found
    @test_throws ArgumentError DoubleMLData(df, :nonexistent, :d)

    # Test 4: Error when d_col not found
    @test_throws ArgumentError DoubleMLData(df, :y, :nonexistent)

    # Test 5: Error when no covariates (only y and d)
    df_no_x = DataFrame(y = randn(10), d = rand(10))
    @test_throws ArgumentError DoubleMLData(df_no_x, :y, :d)

    # Verify error message mentions no covariate columns
    try
        DoubleMLData(df_no_x, :y, :d)
        @test false # Should not reach here
    catch e
        @test e isa ArgumentError
        err_msg = sprint(showerror, e)
        @test occursin("No covariate columns found", err_msg)
    end

    # Test 6: Works with different column orders
    df_reordered = df[:, [:x2, :y, :x1, :d, :x3]]
    data_reordered = DoubleMLData(df_reordered, :y, :d)
    @test data_reordered.x_cols == [:x2, :x1, :x3]  # Preserves order from DataFrame
end
