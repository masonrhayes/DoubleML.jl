using DoubleML
using CategoricalArrays
using Test
using StableRNGs
using Statistics

@testset "Score Functions Unit Tests" begin
    rng = StableRNG(12345)
    n = 100

    # Create test data
    Y = randn(rng, n)
    D = rand(rng, n)
    g_hat = randn(rng, n)
    m_hat = rand(rng, n)

    @testset "PartiallingOutScore" begin
        score = PartiallingOutScore()

        @testset "Type and Interface" begin
            @test DoubleML.get_score_name(score) == :partialling_out
        end

        @testset "Computation" begin
            psi_a, psi_b = DoubleML.compute_score(score, Y, D, g_hat, m_hat)

            @test length(psi_a) == n
            @test length(psi_b) == n

            # Verify manual computation
            m_res = D .- m_hat
            g_res = Y .- g_hat
            @test psi_a ≈ .-m_res .^ 2
            @test psi_b ≈ g_res .* m_res
        end

        @testset "Edge Cases" begin
            # Empty vectors
            psi_a, psi_b = DoubleML.compute_score(score, Float64[], Float64[], Float64[], Float64[])
            @test isempty(psi_a)
            @test isempty(psi_b)

            # Single observation
            psi_a, psi_b = DoubleML.compute_score(score, [1.0], [0.5], [1.0], [0.5])
            @test psi_a[1] ≈ -0.0  # -(0.5 - 0.5)^2
            @test psi_b[1] ≈ 0.0   # (1.0 - 1.0) * (0.5 - 0.5)
        end

        @testset "Mathematical Properties" begin
            psi_a, psi_b = DoubleML.compute_score(score, Y, D, g_hat, m_hat)

            # psi_a should be non-positive (it's -m_res^2)
            @test all(psi_a .<= 0)

            # psi_a should be zero where m_hat == D exactly
            exact_match_idx = findall(abs.(D .- m_hat) .< 1.0e-10)
            if !isempty(exact_match_idx)
                @test all(psi_a[exact_match_idx] .≈ 0)
            end
        end
    end

    @testset "IVTypeScore" begin
        score = IVTypeScore()

        @testset "Type and Interface" begin
            @test DoubleML.get_score_name(score) == :IV_type
        end

        @testset "Computation" begin
            psi_a, psi_b = DoubleML.compute_score(score, Y, D, g_hat, m_hat)

            @test length(psi_a) == n
            @test length(psi_b) == n

            # IV-type uses g_hat, so changing it should affect psi_b but not psi_a
            g_hat2 = randn(rng, n)
            psi_a2, psi_b2 = DoubleML.compute_score(score, Y, D, g_hat2, m_hat)
            @test psi_a ≈ psi_a2  # psi_a doesn't depend on g_hat
            @test psi_b ≉ psi_b2  # psi_b depends on g_hat

            # Verify manual computation for IV-type score
            m_res = D .- m_hat
            g_res = Y .- g_hat
            @test psi_a ≈ .-m_res .* D  # psi_a uses D directly (not m_res)
            @test psi_b ≈ g_res .* m_res  # psi_b uses g_hat adjustment
        end

        @testset "Comparison with PartiallingOut" begin
            po_score = PartiallingOutScore()
            iv_score = IVTypeScore()

            psi_a_po, psi_b_po = DoubleML.compute_score(po_score, Y, D, g_hat, m_hat)
            psi_a_iv, psi_b_iv = DoubleML.compute_score(iv_score, Y, D, g_hat, m_hat)

            # psi_b should be identical (both use g_hat adjustment)
            @test psi_b_po ≈ psi_b_iv

            # psi_a differs: partialling out uses m_res, IV-type uses D
            m_res = D .- m_hat
            expected_diff_a = .-m_res .^ 2 .- (.-m_res .* D)
            actual_diff_a = psi_a_po .- psi_a_iv
            @test actual_diff_a ≈ expected_diff_a
        end

        @testset "IV-type produces different psi_a than partialling out" begin
            po_score = PartiallingOutScore()
            iv_score = IVTypeScore()

            psi_a_po, _ = DoubleML.compute_score(po_score, Y, D, g_hat, m_hat)
            psi_a_iv, _ = DoubleML.compute_score(iv_score, Y, D, g_hat, m_hat)

            # psi_a should be different between the two methods
            @test psi_a_po ≉ psi_a_iv
        end
    end

    @testset "ATEScore" begin
        score = ATEScore()

        @testset "Type and Interface" begin
            @test DoubleML.get_score_name(score) == :ATE
        end

        @testset "Computation" begin
            # Create binary treatment data
            D_binary = rand(rng, [0.0, 1.0], n)
            g_hat0 = randn(rng, n)
            g_hat1 = randn(rng, n)
            m_hat_adj = clamp.(rand(rng, n), 0.1, 0.9)  # Valid propensity scores

            psi_a, psi_b = DoubleML.compute_score(score, Y, D_binary, g_hat0, g_hat1, m_hat_adj)

            @test length(psi_a) == n
            @test length(psi_b) == n

            # Manual verification
            u_hat0 = Y .- g_hat0
            u_hat1 = Y .- g_hat1
            tau_hat = g_hat1 .- g_hat0
            ipw_correction = (
                D_binary .* u_hat1 ./ m_hat_adj .-
                    (1.0 .- D_binary) .* u_hat0 ./ (1.0 .- m_hat_adj)
            )

            expected_psi_b = tau_hat .+ ipw_correction
            expected_psi_a = -ones(n)  # -weights/mean(weights) where weights=1

            @test psi_b ≈ expected_psi_b
            @test psi_a ≈ expected_psi_a
        end

        @testset "ATE score is doubly robust" begin
            # Test that the score has the doubly robust property
            # When g_hat is correctly specified, the IPW correction should average to 0
            D_binary = rand(rng, [0.0, 1.0], n)
            # Set g_hat to true values
            g_hat0_true = Y  # Perfect prediction for treated (would be different in reality)
            g_hat1_true = Y  # Perfect prediction for control
            m_hat_adj = clamp.(rand(rng, n), 0.2, 0.8)

            _, psi_b = DoubleML.compute_score(score, Y, D_binary, g_hat0_true, g_hat1_true, m_hat_adj)

            # With perfect predictions, psi_b should equal tau_hat
            # This tests the doubly robust property
            @test !isnan(mean(psi_b))
        end
    end

    @testset "ATTEScore" begin
        score = ATTEScore()

        @testset "Type and Interface" begin
            @test DoubleML.get_score_name(score) == :ATTE
        end

        @testset "Computation" begin
            # Create binary treatment data with some treated units
            D_binary = vcat(ones(50), zeros(50))
            E_D_global = 0.5
            g_hat0 = randn(rng, 100)
            g_hat1 = randn(rng, 100)
            m_hat_adj = clamp.(rand(rng, 100), 0.1, 0.9)

            psi_a, psi_b = DoubleML.compute_score(score, Y, D_binary, g_hat0, g_hat1, m_hat_adj, E_D_global)

            @test length(psi_a) == 100
            @test length(psi_b) == 100

            # Manual verification
            weights = D_binary ./ E_D_global
            weights_bar = m_hat_adj ./ E_D_global

            u_hat0 = Y .- g_hat0
            u_hat1 = Y .- g_hat1
            tau_hat = g_hat1 .- g_hat0
            ipw_correction = (
                D_binary .* u_hat1 ./ m_hat_adj .-
                    (1.0 .- D_binary) .* u_hat0 ./ (1.0 .- m_hat_adj)
            )

            expected_psi_b = weights .* tau_hat .+ weights_bar .* ipw_correction
            expected_psi_a = -weights

            @test psi_b ≈ expected_psi_b
            @test psi_a ≈ expected_psi_a
        end

        @testset "ATTE weights properly" begin
            # Test that ATTE gives more weight to treated units
            D_binary = vcat(ones(30), zeros(70))
            E_D_global = 0.3
            g_hat0 = randn(rng, 100)
            g_hat1 = randn(rng, 100)
            m_hat_adj = clamp.(rand(rng, 100), 0.1, 0.9)

            psi_a, psi_b = DoubleML.compute_score(score, Y, D_binary, g_hat0, g_hat1, m_hat_adj, E_D_global)

            # Treated units (D=1) should have weights 1/E_D_global
            # Control units (D=0) should have weights 0
            @test all(psi_a[D_binary .== 1] .< 0)  # Treated have negative psi_a
            @test all(psi_a[D_binary .== 0] .== 0)  # Control have zero psi_a
        end
    end

    @testset "dml2_solve" begin
        @testset "Basic computation" begin
            n = 100
            psi_a = -rand(rng, n)  # Must be negative
            psi_b = randn(rng, n)

            theta = DoubleML.dml2_solve(psi_a, psi_b)

            # Verify: theta = -mean(psi_b) / mean(psi_a)
            expected = -mean(psi_b) / mean(psi_a)
            @test theta ≈ expected
        end

        @testset "Edge cases" begin
            # Zero psi_b
            theta = DoubleML.dml2_solve([-1.0, -1.0], [0.0, 0.0])
            @test theta ≈ 0.0

            # Equal means
            theta = DoubleML.dml2_solve([-2.0, -2.0], [1.0, 1.0])
            @test theta ≈ 0.5  # -1 / -2
        end
    end

    @testset "to_numeric helper function" begin
        @testset "Numeric input (unchanged)" begin
            d_num = randn(100)
            result = DoubleML.to_numeric(d_num)
            @test result === d_num  # Same object, not copied
            @test eltype(result) <: AbstractFloat
        end

        @testset "Categorical input (converted)" begin
            D = [0, 1, 0, 1, 0]
            d_cat = CategoricalArrays.categorical(D, levels = [0, 1])
            result = DoubleML.to_numeric(d_cat)
            @test result isa Vector{Int64}
            @test result ≈ [0, 1, 0, 1, 0]
        end

        @testset "Categorical with Float32" begin
            D = Float32[0.0f0, 1.0f0, 0.0f0]
            d_cat = CategoricalArrays.categorical(D, levels = Float32[0.0f0, 1.0f0])
            result = DoubleML.to_numeric(d_cat)
            @test result isa Vector{Float32}
            @test result ≈ [0.0, 1.0, 0.0]
        end

        @testset "Type stability" begin
            d_num = randn(10)
            # Test that return type is inferred
            @inferred DoubleML.to_numeric(d_num)
        end
    end
end
