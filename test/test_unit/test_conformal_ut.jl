using Test
using DoubleML
using ConformalPrediction

const DoubleMLConformalExt = Base.get_extension(DoubleML, :DoubleMLConformalExt)

@testset "Conformal UT validation and warnings" begin
    @test_throws ArgumentError DoubleMLConformalExt._ut_weights(0.0, 2.0, 1.0)
    @test_throws ArgumentError DoubleMLConformalExt._ut_weights(Inf, 2.0, 1.0)
    @test_throws ArgumentError DoubleMLConformalExt._ut_weights(1.0, NaN, 1.0)
    @test_throws ArgumentError DoubleMLConformalExt._ut_weights(1.0, 2.0, -2.0)
    @test_throws ArgumentError DoubleMLConformalExt._ut_weights(1.0, 2.0, Inf)

    invalid_covariance_args = (-10.0, 0.5, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0)
    @test_logs (:warn, r"materially indefinite") DoubleMLConformalExt._ut_propagate(
        invalid_covariance_args...; verbose = 1
    )
    @test_logs DoubleMLConformalExt._ut_propagate(
        invalid_covariance_args...; verbose = 0
    )

    near_zero_denominator_args = (1.0e-12, 0.15, 1.0e-20, 9.0e-5, 0.0, 1.0, 2.0, 1.0)
    @test_logs (:warn, r"denominator was near zero") DoubleMLConformalExt._ut_propagate(
        near_zero_denominator_args...; verbose = 1
    )
    @test_logs DoubleMLConformalExt._ut_propagate(
        near_zero_denominator_args...; verbose = 0
    )
end
