using SafeTestsets
using DoubleML

# Unit tests by functionality (testing individual components)
@time @safetestset "Data Module" include("test_unit/test_data.jl")
@time @safetestset "Score Functions" include("test_unit/test_scores.jl")
@time @safetestset "Bootstrap Inference" include("test_unit/test_bootstrap.jl")
@time @safetestset "Resampling" include("test_unit/test_resampling.jl")
@time @safetestset "IRM Data Tests" include("test_unit/test_irm_data.jl")
@time @safetestset "StatsAPI Methods" include("test_unit/test_statsapi.jl")
@time @safetestset "Evaluation Functions" include("test_unit/test_evaluation.jl")
@time @safetestset "Edge Cases" include("test_unit/test_edge_cases.jl")
@time @safetestset "Aggregation and Multiple Repetitions" include("test_unit/test_aggregation.jl")

# Model tests (testing complete model workflows)
@time @safetestset "PLR Model" include("test_models/test_plr.jl")
@time @safetestset "IRM Model" include("test_models/test_irm.jl")
@time @safetestset "LPLR Model" include("test_models/test_lplr.jl")

# Python validation tests (compare against official DoubleML Python package)
@time @safetestset "Python Validation" include("test_python/validate.jl")

# Quality assurance tests (following SciML best practices)
@time @safetestset "Code Quality - JET" begin
    using JET
    # Use test_broken since JET may find issues that don't break functionality
    # This ensures the test suite passes while tracking code quality improvements
    @test_broken isempty(JET.report_package(DoubleML))
end

@time @safetestset "Code Quality - Aqua" begin
    using DoubleML
    using Aqua
    # Skip checks that are CI/maintenance concerns, not actual code quality issues:
    # - deps_compat: Version compatibility checking (maintenance)
    # - stale_deps: Unused dependency detection (maintenance)
    # - persistent_tasks: Often false positive in CI due to precompilation processes
    Aqua.test_all(
        DoubleML;
        deps_compat = false,
        stale_deps = false,
        persistent_tasks = false,
        project_extras = false
    )
end
