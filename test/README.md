# DoubleML.jl Test Suite

This directory contains the comprehensive test suite for DoubleML.jl.

## Quick Start

Run all tests:

```julia
using Pkg
Pkg.test("DoubleML")
```

Run specific test files:

```julia
include("test/runtests.jl")
```

## Test Organization

### Unit Tests (`test_unit/`)

Core functionality tests:

| File | Description |
|------|-------------|
| `test_data.jl` | DoubleMLData construction and validation |
| `test_scores.jl` | Score computation for PLR, IRM, LPLR |
| `test_bootstrap.jl` | Bootstrap inference methods |
| `test_resampling.jl` | Cross-fitting sample splitting |
| `test_irm_data.jl` | IRM data generation and handling |
| `test_statsapi.jl` | StatsAPI compliance (`coef`, `stderror`, `confint`, etc.) |
| `test_evaluation.jl` | Model evaluation and reporting |
| `test_edge_cases.jl` | Edge cases and error handling |
| `test_aggregation.jl` | Multi-repetition aggregation |

### Model Tests (`test_models/`)

Complete model workflow tests:

| File | Description |
|------|-------------|
| `test_plr.jl` | Partially Linear Regression model |
| `test_irm.jl` | Interactive Regression Model |
| `test_lplr.jl` | Logistic Partially Linear Regression |

### Python Validation Tests (`test_python/`)

Bidirectional validation against [Python DoubleML](https://github.com/DoubleML/doubleml-for-py):

```
test_python/
├── config.toml              # Central configuration (TOML format)
├── setup_python_env.sh      # Python environment setup (requires uv)
├── generate_data_julia.jl   # Julia data generation
├── generate_data_python.py  # Python data generation
├── fit_julia.jl             # Julia model fitting
├── fit_python.py            # Python model fitting
└── validate.jl              # Main comparison tests
```

**Architecture:**

1. **Data Generation**: Both languages generate equivalent IRM/PLR/LPLR datasets
2. **Model Fitting**: Each language fits models on both datasets
3. **Comparison**: 54+ test assertions compare coef/SE within configurable tolerance

**Configuration (`config.toml`):**

```toml
[data_generation]
n_obs = 5000
dim_x = 20
theta = 0.5        # True IRM effect
alpha = 0.5        # True PLR effect
lplr_alpha = 0.5   # True LPLR effect
rng_seed = 42

[model_fitting]
n_folds = 3
n_rep = 1
tolerance = 0.50   # 50% relative difference threshold
```

Adjust parameters to customize validation:

- `n_obs`: Sample size (affects computation time)
- `dim_x`: Number of covariates
- `tolerance`: Pass/fail threshold for coefficient/SE comparison
- Learners: RF (RandomForest) and Ridge (Linear/Logistic)

**Setup:**

1. Install [uv](https://github.com/astral-sh/uv):

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Initialize Python environment:

   ```bash
   cd test/test_python
   bash setup_python_env.sh
   ```

**Running:**

Tests automatically detect Python availability:

- **Python available**: Full bidirectional validation (54+ tests)
- **Python unavailable**: Julia-only validation (skips Python comparison)

Manual execution:

```bash
cd test && julia --project=. test_python/validate.jl
```

See [test_python/README.md](test_python/README.md) for details.

### Experimental Tests (`experimental/`)

Performance benchmarks and research code:

- Threading benchmarks for cross-fitting
- Score computation optimizations
- Memory allocation analysis

See [experimental/README.md](experimental/README.md) for details.

## Troubleshooting

### Python Not Found

If Python validation is skipped:

```
⚠️  Python validation tests skipped - environment not available
```

Run the setup script in `test/test_python/` or ensure `pyproject.toml` exists.

### Adjusting Tolerance

Edit `test_python/config.toml`:

```toml
[model_fitting]
tolerance = 0.60  # More lenient
```

### CondaPkg Conflicts

This test suite uses a `uv`-based Python environment (not CondaPkg) for Python validation. If you see CondaPkg-related errors, ensure you're running tests from the project root with `Pkg.test()`.

## Test Output

Test results generate:

- `test_python/data/*.csv` - Generated datasets (overwritten each run)
- `test_python/data/results_*.json` - Model fitting results
- `test_python/python_comparison.md` - Detailed comparison report

## Coverage

Tests cover:

- ✅ Model correctness vs Python reference
- ✅ Type stability (Float32, Float64)
- ✅ StatsAPI compliance
- ✅ Cross-fitting with multiple repetitions
- ✅ Bootstrap inference
- ✅ Edge cases and error handling
- ✅ JET static analysis
- ✅ Aqua quality checks
