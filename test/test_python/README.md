# Python Validation Tests

Bidirectional validation tests comparing DoubleML.jl against Python DoubleML.

## Quick Start

1. **Setup Python environment** (once):
   ```bash
   bash setup_python_env.sh  # Requires uv
   ```

2. **Run tests**:
   ```bash
   cd ../..
   julia --project=. -e 'using Pkg; Pkg.test()'
   ```

## Configuration

Edit `config.toml` to adjust test parameters:
- `n_obs`: Sample size
- `dim_x`: Number of covariates  
- `tolerance`: Pass/fail threshold
- Learners: RF (RandomForest) and Ridge (Linear/Logistic)

## Key Design Principles

1. **Single Source of Truth**: All configuration is in `config.toml`
2. **Simple Additions**: To add a new model, edit only `config.toml`
3. **Generic Runners**: Julia and Python scripts automatically read the config
4. **Minimal Duplication**: Learners and models defined once, used everywhere

## Files

| File | Purpose |
|------|---------|
| `config.toml` | **Edit this** - Defines all tests, learners, and parameters |
| `fit_julia.jl` | Generic Julia model runner |
| `fit_python.py` | Generic Python model runner |
| `validate.jl` | Main test script that coordinates both runners |
| `generate_data_julia.jl` | Generates Julia test data |
| `generate_data_python.py` | Generates Python test data |

## Adding a New Model

### Step 1: Edit `config.toml`

Add a new `[[models]]` entry:

```toml
[[models]]
name = "LPLR"
julia_type = "DoubleMLLPLR"
python_type = "DoubleMLLPLR"
data_generator_julia = "make_lplr_LZZ2020"
data_generator_python = "make_lplr_LZZ2020"

  [[models.scores]]
  name = "nuisance_space"
  julia_score = "nuisance_space"
  python_score = "nuisance_space"
  learners = ["RF", "Ridge"]
```

### Step 2: Ensure Data Generator Exists

Make sure the data generator function is exported from DoubleML.jl.

### That's it!

The test system will automatically:
- Generate data using your data generator
- Fit models with both RF and Ridge learners
- Compare Julia vs Python results
- Generate a markdown report

## Full Documentation

See [../README.md](../README.md) for full project documentation.
