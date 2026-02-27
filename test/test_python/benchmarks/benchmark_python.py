"""
Benchmark script for Python DoubleML
Reads shared CSV data and runs Python benchmark
"""
import pandas as pd
import numpy as np
import doubleml
import xgboost as xgb
import time
import json
import statistics
from pathlib import Path

# Configuration
N_OBS = 100_000
DIM_X = 1000
ALPHA = 0.5
N_FOLDS = 5
DATA_FILE = Path(__file__).parent / "benchmark_data.csv"
RESULTS_FILE = Path(__file__).parent / "benchmark_results_python.json"

print("=" * 60)
print("Python DoubleML Benchmark")
print("=" * 60)
print("Configuration:")
print(f"  N observations: {N_OBS}")
print(f"  N dimensions: {DIM_X}")
print(f"  Treatment effect: {ALPHA}")
print(f"  Cross-fitting folds: {N_FOLDS}")
print(f"  Learner: XGBRegressor")
print()

# Step 1: Load data
print("Step 1: Loading data from CSV...")
df = pd.read_csv(DATA_FILE)
print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print()

# Step 2: Setup XGBoost learners
print("Step 2: Setting up XGBRegressor...")
learner_l = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)

learner_m = xgb.XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    objective='reg:squarederror',
    n_jobs=-1,
    random_state=42
)
print("  ✓ Learners configured")
print()

# Step 3: Create DoubleML data object
print("Step 3: Creating DoubleML data object...")
# Get covariate columns (all except 'y' and 'd')
x_cols = [col for col in df.columns if col.startswith('X')]
print(f"  Using {len(x_cols)} covariates")

dml_data = doubleml.DoubleMLData(
    df,
    y_col='y',
    d_cols='d',
    x_cols=x_cols
)
print("  ✓ Data object created")
print()

# Step 4: Run benchmark with multiple repetitions
print("Step 4: Running benchmark (5 repetitions)...")
times = []

for i in range(3):
    print(f"  Run {i+1}/3...")
    
    # Create fresh model for each run
    model = doubleml.DoubleMLPLR(
        dml_data,
        learner_l,
        learner_m,
        n_folds=N_FOLDS,
        score='partialling out'
    )
    
    # Time the fit
    start = time.perf_counter()
    model.fit()
    elapsed = time.perf_counter() - start
    
    times.append(elapsed)
    print(f"    Time: {elapsed:.2f}s")

median_time = statistics.median(times)

print()
print("  Timing Results:")
print(f"    Median: {median_time:.2f}s")

# Step 5: Final fit to get coef and SE
print("Step 5: Final fit for coefficient estimation...")
model_final = doubleml.DoubleMLPLR(
    dml_data,
    learner_l,
    learner_m,
    n_folds=N_FOLDS,
    score='partialling out'
)
model_final.fit()

coef_py = model_final.coef[0]
se_py = model_final.se[0]
n_obs_py = model_final.n_obs

print("  Results:")
print(f"    Coefficient: {coef_py}")
print(f"    Std Error:   {se_py}")
print(f"    N obs:       {n_obs_py}")
print()

# Step 6: Save results
print("Step 6: Saving results...")
results = {
    "language": "Python",
    "n_obs": N_OBS,
    "dim_x": DIM_X,
    "alpha": ALPHA,
    "learner": "XGBRegressor",
    "n_folds": N_FOLDS,
    "benchmark_median_sec": median_time,
    "coefficient": float(coef_py),
    "std_error": float(se_py),
    "data_file": str(DATA_FILE),
}

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)

print(f"  ✓ Results saved to: {RESULTS_FILE}")
print()
print("=" * 60)
print("Python benchmark complete!")
print("=" * 60)
