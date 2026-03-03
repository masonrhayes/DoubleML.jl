# DoubleML Performance Benchmark: Julia vs Python

**Generated:** 2026-03-03T12:34:58.805

## Hardware

| Component | Specification |
|-----------|---------------|
| CPU | Intel(R) Core(TM) i7-14700K (28 threads) |
| RAM | 63 GB |
| OS | Linux |

## Test Configuration

| Parameter | Value |
|-----------|-------|
| Observations | 100000 |
| Covariates | 1000 |
| Treatment Effect (true) | 0.5 |
| Cross-fitting Folds | 5 |

## Learners Used

| Language | Learner |
|----------|---------|
| Julia | EvoTreeRegressor (100 trees, max_depth=6, eta=0.1) |
| Python | XGBRegressor (100 trees, max_depth=6, learning_rate=0.1) |

## Timing Results

### Julia (DoubleML.jl)

| Metric | Time (seconds) |
|--------|----------------|
| Median | 41.71s |
### Python (DoubleML)

| Metric | Time (seconds) |
|--------|----------------|
| Median | 306.71s |

### Performance Comparison

**Winner: Julia**

- **Speedup Factor:** 7.35x (Julia is faster)
- **Time Difference:** 264.99s

## Coefficient Estimates

| Metric | Julia | Python | Difference |
|--------|-------|--------|------------|
| Coefficient | 0.487874 | 0.485514 | 0.00236 (0.47%) |
| Std Error | 0.00314 | 0.003144 | 4.0e-6 (0.14%) |

**True Treatment Effect:** 0.5

### Accuracy Assessment

- **Coefficient Accuracy:** Estimates differ by 0.47% from each other
- **Standard Error Agreement:** SEs differ by 0.14%

✓ Coefficient estimates are in good agreement (< 5% difference)
✓ Standard errors are in good agreement (< 10% difference)

## Summary

This benchmark compares DoubleML.jl (Julia) against the Python DoubleML package on identical data:
- 100000 observations with 1000 covariates
- Using gradient boosted trees (EvoTrees in Julia, XGBoost in Python)
- Both implementations use the same partialling_out score function

### Key Findings

1. **Performance:** Julia is 7.35x faster
2. **Accuracy:** Both implementations produce similar coefficient estimates (0.47% difference)
3. **Inference:** Standard errors are consistent between implementations (0.14% difference)

## Raw Data Files

- Julia results: `benchmark_results_julia.json`
- Python results: `benchmark_results_python.json`
- Shared data: `benchmark_data.csv`

---

*Note: Benchmarks run on 100000 observations with 1000 dimensions. Times measured using BenchmarkTools (Julia) with automatic sample size determination and time.perf_counter() (Python) with 3 repetitions.*
