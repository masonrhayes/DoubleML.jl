# DoubleML Performance Benchmark: Julia vs Python

**Generated:** 2026-02-27T21:08:17.148

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
| Julia | EvoTreeRegressor (100 trees, max_depth=6) |
| Python | XGBRegressor (100 trees, max_depth=6) |

## Timing Results

### Julia (DoubleML.jl)

| Metric | Time (seconds) |
|--------|----------------|
| Median | 34.68s |
### Python (DoubleML)

| Metric | Time (seconds) |
|--------|----------------|
| Median | 310.78s |

### Performance Comparison

**Winner: Julia**

- **Speedup Factor:** 8.96x (Julia is faster)
- **Time Difference:** 276.1s

## Coefficient Estimates

| Metric | Julia | Python | Difference |
|--------|-------|--------|------------|
| Coefficient | 0.490186 | 0.485543 | 0.004643 (0.93%) |
| Std Error | 0.003136 | 0.003146 | 1.0e-5 (0.32%) |

**True Treatment Effect:** 0.5

### Accuracy Assessment

- **Coefficient Accuracy:** Estimates differ by 0.93% from each other
- **Standard Error Agreement:** SEs differ by 0.32%

✓ Coefficient estimates are in good agreement (< 5% difference)
✓ Standard errors are in good agreement (< 10% difference)

## Summary

This benchmark compares DoubleML.jl (Julia) against the Python DoubleML package on identical data:
- 100000 observations with 1000 covariates
- Using gradient boosted trees (EvoTrees in Julia, XGBoost in Python)
- Both implementations use the same partialling_out score function

### Key Findings

1. **Performance:** Julia is 8.96x faster
2. **Accuracy:** Both implementations produce similar coefficient estimates (0.93% difference)
3. **Inference:** Standard errors are consistent between implementations (0.32% difference)

## Raw Data Files

- Julia results: `benchmark_results_julia.json`
- Python results: `benchmark_results_python.json`
- Shared data: `benchmark_data.csv`

---

*Note: Benchmarks run on 100000 observations with 1000 dimensions. Times measured using BenchmarkTools (Julia) and time.perf_counter() (Python) with 5 repetitions each.*
