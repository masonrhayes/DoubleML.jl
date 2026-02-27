"""
Generate IRM, PLR, and LPLR data using DoubleML.jl package.
This script is called from the test suite or fit_julia.jl.
"""

using DoubleML
using StableRNGs
using CSV
using DataFrames
using TOML

# Load configuration
config_path = joinpath(@__DIR__, "config.toml")
config = TOML.parsefile(config_path)

data_gen = config["data_generation"]
const RNG = StableRNGs.StableRNG(data_gen["rng_seed"])
const N_OBS = data_gen["n_obs"]
const DIM_X = data_gen["dim_x"]
const THETA = data_gen["theta"]
const ALPHA = data_gen["alpha"]
const LPLR_ALPHA = data_gen["lplr_alpha"]

# Ensure data directory exists
data_dir = joinpath(@__DIR__, "data")
mkpath(data_dir)

println("Generating IRM data (Julia)...")
irm_data = make_irm_data(N_OBS; theta = THETA, rng = RNG)

# Convert to DataFrame and save
df_irm = DataFrame(irm_data.x, irm_data.x_cols)
df_irm.y = irm_data.y
df_irm.d = irm_data.d

irm_path = joinpath(data_dir, "make_irm_data_jl.csv")
CSV.write(irm_path, df_irm)
println("  Saved: $irm_path")

println("Generating PLR data (Julia)...")
plr_data = make_plr_CCDDHNR2018(N_OBS; alpha = ALPHA, rng = RNG)

# Convert to DataFrame and save
df_plr = DataFrame(plr_data.x, plr_data.x_cols)
df_plr.y = plr_data.y
df_plr.d = plr_data.d

plr_path = joinpath(data_dir, "make_plr_CCDDHNR2018_jl.csv")
CSV.write(plr_path, df_plr)
println("  Saved: $plr_path")

println("Generating LPLR data (Julia)...")
lplr_data = make_lplr_LZZ2020(N_OBS; alpha = LPLR_ALPHA, dim_x = DIM_X, rng = RNG, treatment = "continuous")

# Convert to DataFrame and save
df_lplr = DataFrame(lplr_data.x, lplr_data.x_cols)
df_lplr.y = lplr_data.y
df_lplr.d = lplr_data.d

lplr_path = joinpath(data_dir, "make_lplr_LZZ2020_jl.csv")
CSV.write(lplr_path, df_lplr)
println("  Saved: $lplr_path")

println("✓ Julia data generation complete!")
