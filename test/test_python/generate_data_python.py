"""Generate IRM, PLR, and LPLR data using Python DoubleML package."""

import numpy as np
import pandas as pd
import os
import toml

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), 'config.toml')
config = toml.load(config_path)

N_OBS = config['data_generation']['n_obs']
DIM_X = config['data_generation']['dim_x']
THETA = config['data_generation']['theta']
ALPHA = config['data_generation']['alpha']
LPLR_ALPHA = config['data_generation']['lplr_alpha']

# Ensure data directory exists
data_dir = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(data_dir, exist_ok=True)

print("Generating IRM data (Python)...")
from doubleml.irm.datasets import make_irm_data

# Generate IRM data
irm_data = make_irm_data(
    n_obs=N_OBS,
    dim_x=DIM_X,
    theta=THETA,
    return_type="DataFrame"
)

# Save IRM data
irm_path = os.path.join(data_dir, 'make_irm_data_py.csv')
irm_data.to_csv(irm_path, index=False)
print(f"  Saved: {irm_path}")

print("Generating PLR data (Python)...")
from doubleml.plm.datasets import make_plr_CCDDHNR2018

# Generate PLR data
plr_data = make_plr_CCDDHNR2018(
    n_obs=N_OBS,
    dim_x=DIM_X,
    alpha=ALPHA,
    return_type="DataFrame"
)

# Save PLR data
plr_path = os.path.join(data_dir, 'make_plr_CCDDHNR2018_py.csv')
plr_data.to_csv(plr_path, index=False)
print(f"  Saved: {plr_path}")

print("Generating LPLR data (Python)...")
from doubleml.plm.datasets import make_lplr_LZZ2020

# Generate LPLR data
lplr_data = make_lplr_LZZ2020(
    n_obs=N_OBS,
    dim_x=DIM_X,
    alpha=LPLR_ALPHA,
    return_type="DataFrame",
    treatment="continuous"
)

# Drop 'p' column if present (for consistency with Julia default)
if 'p' in lplr_data.columns:
    lplr_data = lplr_data.drop(columns=['p'])

# Save LPLR data
lplr_path = os.path.join(data_dir, 'make_lplr_LZZ2020_py.csv')
lplr_data.to_csv(lplr_path, index=False)
print(f"  Saved: {lplr_path}")

print("✓ Python data generation complete!")
