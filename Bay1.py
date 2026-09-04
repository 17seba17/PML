import xarray as xr
import numpy as np
import pandas as pd
import pymc as pm
import pytensor.tensor as pt
from scipy.stats import norm
import matplotlib.pyplot as plt

###############################################
# 1. LOAD DATA
###############################################

ds_reg = xr.open_dataset("tg_ens_mean_0.1deg_reg_v33.0e.nc")
tg = ds_reg['tg']  # [time, lat, lon]

ds_gmt = xr.open_dataset("HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc")
gmt_var = 'tas_mean' if 'tas_mean' in ds_gmt else 'tas'
gmt = ds_gmt[gmt_var]

###############################################
# 2. RESTRICT TO 1950–PRESENT AND SMOOTH GMT
###############################################

tg = tg.sel(time=slice("1950-01-01", None))
gmt = gmt.sel(time=slice("1950-01-01", None))

gmt_series = gmt.mean(dim=[d for d in gmt.dims if d != 'time']).to_series().sort_index()
T_smooth = gmt_series.rolling(window=120, center=True, min_periods=1).mean()

T_smooth.index = pd.to_datetime(T_smooth.index)
T_smooth = T_smooth.reindex(tg['time'].to_index(), method='ffill').bfill()
T_full = xr.DataArray(T_smooth.values, coords={'time': tg['time']}, dims=['time'])

###############################################
# 3. SELECT FVG REGION AND STACK
###############################################

lat_name = 'latitude' if 'latitude' in tg.coords else 'lat'
lon_name = 'longitude' if 'longitude' in tg.coords else 'lon'

tg_fvg = tg.sel({lat_name: slice(45, 47), lon_name: slice(12, 14)})
time = tg_fvg['time'].to_index()
doy = time.dayofyear
tg_fvg = tg_fvg.assign_coords(doy=('time', doy))
tg_stack = tg_fvg.stack(cell=(lat_name, lon_name))  # [time, cell]

###############################################
# 4. PRECOMPUTE FOURIER BASIS & SAFE SPLIT
###############################################

omega = 2 * np.pi / 365.25

# Continuous temporal split (80% train / 20% test)
n_time = len(tg_stack['time'])
split_idx = int(n_time * 0.8)

tg_train_stack = tg_stack.isel(time=slice(0, split_idx))
T_train_arr = T_full.isel(time=slice(0, split_idx)).values

tg_full_arr = tg_stack.values
T_full_arr = T_full.values

###############################################
# 5. BAYESIAN MODEL PER GRID CELL (OPTIMIZED)
###############################################

cells = tg_stack['cell'].values
counterfactual_stack = xr.full_like(tg_stack, np.nan)

for i, cell in enumerate(cells):
    print(f"Processing cell {i+1}/{len(cells)}: {cell}")

    y_full_orig = tg_full_arr[:, i]
    T_full_raw_cell = T_full_arr
    doy_full_cell = doy.values

    y_train_orig = tg_train_stack.values[:, i]
    T_train_raw = T_train_arr
    doy_train = doy.values[:split_idx]

    # Standardize based strictly on training subset
    y_mean, y_std = np.nanmean(y_train_orig), np.nanstd(y_train_orig)
    y_train = (y_train_orig - y_mean) / y_std

    T_mean, T_std = np.nanmean(T_train_raw), np.nanstd(T_train_raw)
    T_train_norm = (T_train_raw - T_mean) / T_std
    T_full_norm_cell = (T_full_raw_cell - T_mean) / T_std

    t_train_rad = doy_train * omega
    t_full_rad = doy_full_cell * omega

    # Bayesian Model Specification
    with pm.Model() as model:
        a0_intercept = pm.Normal('a0_intercept', mu=0.0, sigma=1.0)
        a_intercept = [
            pm.Normal(f'a{k}_intercept', mu=0.0, sigma=1.0 / (2 * k - 1))
            for k in range(1, 5)
        ]
        a_slope = [
            pm.Normal(f'a{k}_slope', mu=0.0, sigma=0.1)
            for k in range(0, 5)
        ]
        b_intercept = [
            pm.Normal(f'b{k}_intercept', mu=0.0, sigma=1.0 / (2 * k - 1))
            for k in range(1, 5)
        ]
        b_slope = [
            pm.Normal(f'b{k}_slope', mu=0.0, sigma=0.1)
            for k in range(1, 5)
        ]

        # Trajectory mu(T, t) for training set
        a0_T = a0_intercept + a_slope[0] * T_train_norm
        mu_train = a0_T
        for k in range(1, 5):
            a_k_T = a_intercept[k - 1] + a_slope[k] * T_train_norm
            b_k_T = b_intercept[k - 1] + b_slope[k - 1] * T_train_norm
            mu_train += a_k_T * pt.cos(k * t_train_rad) + b_k_T * pt.sin(k * t_train_rad)

        sigma = pm.HalfNormal('sigma', sigma=1.0)
        y_obs = pm.Normal('y_obs', mu=mu_train, sigma=sigma, observed=y_train)

        map_estimate = pm.find_MAP(progressbar=False)

    # MAP Parameter Extraction
    a0_int_hat = map_estimate['a0_intercept']
    a_slope_hat = [map_estimate[f'a{k}_slope'] for k in range(0, 5)]
    a_int_hat = [map_estimate[f'a{k}_intercept'] for k in range(1, 5)]
    b_int_hat = [map_estimate[f'b{k}_intercept'] for k in range(1, 5)]
    b_slope_hat = [map_estimate[f'b{k}_slope'] for k in range(1, 5)]
    sigma_hat = map_estimate['sigma']

    # Factual Reconstruction
    a0_T_full = a0_int_hat + a_slope_hat[0] * T_full_norm_cell
    mu_f_full = a0_T_full.copy()
    for k in range(1, 5):
        a_k_T_full = a_int_hat[k - 1] + a_slope_hat[k] * T_full_norm_cell
        b_k_T_full = b_int_hat[k - 1] + b_slope_hat[k - 1] * T_full_norm_cell
        mu_f_full += a_k_T_full * np.cos(k * t_full_rad) + b_k_T_full * np.sin(k * t_full_rad)

    # Counterfactual Reconstruction (Setting GMT Anomaly to Pre-Industrial baseline: 0°C)
    T0_full = (0.0 - T_mean) / T_std
    a0_T0_full = a0_int_hat + a_slope_hat[0] * T0_full
    mu_cf_full = a0_T0_full.copy()
    for k in range(1, 5):
        a_k_T0_full = a_int_hat[k - 1] + a_slope_hat[k] * T0_full
        b_k_T0_full = b_int_hat[k - 1] + b_slope_hat[k - 1] * T0_full
        mu_cf_full += a_k_T0_full * np.cos(k * t_full_rad) + b_k_T0_full * np.sin(k * t_full_rad)

    # Rescale to physical units
    mu_f_orig = mu_f_full * y_std + y_mean
    mu_cf_orig = mu_cf_full * y_std + y_mean

    # Quantile Mapping
    x_f = y_full_orig
    z_f = (x_f - mu_f_orig) / (sigma_hat * y_std)
    q = norm.cdf(np.clip(z_f, -4.0, 4.0))
    z_cf = norm.ppf(np.clip(q, 1e-6, 1.0 - 1e-6))
    x_cf = mu_cf_orig + z_cf * (sigma_hat * y_std)

    counterfactual_stack.loc[{'cell': cell}] = x_cf

###############################################
# 6. UNSTACK DATASET
###############################################

tg_cf = counterfactual_stack.unstack('cell')

tg_fvg_mean = tg_fvg.mean(dim=(lat_name, lon_name))
tg_cf_mean = tg_cf.mean(dim=(lat_name, lon_name))

###############################################
# 7. ALL DIAGNOSTIC & EVALUATION PLOTS
###############################################

# Plot 1: Daily Time Series Comparison
plt.figure(figsize=(12, 5))
plt.plot(tg_fvg_mean['time'], tg_fvg_mean, label="Factual", color='black', alpha=0.7, linewidth=0.8)
plt.plot(tg_cf_mean['time'], tg_cf_mean, label="Counterfactual", color='darkorange', alpha=0.7, linewidth=0.8)
plt.title("Friuli Venezia Giulia — Daily Temperature (1950–Present)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 2: Annual Mean Temperature Trend
annual_f = tg_fvg_mean.groupby('time.year').mean()
annual_cf = tg_cf_mean.groupby('time.year').mean()

plt.figure(figsize=(10, 5))
plt.plot(annual_f['year'], annual_f, label="Factual", color='black', marker='o', markersize=3)
plt.plot(annual_cf['year'], annual_cf, label="Counterfactual", color='darkorange', marker='o', markersize=3)
plt.title("Friuli Venezia Giulia — Annual Mean Temperature Evolution")
plt.xlabel("Year")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot 3: Anthropogenic Warming Anomaly Bar Chart
anomaly = annual_f - annual_cf

plt.figure(figsize=(10, 4))
plt.bar(annual_f['year'], anomaly, color=np.where(anomaly >= 0, 'crimson', 'navy'), alpha=0.8)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Anthropogenic Warming Impact (Factual − Counterfactual)")
plt.xlabel("Year")
plt.ylabel("Warming Difference (°C)")
plt.grid(True)
plt.tight_layout()
plt.show()