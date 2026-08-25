import os
import numpy as np
import xarray as xr
import torch


def inverse_softplus(y):
    return np.log(np.expm1(y))


def compute_hadcrut_invariant_moments(
    hadcrut_path="../HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc",
    lat_min=45.0,
    lat_max=50.0,
    lon_min=20.0,
    lon_max=25.0,
    split_year=2005
):
    """
    Calcola la sensitività fisica invariante e la sua incertezza spaziale/orografica
    usando l'intero record storico 1850-split_year di HadCRUT tramite proiezione integrale.
    """
    if not os.path.exists(hadcrut_path):
        if os.path.exists("HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"):
            hadcrut_path = "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"
        else:
            raise FileNotFoundError(f"File HadCRUT non trovato in: {hadcrut_path}")

    ds = xr.open_dataset(hadcrut_path)

    ds_train = ds.isel(time=ds.time.dt.year <= split_year)

    weights = np.cos(np.deg2rad(ds_train.latitude))
    gmt_series = ds_train["tas_mean"].weighted(weights).mean(dim=("latitude", "longitude")).values
    
    gmt_c = gmt_series - np.nanmean(gmt_series)
    var_gmt = np.nanvar(gmt_series)

    balkans_sub = ds_train["tas_mean"].sel(
        latitude=slice(lat_min, lat_max),
        longitude=slice(lon_min, lon_max)
    )
    
    tas_balkans = balkans_sub.values  # [T, n_lat, n_lon]
    T, n_lat, n_lon = tas_balkans.shape
    tas_2d = tas_balkans.reshape(T, -1)

    ds.close()

    alphas = []
    for col in range(tas_2d.shape[1]):
        b_cell = tas_2d[:, col]
        valid = ~np.isnan(b_cell) & ~np.isnan(gmt_series)
        
        if np.sum(valid) > 360:
            b_c = b_cell[valid] - np.mean(b_cell[valid])
            g_c = gmt_series[valid] - np.mean(gmt_series[valid])
            
            cov_i = np.mean(b_c * g_c)
            var_g_i = np.var(g_c)
            
            s_i = cov_i / (var_g_i + 1e-8)
            alphas.append(s_i)

    alphas = np.array(alphas)

    target_mu_phys = float(np.mean(alphas))
    target_std_phys = float(np.std(alphas))

    if len(alphas) <= 1 or target_std_phys < 1e-4:
        res = b_c - target_mu_phys * g_c
        target_std_phys = float(np.std(res) / (np.std(g_c) + 1e-8) / np.sqrt(len(g_c)))

    print(f"years: (1850 - {split_year}):")
    print(f"cells          : {len(alphas)}")
    print(f"(target_mu)  : {target_mu_phys:.4f} °C/°C")
    print(f"(target_std)   : {target_std_phys:.4f} °C/°C")

    return target_mu_phys, target_std_phys


def scale(
    norm_stats,
    hadcrut_path="../HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc",
    lat_min=45.0,
    lat_max=50.0,
    lon_min=20.0,
    lon_max=25.0,
    split_year=2005,
    target_mu_phys=None,
    target_std_phys=None
):
    if target_mu_phys is None:
        target_mu_phys, target_std_phys = compute_hadcrut_invariant_moments(
            hadcrut_path=hadcrut_path,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            split_year=split_year
        )

    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()

    fgmt_std = float(norm_stats["fgmt_std"])
    mean_tg_std = float(np.mean(tg_std))

    k_scale = fgmt_std / mean_tg_std

    target_mu_norm = float(target_mu_phys * k_scale)
    target_std_norm = float(target_std_phys * k_scale)
    exact_base_sens = float(inverse_softplus(target_mu_norm))

    return target_mu_norm, target_std_norm, exact_base_sens
