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
    lon_min=5.0,
    lon_max=10.0,
    split_year=2005,
):
    if not os.path.exists(hadcrut_path):
        if os.path.exists("HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"):
            hadcrut_path = "HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"
        else:
            raise FileNotFoundError(f"File HadCRUT non trovato in: {hadcrut_path}")

    ds = xr.open_dataset(hadcrut_path)
    ds_train = ds.isel(time=ds.time.dt.year <= split_year)

    # 1. fGMT globale (pesato per il coseno e smussato a 60 mesi)
    cos_lat = np.cos(np.deg2rad(ds_train.latitude))
    gmt_raw = ds_train["tas_mean"].weighted(cos_lat).mean(dim=("latitude", "longitude"))
    fgmt_series = gmt_raw.to_series().ewm(span=60, adjust=False).mean().values

    # Funzione interna per estrarre le celle valide dato un Bounding Box
    def extract_box(l_min, l_max, ln_min, ln_max):
        if ds_train.latitude.values[0] > ds_train.latitude.values[-1]:
            lat_slice = slice(l_max, l_min)
        else:
            lat_slice = slice(l_min, l_max)
        lon_slice = slice(ln_min, ln_max)

        sub_region = ds_train["tas_mean"].sel(latitude=lat_slice, longitude=lon_slice)
        tas_vals = sub_region.values  # [time, lat, lon]
        T = tas_vals.shape[0]
        tas_2d = tas_vals.reshape(T, -1)

        alphas_box = []
        stds_box = []

        for col in range(tas_2d.shape[1]):
            b_cell = tas_2d[:, col]
            valid = ~np.isnan(b_cell) & ~np.isnan(fgmt_series)

            if np.sum(valid) > 360:
                b_c = b_cell[valid] - np.mean(b_cell[valid])
                g_c = fgmt_series[valid] - np.mean(fgmt_series[valid])

                cov_i = np.mean(b_c * g_c)
                var_g_i = np.var(g_c)
                s_i = cov_i / (var_g_i + 1e-8)
                alphas_box.append(s_i)

                res = b_c - s_i * g_c
                se_i = float(np.std(res) / (np.std(g_c) + 1e-8) / np.sqrt(len(g_c)))
                stds_box.append(se_i)

        return np.array(alphas_box), np.array(stds_box)

    # 2. Primo tentativo con il Bounding Box esatto
    alphas, stds_residui = extract_box(lat_min, lat_max, lon_min, lon_max)

    # 3. Fallback: Media dei vicini (Allargamento dinamico)
    expansion_step = 5.0  # Gradi (pari a 1 cella HadCRUT)
    expansion_level = 1
    
    while len(alphas) == 0 and expansion_level <= 3:
        print(f" [!] Box [{lat_min:.1f}, {lon_min:.1f}] vuoto in HadCRUT. Estrapolo la media dai vicini (+{expansion_step * expansion_level}°)...")
        alphas, stds_residui = extract_box(
            lat_min - expansion_step * expansion_level,
            lat_max + expansion_step * expansion_level,
            lon_min - expansion_step * expansion_level,
            lon_max + expansion_step * expansion_level
        )
        expansion_level += 1

    ds.close()

    # Se fallisce anche prendendo tutta l'Europa (praticamente impossibile su terra)
    if len(alphas) == 0:
        target_mu_phys = 1.05
        target_std_phys = 0.20
    else:
        target_mu_phys = float(np.mean(alphas))
        if len(alphas) > 1 and np.std(alphas) > 0.05:
            target_std_phys = float(np.std(alphas))
        else:
            target_std_phys = float(np.mean(stds_residui))

    print("=" * 60)
    print(f"HADCRUT INVARIANT PRIOR (1850 - {split_year})")
    print(f"Bounding Box       : Lat [{lat_min}, {lat_max}] | Lon [{lon_min}, {lon_max}]")
    print(f"Celle valide       : {len(alphas)}")
    print(f"Target Mu Fisico   : {target_mu_phys:.4f} °C/°C")
    print(f"Target Std Fisico  : {target_std_phys:.4f} °C/°C")
    print("=" * 60)

    return target_mu_phys, target_std_phys


def scale(
    norm_stats,
    hadcrut_path="../HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc",
    lat_min=45.0,
    lat_max=50.0,
    lon_min=5.0,
    lon_max=10.0,
    split_year=2005,
    target_mu_phys=None,
    target_std_phys=None,
):
    if target_mu_phys is None or target_std_phys is None:
        target_mu_phys, target_std_phys = compute_hadcrut_invariant_moments(
            hadcrut_path=hadcrut_path,
            lat_min=lat_min,
            lat_max=lat_max,
            lon_min=lon_min,
            lon_max=lon_max,
            split_year=split_year,
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
