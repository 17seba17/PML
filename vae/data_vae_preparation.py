import gc
import sys
import numpy as np
import xarray as xr


def impute_and_mask_variable(arr_3d):
    T, H, W = arr_3d.shape
    is_nan = np.isnan(arr_3d)

    t_idx, h_idx, w_idx = np.where(is_nan)

    if len(t_idx) > 0:
        val_yesterday = np.full(len(t_idx), np.nan, dtype=np.float32)
        valid_t = t_idx > 0
        val_yesterday[valid_t] = arr_3d[
            t_idx[valid_t] - 1, h_idx[valid_t], w_idx[valid_t]
        ]

        val_north = np.full(len(t_idx), np.nan, dtype=np.float32)
        valid_h_north = h_idx > 0
        val_north[valid_h_north] = arr_3d[
            t_idx[valid_h_north], h_idx[valid_h_north] - 1, w_idx[valid_h_north]
        ]

        val_south = np.full(len(t_idx), np.nan, dtype=np.float32)
        valid_h_south = h_idx < H - 1
        val_south[valid_h_south] = arr_3d[
            t_idx[valid_h_south], h_idx[valid_h_south] + 1, w_idx[valid_h_south]
        ]

        val_west = np.full(len(t_idx), np.nan, dtype=np.float32)
        valid_w_west = w_idx > 0
        val_west[valid_w_west] = arr_3d[
            t_idx[valid_w_west], h_idx[valid_w_west], w_idx[valid_w_west] - 1
        ]

        val_east = np.full(len(t_idx), np.nan, dtype=np.float32)
        valid_w_east = w_idx < W - 1
        val_east[valid_w_east] = arr_3d[
            t_idx[valid_w_east], h_idx[valid_w_east], w_idx[valid_w_east] + 1
        ]

        neighbors = np.stack(
            [val_yesterday, val_north, val_south, val_west, val_east], axis=0
        )
        del val_yesterday, val_north, val_south, val_west, val_east

        valid_mask = ~np.isnan(neighbors)
        n_valid = valid_mask.sum(axis=0)
        sum_valid = np.where(valid_mask, neighbors, 0.0).sum(axis=0)
        del neighbors, valid_mask

        can_impute = n_valid > 0
        imputed = np.zeros(len(t_idx), dtype=np.float32)
        imputed[can_impute] = sum_valid[can_impute] / n_valid[can_impute]
        del n_valid, sum_valid

        arr_3d[
            t_idx[can_impute], h_idx[can_impute], w_idx[can_impute]
        ] = imputed[can_impute]
        del imputed, can_impute

    del t_idx, h_idx, w_idx, is_nan
    gc.collect()

    valid_spatial_mask = ~np.isnan(arr_3d).any(axis=0)
    return arr_3d, valid_spatial_mask


def data_preparation(x1, y1, x2, y2, output_path="vae_dataset.nc"):
    print("Executing data_preparation for VAE...")

    # opening first dataset
    ds_hadcrut = xr.open_dataset(
        "../HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"
    )

    # building GMT

    weights = np.cos(np.deg2rad(ds_hadcrut.latitude))

    gmt_monthly = (
        ds_hadcrut["tas_mean"]
        .weighted(weights)
        .mean(dim=("latitude", "longitude"))
    )
    fGMT = gmt_monthly.rolling(time=120, center=True, min_periods=12).mean()
    ds_hadcrut.close()
    del ds_hadcrut, weights, gmt_monthly
    gc.collect()

    lat_min, lat_max = min(y1, y2), max(y1, y2)
    lon_min, lon_max = min(x1, x2), max(x1, x2)

    # opening second dataset
    
    ds_tg = xr.open_dataset("../tg_ens_mean_0.1deg_reg_v33.0e.nc")

    # local temperature

    if ds_tg.latitude.values[0] > ds_tg.latitude.values[-1]:
        lat_slice = slice(lat_max, lat_min)
    else:
        lat_slice = slice(lat_min, lat_max)
    lon_slice = slice(lon_min, lon_max)

    is_single_point = lat_min == lat_max and lon_min == lon_max

    if is_single_point:
        tg_sub = ds_tg["tg"].sel(
            latitude=lat_min, longitude=lon_min, method="nearest"
        )
        tg_sub = tg_sub.expand_dims(
            dim={
                "latitude": [float(tg_sub.latitude)],
                "longitude": [float(tg_sub.longitude)],
            }
        )
    else:
        tg_sub = ds_tg["tg"].sel(latitude=lat_slice, longitude=lon_slice)

    ds_tg.close()
    del ds_tg
    gc.collect()

    # opening third dataset
    
    ds_pp = xr.open_dataset("../pp_ens_mean_0.1deg_reg_v33.0e.nc")

    # pressure

    if is_single_point:
        pp_sub = ds_pp["pp"].sel(
            latitude=lat_min, longitude=lon_min, method="nearest"
        )
        pp_sub = pp_sub.expand_dims(
            dim={
                "latitude": [float(pp_sub.latitude)],
                "longitude": [float(pp_sub.longitude)],
            }
        )
    else:
        pp_sub = ds_pp["pp"].sel(latitude=lat_slice, longitude=lon_slice)

    ds_pp.close()
    del ds_pp
    gc.collect()

    # GMT monthly -> GMT daily

    fgmt_daily = fGMT.interp(time=tg_sub.time, method="linear")
    valid_times = fgmt_daily.notnull().values
    del fGMT

    

    tg_sub = tg_sub.isel(time=valid_times)
    pp_sub = pp_sub.isel(time=valid_times)
    fgmt_clean = fgmt_daily.isel(time=valid_times)
    del fgmt_daily, valid_times
    gc.collect()

    # imputing
    
    print(f"Starting imputing...")

    T, H, W = tg_sub.shape
    lats_1d = tg_sub.latitude.values
    lons_1d = tg_sub.longitude.values
    times = tg_sub.time.values

    tg_arr = tg_sub.values.astype(np.float32)
    del tg_sub
    gc.collect()

    pp_arr = pp_sub.values.astype(np.float32)
    del pp_sub
    gc.collect()

    if H >= 3 and W >= 3:
        tg_arr, land_mask_tg = impute_and_mask_variable(tg_arr)
        pp_arr, land_mask_pp = impute_and_mask_variable(pp_arr)
        land_mask = land_mask_tg & land_mask_pp
        del land_mask_tg, land_mask_pp
    else:
        land_mask = (~np.isnan(tg_arr).any(axis=0)) & (~np.isnan(pp_arr).any(axis=0))

    # end immputing

    num_valid_points = int(np.sum(land_mask))
    print(f"Total cells: {H * W} | Valid cells: {num_valid_points}")

    if num_valid_points == 0:
        print("ERROR: The dataset built has no valid land points.")
        sys.exit(1)

    # getting coordinates

    lats_2d, lons_2d = np.meshgrid(lats_1d, lons_1d, indexing="ij")
    lats_clean = lats_2d[land_mask].astype(np.float32)
    lons_clean = lons_2d[land_mask].astype(np.float32)
    del lats_2d, lons_2d

    tg_clean = tg_arr[:, land_mask]
    del tg_arr
    gc.collect()

    pp_clean = pp_arr[:, land_mask]
    del pp_arr
    gc.collect()

    # building final dataset

    ds_out = xr.Dataset(
        data_vars={
            "tg": (("time", "point"), tg_clean),
            "pp": (("time", "point"), pp_clean),
            "fgmt": ("time", fgmt_clean.values.astype(np.float32)),
        },
        coords={
            "time": times,
            "point": np.arange(num_valid_points),
            "lat": ("point", lats_clean),
            "lon": ("point", lons_clean),
        },
        attrs={
            "orig_n_lat": H,
            "orig_n_lon": W,
            "lat_min": float(lat_min),
            "lat_max": float(lat_max),
            "lon_min": float(lon_min),
            "lon_max": float(lon_max),
            "lat_grid": lats_1d.tolist(),
            "lon_grid": lons_1d.tolist(),
        },
    )

    del tg_clean, pp_clean, fgmt_clean, lats_clean, lons_clean
    gc.collect()

    ds_out.to_netcdf(output_path)
    print(f"Dataset successfully saved to '{output_path}'.")
    return ds_out


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        x1, y1, x2, y2 = map(float, sys.argv[1:5])
        out_name = sys.argv[5] if len(sys.argv) > 5 else "vae_dataset.nc"
        data_preparation(x1, y1, x2, y2, output_path=out_name)