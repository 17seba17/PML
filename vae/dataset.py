import gc
import sys

import xarray as xr


import numpy as np
import torch


def buildingTensors(file_path="vae_dataset.nc"):
    ds = xr.open_dataset(file_path)
    N_time = ds.sizes["time"]

    # temperature anomaly
    tg_climatology = ds["tg"].groupby("time.dayofyear").mean(dim="time")
    tg_anomalies = (ds["tg"].groupby("time.dayofyear") - tg_climatology)

    tg_raw = tg_anomalies.values.astype(np.float32)
    tg_dim = tg_raw.shape[1]

    tg_std = tg_raw.std(axis=0, keepdims=True) + 1e-6
    tg_raw /= tg_std




    # building Y

    nan_tg = np.isnan(tg_raw).sum()
    if nan_tg>0:
        print(f"dataset is NOT valid")


    Y = torch.from_numpy(tg_raw)
    del tg_raw, tg_anomalies
    gc.collect()

    # pressure anomaly

    pp_climatology = ds["pp"].groupby("time.dayofyear").mean(dim="time")
    pp_anomalies = (ds["pp"].groupby("time.dayofyear") - pp_climatology)

    pp_raw = pp_anomalies.values.astype(np.float32)
    pp_dim = pp_raw.shape[1]
    cond_dim = pp_dim + 1

    pp_std = pp_raw.std(axis=0, keepdims=True) + 1e-6
    pp_raw /= pp_std

    # normalization of fGMT
    fgmt_raw = ds["fgmt"].values.reshape(N_time, 1).astype(np.float32)
    fgmt_mean = fgmt_raw.mean(axis=0, keepdims=True)
    fgmt_std = fgmt_raw.std(axis=0, keepdims=True) + 1e-6
    fgmt_raw -= fgmt_mean
    fgmt_raw /= fgmt_std

    # building X
    X_mat = np.concatenate([pp_raw, fgmt_raw], axis=1)
    X = torch.from_numpy(X_mat)

    nan_pp = np.isnan(pp_raw).sum()
    nan_fgmt = np.isnan(fgmt_raw).sum()
    if nan_pp>0 or nan_fgmt>0:
        print(f"dataset is NOT valid")


    del pp_raw, fgmt_raw, X_mat
    gc.collect()

    # norm stats
    norm_stats = {
        "tg_std": tg_std.squeeze(),
        "pp_std": pp_std.squeeze(),
        "fgmt_mean": fgmt_mean.squeeze(),
        "fgmt_std": fgmt_std.squeeze(),
    }

    return X, Y, tg_dim, pp_dim, cond_dim, norm_stats, ds


def temporal_train_test_split(ds, split_year=2000):
    train_mask = (ds.time.dt.year <= split_year).values
    test_mask = (ds.time.dt.year > split_year).values

    return train_mask, test_mask
