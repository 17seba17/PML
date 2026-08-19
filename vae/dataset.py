import gc
import sys

import xarray as xr


import numpy as np
import torch


def buildingTensors(file_path="vae_dataset.nc", split_year=2000):
    ds = xr.open_dataset(file_path)
    train_mask = ds.time.dt.year <= split_year
    test_mask = ds.time.dt.year > split_year

    ds_train = ds.isel(time=train_mask)
    ds_test = ds.isel(time=test_mask)

    del train_mask, test_mask
    gc.collect()

    N_train = ds_train.sizes["time"]
    N_test = ds_test.sizes["time"]

    # temperature

    tg_clim = ds_train["tg"].groupby("time.dayofyear").mean(dim="time")

    ## anomaly 

    tg_anom_train = (
        ds_train["tg"].groupby("time.dayofyear") - tg_clim
    ).values.astype(np.float32)
    tg_anom_test = (
        ds_test["tg"].groupby("time.dayofyear") - tg_clim
    ).values.astype(np.float32)

    tg_dim = tg_anom_train.shape[1]


    ## normalization 

    tg_std = tg_anom_train.std(axis=0, keepdims=True) + 1e-6
    tg_train_norm = tg_anom_train / tg_std
    tg_test_norm = tg_anom_test / tg_std

    ## building Y

    Y_train = torch.from_numpy(tg_train_norm)
    Y_test = torch.from_numpy(tg_test_norm)


    ## cleaning

    nan = np.isnan(tg_train_norm).sum()+np.isnan(tg_test_norm).sum()
    if nan>0:
        print(f"dataset is NOT valid")
        sys.exit(1)

    del tg_anom_train, tg_anom_test, tg_train_norm, tg_test_norm
    gc.collect()



    # pressure

    pp_clim = ds_train["pp"].groupby("time.dayofyear").mean(dim="time")


    ## anomaly

    pp_anom_train = (
        ds_train["pp"].groupby("time.dayofyear") - pp_clim
    ).values.astype(np.float32)
    pp_anom_test = (
        ds_test["pp"].groupby("time.dayofyear") - pp_clim
    ).values.astype(np.float32)

    pp_dim = pp_anom_train.shape[1]
    cond_dim = pp_dim + 1

    ## normalization

    pp_std = pp_anom_train.std(axis=0, keepdims=True) + 1e-6

    pp_train_norm = pp_anom_train / pp_std
    pp_test_norm = pp_anom_test / pp_std

    ## cleaning
    nan = np.isnan(pp_train_norm).sum()+np.isnan(pp_test_norm).sum()
    if nan>0:
        print(f"dataset is NOT valid")
        sys.exit(1)
    del pp_anom_train, pp_anom_test
    gc.collect()


    #  fGMT

    fgmt_train_raw = ds_train["fgmt"].values.reshape(N_train, 1).astype(np.float32)
    fgmt_test_raw = ds_test["fgmt"].values.reshape(N_test, 1).astype(np.float32)

    ## normalization

    fgmt_mean = fgmt_train_raw.mean(axis=0, keepdims=True)
    fgmt_std = fgmt_train_raw.std(axis=0, keepdims=True) + 1e-6

    fgmt_train_norm = (fgmt_train_raw - fgmt_mean) / fgmt_std
    fgmt_test_norm = (fgmt_test_raw - fgmt_mean) / fgmt_std

    ## cleaning
    nan = np.isnan(fgmt_train_norm).sum()+np.isnan(fgmt_test_norm).sum()
    if nan>0:
        print(f"dataset is NOT valid")
        sys.exit(1)
    del fgmt_train_raw, fgmt_test_raw
    gc.collect()

    # building X
    X_train_mat = np.concatenate([pp_train_norm, fgmt_train_norm], axis=1)
    X_test_mat = np.concatenate([pp_test_norm, fgmt_test_norm], axis=1)

    X_train = torch.from_numpy(X_train_mat)
    X_test = torch.from_numpy(X_test_mat)

    del (
        pp_train_norm,
        pp_test_norm,
        fgmt_train_norm,
        fgmt_test_norm,
        X_train_mat,
        X_test_mat,
    )
    gc.collect()

    # norm stats
    norm_stats = {
        "tg_std": tg_std.squeeze(),
        "pp_std": pp_std.squeeze(),
        "fgmt_mean": fgmt_mean.squeeze(),
        "fgmt_std": fgmt_std.squeeze(),
    }


    return (
        X_train,
        Y_train,
        X_test,
        Y_test,
        tg_dim,
        pp_dim,
        cond_dim,
        norm_stats,
        ds,
    )
