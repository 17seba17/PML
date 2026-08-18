import sys
import numpy as np
import xarray as xr


def data_preparation(x1, y1, x2, y2, output_path="vae_dataset.nc"):
    """description of this function...

    [TBD]
    """
    print("Executing data_preparation for VAE...")
    # database importing

    ds_tg = xr.open_dataset("../tg_ens_mean_0.1deg_reg_v33.0e.nc")
    ds_pp = xr.open_dataset("../pp_ens_mean_0.1deg_reg_v33.0e.nc")
    ds_hadcrut = xr.open_dataset(
        "../HadCRUT.5.1.0.0.noninfilled.anomalies.ensemble_mean.nc"
    )

    # building fGMT

    weights = np.cos(np.deg2rad(ds_hadcrut.latitude))
    gmt_monthly = (
        ds_hadcrut["tas_mean"]
        .weighted(weights)
        .mean(dim=("latitude", "longitude"))
    )
    fGMT = gmt_monthly.rolling(time=120, center=True, min_periods=12).mean()

    # local values

    lat_min, lat_max = min(y1, y2), max(y1, y2)
    lon_min, lon_max = min(x1, x2), max(x1, x2)

    if lat_min == lat_max and lon_min == lon_max:
        tg_local = ds_tg["tg"].sel(
            latitude=lat_min, longitude=lon_min, method="nearest"
        )
        pp_local = ds_pp["pp"].sel(
            latitude=lat_min, longitude=lon_min, method="nearest"
        )
    else:
        tg_local = ds_tg["tg"].sel(
            latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
        )
        pp_local = ds_pp["pp"].sel(
            latitude=slice(lat_min, lat_max), longitude=slice(lon_min, lon_max)
        )

    # fGMT monthly -> fGMT daily

    fgmt_daily = fGMT.interp(time=tg_local.time, method="linear")

    # deleting NAN values

    land_mask = tg_local.notnull().any(dim="time")


    valid_days = (
        fgmt_daily.notnull()
        & tg_local.where(land_mask)
        .notnull()
        .any(dim=[d for d in tg_local.dims if d != "time"])
    )

    tg_clean = tg_local.sel(time=valid_days)
    pp_clean = pp_local.sel(time=valid_days)
    fgmt_clean = fgmt_daily.sel(time=valid_days)

    coords_dict = {"time": tg_clean.time.values}
    if "latitude" in tg_clean.coords and "longitude" in tg_clean.coords:
        coords_dict["latitude"] = tg_clean.latitude.values
        coords_dict["longitude"] = tg_clean.longitude.values


    ds_out = xr.Dataset(
        data_vars={
            "tg": (tg_clean.dims, tg_clean.values),
            "pp": (pp_clean.dims, pp_clean.values),
            "fgmt": ("time", fgmt_clean.values),
        },
        coords=coords_dict,
    )

    if ds_out["tg"].size == 0 or ds_out.sizes["time"] == 0:        
        print("WARNING the dataset built has no point")

    # exporting dataset

    ds_out.to_netcdf(output_path)
    print("End execution data_preparation for VAE...")
    return ds_out


if __name__ == "__main__":
    if len(sys.argv) >= 5:
        x1, y1, x2, y2 = map(float, sys.argv[1:5])
        out_name = sys.argv[5] if len(sys.argv) > 5 else "vae_dataset.nc"
        data_preparation(x1, y1, x2, y2, output_path=out_name)