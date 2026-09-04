import xarray as xr
import numpy as np
import pandas as pd


def prepara_dati(
    file_anomalies="anomalies.nc",
    file_tg="tg.nc"
):
    # -------------  CARICAMENTO DATASET

    anomalies = xr.open_dataset(file_anomalies)
    tg = xr.open_dataset(file_tg)

    # -------------  PERIODO COMUNE 1950-2025

    anomalies_common = anomalies.sel(
        time=slice("1950-01-01", "2025-12-31")
    )

    tas = anomalies_common["tas_mean"]

    # -------------  CALCOLO GMTA - EMISFERO NORD

    tas_north = tas.where(
        anomalies_common.latitude > 0,
        drop=True
    )

    weights_north = np.cos(
        np.deg2rad(tas_north.latitude)
    )

    gmt_north = tas_north.weighted(
        weights_north
    ).mean(
        dim=("latitude", "longitude")
    )

    # -------------  CALCOLO GMTA - EMISFERO SUD

    tas_south = tas.where(
        anomalies_common.latitude < 0,
        drop=True
    )

    weights_south = np.cos(
        np.deg2rad(tas_south.latitude)
    )

    gmt_south = tas_south.weighted(
        weights_south
    ).mean(
        dim=("latitude", "longitude")
    )


    # -------------  Media globale: 50% Nord + 50% Sud

    gmt_monthly = (
        0.5 * gmt_north
        + 0.5 * gmt_south
    )

    # -------------  ALLINEAMENTO DATE

    gmt_monthly = gmt_monthly.assign_coords(
        time=pd.to_datetime(
            gmt_monthly.time.values
        ).to_period("M").to_timestamp()
    )

    # -------------  SMOOTHING GMTA

    gmt_6m = gmt_monthly.rolling(
        time=6,
        center=True
    ).mean()

    gmt_12m = gmt_monthly.rolling(
        time=12,
        center=True
    ).mean()

    gmt_24m = gmt_monthly.rolling(
        time=24,
        center=True
    ).mean()

    # -------------  DOMINIO GEOGRAFICO CHE CONTIENE L'ITALIA

    italy = tg.sel(
        latitude=slice(36, 47.5),
        longitude=slice(6, 19)
    )

    # -------------  E-OBS: DA GIORNALIERO A MENSILE

    tg_mensile = italy["tg"].resample(
        time="MS"
    ).mean()

    # -------------  CLIMATOLOGIA 1961-1990

    baseline_locale = tg_mensile.sel(
        time=slice("1961-01-01", "1990-12-31")
    )

    climatologia_mensile = (
        baseline_locale
        .groupby("time.month")
        .mean("time")
    )

    # -------------  ANOMALIE LOCALI

    tg_anomalia = (
        tg_mensile.groupby("time.month")
        - climatologia_mensile
    )

    # -------------  CONTROLLI

    anomalia_baseline = tg_anomalia.sel(
        time=slice("1961-01-01", "1990-12-31")
    )

    media_baseline = (
        anomalia_baseline.mean().values
    )

    stesse_date = np.array_equal(
        pd.to_datetime(gmt_monthly.time.values),
        pd.to_datetime(tg_anomalia.time.values)
    )

    print("\n--- CONTROLLO DATI ---")

    print(
        "Numero mesi GMTA:",
        gmt_monthly.sizes["time"]
    )

    print(
        "Numero mesi E-OBS:",
        tg_mensile.sizes["time"]
    )

    print(
        "Media anomalie 1961-1990:",
        media_baseline
    )

    print(
        "GMTA ed E-OBS hanno gli stessi mesi?",
        stesse_date
    )


    # -------------  Ciò che servirà agli altri file

    return (
        gmt_monthly,
        gmt_6m,
        gmt_12m,
        gmt_24m,
        tg_anomalia
    )
