import xarray as xr

# -------------  COSTRUZIONE DELLA GMTA CONTROFATTUALE

def costruisci_gmta_counterfactual(
    gmt_factual,
    inizio_riferimento="1951-01-01",
    fine_riferimento="1960-12-01"
):

    periodo_riferimento = gmt_factual.sel(
        time=slice(
            inizio_riferimento,
            fine_riferimento
        )
    )

    gmt_riferimento = float(
        periodo_riferimento
        .mean()
        .values
    )

    gmt_counterfactual = xr.full_like(
        gmt_factual,
        gmt_riferimento
    )

    gmt_counterfactual.name = (
        "gmt_counterfactual"
    )

    return (
        gmt_riferimento,
        gmt_counterfactual
    )

# -------------  CALCOLO DELL'IMPATTO

def calcola_impatto(
    beta_map,
    gmt_factual,
    gmt_counterfactual,
    tg_factual
):

    delta_gmt = (
        gmt_factual
        -
        gmt_counterfactual
    )

    impatto_tempo = (
        beta_map
        *
        delta_gmt
    )

    impatto_tempo = impatto_tempo.transpose(
        "time",
        "latitude",
        "longitude"
    )

    impatto_tempo.name = (
        "climate_impact"
    )


    # Temperatura counterfactual locale

    tg_counterfactual = (
        tg_factual
        -
        impatto_tempo
    )

    tg_counterfactual.name = (
        "counterfactual_temperature_anomaly"
    )


    return (
        impatto_tempo,
        tg_counterfactual
    )

# -------------  INDICE DI IMPATTO SU UN PERIODO

def indice_impatto_periodo(
    impatto,
    data_inizio,
    data_fine
):

    periodo = impatto.sel(
        time=slice(
            data_inizio,
            data_fine
        )
    )

    if periodo.sizes["time"] == 0:
        raise ValueError(
            "Il periodo selezionato non contiene dati."
        )

    return periodo.mean(
        dim="time",
        skipna=True
    )

# -------------  STATISTICHE DELL'IMPATTO

def statistiche_impatto(
    impatto
):

    quantili = impatto.quantile(
        [
            0.05,
            0.50,
            0.95
        ],
        dim=(
            "latitude",
            "longitude"
        ),
        skipna=True
    ).values


    return {

        "medio":
            float(
                impatto.mean(
                    skipna=True
                ).values
            ),

        "minimo":
            float(
                impatto.min(
                    skipna=True
                ).values
            ),

        "massimo":
            float(
                impatto.max(
                    skipna=True
                ).values
            ),

        "quantili":
            quantili
    }
