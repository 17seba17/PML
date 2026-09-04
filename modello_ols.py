import numpy as np
import pandas as pd
import statsmodels.api as sm

# -------------  ESTRAZIONE DI UNA LOCALITA'
def estrai_anomalia_locale(
    tg_anomalia,
    latitudine,
    longitudine
):

    serie = tg_anomalia.sel(
        latitude=latitudine,
        longitude=longitudine,
        method="nearest"
    )

    if np.isnan(serie.values).all():
        raise ValueError(
            "La cella selezionata non contiene dati validi."
        )

    print("\nCoordinate richieste:")

    print(
        "Latitudine:",
        latitudine,
        "Longitudine:",
        longitudine
    )

    print("Cella E-OBS utilizzata:")

    print(
        "Latitudine:",
        float(serie.latitude.values),
        "Longitudine:",
        float(serie.longitude.values)
    )

    return serie

# -------------  FUNZIONE GENERALE OLS

def valuta_ols(
    dati,
    nome_x,
    nome_y
):

    # Split cronologico 80% / 20%

    n = len(dati)
    split = int(n * 0.8)

    train = dati.iloc[:split]
    test = dati.iloc[split:]


    # TRAIN

    X_train = sm.add_constant(
        train[nome_x],
        has_constant="add"
    )

    y_train = train[nome_y]

    modello = sm.OLS(
        y_train,
        X_train
    ).fit()


    # TEST

    X_test = sm.add_constant(
        test[nome_x],
        has_constant="add"
    )

    y_test = test[nome_y]

    y_pred = modello.predict(
        X_test
    )


    # METRICHE

    errori = (
        y_test.values
        - y_pred.values
    )

    mae = np.mean(
        np.abs(errori)
    )

    rmse = np.sqrt(
        np.mean(errori ** 2)
    )

    r2_test = 1 - (
        np.sum(errori ** 2)
        /
        np.sum(
            (
                y_test.values
                - y_test.values.mean()
            ) ** 2
        )
    )


    return {

        "X": nome_x,
        "Y": nome_y,

        "alpha":
            modello.params["const"],

        "beta":
            modello.params[nome_x],

        "r2_train":
            modello.rsquared,

        "r2_test":
            r2_test,

        "mae":
            mae,

        "rmse":
            rmse
    }

# -------------  ANALISI DELLA LOCALITA' DI ESEMPIO

def analisi_localita(
    tg_anomalia,
    gmt_monthly,
    gmt_6m,
    gmt_12m,
    gmt_24m,
    latitudine,
    longitudine
):

    serie = estrai_anomalia_locale(
        tg_anomalia,
        latitudine,
        longitudine
    )


    # Smoothing locale

    local_6m = serie.rolling(
        time=6,
        center=True
    ).mean()

    local_12m = serie.rolling(
        time=12,
        center=True
    ).mean()

    local_24m = serie.rolling(
        time=24,
        center=True
    ).mean()


    # Dataset comune

    dataset = pd.DataFrame({

        "data":
            pd.to_datetime(
                gmt_monthly.time.values
            ),

        "gmt_monthly":
            gmt_monthly.values,

        "gmt_6m":
            gmt_6m.values,

        "gmt_12m":
            gmt_12m.values,

        "gmt_24m":
            gmt_24m.values,

        "local_monthly":
            serie.values,

        "local_6m":
            local_6m.values,

        "local_12m":
            local_12m.values,

        "local_24m":
            local_24m.values
    })


    # Stessi mesi per tutti i confronti

    dataset = dataset.dropna().copy()

    # -------------  ESPERIMENTO 1: risposta locale sempre mensile


    risultati_mensili = []

    for nome_x in [
        "gmt_monthly",
        "gmt_6m",
        "gmt_12m",
        "gmt_24m"
    ]:

        risultati_mensili.append(
            valuta_ols(
                dataset,
                nome_x,
                "local_monthly"
            )
        )

    tabella_mensili = pd.DataFrame(
        risultati_mensili
    )

    # -------------  ESPERIMENTO 2: stessa scala temporale X e Y

    coppie = [

        (
            "gmt_monthly",
            "local_monthly"
        ),

        (
            "gmt_6m",
            "local_6m"
        ),

        (
            "gmt_12m",
            "local_12m"
        ),

        (
            "gmt_24m",
            "local_24m"
        )
    ]


    risultati_smooth = []

    for nome_x, nome_y in coppie:

        risultati_smooth.append(
            valuta_ols(
                dataset,
                nome_x,
                nome_y
            )
        )


    tabella_smooth = pd.DataFrame(
        risultati_smooth
    )


    return (
        serie,
        dataset,
        tabella_mensili,
        tabella_smooth
    )

# -------------  OLS SPAZIALE

def calcola_mappe_ols(
    gmt_12m,
    tg_anomalia,
    soglia_copertura=0.8,
    soglia_r2=0.20
):

    # Smoothing locale a 12 mesi

    tg_anomalia_12m = tg_anomalia.rolling(
        time=12,
        center=True
    ).mean()


    # Mesi validi

    gmt_12m_valida = gmt_12m.dropna(
        dim="time"
    )

    tg_12m_valida = tg_anomalia_12m.sel(
        time=gmt_12m_valida.time
    )


    x = gmt_12m_valida
    y = tg_12m_valida


    # Osservazioni valide

    validi = (
        y.notnull()
        &
        x.notnull()
    )


    # Medie

    x_media = x.where(
        validi
    ).mean(
        dim="time",
        skipna=True
    )

    y_media = y.where(
        validi
    ).mean(
        dim="time",
        skipna=True
    )


    # Beta OLS

    numeratore = (
        (
            x - x_media
        )
        *
        (
            y - y_media
        )
    ).where(
        validi
    ).sum(
        dim="time",
        skipna=True
    )


    denominatore = (
        (
            x - x_media
        ) ** 2
    ).where(
        validi
    ).sum(
        dim="time",
        skipna=True
    )


    beta_map = (
        numeratore
        /
        denominatore
    )


    # Controllo copertura

    numero_osservazioni = validi.sum(
        dim="time"
    )

    min_osservazioni = int(
        soglia_copertura
        *
        gmt_12m_valida.sizes["time"]
    )

    beta_map = beta_map.where(
        numero_osservazioni
        >=
        min_osservazioni
    )


    # Alpha

    alpha_map = (
        y_media
        -
        beta_map * x_media
    )


    # Previsioni

    y_pred_map = (
        alpha_map
        +
        beta_map * x
    )


    # R-quadro

    residui = (
        y - y_pred_map
    ).where(
        validi
    )

    sse = (
        residui ** 2
    ).sum(
        dim="time",
        skipna=True
    )

    sst = (
        (
            y - y_media
        ) ** 2
    ).where(
        validi
    ).sum(
        dim="time",
        skipna=True
    )

    r2_map = (
        1
        -
        sse / sst
    )

    r2_map = r2_map.where(
        numero_osservazioni
        >=
        min_osservazioni
    )

    # -------------  STATISTICHE RIASSUNTIVE

    quantili_beta = beta_map.quantile(
        [
            0.01,
            0.05,
            0.50,
            0.95,
            0.99
        ],
        skipna=True
    ).values


    celle_valide = int(
        beta_map.notnull()
        .sum()
        .values
    )

    celle_r2 = int(
        (
            beta_map.notnull()
            &
            (r2_map >= soglia_r2)
        )
        .sum()
        .values
    )

    percentuale_r2 = (
        celle_r2
        /
        celle_valide
        *
        100
    )


    statistiche = {

        "beta_min":
            float(beta_map.min().values),

        "beta_max":
            float(beta_map.max().values),

        "beta_medio":
            float(beta_map.mean().values),

        "beta_quantili":
            quantili_beta,

        "r2_min":
            float(r2_map.min().values),

        "r2_max":
            float(r2_map.max().values),

        "r2_medio":
            float(r2_map.mean().values),

        "celle_valide":
            celle_valide,

        "celle_r2":
            celle_r2,

        "percentuale_r2":
            percentuale_r2,

        "min_osservazioni":
            min_osservazioni
    }


    return (
        beta_map,
        r2_map,
        gmt_12m_valida,
        tg_12m_valida,
        statistiche
    )
