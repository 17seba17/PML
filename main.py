from preprocessing import prepara_dati

from modello_ols import (
    analisi_localita,
    calcola_mappe_ols
)

from counterfactual import (
    costruisci_gmta_counterfactual,
    calcola_impatto,
    indice_impatto_periodo,
    statistiche_impatto
)

from grafici import (
    grafico_smoothing,
    grafico_beta,
    grafico_r2,
    grafico_beta_affidabile,
    grafico_gmta_counterfactual,
    grafico_impatto,
    grafico_localita_counterfactual
)
# -------------  PREPARAZIONE DEI DATI

(
    gmt_monthly,
    gmt_6m,
    gmt_12m,
    gmt_24m,
    tg_anomalia
) = prepara_dati()

# -------------  CONFRONTO DELLO SMOOTHING
grafico_smoothing(
    gmt_monthly,
    gmt_6m,
    gmt_12m,
    gmt_24m
)

# -------------  TEST DEL MODELLO SU ROMA

(
    roma,
    dataset_roma,
    tabella_mensili,
    tabella_smooth
) = analisi_localita(
    tg_anomalia,
    gmt_monthly,
    gmt_6m,
    gmt_12m,
    gmt_24m,
    latitudine=41.90,
    longitudine=12.50
)


print(
    "\n--- OLS: RISPOSTA LOCALE MENSILE ---"
)

print(
    tabella_mensili
)


print(
    "\n--- OLS: SMOOTHING ABBINATO ---"
)

print(
    tabella_smooth
)


tabella_mensili.to_csv(
    "risultati_ols_roma_mensile.csv",
    index=False
)

tabella_smooth.to_csv(
    "risultati_ols_roma_smoothing.csv",
    index=False
)

#  -------------  OLS SPAZIALE A 12 MESI

(
    beta_map,
    r2_map,
    gmt_12m_valida,
    tg_12m_valida,
    statistiche
) = calcola_mappe_ols(
    gmt_12m,
    tg_anomalia
)


print(
    "\n--- RISULTATI SPAZIALI OLS ---"
)

print(
    "Beta minimo:",
    statistiche["beta_min"]
)

print(
    "Beta massimo:",
    statistiche["beta_max"]
)

print(
    "Beta medio:",
    statistiche["beta_medio"]
)

print(
    "Quantili beta:",
    statistiche["beta_quantili"]
)

print(
    "R² medio:",
    statistiche["r2_medio"]
)

print(
    "Percentuale celle con R² >= 0.20:",
    statistiche["percentuale_r2"]
)

#  -------------  Controllo della cella di Roma

beta_roma = beta_map.sel(
    latitude=41.90,
    longitude=12.50,
    method="nearest"
)

print(
    "Beta Roma:",
    beta_roma.values
)

# -------------  MAPPE OLS

grafico_beta(
    beta_map
)

grafico_r2(
    r2_map
)

grafico_beta_affidabile(
    beta_map,
    r2_map,
    soglia=0.20
)


# -------------  Salvataggio mappe

beta_map.name = "beta"
r2_map.name = "r2"

beta_map.to_netcdf(
    "beta_map_ols_12m.nc"
)

r2_map.to_netcdf(
    "r2_map_ols_12m.nc"
)

# -------------  SCENARIO COUNTERFACTUAL

(
    gmt_riferimento,
    gmt_counterfactual
) = costruisci_gmta_counterfactual(
    gmt_12m_valida
)


print(
    "\nGMTA media di riferimento 1951-1960:",
    gmt_riferimento
)


grafico_gmta_counterfactual(
    gmt_12m_valida,
    gmt_counterfactual
)

# -------------  IMPATTO FACTUAL - COUNTERFACTUAL

(
    impatto_tempo,
    tg_counterfactual_12m
) = calcola_impatto(
    beta_map,
    gmt_12m_valida,
    gmt_counterfactual,
    tg_12m_valida
)


# -------------  Impatto medio 2015-2024

impatto_2015_2024 = indice_impatto_periodo(
    impatto_tempo,
    "2015-01-01",
    "2024-12-01"
)

impatto_2015_2024.name = (
    "mean_climate_impact_2015_2024"
)


stat_impatto = statistiche_impatto(
    impatto_2015_2024
)


print(
    "\n--- IMPATTO MEDIO 2015-2024 ---"
)

print(
    "Impatto medio:",
    stat_impatto["medio"]
)

print(
    "Impatto minimo:",
    stat_impatto["minimo"]
)

print(
    "Impatto massimo:",
    stat_impatto["massimo"]
)

print(
    "Quantili 5%, 50%, 95%:",
    stat_impatto["quantili"]
)


grafico_impatto(
    impatto_2015_2024
)

# -------------  ROMA: FACTUAL VS COUNTERFACTUAL

roma_factual_12m = tg_12m_valida.sel(
    latitude=41.90,
    longitude=12.50,
    method="nearest"
)

roma_counterfactual_12m = (
    tg_counterfactual_12m.sel(
        latitude=41.90,
        longitude=12.50,
        method="nearest"
    )
)


grafico_localita_counterfactual(
    roma_factual_12m,
    roma_counterfactual_12m,
    nome_localita="Roma"
)


impatto_roma = impatto_2015_2024.sel(
    latitude=41.90,
    longitude=12.50,
    method="nearest"
)


print(
    "\nImpatto medio Roma 2015-2024:",
    impatto_roma.values,
    "°C"
)

# -------------  SALVATAGGIO RISULTATI FINALI

gmt_counterfactual.to_netcdf(
    "gmt_counterfactual_12m.nc"
)

impatto_tempo.to_netcdf(
    "impact_ols_12m.nc"
)

tg_counterfactual_12m.to_netcdf(
    "temperature_counterfactual_ols_12m.nc"
)

impatto_2015_2024.to_netcdf(
    "impact_index_ols_2015_2024.nc"
)


print(
    "\n=============================="
)

print(
    "ANALISI OLS COMPLETATA"
)

print(
    "=============================="
)
