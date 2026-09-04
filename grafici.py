import matplotlib.pyplot as plt

# -------------  GMTA E SMOOTHING

def grafico_smoothing(
    gmt_monthly,
    gmt_6m,
    gmt_12m,
    gmt_24m
):

    plt.figure(figsize=(12, 6))

    plt.plot(
        gmt_monthly.time,
        gmt_monthly,
        linewidth=0.5,
        label="Mensile"
    )

    plt.plot(
        gmt_6m.time,
        gmt_6m,
        linewidth=1.2,
        label="6 mesi"
    )

    plt.plot(
        gmt_12m.time,
        gmt_12m,
        linewidth=1.8,
        label="12 mesi"
    )

    plt.plot(
        gmt_24m.time,
        gmt_24m,
        linewidth=2,
        label="24 mesi"
    )

    plt.xlabel("Anno")
    plt.ylabel(
        "Anomalia globale di temperatura (K)"
    )

    plt.title(
        "Global Mean Temperature Anomaly"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# -------------  MAPPA BETA

def grafico_beta(
    beta_map
):

    plt.figure(figsize=(9, 10))

    beta_map.plot(
        x="longitude",
        y="latitude",
        robust=True,
        cbar_kwargs={
            "label":
                "β (°C locale / K globale)"
        }
    )

    plt.title(
        "OLS - Sensibilità locale alla GMTA (12 mesi)"
    )

    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")

    plt.tight_layout()
    plt.show()

# -------------  MAPPA R²

def grafico_r2(
    r2_map
):

    plt.figure(figsize=(9, 10))

    r2_map.plot(
        x="longitude",
        y="latitude",
        cbar_kwargs={
            "label": "R²"
        }
    )

    plt.title(
        "OLS - R² del modello GMTA 12 mesi"
    )

    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")

    plt.tight_layout()
    plt.show()

# -------------  MAPPA BETA CON SOGLIA R²

def grafico_beta_affidabile(
    beta_map,
    r2_map,
    soglia=0.20
):

    beta_affidabile = beta_map.where(
        r2_map >= soglia
    )

    plt.figure(figsize=(9, 10))

    beta_affidabile.plot(
        x="longitude",
        y="latitude",
        robust=True,
        cbar_kwargs={
            "label":
                "β (°C locale / K globale)"
        }
    )

    plt.title(
        "OLS - Sensibilità locale alla GMTA\n"
        f"celle con R² >= {soglia}"
    )

    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")

    plt.tight_layout()
    plt.show()

# -------------  GMTA FACTUAL VS COUNTERFACTUAL

def grafico_gmta_counterfactual(
    gmt_factual,
    gmt_counterfactual
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        gmt_factual.time,
        gmt_factual,
        label="GMTA factual - 12 mesi"
    )

    plt.plot(
        gmt_counterfactual.time,
        gmt_counterfactual,
        linewidth=2,
        label="GMTA counterfactual"
    )

    plt.xlabel("Anno")

    plt.ylabel(
        "Anomalia globale di temperatura (K)"
    )

    plt.title(
        "GMTA factual e scenario counterfactual"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# -------------  MAPPA IMPATTO

def grafico_impatto(
    impatto
):

    plt.figure(figsize=(9, 10))

    impatto.plot(
        x="longitude",
        y="latitude",
        robust=True,
        cbar_kwargs={
            "label":
                "Factual - Counterfactual (°C)"
        }
    )

    plt.title(
        "OLS - Differenza media factual-counterfactual\n"
        "2015-2024"
    )

    plt.xlabel("Longitudine")
    plt.ylabel("Latitudine")

    plt.tight_layout()
    plt.show()

# -------------  LOCALITA': FACTUAL VS COUNTERFACTUAL

def grafico_localita_counterfactual(
    factual,
    counterfactual,
    nome_localita="Roma"
):

    plt.figure(figsize=(12, 5))

    plt.plot(
        factual.time,
        factual,
        label=f"{nome_localita} factual"
    )

    plt.plot(
        counterfactual.time,
        counterfactual,
        label=f"{nome_localita} counterfactual"
    )

    plt.axhline(
        0,
        linewidth=0.8
    )

    plt.xlabel("Anno")

    plt.ylabel(
        "Anomalia temperatura locale (°C)"
    )

    plt.title(
        f"{nome_localita} - clima factual vs counterfactual"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
