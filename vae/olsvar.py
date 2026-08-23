import os
import numpy as np
import matplotlib.pyplot as plt
import torch

from dataset import buildingTensors

def to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return np.array(tensor)

def build_2d_matrix(values, lat_indices, lon_indices, n_lat, n_lon, invert_lat=False):
    """Mappa il vettore 1D dei punti validi sulla griglia 2D (n_lat, n_lon)."""
    matrix = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    matrix[lat_indices, lon_indices] = values
    if invert_lat:
        matrix = matrix[::-1, :]
    return matrix

def main():
    print("=" * 80)
    print("       PASSO 0: BASELINE OLS & MAPPA DELLA VARIANZA TEMPERATURA")
    print("=" * 80)

    # 1. Caricamento Dati
    print("Caricamento dataset...")
    (
        X_train,
        Y_train,
        X_test,
        Y_test,
        tg_dim,
        pp_dim,
        cond_dim,
        norm_stats,
        ds,
    ) = buildingTensors("vae_dataset.nc", split_year=2005)

    # Conversione in NumPy
    X_train_np = to_numpy(X_train)
    Y_train_np = to_numpy(Y_train)
    X_test_np = to_numpy(X_test)
    Y_test_np = to_numpy(Y_test)

    tg_std = to_numpy(norm_stats["tg_std"])
    fgmt_std = float(norm_stats["fgmt_std"])
    fgmt_mean = float(norm_stats["fgmt_mean"])
    pp_std = to_numpy(norm_stats["pp_std"])

    # 2. Ricostruzione delle unità fisiche reali (°C / hPa)
    print("Denormalizzazione dei dati alle unità fisiche reali (°C)...")
    Y_train_phys = Y_train_np * tg_std
    Y_test_phys = Y_test_np * tg_std

    # FGMT in °C effettivi (anomalia globale rispetto al pre-industriale)
    fgmt_train_phys = X_train_np[:, -1] * fgmt_std + fgmt_mean
    fgmt_test_phys = X_test_np[:, -1] * fgmt_std + fgmt_mean

    # Circolazione (PP)
    pp_train_phys = X_train_np[:, :-1] * pp_std
    pp_test_phys = X_test_np[:, :-1] * pp_std

    n_points = Y_train_phys.shape[1]
    n_train_samples = Y_train_phys.shape[0]

    # Warming medio globale nel periodo di test (es. > 2005) rispetto al pre-industriale (0.0 °C)
    delta_fgmt_test = float(np.mean(fgmt_test_phys)) - 0.0
    print(f"Delta FGMT medio periodo di Test (Factual - CF 0°C): {delta_fgmt_test:.3f} °C\n")

    # 3. Regressione OLS punto per punto
    print(f"Calcolo regressione OLS su {n_points} punti griglia...")
    ols_alpha = np.zeros(n_points, dtype=np.float32)       # Sensibilità dTG / dFGMT [°C/°C]
    ols_impact = np.zeros(n_points, dtype=np.float32)      # Impatto climatico test [°C]
    tg_variance = np.zeros(n_points, dtype=np.float32)     # Varianza totale [°C^2]
    tg_std_phys = np.zeros(n_points, dtype=np.float32)     # Deviazione standard [°C]
    ols_r2 = np.zeros(n_points, dtype=np.float32)          # R^2 del fit OLS

    for i in range(n_points):
        y_i = Y_train_phys[:, i]
        fgmt_i = fgmt_train_phys
        pp_i = pp_train_phys[:, i]

        # Varianza e Std totale della temperatura (Train set)
        tg_variance[i] = np.var(y_i)
        tg_std_phys[i] = np.std(y_i)

        # Matrice di design: [FGMT, PP, Costante/Intercetta]
        A = np.column_stack([fgmt_i, pp_i, np.ones(n_train_samples)])
        
        # Fit OLS: theta = [alpha, beta, intercept]
        theta, residuals, rank, s = np.linalg.lstsq(A, y_i, rcond=None)
        
        alpha_i = theta[0]
        ols_alpha[i] = alpha_i
        ols_impact[i] = alpha_i * delta_fgmt_test

        # Calcolo R^2
        y_pred = A @ theta
        ss_tot = np.sum((y_i - np.mean(y_i)) ** 2)
        ss_res = np.sum((y_i - y_pred) ** 2)
        ols_r2[i] = 1.0 - (ss_res / (ss_tot + 1e-8))

    # 4. Coordinate e Griglia Geografica
    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])
    n_lat = len(lat_grid)
    n_lon = len(lon_grid)
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    invert_lat = lat_grid[0] > lat_grid[-1]
    plot_lat_grid = lat_grid[::-1] if invert_lat else lat_grid

    # 5. Analisi comparativa Ovest vs Est
    mask_west = lons < 12.0
    mask_east = lons > 18.0

    print("-" * 80)
    print("CONFRONTO STATISTICO OVEST (Lon < 12°E) vs EST (Lon > 18°E)")
    print("-" * 80)
    print(f"Numero punti Ovest : {np.sum(mask_west):<5} | Numero punti Est : {np.sum(mask_east):<5}")
    print(f"OLS Impatto Ovest  : {np.mean(ols_impact[mask_west]):.2f} ± {np.std(ols_impact[mask_west]):.2f} °C")
    print(f"OLS Impatto Est    : {np.mean(ols_impact[mask_east]):.2f} ± {np.std(ols_impact[mask_east]):.2f} °C")
    print(f"Std Dev (TG) Ovest : {np.mean(tg_std_phys[mask_west]):.2f} °C")
    print(f"Std Dev (TG) Est   : {np.mean(tg_std_phys[mask_east]):.2f} °C")
    print(f"OLS R² medio Ovest : {np.mean(ols_r2[mask_west]):.2f}")
    print(f"OLS R² medio Est   : {np.mean(ols_r2[mask_east]):.2f}")
    print("-" * 80 + "\n")

    # 6. Costruzione Matrici 2D
    matrix_impact = build_2d_matrix(ols_impact, lat_indices, lon_indices, n_lat, n_lon, invert_lat)
    matrix_std = build_2d_matrix(tg_std_phys, lat_indices, lon_indices, n_lat, n_lon, invert_lat)
    matrix_r2 = build_2d_matrix(ols_r2, lat_indices, lon_indices, n_lat, n_lon, invert_lat)

    extent = [lon_grid.min(), lon_grid.max(), plot_lat_grid.min(), plot_lat_grid.max()]

    cmap_impact = plt.cm.Reds.copy()
    cmap_impact.set_bad(color="lightgray")

    cmap_var = plt.cm.viridis.copy()
    cmap_var.set_bad(color="lightgray")

    # 7. GENERAZIONE IMMAGINE 1: OLS Climate Impact Map
    plt.figure(figsize=(10, 8))
    im1 = plt.imshow(
        matrix_impact,
        origin="lower",
        extent=extent,
        cmap=cmap_impact,
        vmin=0.0,
        vmax=max(float(np.nanpercentile(ols_impact, 98)), 1.5),
    )
    cbar1 = plt.colorbar(im1, fraction=0.046, pad=0.04)
    cbar1.set_label("OLS Climate Impact: $T_{factual} - T_{counterfactual}$ [°C]", size=12)
    plt.title("Baseline Climate Impact Map (Linear OLS Regression)", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("ols_climate_impact.png", dpi=300)
    plt.close()
    print(" Salvata: 'ols_climate_impact.png'")

    # 8. GENERAZIONE IMMAGINE 2: Temperature Variability (Std Dev) Map
    plt.figure(figsize=(10, 8))
    im2 = plt.imshow(
        matrix_std,
        origin="lower",
        extent=extent,
        cmap=cmap_var,
        vmin=float(np.nanpercentile(tg_std_phys, 2)),
        vmax=float(np.nanpercentile(tg_std_phys, 98)),
    )
    cbar2 = plt.colorbar(im2, fraction=0.046, pad=0.04)
    cbar2.set_label("Temperature Standard Deviation $\\sigma(T)$ [°C]", size=12)
    plt.title("Local Temperature Natural Variability (Std Dev)", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("temperature_variance.png", dpi=300)
    plt.close()
    print(" Salvata: 'temperature_variance.png'")

    # 9. GENERAZIONE PANNELLO RIASSUNTIVO (Side-by-side)
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    
    # Subplot 1: OLS Impact
    im_a = axes[0].imshow(matrix_impact, origin="lower", extent=extent, cmap=cmap_impact, vmin=0.0, vmax=2.0)
    axes[0].set_title("(A) Baseline OLS Climate Impact [°C]", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Longitude", fontsize=11)
    axes[0].set_ylabel("Latitude", fontsize=11)
    cbar_a = fig.colorbar(im_a, ax=axes[0], fraction=0.046, pad=0.04)
    cbar_a.set_label("Impact [°C]")

    # Subplot 2: Std Dev
    im_b = axes[1].imshow(matrix_std, origin="lower", extent=extent, cmap=cmap_var)
    axes[1].set_title("(B) Temperature Std Dev $\\sigma(T)$ [°C]", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Longitude", fontsize=11)
    axes[1].set_ylabel("Latitude", fontsize=11)
    cbar_b = fig.colorbar(im_b, ax=axes[1], fraction=0.046, pad=0.04)
    cbar_b.set_label("Std Dev [°C]")

    plt.suptitle("Step 0 Diagnostic: Linear Attribution Benchmark & Variance Field", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("step0_summary.png", dpi=300)
    plt.close()
    print(" Salvata: 'step0_summary.png'")
    print("=" * 80)

if __name__ == "__main__":
    main()
