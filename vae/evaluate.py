import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_counterfactuals(model, X_test, Y_test, norm_stats, device="cpu"):
    model = model.to(device)
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pp_test = X_test[:, :model.pp_dim]
    fgmt_test = X_test[:, model.pp_dim:]

    with torch.no_grad():
        sensitivity = model.decoder.get_sensitivity(pp_test)
        y_dyn_test = Y_test - sensitivity * fgmt_test
        
        mu_z, _ = model.encoder(y_dyn_test, pp_test)
        z = mu_z

        mu_fact, logvar_fact, _ = model.decoder(z, pp_test, fgmt_test)
        
        fgmt_zero = torch.full_like(
                fgmt_test,
                float((0.0 - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"])
        )
        mu_cf, _, _ = model.decoder(z, pp_test, fgmt_zero)

    # Denormalization
    y_factual = mu_fact.cpu().numpy() * norm_stats["tg_std"]
    y_counterfactual = mu_cf.cpu().numpy() * norm_stats["tg_std"]
    y_true = Y_test.cpu().numpy() * norm_stats["tg_std"]

    climate_impact = y_factual - y_counterfactual
    impact_mean = climate_impact.mean(axis=0)
    sigma_fact = torch.exp(0.5 * logvar_fact).cpu().numpy() * norm_stats["tg_std"]
    sigma_mean = sigma_fact.mean(axis=0)

    return y_factual, y_counterfactual, y_true, impact_mean, sigma_mean

def aggregate_and_plot_all_methods(point_predictions, ds, radius=50.0):
    n_points = len(point_predictions)
    eps = 1e-4

    methods = {
        "1_IDW_ExpSigma": np.full(n_points, np.nan, dtype=np.float32),
        "2_IDW_ExpMinusSigma": np.full(n_points, np.nan, dtype=np.float32),
        "3_Standard_IDW": np.full(n_points, np.nan, dtype=np.float32),
        "4_Nearest_Model": np.full(n_points, np.nan, dtype=np.float32),
        "5_Min_Sigma": np.full(n_points, np.nan, dtype=np.float32),
        "6_Gaussian_RBF_Bayes": np.full(n_points, np.nan, dtype=np.float32),
        "7_Ensemble_Mean": np.full(n_points, np.nan, dtype=np.float32),
    }

    # Calcolo per ciascun punto
    for p in range(n_points):
        preds = point_predictions[p]
        if not preds:
            continue

        t_arr = np.array([item["t"] for item in preds])
        sigma_arr = np.array([item["sigma"] for item in preds])
        d_arr = np.array([item["dist"] for item in preds])

        # Metodo 1: d_i^-1 * exp(sigma_i)
        w1 = np.exp(sigma_arr) / (d_arr + eps)
        methods["1_IDW_ExpSigma"][p] = np.sum(w1 * t_arr) / np.sum(w1)

        # Metodo 2: d_i^-1 * exp(-sigma_i) (Pesato sull'affidabilità)
        w2 = np.exp(-sigma_arr) / (d_arr + eps)
        methods["2_IDW_ExpMinusSigma"][p] = np.sum(w2 * t_arr) / np.sum(w2)

        # Metodo 3: IDW classico (d_i^-1)
        w3 = 1.0 / (d_arr + eps)
        methods["3_Standard_IDW"][p] = np.sum(w3 * t_arr) / np.sum(w3)

        # Metodo 4: Modello più vicino (argmin d_i)
        methods["4_Nearest_Model"][p] = t_arr[np.argmin(d_arr)]

        # Metodo 5: Modello più sicuro (argmin sigma_i)
        methods["5_Min_Sigma"][p] = t_arr[np.argmin(sigma_arr)]

        # Metodo 6: Gaussian RBF + Precisione
        rbf_sigma = max(radius / 2.0, 1.0)
        w6 = np.exp(-0.5 * (d_arr / rbf_sigma) ** 2) * np.exp(-sigma_arr)
        methods["6_Gaussian_RBF_Bayes"][p] = np.sum(w6 * t_arr) / np.sum(w6)

        # Metodo 7: Media semplice uniforme
        methods["7_Ensemble_Mean"][p] = np.mean(t_arr)

    # Griglia 2D
    n_lat = ds.attrs["orig_n_lat"]
    n_lon = ds.attrs["orig_n_lon"]
    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    invert_lat = lat_grid[0] > lat_grid[-1]
    plot_lat_grid = lat_grid[::-1] if invert_lat else lat_grid

    print("\n" + "=" * 80)
    print("                      CONFRONTO METODI DI AGGREGAZIONE                  ")
    print("=" * 80)
    print(f"{'Metodo':<25} | {'Min (°C)':<10} | {'Max (°C)':<10} | {'Media (°C)':<10}")
    print("-" * 80)

    grid_results = {}
    for name, values in methods.items():
        print(
            f"{name:<25} | {np.nanmin(values):<10.2f} | {np.nanmax(values):<10.2f} | {np.nanmean(values):<10.2f}"
        )

        matrix = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        matrix[lat_indices, lon_indices] = values
        if invert_lat:
            matrix = matrix[::-1, :]
        grid_results[name] = matrix

    print("=" * 80 + "\n")

    # Plot Multi-Pannello Comparativo
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()

    all_vals = np.concatenate([v for v in methods.values() if not np.isnan(v).all()])
    vmin = 0.0
    vmax = max(float(np.nanpercentile(all_vals, 98)), 0.5)

    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="lightgray")

    for ax_idx, (name, matrix) in enumerate(grid_results.items()):
        ax = axes[ax_idx]
        im = ax.imshow(
            matrix,
            origin="lower",
            extent=[lon_grid.min(), lon_grid.max(), plot_lat_grid.min(), plot_lat_grid.max()],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        ax.set_title(name.replace("_", " "), fontsize=11, fontweight="bold")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    # Rimuoviamo gli assi non utilizzati nel layout 3x3
    for k in range(len(grid_results), len(axes)):
        fig.delaxes(axes[k])

    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Climate Impact: $T_{fact} - T_{cf}$ [°C]", size=12)

    plt.suptitle(
        f"Confronto Aggregazioni CVAE (dx={int(lon_grid[1]-lon_grid[0]) if len(lon_grid)>1 else 0}, radius={radius})",
        fontsize=16,
        fontweight="bold",
    )
    plt.savefig("vae_climate_impact_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    print("Grafico comparativo salvato in: 'vae_climate_impact_comparison.png'")

    # Salviamo anche la mappa raccomandata singola (Metodo 6 o Metodo 2) ad altissima definizione
    plt.figure(figsize=(10, 8))
    im_best = plt.imshow(
        grid_results["6_Gaussian_RBF_Bayes"],
        origin="lower",
        extent=[lon_grid.min(), lon_grid.max(), plot_lat_grid.min(), plot_lat_grid.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(im_best, fraction=0.046, pad=0.04)
    cbar.set_label("Climate Impact: $T_{factual} - T_{counterfactual}$ [°C]", size=12)
    plt.title("Local Climate Impact Map (Gaussian RBF Blending)", fontsize=14, fontweight="bold")
    plt.xlabel("Longitude", fontsize=12)
    plt.ylabel("Latitude", fontsize=12)
    plt.tight_layout()
    plt.savefig("vae_climate_impact_best.png", dpi=300)
    plt.close()
    print("Saved map: 'vae_climate_impact_best.png'")

