import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_counterfactuals(model, X_test, Y_test, norm_stats, device="cpu"):
    model = model.to(device)
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pp_test = X_test[:, : model.pp_dim]
    fgmt_test = X_test[:, model.pp_dim :]

    with torch.no_grad():
        sensitivity = model.decoder.get_sensitivity(pp_test)
        y_dyn_test = Y_test - sensitivity * fgmt_test

        mu_z, _ = model.encoder(y_dyn_test, pp_test)
        z = mu_z

        mu_fact, _, _ = model.decoder(z, pp_test, fgmt_test)

        fgmt_zero = torch.full_like(
            fgmt_test,
            float((0.0 - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"]),
        )
        mu_cf, _, _ = model.decoder(z, pp_test, fgmt_zero)

    y_factual = mu_fact.cpu().numpy() * norm_stats["tg_std"]
    y_counterfactual = mu_cf.cpu().numpy() * norm_stats["tg_std"]
    y_true = Y_test.cpu().numpy() * norm_stats["tg_std"]

    climate_impact = y_factual - y_counterfactual
    return y_factual, y_counterfactual, y_true, climate_impact


def save_impact_frame(climate_impact, ds, epoch, total_epochs, loss, output_path, vmin=0.0, vmax=3.5):
    mean_impact_1d = np.nanmean(climate_impact, axis=0)

    n_lat = ds.attrs["orig_n_lat"]
    n_lon = ds.attrs["orig_n_lon"]
    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])

    impact_matrix = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)
    impact_matrix[lat_indices, lon_indices] = mean_impact_1d

    if lat_grid[0] > lat_grid[-1]:
        lat_grid = lat_grid[::-1]
        impact_matrix = impact_matrix[::-1, :]

    plt.figure(figsize=(9, 7.5))
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="lightgray")

    im = plt.imshow(
        impact_matrix,
        origin="lower",
        extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Climate Impact: $T_{factual} - T_{counterfactual}$ [°C]", size=11)

    plt.title(f"CVAE Climate Attribution Evolution — Epoch {epoch:03d}/{total_epochs:03d} (Loss: {loss:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Longitude", fontsize=11)
    plt.ylabel("Latitude", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
