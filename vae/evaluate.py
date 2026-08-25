import matplotlib.pyplot as plt
import numpy as np
import torch

def compute_physical_sensitivity_stats(model, X_test, norm_stats, device="cpu"):
    model.eval()
    X_test = X_test.to(device)
    pp_test = X_test[:, :model.pp_dim]
    with torch.no_grad():
        sens_norm = model.decoder.get_sensitivity(pp_test).cpu().numpy()
        sens_norm_1d = np.mean(sens_norm, axis=0)
    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()
    fgmt_std = float(norm_stats["fgmt_std"])
    sens_phys = sens_norm_1d * (tg_std / fgmt_std)
    mean_s = float(np.mean(sens_phys))
    std_s = float(np.std(sens_phys))
    min_s = float(np.min(sens_phys))
    max_s = float(np.max(sens_phys))
    return mean_s, std_s, min_s, max_s, sens_phys



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

        mu_fact, _, _, _ = model.decoder(z, pp_test, fgmt_test)

        fgmt_zero = torch.full_like(
            fgmt_test,
            float((0.0 - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"]),
        )
        mu_cf, _, _, _ = model.decoder(z, pp_test, fgmt_zero)

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

    valid_data = impact_matrix[~np.isnan(impact_matrix)]
    if len(valid_data) > 0:
        dyn_vmin = float(np.nanmin(impact_matrix))
        dyn_vmax = float(np.nanmax(impact_matrix))
        if dyn_vmin == dyn_vmax:
            dyn_vmin -= 0.1
            dyn_vmax += 0.1
    else:
        dyn_vmin, dyn_vmax = 0.0, 1.0
    plt.figure(figsize=(9, 7.5))
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="lightgray")
    im = plt.imshow(
        impact_matrix,
        origin="lower",
        extent=[lon_grid.min(), lon_grid.max(), lat_grid.min(), lat_grid.max()],
        cmap=cmap,
        vmin=dyn_vmin,
        vmax=dyn_vmax,
        )
    cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
    cbar.set_label("Climate Impact: $T_{factual} - T_{counterfactual}$ [°C]", size=11)
    plt.title(
        f"CVAE Climate Attribution — Epoch {epoch:03d}/{total_epochs:03d}\n"
        f"Loss: {loss:.3f} | Dynamic Range: [{dyn_vmin:.2f} °C, {dyn_vmax:.2f} °C]",
        fontsize=11,
        fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


