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

        mu_fact, _, _ = model.decoder(z, pp_test, fgmt_test)
        
        fgmt_zero = torch.full_like(fgmt_test, float((0.0 - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"]))
        mu_cf, _, _ = model.decoder(z, pp_test, fgmt_zero)

    # Denormalization
    y_factual = mu_fact.cpu().numpy() * norm_stats["tg_std"]
    y_counterfactual = mu_cf.cpu().numpy() * norm_stats["tg_std"]
    y_true = Y_test.cpu().numpy() * norm_stats["tg_std"]

    climate_impact = y_factual - y_counterfactual

    return y_factual, y_counterfactual, y_true, climate_impact



def plot_and_print_impact_matrix(
    climate_impact, ds, output_path="vae_climate_impact.png"
):
    mean_impact_1d = climate_impact.mean(axis=0)

    has_grid_attrs = (
        "orig_n_lat" in ds.attrs
        and "orig_n_lon" in ds.attrs
        and "lat_grid" in ds.attrs
        and "lon_grid" in ds.attrs
    )

    if has_grid_attrs and ds.attrs["orig_n_lat"] > 1 and ds.attrs["orig_n_lon"] > 1:
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

        print(f"\n--- CLIMATE IMPACT MATRIX [°C] ({n_lat} x {n_lon} grid, {len(mean_impact_1d)} valid points) ---")
        print(f"Minimum Value: {np.nanmin(mean_impact_1d):.2f} °C")
        print(f"Maximum Value: {np.nanmax(mean_impact_1d):.2f} °C")
        print(f"Mean Value:    {np.nanmean(mean_impact_1d):.2f} °C")
        print("-----------------------------------------------------------------------------------------")

        plt.figure(figsize=(10, 8))

        vmax = max(float(np.nanpercentile(mean_impact_1d, 98)), 0.5)
        vmin = 0.0

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
        cbar.set_label("Climate Impact: $T_{factual} - T_{counterfactual}$ [°C]", size=12)

        plt.title("Local Climate Impact / Sensitivity Map", fontsize=14, fontweight="bold")
        plt.xlabel("Longitude", fontsize=12)
        plt.ylabel("Latitude", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"Graph successfully saved to: '{output_path}'")
        plt.close()

    else:
        print(f"\n1D Dataset | Mean Climate Impact: {mean_impact_1d.mean():.2f} °C")
