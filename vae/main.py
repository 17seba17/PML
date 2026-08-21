import gc
import sys
import numpy as np
import torch

from dataset import buildingTensors
from evaluate import generate_counterfactuals, plot_and_print_impact_matrix
from models import CVAE
from train import train_cvae

if __name__ == "__main__":

    dx = 50
    dy = 50

    if len(sys.argv) > 1:
        dx = int(sys.argv[1])
        dy = dx

    if len(sys.argv) > 2:
        dy = int(sys.argv[2])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Creating tensors...")

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

    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])

    n_lat = len(lat_grid)
    n_lon = len(lon_grid)

    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    total_impact = np.full((len(Y_test), len(lats)), np.nan, dtype=np.float32)

    for i in range(0, n_lat, dy):
        for j in range(0, n_lon, dx):
            i_end = min(i + dy, n_lat)
            j_end = min(j + dx, n_lon)

            tile_mask = (
                (lat_indices >= i)
                & (lat_indices < i_end)
                & (lon_indices >= j)
                & (lon_indices < j_end)
            )
            idx = np.where(tile_mask)[0]

            if len(idx) == 0:
                print(f"Sea tile: skipping")
                continue

            print(f"Training ({i}:{i_end}, {j}:{j_end}) | Punti validi: {len(idx)}")

            print(f"Creating CVAE on device: {device}...")

            tile_tg_dim = len(idx)
            tile_pp_dim = len(idx)

            Y_train_tile = Y_train[:, idx]
            Y_test_tile = Y_test[:, idx]

            fgmt_train = X_train[:, -1:]
            fgmt_test = X_test[:, -1:]

            X_train_tile = torch.cat([X_train[:, idx], fgmt_train], dim=1)
            X_test_tile = torch.cat([X_test[:, idx], fgmt_test], dim=1)

            norm_stats_tile = {
                "tg_std": norm_stats["tg_std"][idx],
                "pp_std": norm_stats["pp_std"][idx],
                "fgmt_mean": norm_stats["fgmt_mean"],
                "fgmt_std": norm_stats["fgmt_std"],
            }

            model = CVAE(
                tg_dim=tile_tg_dim,
                pp_dim=tile_pp_dim,
                hidden_dim=128,
                latent_dim=20,
            )

            print(f"Training...")

            model = train_cvae(
                model, X_train_tile, Y_train_tile, epochs=30, device=device
            )

            y_fact, y_cf, y_true, impact = generate_counterfactuals(
                model, X_test_tile, Y_test_tile, norm_stats_tile, device=device
            )

            total_impact[:, idx] = impact

    plot_and_print_impact_matrix(
        total_impact, ds, output_path="vae_climate_impact.png"
    )
