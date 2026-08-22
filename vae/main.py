import gc
import os
import sys
import numpy as np
import torch

from dataset import buildingTensors
from evaluate import generate_counterfactuals, aggregate_and_plot_all_methods
from models import CVAE
from train import train_cvae

def get_grid_anchors(n_points, step):
    anchors = list(range(0, n_points, step))
    if anchors[-1] != n_points - 1:
        anchors.append(n_points - 1)
    return anchors

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Use: python main.py <dx> <dy> <radius> [epochs]")
        print("Esempio: python main.py 50 50 71 30")
        sys.exit(1)

    dx = int(sys.argv[1])
    dy = int(sys.argv[2])
    radius = float(sys.argv[3])
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 30
    min_radius_sq = dx**2 + dy**2
    if radius**2 < min_radius_sq:
        raise ValueError(
            f"radius^2 ({radius**2:.1f}) must be >= dx^2 + dy^2 ({min_radius_sq})"
        )

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

    total_valid_points = len(lats)

    lat_anchors = get_grid_anchors(n_lat, dy)
    lon_anchors = get_grid_anchors(n_lon, dx)

    print(f"\nDim grid: {n_lat} x {n_lon}")
    print(f"Anchors Latitude  ({len(lat_anchors)}): {lat_anchors}")
    print(f"Anchors Longitude ({len(lon_anchors)}): {lon_anchors}")
    print(f"Total models: {len(lat_anchors) * len(lon_anchors)}\n")
    point_predictions = [[] for _ in range(total_valid_points)]

    model_idx = 0
    total_models = len(lat_anchors) * len(lon_anchors)


    for i0 in lat_anchors:
        for j0 in lon_anchors:

            model_idx += 1



            distances = np.sqrt((lat_indices - i0) ** 2 + (lon_indices - j0) ** 2)
            circle_mask = distances <= radius
            idx = np.where(circle_mask)[0]

            if len(idx) == 0:
                print(f"[{model_idx}/{total_models}] Anchor ({i0}, {j0}) - Only sea - skipping...")
                continue

            print(
                f"[{model_idx}/{total_models}] Training model on anchor ({i0}, {j0}) | "
                f"Points {radius}: {len(idx)}"
            )

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




            print(f"Creating CVAE...")

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


            _, _, _, impact_mean, sigma_mean = generate_counterfactuals(
                model, X_test_tile, Y_test_tile, norm_stats_tile, device=device
            )

            dists_sub = distances[idx]

            for local_k, global_pt in enumerate(idx):
                point_predictions[global_pt].append({
                    "t": float(impact_mean[local_k]),
                    "sigma": float(sigma_mean[local_k]),
                    "dist": float(dists_sub[local_k]),
                })


    aggregate_and_plot_all_methods(point_predictions, ds, radius=radius)
