import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import imageio.v2 as imageio
from dataset import buildingTensors
from models import CVAE

def read_json(path):
    with open(path, 'r') as f: return json.load(f)

def generate_counterfactuals_for_anchor(model, X_test, Y_test, norm_stats, device="cpu"):
    model = model.to(device)
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    pp_test = X_test[:, :model.pp_dim]
    fgmt_test = X_test[:, -1:] 

    with torch.no_grad():
        sensitivity = model.decoder.get_sensitivity(pp_test)
        y_dyn_test = Y_test - sensitivity * fgmt_test
        
        mu_z, _ = model.encoder(y_dyn_test, pp_test)
        z = mu_z

        mu_fact, logvar_fact, _, _ = model.decoder(z, pp_test, fgmt_test)
        
        fgmt_zero = torch.full_like(
                fgmt_test,
                float((0.0 - norm_stats["fgmt_mean"]) / norm_stats["fgmt_std"])
        )
        mu_cf, _, _, _ = model.decoder(z, pp_test, fgmt_zero)

    tg_std = torch.from_numpy(norm_stats["tg_std"]).to(device)
    
    y_factual = mu_fact * tg_std
    y_counterfactual = mu_cf * tg_std
    y_true = Y_test * tg_std

    climate_impact = (y_factual - y_counterfactual).cpu().numpy()
    impact_mean = climate_impact.mean(axis=0)
    
    sigma_fact = torch.exp(0.5 * logvar_fact) * tg_std
    sigma_mean = sigma_fact.cpu().numpy().mean(axis=0)

    rmse = np.sqrt(np.mean((y_true.cpu().numpy() - y_factual.cpu().numpy()) ** 2))
    mae = np.mean(np.abs(y_true.cpu().numpy() - y_factual.cpu().numpy()))
    
    ss_res = np.sum((y_true.cpu().numpy() - y_factual.cpu().numpy()) ** 2)
    ss_tot = np.sum((y_true.cpu().numpy() - np.mean(y_true.cpu().numpy(), axis=0)) ** 2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    return impact_mean, sigma_mean, rmse, mae, r2

def aggregate_and_plot_all_methods(point_predictions, ds, radius, out_dir, frame_id="latest"):
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

    for p in range(n_points):
        preds = point_predictions[p]
        if not preds:
            continue

        t_arr = np.array([item["t"] for item in preds])
        sigma_arr = np.array([item["sigma"] for item in preds])
        d_arr = np.array([item["dist"] for item in preds])

        w1 = np.exp(sigma_arr) / (d_arr + eps)
        methods["1_IDW_ExpSigma"][p] = np.sum(w1 * t_arr) / np.sum(w1)

        w2 = np.exp(-sigma_arr) / (d_arr + eps)
        methods["2_IDW_ExpMinusSigma"][p] = np.sum(w2 * t_arr) / np.sum(w2)

        w3 = 1.0 / (d_arr + eps)
        methods["3_Standard_IDW"][p] = np.sum(w3 * t_arr) / np.sum(w3)

        methods["4_Nearest_Model"][p] = t_arr[np.argmin(d_arr)]
        methods["5_Min_Sigma"][p] = t_arr[np.argmin(sigma_arr)]

        rbf_sigma = max(radius / 2.0, 1.0)
        w6 = np.exp(-0.5 * (d_arr / rbf_sigma) ** 2) * np.exp(-sigma_arr)
        methods["6_Gaussian_RBF_Bayes"][p] = np.sum(w6 * t_arr) / np.sum(w6)

        methods["7_Ensemble_Mean"][p] = np.mean(t_arr)

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

    grid_results = {}
    for name, values in methods.items():
        matrix = np.full((n_lat, n_lon), np.nan, dtype=np.float32)
        matrix[lat_indices, lon_indices] = values
        if invert_lat:
            matrix = matrix[::-1, :]
        grid_results[name] = matrix

    rbf_map = grid_results["6_Gaussian_RBF_Bayes"]
    valid_pixels = rbf_map[~np.isnan(rbf_map)]
    
    if len(valid_pixels) > 0:
        dyn_vmin = max(0.0, np.percentile(valid_pixels, 1))
        dyn_vmax = np.percentile(valid_pixels, 99)
    else:
        dyn_vmin, dyn_vmax = 0.0, 2.0

   if frame_id == "latest":
        fig, axes = plt.subplots(3, 3, figsize=(18, 14))
        axes = axes.flatten()

        cmap = plt.cm.Reds.copy()
        cmap.set_bad(color="lightgray")

        for ax_idx, (name, matrix) in enumerate(grid_results.items()):
            ax = axes[ax_idx]
            im = ax.imshow(matrix, origin="lower", extent=[lon_grid.min(), lon_grid.max(), plot_lat_grid.min(), plot_lat_grid.max()], cmap=cmap, vmin=dyn_vmin, vmax=dyn_vmax)
            ax.set_title(name.replace("_", " "), fontsize=11, fontweight="bold")
        
        for k in range(len(grid_results), len(axes)): fig.delaxes(axes[k])
        cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])
        fig.colorbar(im, cax=cbar_ax).set_label("Climate Impact: $T_{fact} - T_{cf}$ [°C]", size=12)
        plt.suptitle(f"Confronto Aggregazioni CVAE (radius={radius})", fontsize=16, fontweight="bold")
        plt.savefig(os.path.join(out_dir, "vae_climate_impact_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 8))
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="lightgray")
    
    im_best = plt.imshow(
        rbf_map,
        origin="lower", extent=[lon_grid.min(), lon_grid.max(), plot_lat_grid.min(), plot_lat_grid.max()],
        cmap=cmap, vmin=dyn_vmin, vmax=dyn_vmax
    )
    plt.colorbar(im_best, fraction=0.046, pad=0.04).set_label("Impact [°C]", size=12)
    plt.title(f"Gaussian RBF Blending Impact | Epoch/State: {frame_id}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    
    frame_path = os.path.join(out_dir, f"frame_{str(frame_id).zfill(4)}.png")
    plt.savefig(frame_path, dpi=150)
    plt.close()
    
    return frame_path


def evaluate_experiment(exp_name, target_epoch="latest", generate_gif=False):
    exp_dir = os.path.join("experiments", exp_name)
    out_dir = os.path.join(exp_dir, "evaluations")
    os.makedirs(out_dir, exist_ok=True)
    
    config = read_json(os.path.join(exp_dir, "config.json"))
    anchors_info = read_json(os.path.join(exp_dir, "anchors_info.json"))
    state = read_json(os.path.join(exp_dir, "state.json"))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating '{exp_name}' su {device}...")

    ds_name = os.path.join(exp_dir, "dataset.nc")
    X_train, Y_train, X_test, Y_test, _, _, _, norm_stats, ds = buildingTensors(ds_name, split_year=config["split_year"])

    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    if target_epoch == "latest":
        epochs_to_eval = ["latest"]
    elif target_epoch == "all" and generate_gif:
        max_ep = max(list(state.values())) if state else 1
        epochs_to_eval = list(range(1, max_ep + 1))
    else:
        epochs_to_eval = [int(target_epoch)]

    frames = []

    for ep in epochs_to_eval:
        print(f"\n--- Valutazione Spaziale: Frame {ep} ---")
        point_predictions = [[] for _ in range(len(lats))]
        metrics_list = []

        for anchor_id, info in anchors_info.items():
            anchor_dir = os.path.join(exp_dir, "checkpoints", anchor_id)
            
            if ep == "latest":
                curr_ep = state.get(anchor_id, 0)
            else:
                curr_ep = min(ep, state.get(anchor_id, 0))
                
            if curr_ep == 0: 
                continue

            ckpt_path = os.path.join(anchor_dir, f"epoch_{curr_ep:04d}.pt")
            if not os.path.exists(ckpt_path): 
                continue

            distances = np.maximum(np.abs(lat_indices - info["i0"]), np.abs(lon_indices - info["j0"]))
            idx = np.where(distances <= config["radius"])[0]

            Y_test_tile = Y_test[:, idx]
            X_test_tile = torch.cat([X_test[:, idx], X_test[:, -1:]], dim=1)

            norm_stats_tile = {
                "tg_std": norm_stats["tg_std"][idx],
                "pp_std": norm_stats["pp_std"][idx],
                "fgmt_mean": norm_stats["fgmt_mean"],
                "fgmt_std": norm_stats["fgmt_std"],
            }

            model = CVAE(
                tg_dim=info["n_points"], pp_dim=info["n_points"],
                hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"],
                base_sensitivity=info["base_sens_init"]
            ).to(device)

            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt['model_state'])

            impact_mean, sigma_mean, rmse, mae, r2 = generate_counterfactuals_for_anchor(
                model, X_test_tile, Y_test_tile, norm_stats_tile, device=device
            )
            metrics_list.append([rmse, mae, r2])

            dists_sub = distances[idx]
            for local_k, global_pt in enumerate(idx):
                point_predictions[global_pt].append({
                    "t": float(impact_mean[local_k]),
                    "sigma": float(sigma_mean[local_k]),
                    "dist": float(dists_sub[local_k]),
                })

        if not metrics_list:
            print(f"Nessun dato ancora salvato per comporre il frame {ep}.")
            continue

        metrics_arr = np.array(metrics_list)
        print(f"Global RMSE: {np.mean(metrics_arr[:, 0]):.3f} °C")
        
        frame_name = ep if ep == "latest" else f"{ep:04d}"
        frame_path = aggregate_and_plot_all_methods(point_predictions, ds, config["radius"], out_dir, frame_id=frame_name)
        frames.append(frame_path)

    if generate_gif and len(frames) > 1:
        print("\nGenerazione GIF Animata dell'apprendimento...")
        gif_path = os.path.join(out_dir, "climate_evolution_rbf.gif")
        
        with imageio.get_writer(gif_path, mode='I', duration=0.15) as writer:
            for filename in frames:
                image = imageio.imread(filename)
                writer.append_data(image)
        print(f"GIF salvata con successo in: {gif_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("exp_name", type=str)
    parser.add_argument("--epoch", type=str, default="latest")
    parser.add_argument("--gif", action="store_true")
    
    args = parser.parse_args()
    evaluate_experiment(args.exp_name, target_epoch=args.epoch, generate_gif=args.gif)
