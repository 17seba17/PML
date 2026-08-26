import os
import sys
import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from dataset import buildingTensors
from models import CVAE
from scales import scale
from train import vae_loss_function
from data_vae_preparation import data_preparation

def get_grid_anchors(n_points, step):
    anchors = list(range(0, n_points, step))
    if anchors[-1] != n_points - 1:
        anchors.append(n_points - 1)
    return anchors


def read_json(path):
    with open(path, 'r') as f: return json.load(f)

def write_json(path, data):
    with open(path, 'w') as f: json.dump(data, f, indent=4)


def init_experiment(args):
    exp_dir = os.path.join("experiments", args.exp_name)
    if os.path.exists(exp_dir):
        print(f"Errore: L'esperimento '{args.exp_name}' esiste già.")
        sys.exit(1)
        
    os.makedirs(exp_dir)

    ds_name = os.path.join(exp_dir, "dataset.nc")
    
    data_preparation(
        args.lon_min, 
        args.lat_min, 
        args.lon_max, 
        args.lat_max, 
        output_path=ds_name
    )

    os.makedirs(os.path.join(exp_dir, "checkpoints"))
    
    config = {
        "dx": args.dx, "dy": args.dy, "radius": args.radius,
        "beta": args.beta, "lambda_moments": args.lambda_moments, "lambda_dyn": args.lambda_dyn,
        "batch_size": args.batch_size, "lr": args.lr,
        "lr_base_mult": args.lr_base_mult, "lr_decay": args.lr_decay,
        "hidden_dim": args.hidden_dim, "latent_dim": args.latent_dim,
        "split_year": args.split_year
    }
    write_json(os.path.join(exp_dir, "config.json"), config)
    
    _, _, _, _, _, _, _, norm_stats, ds = buildingTensors(ds_name, split_year=args.split_year)

    lat_grid = np.array(ds.attrs["lat_grid"])
    lon_grid = np.array(ds.attrs["lon_grid"])
    lats = ds["lat"].values
    lons = ds["lon"].values

    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    lat_anchors = get_grid_anchors(len(lat_grid), args.dy)
    lon_anchors = get_grid_anchors(len(lon_grid), args.dx)

    anchors_info = {}
    epochs_state = {}

    for i0 in lat_anchors:
        for j0 in lon_anchors:
            distances = np.maximum(np.abs(lat_indices - i0), np.abs(lon_indices - j0))
            idx = np.where(distances <= args.radius)[0]

            if len(idx) == 0:
                continue
            
            lat_min_tile, lat_max_tile = float(lats[idx].min()), float(lats[idx].max())
            lon_min_tile, lon_max_tile = float(lons[idx].min()), float(lons[idx].max())
            
            target_mu_norm, target_std_norm, base_sens_init = scale(
                norm_stats,
                lat_min=lat_min_tile, lat_max=lat_max_tile,
                lon_min=lon_min_tile, lon_max=lon_max_tile,
                split_year=args.split_year
            )
            
            anchor_id = f"{i0}_{j0}"
            anchors_info[anchor_id] = {
                "i0": i0, "j0": j0, "n_points": len(idx),
                "target_mu_norm": target_mu_norm,
                "target_std_norm": target_std_norm,
                "base_sens_init": base_sens_init
            }
            epochs_state[anchor_id] = 0
            
            os.makedirs(os.path.join(exp_dir, "checkpoints", anchor_id))

    write_json(os.path.join(exp_dir, "anchors_info.json"), anchors_info)
    write_json(os.path.join(exp_dir, "state.json"), epochs_state)
    
    control_data = {
        "action": "run",
        "active_anchors": list(anchors_info.keys()),
        "description": "Edita 'active_anchors' per scegliere quali addestrare contemporaneamente (es. ['0_0', '50_50']). Metti 'action':'stop' per fermare."
    }
    write_json(os.path.join(exp_dir, "control.json"), control_data)
    
    print(f"Execute 'python main.py resume {args.exp_name}' to run.")



def resume_experiment(args):
    exp_dir = os.path.join("experiments", args.exp_name)
    if not os.path.exists(exp_dir):
        print(f"Error: the experiment '{args.exp_name}' does not exist.")
        sys.exit(1)

    config = read_json(os.path.join(exp_dir, "config.json"))
    anchors_info = read_json(os.path.join(exp_dir, "anchors_info.json"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading dataset in memory...")
    ds_name=os.path.join(exp_dir, "dataset.nc")
    X_train, Y_train, _, _, _, _, _, _, ds = buildingTensors(ds_name, split_year=config["split_year"])

    lat_grid, lon_grid = np.array(ds.attrs["lat_grid"]), np.array(ds.attrs["lon_grid"])
    lats, lons = ds["lat"].values, ds["lon"].values
    lat_indices = np.abs(lat_grid[:, None] - lats).argmin(axis=0)
    lon_indices = np.abs(lon_grid[:, None] - lons).argmin(axis=0)

    active_sessions = {}


    while True:
        control = read_json(os.path.join(exp_dir, "control.json"))
        if control.get("action") == "stop":
            print("\nReceived  STOP from control.json.")
            break

        active_list = control.get("active_anchors", [])
        if not active_list:
            print("No anchor in the file")
            import time
            time.sleep(15)
            continue

        state = read_json(os.path.join(exp_dir, "state.json"))

        for anchor_id in list(active_sessions.keys()):
            if anchor_id not in active_list:
                del active_sessions[anchor_id]

        for anchor_id in active_list:
            if anchor_id not in anchors_info:
                print(f"WARNING: this anchor does not exist.")
                continue

            info = anchors_info[anchor_id]
            curr_epoch = state.get(anchor_id, 0)
            anchor_dir = os.path.join(exp_dir, "checkpoints", anchor_id)

            if anchor_id not in active_sessions:
                distances = np.maximum(np.abs(lat_indices - info["i0"]), np.abs(lon_indices - info["j0"]))

                idx = np.where(distances <= config["radius"])[0]

                Y_tile = Y_train[:, idx]
                X_tile = torch.cat([X_train[:, idx], X_train[:, -1:]], dim=1)
                
                dataset = TensorDataset(X_tile, Y_tile)
                loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

                model = CVAE(
                    tg_dim=info["n_points"], pp_dim=info["n_points"],
                    hidden_dim=config["hidden_dim"], latent_dim=config["latent_dim"],
                    base_sensitivity=info["base_sens_init"]
                ).to(device)

                base_param = [model.decoder.base_sensitivity]
                base_id = set(map(id, base_param))
                general_params = [p for p in model.parameters() if id(p) not in base_id]

                optimizer = torch.optim.Adam([
                    {'params': general_params, 'lr': config["lr"]},
                    {'params': base_param, 'lr': config["lr_base_mult"] * config["lr"]}
                ])
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config["lr_decay"])

                if curr_epoch > 0:
                    ckpt_path = os.path.join(anchor_dir, f"epoch_{curr_epoch:04d}.pt")
                    if os.path.exists(ckpt_path):
                        ckpt = torch.load(ckpt_path, map_location=device)
                        model.load_state_dict(ckpt['model_state'])
                        optimizer.load_state_dict(ckpt['optimizer_state'])
                        scheduler.load_state_dict(ckpt['scheduler_state'])
                
                active_sessions[anchor_id] = {
                    "model": model, "opt": optimizer, "sch": scheduler, "loader": loader
                }

            session = active_sessions[anchor_id]
            model, optimizer, scheduler, loader = session["model"], session["opt"], session["sch"], session["loader"]
            
            model.train()
            total_loss, total_recon = 0.0, 0.0

            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                mu_y, logvar_y, mu_z, logvar_z, sensitivity, mu_dyn = model(batch_y, batch_x)

                loss, recon, _ = vae_loss_function(
                    mu_y, logvar_y, batch_y, mu_z, logvar_z, sensitivity, mu_dyn,
                    beta=config["beta"],
                    target_mu_norm=info["target_mu_norm"],
                    target_std_norm=info["target_std_norm"],
                    lambda_moments=config["lambda_moments"],
                    lambda_dyn=config["lambda_dyn"]
                )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item() * len(batch_x)
                total_recon += recon.item() * len(batch_x)

            scheduler.step()
            curr_epoch += 1

            N = len(loader.dataset)
            print(f"Anchor [{anchor_id:>7}] | Epoca {curr_epoch:04d} | Loss: {total_loss/N:6.3f} | NLL: {total_recon/N:6.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': curr_epoch
            }, os.path.join(anchor_dir, f"epoch_{curr_epoch:04d}.pt"))

            state[anchor_id] = curr_epoch
            write_json(os.path.join(exp_dir, "state.json"), state)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gestore Addestramento Interlacciato CVAE")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init")
    init_parser.add_argument("exp_name", type=str)
    init_parser.add_argument("--dx", type=int, default=50)
    init_parser.add_argument("--dy", type=int, default=50)
    init_parser.add_argument("--radius", type=float, default=71.0)
    
    init_parser.add_argument("--beta", type=float, default=0.1)
    init_parser.add_argument("--lambda_moments", type=float, default=150.0)
    init_parser.add_argument("--lambda_dyn", type=float, default=10.0)
    init_parser.add_argument("--batch_size", type=int, default=128)
    init_parser.add_argument("--lr", type=float, default=1e-3)
    init_parser.add_argument("--lr_base_mult", type=float, default=0.1)
    init_parser.add_argument("--lr_decay", type=float, default=0.99)
    init_parser.add_argument("--hidden_dim", type=int, default=128)
    init_parser.add_argument("--latent_dim", type=int, default=20)
    init_parser.add_argument("--split_year", type=int, default=2005)

    init_parser.add_argument("--lat_min", type=int, default=45)
    init_parser.add_argument("--lat_max", type=int, default=50)
    init_parser.add_argument("--lon_min", type=int, default=10)
    init_parser.add_argument("--lon_max", type=int, default=15)



    resume_parser = subparsers.add_parser("resume")
    resume_parser.add_argument("exp_name", type=str)

    args = parser.parse_args()

    if args.command == "init":
        init_experiment(args)
    elif args.command == "resume":
        resume_experiment(args)
