import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

from dataset import buildingTensors
from models import CVAE
from train import vae_loss_function
from evaluate import generate_counterfactuals, save_impact_frame

if __name__ == "__main__":
    epochs = 100
    batch_size = 64
    lr = 1e-3
    beta = 1e-1
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

    print(f"Instantiating single CVAE on device: {device} | Total points: {tg_dim}...")
    model = CVAE(tg_dim=tg_dim, pp_dim=pp_dim, hidden_dim=128, latent_dim=16).to(device)

    sensitivity_params = [model.decoder.base_sensitivity,*model.decoder.fc_sensitivity.parameters()]
    sensitivity_ids = set(map(id, sensitivity_params))
    general_params = [p for p in model.parameters() if id(p) not in sensitivity_ids]
    optimizer = torch.optim.Adam([{'params': general_params, 'lr': lr},{'params': sensitivity_params, 'lr': 0.5*lr}])
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    os.makedirs("frames", exist_ok=True)
    frame_paths = []

    print(f"\n--- Starting 100 Epochs Evolution Training ---")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0

        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()

            mu_y, logvar_y, mu_z, logvar_z = model(batch_y, batch_x)
            loss, recon, kl = vae_loss_function(mu_y, logvar_y, batch_y, mu_z, logvar_z, beta=beta)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_recon += recon.item() * len(batch_x)
            total_kl += kl.item() * len(batch_x)

        epoch_loss = total_loss / len(X_train)
        print(f"Epoch [{epoch:03d}/{epochs:03d}] | Loss: {epoch_loss:.4f} | Generating frame...")

        _, _, _, impact = generate_counterfactuals(model, X_test, Y_test, norm_stats, device=device)
        frame_name = f"frames/frame_{epoch:03d}.png"
        save_impact_frame(impact, ds, epoch, epochs, epoch_loss, frame_name, vmin=0.0, vmax=3.5)
        frame_paths.append(frame_name)

    print("\nAssembling all 100 frames into 'climate_evolution.gif'...")
    images = [Image.open(f) for f in frame_paths]
    images[0].save(
        "climate_evolution.gif",
        save_all=True,
        append_images=images[1:],
        duration=100,
        loop=0
    )
    print("Animation successfully saved to 'climate_evolution.gif'!")
