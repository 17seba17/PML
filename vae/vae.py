import xarray as xr

import gc

import numpy as np


import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

def buildingTensors(file_path="vae_dataset.nc"):
    ds = xr.open_dataset(file_path)
    N_time = ds.sizes["time"]

    # temperature elaboration (Y) - In-place without copy
    tg_raw = ds["tg"].values.reshape(N_time, -1)
    np.nan_to_num(tg_raw, copy=False, nan=0.0)
    tg_raw = tg_raw.astype(np.float32, copy=False)

    tg_dim = tg_raw.shape[1]
    tg_mean = tg_raw.mean(axis=0, keepdims=True)
    tg_std = tg_raw.std(axis=0, keepdims=True) + 1e-6

    # Normalization in place
    tg_raw -= tg_mean
    tg_raw /= tg_std

    # building Y
    
    Y = torch.from_numpy(tg_raw)
    del tg_raw
    gc.collect()

    #  pressure & fGMT (X) - In-place without copy
    pp_raw = ds["pp"].values.reshape(N_time, -1)
    np.nan_to_num(pp_raw, copy=False, nan=0.0)
    pp_raw = pp_raw.astype(np.float32, copy=False)

    pp_dim = pp_raw.shape[1]
    cond_dim = pp_dim + 1

    pp_mean = pp_raw.mean(axis=0, keepdims=True)
    pp_std = pp_raw.std(axis=0, keepdims=True) + 1e-6

    pp_raw -= pp_mean
    pp_raw /= pp_std

    fgmt_raw = np.nan_to_num(ds["fgmt"].values.reshape(N_time, 1), nan=0.0).astype(
        np.float32
    )
    fgmt_mean = fgmt_raw.mean(axis=0, keepdims=True)
    fgmt_std = fgmt_raw.std(axis=0, keepdims=True) + 1e-6
    fgmt_raw -= fgmt_mean
    fgmt_raw /= fgmt_std

    # building X
    
    X_mat = np.concatenate([pp_raw, fgmt_raw], axis=1)
    X = torch.from_numpy(X_mat)

    del pp_raw, fgmt_raw, X_mat
    gc.collect()

    # norm stats

    norm_stats = {
        "tg_mean": tg_mean.squeeze(),
        "tg_std": tg_std.squeeze(),
        "pp_mean": pp_mean.squeeze(),
        "pp_std": pp_std.squeeze(),
        "fgmt_mean": fgmt_mean.squeeze(),
        "fgmt_std": fgmt_std.squeeze(),
    }

    return X, Y, tg_dim, pp_dim, cond_dim, norm_stats, ds

# encoder and decoder

class Encoder(nn.Module):
    def __init__(self, tg_dim, cond_dim, hidden_dim=64, latent_dim=10):
        super().__init__()
        in_dim = tg_dim + cond_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y, x):
        h = self.fc(torch.cat([y, x], dim=-1))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim=10, cond_dim=2, hidden_dim=64, tg_dim=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, tg_dim)
        )

    def forward(self, z, x):
        return self.fc(torch.cat([z, x], dim=-1))

class CVAE(nn.Module):
    '''
    '''
    def __init__(self, tg_dim, cond_dim, hidden_dim=64, latent_dim=10):
        super().__init__()
        self.encoder = Encoder(tg_dim, cond_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, cond_dim, hidden_dim, tg_dim)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick:
        std = exp(0.5 * logvar)
        z = mu + sigma * eps, con eps ~ N(0, 1)
        """
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, y, x):
        mu, logvar = self.encoder(y, x)
        z = self.reparameterize(mu, logvar)
        y_recon = self.decoder(z, x)
        return y_recon, mu, logvar

def vae_loss_function(y_recon, y, mu, logvar, beta=1e-3):
    # reconstruction error
    recon_loss = nn.functional.mse_loss(y_recon, y, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    #total loss
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


def temporal_train_test_split(ds, split_year=2000):
    train_mask = (ds.time.dt.year <= split_year).values
    test_mask = (ds.time.dt.year > split_year).values

    return train_mask, test_mask

def train_cvae(
    model,
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    beta=1e-3,
    device="cpu",
):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, Y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss, total_recon, total_kl = 0.0, 0.0, 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            y_recon, mu, logvar = model(batch_y, batch_x)
            loss, recon, kl = vae_loss_function(
                y_recon, batch_y, mu, logvar, beta=beta
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_recon += recon.item() * len(batch_x)
            total_kl += kl.item() * len(batch_x)

        N = len(X_train)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/N:.4f} | MSE: {total_recon/N:.4f} | KL: {total_kl/N:.4f}")

    return model


def generate_counterfactuals(model, X_test, Y_test, norm_stats, device="cpu"):
    model.eval()
    X_test = X_test.to(device)
    Y_test = Y_test.to(device)

    with torch.no_grad():
        mu, logvar = model.encoder(Y_test, X_test)
        z = mu

        y_pred_fact_norm = model.decoder(z, X_test).cpu().numpy()

        X_test_cf = X_test.clone()
        fgmt_zero_norm = (0.0 - norm_stats["fgmt_mean"]) / norm_stats[
            "fgmt_std"
        ]
        X_test_cf[:, -1] = torch.tensor(fgmt_zero_norm, dtype=torch.float32)

        y_pred_cf_norm = model.decoder(z, X_test_cf).cpu().numpy()

    y_factual = y_pred_fact_norm * norm_stats["tg_std"] + norm_stats["tg_mean"]
    y_counterfactual = (
        y_pred_cf_norm * norm_stats["tg_std"] + norm_stats["tg_mean"]
    )
    y_true = Y_test.cpu().numpy() * norm_stats["tg_std"] + norm_stats["tg_mean"]

    climate_impact = y_factual - y_counterfactual

    return y_factual, y_counterfactual, y_true, climate_impact


import matplotlib.pyplot as plt
import numpy as np


def plot_and_print_impact_matrix(
    climate_impact, ds, output_path="climate_impact_map.png"
):

    mean_impact_1d = climate_impact.mean(axis=0)

    if "latitude" in ds.sizes and "longitude" in ds.sizes:
        n_lat = ds.sizes["latitude"]
        n_lon = ds.sizes["longitude"]

        impact_matrix = mean_impact_1d.reshape(n_lat, n_lon)

        print(
            f"\n--- MATRICE IMPATTO CLIMATICO [°C] ({n_lat} x {n_lon} celle) ---"
        )
        print(f"Valore Minimo:  {np.nanmin(impact_matrix):.2f} °C")
        print(f"Valore Massimo: {np.nanmax(impact_matrix):.2f} °C")
        print(f"Valore Medio:   {np.nanmean(impact_matrix):.2f} °C")
        print("---------------------------------------------------------------")

        lats = ds.latitude.values if "latitude" in ds.coords else np.arange(n_lat)
        lons = (
            ds.longitude.values if "longitude" in ds.coords else np.arange(n_lon)
        )

        plt.figure(figsize=(10, 8))

        limit = max(
            abs(np.nanmin(impact_matrix)), abs(np.nanmax(impact_matrix)), 0.5
        )

        im = plt.imshow(
            impact_matrix,
            origin="lower",
            extent=[lons.min(), lons.max(), lats.min(), lats.max()],
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(
            "Impatto Climatico: $T_{factual} - T_{controfattuale}$ [°C]", size=12
        )

        plt.title(
            "Mappa di Sensibilità / Impatto Climatico Locale (VAE 50x50)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel(
            "Longitudine" if "longitude" in ds.coords else "Pixel X",
            fontsize=12,
        )
        plt.ylabel(
            "Latitudine" if "latitude" in ds.coords else "Pixel Y", fontsize=12
        )

        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f" Grafico salvato con successo in: '{output_path}'")
        plt.show()

    else:
        print(
            f"\n Dataset 1D | Climate impact mean: {mean_impact_1d.mean():.2f} °C"
        )

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Creating tensors...")
    
    X, Y, tg_dim, pp_dim, cond_dim, norm_stats, ds = buildingTensors(
        "vae_dataset.nc"
    )
    
    print(f"Splitting data...")

    train_mask, test_mask = temporal_train_test_split(ds, split_year=2005)
    X_train, Y_train = X[train_mask], Y[train_mask]
    X_test, Y_test = X[test_mask], Y[test_mask]

    del X, Y
    gc.collect()

    print(f"Creating CVAE...")

    model = CVAE(
        tg_dim=tg_dim, cond_dim=cond_dim, hidden_dim=128, latent_dim=20
    )

    print(f"Training...")

    model = train_cvae(model, X_train, Y_train, epochs=100, device=device)

    y_fact, y_cf, y_true, impact = generate_counterfactuals(
        model, X_test, Y_test, norm_stats, device=device
    )

    plot_and_print_impact_matrix(
        impact, ds, output_path="vae_climate_impact.png"
    )