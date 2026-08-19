import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def vae_loss_function(y_recon, y, mu, logvar, beta=1e-3):
    # reconstruction error
    recon_loss = nn.functional.mse_loss(y_recon, y, reduction="mean")

    # KL divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

    # total loss
    return recon_loss + beta * kl_loss, recon_loss, kl_loss


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
