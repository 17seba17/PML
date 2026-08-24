import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def vae_loss_function(    mu_y, logvar_y, y_true, mu_z, logvar_z, sensitivity, beta=0.1, target_mu_norm=0.05530, target_std_norm=0.01053, lambda_moments=150.0 ):
    var_y = torch.exp(logvar_y)    
#    recon_nll = 0.5 * torch.mean(torch.sum(logvar_y + ((y_true - mu_y) ** 2) / var_y, dim=-1))
    recon_nll = 0.5 * torch.mean(logvar_y + ((y_true - mu_y) ** 2) / var_y)
#    kl_div = -0.5 * torch.mean(torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=-1))
    kl_div = -0.5 * torch.mean(1 + logvar_z - mu_z.pow(2) - logvar_z.exp())

    sens_mean = torch.mean(sensitivity)
    sens_std = torch.std(sensitivity)
    loss_mean = (sens_mean - target_mu_norm) ** 2
    loss_std = (sens_std - target_std_norm) ** 2
    moment_loss = loss_mean + loss_std
    total_loss = recon_nll + beta * kl_div + lambda_moments * moment_loss
    return total_loss, recon_nll, kl_div




def train_cvae(
    model,
    X_train,
    Y_train,
    epochs=100,
    batch_size=64,
    lr=1e-3,
    beta=1e-1,
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
            
            mu_y, logvar_y, mu_z, logvar_z, sensitivity = model(batch_y, batch_x)
            
            loss, recon, kl = vae_loss_function(
                mu_y, logvar_y, batch_y, mu_z, logvar_z, sensitivity, beta=beta
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_x)
            total_recon += recon.item() * len(batch_x)
            total_kl += kl.item() * len(batch_x)

        N = len(X_train)
        print(f"Epoch [{epoch+1}/{epochs}] | Loss: {total_loss/N:.4f} | NLL: {total_recon/N:.4f} | KL: {total_kl/N:.4f}")

    return model
