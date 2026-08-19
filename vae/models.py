import torch
import torch.nn as nn


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
    Description [TBD]
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
