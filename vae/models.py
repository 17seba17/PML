import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, tg_dim, pp_dim, hidden_dim=128, latent_dim=20):
        super().__init__()
        in_dim = tg_dim + pp_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y, pp):
        h = self.fc(torch.cat([y, pp], dim=-1))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class Decoder(nn.Module):
    def __init__(self, latent_dim, pp_dim, tg_dim, hidden_dim=128):
        super().__init__()
        self.fc_dynamic = nn.Sequential(
            nn.Linear(latent_dim + pp_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU()
        )
        self.out_mu_dyn = nn.Linear(hidden_dim, tg_dim)
        self.out_logvar = nn.Linear(hidden_dim, tg_dim)
        
        self.fc_sensitivity = nn.Sequential(
            nn.Linear(pp_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, tg_dim)
        )

    def forward(self, z, pp, fgmt):
        h_dyn = self.fc_dynamic(torch.cat([z, pp], dim=-1))
        mu_dyn = self.out_mu_dyn(h_dyn)
        logvar_y = torch.clamp(self.out_logvar(h_dyn), min=-8.0, max=4.0)

        sensitivity = F.softplus(self.fc_sensitivity(pp))
        mu_forced = sensitivity * fgmt
        mu_total = mu_dyn + mu_forced

        return mu_total, logvar_y, sensitivity


class CVAE(nn.Module):
    def __init__(self, tg_dim, pp_dim, hidden_dim=128, latent_dim=20):
        super().__init__()
        self.pp_dim = pp_dim
        self.encoder = Encoder(tg_dim, pp_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, pp_dim, tg_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, y, x):
        pp = x[:, :self.pp_dim]
        fgmt = x[:, self.pp_dim:]
        mu_z, logvar_z = self.encoder(y, pp)
        z = self.reparameterize(mu_z, logvar_z)
        mu_y, logvar_y, sensitivity = self.decoder(z, pp, fgmt)
        return mu_y, logvar_y, mu_z, logvar_z