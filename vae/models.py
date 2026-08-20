import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialMapper(nn.Module):
    def __init__(self, spatial_mask):
        super().__init__()
        self.register_buffer("mask", spatial_mask.bool())
        self.H, self.W = self.mask.shape
        self.num_valid = int(self.mask.sum().item())

    def vec_to_grid(self, vec, fill_val=0.0):
        if vec.dim() == 2:
            B, N = vec.shape
            C = 1
            vec = vec.unsqueeze(1)
        else:
            B, C, N = vec.shape

        grid = torch.full(
            (B, C, self.H, self.W), fill_val, device=vec.device, dtype=vec.dtype
        )
        grid[:, :, self.mask] = vec
        return grid

    def grid_to_vec(self, grid):
        vec = grid[:, :, self.mask]
        if vec.shape[1] == 1:
            return vec.squeeze(1)
        return vec


class ConvEncoder(nn.Module):
    def __init__(self, spatial_mapper, hidden_dim=128, latent_dim=20):
        super().__init__()
        self.mapper = spatial_mapper
        H, W = self.mapper.H, self.mapper.W

        self.conv_net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((min(H, 4), min(W, 4))),
            nn.Flatten(),
        )

        flat_dim = 64 * min(H, 4) * min(W, 4)
        self.fc = nn.Sequential(
            nn.Linear(flat_dim, hidden_dim),
            nn.SiLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, y_dyn, pp):
        B = y_dyn.size(0)
        grid_y = self.mapper.vec_to_grid(y_dyn)
        grid_pp = self.mapper.vec_to_grid(pp)
        grid_mask = self.mapper.mask.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        x_2d = torch.cat([grid_y, grid_pp, grid_mask], dim=1)

        h = self.fc(self.conv_net(x_2d))
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), min=-10.0, max=10.0)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, spatial_mapper, tg_dim, latent_dim=20, hidden_dim=128):
        super().__init__()
        self.mapper = spatial_mapper
        self.H, self.W = self.mapper.H, self.mapper.W

        self.base_sensitivity = nn.Parameter(torch.ones(tg_dim) * 0.8)
        self.conv_sensitivity = nn.Sequential(
            nn.Conv2d(2, 16, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

        self.fc_z = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 16 * self.H * self.W),
            nn.SiLU()
        )

        self.conv_dyn = nn.Sequential(
            nn.Conv2d(18, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.GroupNorm(4, 32),
            nn.SiLU(),
            nn.Conv2d(32, 2, kernel_size=1)
        )

    def get_sensitivity(self, pp):
        B = pp.size(0)
        grid_pp = self.mapper.vec_to_grid(pp)
        grid_mask = self.mapper.mask.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)
        
        in_sens = torch.cat([grid_pp, grid_mask], dim=1)
        delta_grid = self.conv_sensitivity(in_sens)
        delta_sens_1d = self.mapper.grid_to_vec(delta_grid)

        return F.softplus(self.base_sensitivity + 0.1 * delta_sens_1d)

    def forward(self, z, pp, fgmt):
        B = z.size(0)
        grid_pp = self.mapper.vec_to_grid(pp)
        grid_mask = self.mapper.mask.float().unsqueeze(0).unsqueeze(0).expand(B, 1, -1, -1)

        z_map = self.fc_z(z).view(B, 16, self.H, self.W)

        in_dyn = torch.cat([z_map, grid_pp, grid_mask], dim=1)
        dyn_grid = self.conv_dyn(in_dyn)

        mu_dyn_grid = dyn_grid[:, 0:1, :, :]
        logvar_grid = dyn_grid[:, 1:2, :, :]

        mu_dyn = self.mapper.grid_to_vec(mu_dyn_grid)
        logvar_y = torch.clamp(self.mapper.grid_to_vec(logvar_grid), min=-8.0, max=4.0)

        sensitivity = self.get_sensitivity(pp)
        mu_forced = sensitivity * fgmt
        mu_total = mu_dyn + mu_forced

        return mu_total, logvar_y, sensitivity


class CVAE(nn.Module):
    def __init__(self, spatial_mask, tg_dim, pp_dim, hidden_dim=128, latent_dim=20):
        super().__init__()
        self.pp_dim = pp_dim
        self.spatial_mapper = SpatialMapper(spatial_mask)
        self.encoder = ConvEncoder(self.spatial_mapper, hidden_dim, latent_dim)
        self.decoder = ConvDecoder(self.spatial_mapper, tg_dim, latent_dim, hidden_dim)

    def reparameterize(self, mu, logvar):
        sigma = torch.exp(0.5 * logvar)
        eps = torch.randn_like(sigma)
        return mu + eps * sigma

    def forward(self, y, x):
        pp = x[:, :self.pp_dim]
        fgmt = x[:, self.pp_dim:]

        sensitivity = self.decoder.get_sensitivity(pp)
        y_dyn = y - sensitivity * fgmt

        mu_z, logvar_z = self.encoder(y_dyn, pp)
        z = self.reparameterize(mu_z, logvar_z)

        mu_y, logvar_y, _ = self.decoder(z, pp, fgmt)
        return mu_y, logvar_y, mu_z, logvar_z
