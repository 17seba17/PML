import numpy as np
import torch

def inverse_softplus(y):
    return np.log(np.expm1(y))

def scale(norm_stats, target_mu_phys=1.05, target_std_phys=0.20):    
    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()
    
    fgmt_std = float(norm_stats["fgmt_std"])
    mean_tg_std = float(np.mean(tg_std))

    k_scale = fgmt_std / mean_tg_std

    target_mu_norm = float(target_mu_phys * k_scale)
    target_std_norm = float(target_std_phys * k_scale)
    exact_base_sens = float(inverse_softplus(target_mu_norm))

    return target_mu_norm, target_std_norm, exact_base_sens
