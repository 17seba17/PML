import sys
import numpy as np
import torch
from dataset import buildingTensors

def inverse_softplus(y):
    return np.log(np.expm1(y))

def main():
    file_path = sys.argv[1] if len(sys.argv) > 1 else "vae_dataset.nc"

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
    ) = buildingTensors(file_path, split_year=2005)

    tg_std = norm_stats["tg_std"]
    if isinstance(tg_std, torch.Tensor):
        tg_std = tg_std.cpu().numpy()
    
    fgmt_std = float(norm_stats["fgmt_std"])
    mean_tg_std = float(np.mean(tg_std))

    k_scale = fgmt_std / mean_tg_std

    target_mu_phys = 1.05
    target_std_phys = 0.20 

    # CONVERSIONE IN SCALE NORMALIZZATE
    target_mu_norm = target_mu_phys * k_scale
    target_std_norm = target_std_phys * k_scale
    exact_base_sens = float(inverse_softplus(target_mu_norm))

    print(f"(fgmt_std / mean_tg_std) :             {k_scale:.5f}")
    print(f"target_mu_phys                       : {target_mu_phys:.5f} °C/°C  --> Normalized: {target_mu_norm:.5f}")
    print(f"target_std_phys                    :   {target_std_phys:.5f} °C/°C  --> Normalized: {target_std_norm:.5f}")
    print(f"base_sensitivity         :             {exact_base_sens:.5f}")

if __name__ == "__main__":
    main()
