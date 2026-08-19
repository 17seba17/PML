import gc
import torch

from dataset import buildingTensors
from models import CVAE
from train import train_cvae
from evaluate import generate_counterfactuals, plot_and_print_impact_matrix

if __name__ == "__main__":
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

    print(f"Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"tg_dim: {tg_dim}, pp_dim: {pp_dim}")

    print(f"Creating CVAE on device: {device}...")

    model = CVAE(
        tg_dim=tg_dim, pp_dim=pp_dim, hidden_dim=128, latent_dim=20
    )

    print(f"Training...")

    model = train_cvae(model, X_train, Y_train, epochs=30, device=device)

    y_fact, y_cf, y_true, impact = generate_counterfactuals(
        model, X_test, Y_test, norm_stats, device=device
    )

    plot_and_print_impact_matrix(
        impact, ds, output_path="vae_climate_impact.png"
    )
