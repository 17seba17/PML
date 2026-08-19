import gc
import torch

from dataset import buildingTensors, temporal_train_test_split
from models import CVAE
from train import train_cvae
from evaluate import generate_counterfactuals, plot_and_print_impact_matrix

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
