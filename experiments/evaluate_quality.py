import torch
import numpy as np
from scipy.stats import wasserstein_distance

from models.networks import SimpleMLP
from models.ddpm import DDPM2D
from models.flow_matching import FlowMatching2D
from data.toy_datasets import TwoClusterDataset


def compute_wasserstein_2d(samples1, samples2):
    w_x = wasserstein_distance(samples1[:, 0], samples2[:, 0])
    w_y = wasserstein_distance(samples1[:, 1], samples2[:, 1])
    return (w_x + w_y) / 2


def evaluate_models():
    dataset = TwoClusterDataset(n_points=1000, seed=42)
    real_data = dataset.get_all_data().numpy()

    network_ddpm = SimpleMLP(
        input_dim=2, hidden_dim=128, time_embed_dim=32, num_layers=3
    )
    ddpm = DDPM2D(network_ddpm, schedule_type="cosine", timesteps=1000)
    ddpm_ckpt = torch.load(
        "outputs/checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    ddpm.load_state_dict(ddpm_ckpt["model_state_dict"])

    network_flow = SimpleMLP(
        input_dim=2, hidden_dim=128, time_embed_dim=32, num_layers=3
    )
    flow = FlowMatching2D(network_flow)
    flow_ckpt = torch.load(
        "outputs/checkpoints_flow/best.pt", map_location="cpu", weights_only=False
    )
    flow.load_state_dict(flow_ckpt["model_state_dict"])

    n_samples = 1000
    ddpm_samples, _ = ddpm.sample(n_samples=n_samples, n_steps=50)
    flow_samples, _ = flow.sample(n_samples=n_samples, n_steps=50)

    ddpm_samples = ddpm_samples.numpy()
    flow_samples = flow_samples.numpy()

    w_ddpm = compute_wasserstein_2d(real_data, ddpm_samples)
    w_flow = compute_wasserstein_2d(real_data, flow_samples)

    print("=" * 60)
    print("Sample Quality Evaluation")
    print("=" * 60)
    print(f"Training Loss:")
    print(f"  DDPM: {ddpm_ckpt['best_loss']:.4f}")
    print(f"  Flow: {flow_ckpt['best_loss']:.4f}")
    print()
    print(f"Wasserstein Distance (lower = better):")
    print(f"  DDPM: {w_ddpm:.4f}")
    print(f"  Flow: {w_flow:.4f}")
    print()

    print("Sample Statistics:")
    print(f"Real Data:")
    print(f"  Mean: ({real_data[:, 0].mean():.3f}, {real_data[:, 1].mean():.3f})")
    print(f"  Std:  ({real_data[:, 0].std():.3f}, {real_data[:, 1].std():.3f})")
    print(f"DDPM:")
    print(f"  Mean: ({ddpm_samples[:, 0].mean():.3f}, {ddpm_samples[:, 1].mean():.3f})")
    print(f"  Std:  ({ddpm_samples[:, 0].std():.3f}, {ddpm_samples[:, 1].std():.3f})")
    print(f"Flow:")
    print(f"  Mean: ({flow_samples[:, 0].mean():.3f}, {flow_samples[:, 1].mean():.3f})")
    print(f"  Std:  ({flow_samples[:, 0].std():.3f}, {flow_samples[:, 1].std():.3f})")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_models()
