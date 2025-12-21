import torch
import matplotlib.pyplot as plt
from pathlib import Path

from models.networks import SimpleMLP
from models.ddpm import DDPM2D
from models.flow_matching import FlowMatching2D
from data.toy_datasets import TwoClusterDataset


def compare_losses():
    dataset = TwoClusterDataset(n_points=100, seed=42)
    x0 = dataset.get_all_data()

    network_ddpm = SimpleMLP(
        input_dim=2, hidden_dim=128, time_embed_dim=32, num_layers=3
    )
    network_flow = SimpleMLP(
        input_dim=2, hidden_dim=128, time_embed_dim=32, num_layers=3
    )

    ddpm = DDPM2D(network_ddpm, schedule_type="cosine", timesteps=1000)
    flow = FlowMatching2D(network_flow)

    loss_ddpm = ddpm.compute_loss(x0)
    loss_flow = flow.compute_loss(x0)

    print("Untrained Model Losses:")
    print(f"  DDPM: {loss_ddpm.item():.4f}")
    print(f"  Flow: {loss_flow.item():.4f}")
    print()

    print("Loading trained models...")

    ddpm_ckpt = torch.load(
        "outputs/checkpoints/best.pt", map_location="cpu", weights_only=False
    )
    ddpm.load_state_dict(ddpm_ckpt["model_state_dict"])

    flow_ckpt = torch.load(
        "outputs/checkpoints_flow/best.pt", map_location="cpu", weights_only=False
    )
    flow.load_state_dict(flow_ckpt["model_state_dict"])

    loss_ddpm_trained = ddpm.compute_loss(x0)
    loss_flow_trained = flow.compute_loss(x0)

    print("Trained Model Losses:")
    print(f"  DDPM: {loss_ddpm_trained.item():.4f} (epoch {ddpm_ckpt['epoch']})")
    print(f"  Flow: {loss_flow_trained.item():.4f} (epoch {flow_ckpt['epoch']})")
    print()

    print("Generating samples...")
    samples_ddpm_sde, _ = ddpm.sample(n_samples=500, n_steps=50)
    samples_ddpm_ode, _ = ddpm.sample_ode(n_samples=500, n_steps=50)
    samples_flow, traj_flow = flow.sample(
        n_samples=500, n_steps=50, return_trajectory=True
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    for i in range(2):
        axes[i, 0].scatter(x0[:, 0], x0[:, 1], alpha=0.5, s=10, c="blue")
        axes[i, 0].set_xlim(-4, 4)
        axes[i, 0].set_ylim(-4, 4)
        axes[i, 0].set_aspect("equal")
        axes[i, 0].grid(True, alpha=0.3)
        axes[i, 0].set_title("Original Data")

    axes[0, 1].scatter(
        samples_ddpm_sde[:, 0], samples_ddpm_sde[:, 1], alpha=0.5, s=10, c="red"
    )
    axes[0, 1].set_xlim(-4, 4)
    axes[0, 1].set_ylim(-4, 4)
    axes[0, 1].set_aspect("equal")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_title(f"DDPM SDE\nLoss: {loss_ddpm_trained.item():.4f}")

    axes[0, 2].scatter(
        samples_ddpm_ode[:, 0], samples_ddpm_ode[:, 1], alpha=0.5, s=10, c="purple"
    )
    axes[0, 2].set_xlim(-4, 4)
    axes[0, 2].set_ylim(-4, 4)
    axes[0, 2].set_aspect("equal")
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_title("DDPM ODE")

    axes[1, 1].scatter(
        samples_flow[:, 0], samples_flow[:, 1], alpha=0.5, s=10, c="green"
    )
    axes[1, 1].set_xlim(-4, 4)
    axes[1, 1].set_ylim(-4, 4)
    axes[1, 1].set_aspect("equal")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_title(f"Flow Matching\nLoss: {loss_flow_trained.item():.4f}")

    for i in range(10):
        path = traj_flow[:, i, :].numpy()
        axes[1, 2].plot(path[:, 0], path[:, 1], "g-", alpha=0.3, linewidth=1)
        axes[1, 2].scatter(path[0, 0], path[0, 1], c="red", s=20, marker="x")
        axes[1, 2].scatter(path[-1, 0], path[-1, 1], c="green", s=20, marker="o")
    axes[1, 2].set_xlim(-4, 4)
    axes[1, 2].set_ylim(-4, 4)
    axes[1, 2].set_aspect("equal")
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].set_title("Flow Trajectories")

    plt.tight_layout()
    plt.savefig("outputs/comparison_ddpm_vs_flow.png", dpi=150)
    print("Saved comparison to outputs/comparison_ddpm_vs_flow.png")


if __name__ == "__main__":
    compare_losses()
