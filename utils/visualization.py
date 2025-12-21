import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


@torch.no_grad()
def visualize_score_field(
    model,
    t_values=[0.2, 0.5, 0.8],
    grid_size=30,
    data=None,
    save_path="outputs/score_field.png",
):
    model.eval()
    device = next(model.parameters()).device

    x = np.linspace(-4, 4, grid_size)
    y = np.linspace(-4, 4, grid_size)
    X, Y = np.meshgrid(x, y)

    grid_points = torch.tensor(
        np.stack([X.flatten(), Y.flatten()], axis=1), dtype=torch.float32, device=device
    )

    n_points = grid_points.shape[0]

    fig, axes = plt.subplots(2, len(t_values), figsize=(6 * len(t_values), 12))
    if len(t_values) == 1:
        axes = axes.reshape(2, 1)

    for idx, t_val in enumerate(t_values):
        t = torch.full((n_points,), t_val, device=device)

        alpha_t, sigma_t = model.schedule.get_alpha_sigma(t)

        epsilon_pred = model.forward(grid_points, t)

        score = -epsilon_pred / sigma_t.unsqueeze(1)
        score = score.cpu().numpy()

        U = score[:, 0].reshape(X.shape)
        V = score[:, 1].reshape(Y.shape)
        magnitude = np.sqrt(U**2 + V**2)

        ax = axes[0, idx]

        skip = 2
        ax.quiver(
            X[::skip, ::skip],
            Y[::skip, ::skip],
            U[::skip, ::skip],
            V[::skip, ::skip],
            magnitude[::skip, ::skip],
            cmap="viridis",
            alpha=0.7,
            scale=50,
            width=0.003,
        )

        if data is not None:
            ax.scatter(data[:, 0], data[:, 1], c="red", s=10, alpha=0.3, label="Data")

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"Score Field at t={t_val:.1f}\n"
            f"(α={alpha_t[0].item():.3f}, σ={sigma_t[0].item():.3f})",
            fontsize=12,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("y")

        ax = axes[1, idx]

        im = ax.contourf(X, Y, magnitude, levels=20, cmap="hot")
        plt.colorbar(im, ax=ax, label="Score Magnitude")

        if data is not None:
            ax.scatter(
                data[:, 0],
                data[:, 1],
                c="cyan",
                s=10,
                alpha=0.5,
                edgecolors="white",
                linewidths=0.5,
            )

        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        ax.set_aspect("equal")
        ax.set_title(f"Score Magnitude at t={t_val:.1f}", fontsize=12)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Score field visualization saved to {save_path}")

    return fig


@torch.no_grad()
def visualize_denoising_process(
    model, n_samples=5, n_steps=50, save_path="outputs/denoising_process.png"
):
    """
    Visualize the full denoising process for a few samples
    Shows how noise gradually becomes data
    """
    model.eval()
    device = next(model.parameters()).device

    # Sample with trajectory
    samples, trajectory = model.sample(
        n_samples=n_samples, n_steps=n_steps, return_trajectory=True, device=device
    )

    trajectory = trajectory.cpu().numpy()

    # Select time steps to visualize (start, 25%, 50%, 75%, end)
    time_indices = [0, n_steps // 4, n_steps // 2, 3 * n_steps // 4, n_steps]

    fig, axes = plt.subplots(n_samples, len(time_indices), figsize=(15, 3 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(n_samples):
        for time_idx, step in enumerate(time_indices):
            ax = axes[sample_idx, time_idx]

            # Get position at this time step
            pos = trajectory[step, sample_idx, :]

            # Plot the trajectory up to this point
            path = trajectory[: step + 1, sample_idx, :]
            ax.plot(path[:, 0], path[:, 1], "b-", alpha=0.5, linewidth=2)

            # Mark current position
            ax.scatter(
                pos[0],
                pos[1],
                c="red",
                s=200,
                marker="o",
                edgecolors="black",
                linewidths=2,
                zorder=5,
            )

            # Mark start (noise)
            start = trajectory[0, sample_idx, :]
            ax.scatter(
                start[0], start[1], c="red", s=100, marker="x", linewidths=2, zorder=5
            )

            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

            # Title with time step
            t_val = 1 - step / n_steps
            ax.set_title(f"Step {step}/{n_steps}\nt={t_val:.2f}", fontsize=10)

            if sample_idx == 0:
                ax.set_xlabel("x")
            if time_idx == 0:
                ax.set_ylabel(f"Sample {sample_idx+1}\ny", fontsize=10)

    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Denoising process visualization saved to {save_path}")

    return fig
