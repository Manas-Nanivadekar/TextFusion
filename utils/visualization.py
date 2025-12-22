import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torchvision.utils as vutils


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


def visualize_mnist_samples(
    samples: torch.Tensor,
    save_path: str,
    title: str = "Generated MNIST Samples",
    nrow: int = 8,
):
    """
    Visualize a grid of MNIST samples

    Args:
        samples: (N, 1, 28, 28) tensor in [-1, 1]
        save_path: Where to save the image
        title: Title for the plot
        nrow: Number of images per row
    """
    samples = (samples.cpu() + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    print(
        f"    Sample stats after denorm: min={samples.min():.3f}, max={samples.max():.3f}, mean={samples.mean():.3f}"
    )

    grid = vutils.make_grid(samples, nrow=nrow, padding=2, normalize=False)

    grid_np = grid.permute(1, 2, 0).squeeze().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.imshow(grid_np, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=20)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"    Saved visualization to {save_path}")


def visualize_mnist_denoising(
    model,
    n_samples: int = 8,
    n_steps: int = 50,
    steps_to_show: list = None,
    save_path: str = "outputs/denoising_process.png",
    device: str = "cpu",
):
    if steps_to_show is None:
        steps_to_show = [
            0,
            n_steps // 5,
            2 * n_steps // 5,
            3 * n_steps // 5,
            4 * n_steps // 5,
            n_steps,
        ]

    model.eval()

    samples, trajectory = model.sample(
        n_samples=n_samples,
        image_shape=(1, 28, 28),
        n_steps=n_steps,
        return_trajectory=True,
        device=device,
    )

    trajectory = trajectory.cpu()

    print(f"    Trajectory shape: {trajectory.shape}")
    print(f"    Trajectory range: [{trajectory.min():.3f}, {trajectory.max():.3f}]")

    fig, axes = plt.subplots(n_samples, len(steps_to_show), figsize=(15, 2 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for sample_idx in range(n_samples):
        for col_idx, step in enumerate(steps_to_show):
            ax = axes[sample_idx, col_idx]

            img = trajectory[step, sample_idx, 0].numpy()

            img = (img + 1) / 2
            img = np.clip(img, 0, 1)

            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.axis("off")

            if sample_idx == 0:
                t_val = 1 - step / n_steps
                ax.set_title(f"Step {step}\nt={t_val:.2f}", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"    Saved denoising visualization to {save_path}")


def compare_sampling_methods(
    model,
    n_samples: int = 64,
    step_counts: list = [10, 25, 50, 100],
    save_path: str = "outputs/sampling_comparison.png",
    device: str = "cpu",
):
    model.eval()

    fig, axes = plt.subplots(1, len(step_counts), figsize=(20, 5))

    for idx, n_steps in enumerate(step_counts):
        print(f"    Generating with {n_steps} steps...")

        samples, _ = model.sample(
            n_samples=n_samples,
            image_shape=(1, 28, 28),
            n_steps=n_steps,
            return_trajectory=False,
            device=device,
        )

        samples = (samples.cpu() + 1) / 2
        samples = torch.clamp(samples, 0, 1)

        print(f"      Sample range: [{samples.min():.3f}, {samples.max():.3f}]")

        grid = vutils.make_grid(samples, nrow=8, padding=2, normalize=False)
        grid_np = grid.permute(1, 2, 0).squeeze().numpy()

        ax = axes[idx]
        ax.imshow(grid_np, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"{n_steps} Steps", fontsize=14)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"    Saved sampling comparison to {save_path}")


def visualize_interpolation(
    model,
    start_noise: torch.Tensor,
    end_noise: torch.Tensor,
    n_interp: int = 8,
    n_steps: int = 50,
    save_path: str = "outputs/interpolation.png",
    device: str = "cpu",
):
    model.eval()

    alphas = torch.linspace(0, 1, n_interp, device=device)
    all_samples = []

    for alpha in alphas:
        noise = (1 - alpha) * start_noise + alpha * end_noise

        x = noise.clone()
        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((1,), t_current, device=device)
            v_pred = model.network(x, t)
            x = x + dt * v_pred

            if model.clip_samples:
                x = torch.clamp(x, -1, 1)

        all_samples.append(x[0])

    all_samples = torch.stack(all_samples)

    all_samples = (all_samples.cpu() + 1) / 2
    all_samples = torch.clamp(all_samples, 0, 1)

    fig, axes = plt.subplots(1, n_interp, figsize=(16, 2))

    for idx, (ax, img) in enumerate(zip(axes, all_samples)):
        img_np = img.squeeze().detach().numpy()
        ax.imshow(img_np, cmap="gray", vmin=0, vmax=1)
        ax.axis("off")
        ax.set_title(f"α={idx/(n_interp-1):.2f}", fontsize=10)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"    Saved interpolation to {save_path}")
