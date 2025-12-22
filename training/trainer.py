import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import Optional, Dict, Any
import numpy as np


class DiffusionTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        log_dir: str = "outputs/logs",
        checkpoint_dir: str = "outputs/checkpoints",
        sample_dir: str = "outputs/samples",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir = Path(sample_dir)

        for dir in [self.log_dir, self.checkpoint_dir, self.sample_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        print(f"Trainer initialized:")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Checkpoint dir: {self.checkpoint_dir}")
        print(f"  Sample dir: {self.sample_dir}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for batch in pbar:
            if isinstance(batch, (list, tuple)):
                x0 = batch[0].to(self.device)
            else:
                x0 = batch.to(self.device)

            loss = self.model.compute_loss(x0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)

        avg_loss = epoch_loss / num_batches
        return {"loss": avg_loss}

    def train(
        self,
        epochs: int,
        sample_every: int = 10,
        save_every: int = 50,
        vis_n_samples: int = 500,
        vis_n_trajectories: int = 10,
    ):
        print(f"\nStarting training for {epochs} epochs")
        print(f"Device: {self.device}")

        for epoch in range(epochs):
            self.epoch = epoch

            metrics = self.train_epoch()

            self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.6f}")

            if (epoch + 1) % sample_every == 0:
                print(f"  Generating samples...")
                self.visualize_samples(
                    epoch=epoch,
                    n_samples=vis_n_samples,
                    n_trajectories=vis_n_trajectories,
                )

            if (epoch + 1) % save_every == 0 or metrics["loss"] < self.best_loss:
                self.save_checkpoint(epoch, metrics)
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]

        print("\n" + "=" * 60)
        print("Training complete!")
        print(f"Best loss: {self.best_loss:.6f}")
        self.writer.close()

    @torch.no_grad()
    def visualize_samples(
        self, epoch: int, n_samples: int = 500, n_trajectories: int = 10
    ):
        self.model.eval()
        data = self.train_loader.dataset.get_all_data()

        samples_sde, traj_sde = self.model.sample(
            n_samples=n_samples, n_steps=50, return_trajectory=True, device=self.device
        )
        samples_sde = samples_sde.cpu()

        samples_ode, traj_ode = self.model.sample_ode(
            n_samples=n_samples, n_steps=50, return_trajectory=True, device=self.device
        )
        samples_ode = samples_ode.cpu()

        if (epoch + 1) % sample_every == 0:
            print(f"  Generating samples...")

            # Check if we're working with images or 2D data
            sample_batch = next(iter(self.train_loader))
            if isinstance(sample_batch, (list, tuple)):
                sample_data = sample_batch[0]
            else:
                sample_data = sample_batch

            if len(sample_data.shape) == 4:  # Images (B, C, H, W)
                self.visualize_samples_images(
                    epoch=epoch,
                    n_samples=vis_n_samples,
                    n_trajectories=vis_n_trajectories,
                    image_shape=tuple(sample_data.shape[1:]),
                )
            else:  # 2D data (B, 2)
                self.visualize_samples(
                    epoch=epoch,
                    n_samples=vis_n_samples,
                    n_trajectories=vis_n_trajectories,
                )

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes[0, 0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, c="blue")
        axes[0, 0].set_xlim(-4, 4)
        axes[0, 0].set_ylim(-4, 4)
        axes[0, 0].set_aspect("equal")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_title("Original Data")

        axes[0, 1].scatter(
            samples_sde[:, 0], samples_sde[:, 1], alpha=0.5, s=10, c="red"
        )
        axes[0, 1].set_xlim(-4, 4)
        axes[0, 1].set_ylim(-4, 4)
        axes[0, 1].set_aspect("equal")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_title(f"SDE Samples (Epoch {epoch})")

        for i in range(min(n_trajectories, n_samples)):
            path = traj_sde[:, i, :].cpu().numpy()
            axes[0, 2].plot(path[:, 0], path[:, 1], "b-", alpha=0.3, linewidth=1)
            axes[0, 2].scatter(path[0, 0], path[0, 1], c="red", s=20, marker="x")
            axes[0, 2].scatter(path[-1, 0], path[-1, 1], c="green", s=20, marker="o")
        axes[0, 2].set_xlim(-4, 4)
        axes[0, 2].set_ylim(-4, 4)
        axes[0, 2].set_aspect("equal")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_title("SDE Trajectories")

        axes[1, 0].scatter(data[:, 0], data[:, 1], alpha=0.5, s=10, c="blue")
        axes[1, 0].set_xlim(-4, 4)
        axes[1, 0].set_ylim(-4, 4)
        axes[1, 0].set_aspect("equal")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_title("Original Data")

        axes[1, 1].scatter(
            samples_ode[:, 0], samples_ode[:, 1], alpha=0.5, s=10, c="purple"
        )
        axes[1, 1].set_xlim(-4, 4)
        axes[1, 1].set_ylim(-4, 4)
        axes[1, 1].set_aspect("equal")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_title(f"ODE Samples (Epoch {epoch})")

        for i in range(min(n_trajectories, n_samples)):
            path = traj_ode[:, i, :].cpu().numpy()
            axes[1, 2].plot(path[:, 0], path[:, 1], "r-", alpha=0.3, linewidth=1)
            axes[1, 2].scatter(path[0, 0], path[0, 1], c="red", s=20, marker="x")
            axes[1, 2].scatter(path[-1, 0], path[-1, 1], c="green", s=20, marker="o")
        axes[1, 2].set_xlim(-4, 4)
        axes[1, 2].set_ylim(-4, 4)
        axes[1, 2].set_aspect("equal")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_title("ODE Trajectories")

        plt.tight_layout()

        save_path = self.sample_dir / f"samples_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.writer.add_figure("samples", fig, epoch)

        print(f"    → Samples saved to {save_path}")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self.best_loss,
        }

        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)

        if metrics["loss"] <= self.best_loss:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"    → Best model saved (loss: {metrics['loss']:.6f})")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]

        print(f"Loaded checkpoint from epoch {self.epoch}")
        return checkpoint

    # training/trainer.py - Add this method to DiffusionTrainer class


@torch.no_grad()
def visualize_samples_images(
    self,
    epoch: int,
    n_samples: int = 64,
    n_trajectories: int = 16,
    image_shape: tuple = (1, 28, 28),
):
    """Generate and visualize image samples"""
    self.model.eval()

    # Determine grid size
    grid_size = int(n_samples**0.5)

    # Sample images
    samples, _ = self.model.sample(
        n_samples=n_samples,
        image_shape=image_shape,
        n_steps=50,
        return_trajectory=False,
        device=self.device,
    )
    samples = samples.cpu()

    # Denormalize from [-1, 1] to [0, 1]
    samples = (samples + 1) / 2
    samples = torch.clamp(samples, 0, 1)

    # Create grid
    import torchvision.utils as vutils

    grid = vutils.make_grid(samples, nrow=grid_size, padding=2, normalize=False)

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
    ax.axis("off")
    ax.set_title(f"Generated Samples (Epoch {epoch})", fontsize=16)

    # Save
    save_path = self.sample_dir / f"samples_epoch_{epoch:04d}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    # Log to TensorBoard
    self.writer.add_image("samples", grid, epoch)

    print(f"    → Samples saved to {save_path}")
