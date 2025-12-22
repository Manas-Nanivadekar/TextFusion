import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from typing import Dict, Any


class ConditionalDiffusionTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        log_dir: str = "outputs/logs_conditional",
        checkpoint_dir: str = "outputs/checkpoints_conditional",
        sample_dir: str = "outputs/samples_conditional",
        num_classes: int = 10,
        guidance_scale: float = 3.0,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.num_classes = num_classes
        self.guidance_scale = guidance_scale

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir = Path(sample_dir)

        for dir in [self.log_dir, self.checkpoint_dir, self.sample_dir]:
            dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        print(f"Conditional Trainer initialized:")
        print(f"  Num classes: {num_classes}")
        print(f"  Guidance scale: {guidance_scale}")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            loss = self.model.compute_loss(images, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            if self.global_step % 10 == 0:
                self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)

        return {"loss": epoch_loss / num_batches}

    def train(
        self,
        epochs: int,
        sample_every: int = 5,
        save_every: int = 10,
        vis_n_samples: int = 100,
    ):
        print(f"\nTraining for {epochs} epochs")
        print("=" * 60)

        for epoch in range(epochs):
            self.epoch = epoch

            metrics = self.train_epoch()

            self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.6f}")

            if (epoch + 1) % sample_every == 0:
                print(f"  Generating conditional samples...")
                self.visualize_conditional_samples(epoch, vis_n_samples)

            if (epoch + 1) % save_every == 0 or metrics["loss"] < self.best_loss:
                self.save_checkpoint(epoch, metrics)
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]

        print("\n" + "=" * 60)
        print(f"Training complete! Best loss: {self.best_loss:.6f}")
        self.writer.close()

    @torch.no_grad()
    def visualize_conditional_samples(self, epoch: int, n_samples: int = 100):
        self.model.eval()

        samples_per_class = n_samples // self.num_classes
        all_samples = []

        for class_idx in range(self.num_classes):
            labels = torch.full((samples_per_class,), class_idx, device=self.device)

            samples, _ = self.model.sample(
                n_samples=samples_per_class,
                labels=labels,
                image_shape=(1, 28, 28),
                n_steps=50,
                guidance_scale=self.guidance_scale,
                device=self.device,
            )

            all_samples.append(samples)

        all_samples = torch.cat(all_samples, dim=0).cpu()

        all_samples = (all_samples + 1) / 2
        all_samples = torch.clamp(all_samples, 0, 1)

        grid = vutils.make_grid(all_samples, nrow=samples_per_class, padding=2)

        fig, ax = plt.subplots(1, 1, figsize=(15, 12))
        ax.imshow(grid.permute(1, 2, 0).squeeze(), cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"Conditional Samples (Epoch {epoch}, Guidance={self.guidance_scale})",
            fontsize=16,
        )

        save_path = self.sample_dir / f"conditional_epoch_{epoch:04d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()

        self.writer.add_image("samples", grid, epoch)

        print(f"    → Saved to {save_path}")

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
