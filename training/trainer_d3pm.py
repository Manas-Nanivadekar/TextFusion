import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Dict, Any


class D3PMTrainer:
    def __init__(
        self,
        model,
        train_loader: DataLoader,
        optimizer: optim.Optimizer,
        device: str = "cpu",
        log_dir: str = "outputs/logs_d3pm",
        checkpoint_dir: str = "outputs/checkpoints_d3pm",
        sample_dir: str = "outputs/samples_d3pm",
        dataset=None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.dataset = dataset

        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.sample_dir = Path(sample_dir)

        for d in [self.log_dir, self.checkpoint_dir, self.sample_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(log_dir=str(self.log_dir))

        self.epoch = 0
        self.global_step = 0
        self.best_loss = float("inf")

        print("D3PM Trainer initialized")

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")
        for tokens in pbar:
            tokens = tokens.to(self.device)

            loss = self.model.compute_loss(tokens)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
        vis_n_samples: int = 5,
        vis_seq_len: int = 64,
        vis_n_steps: int = 50,
    ):
        print(f"\nTraining for {epochs} epochs")
        print("=" * 60)

        for epoch in range(epochs):
            self.epoch = epoch

            metrics = self.train_epoch()

            self.writer.add_scalar("train/loss_epoch", metrics["loss"], epoch)
            print(f"Epoch {epoch}: Loss = {metrics['loss']:.4f}")

            if (epoch + 1) % sample_every == 0:
                print(f"  Generating samples...")
                self.generate_samples(epoch, vis_n_samples, vis_seq_len, vis_n_steps)

            if (epoch + 1) % save_every == 0 or metrics["loss"] < self.best_loss:
                self.save_checkpoint(epoch, metrics)
                if metrics["loss"] < self.best_loss:
                    self.best_loss = metrics["loss"]

        print("\n" + "=" * 60)
        print(f"Training complete! Best loss: {self.best_loss:.4f}")
        self.writer.close()

    @torch.no_grad()
    def generate_samples(self, epoch: int, n_samples: int, seq_len: int, n_steps: int):
        self.model.eval()

        samples, _ = self.model.sample(
            n_samples=n_samples, seq_len=seq_len, n_steps=n_steps, device=self.device
        )

        save_path = self.sample_dir / f"samples_epoch_{epoch:04d}.txt"
        with open(save_path, "w") as f:
            f.write(f"Epoch {epoch} - Generated Samples\n")
            f.write("=" * 60 + "\n\n")

            for i, sample_tokens in enumerate(samples):
                if self.dataset is not None:
                    text = self.dataset.decode(sample_tokens.cpu().tolist())
                else:
                    text = " ".join(map(str, sample_tokens.cpu().tolist()))

                f.write(f"Sample {i+1}:\n{text}\n\n")

        print(f"Samples saved to {save_path}")

    def save_checkpoint(self, epoch: int, metrics: Dict[str, Any]):
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "best_loss": self.best_loss,
        }

        torch.save(checkpoint, self.checkpoint_dir / "latest.pt")

        if metrics["loss"] <= self.best_loss:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            print(f"Best model saved (loss: {metrics['loss']:.4f})")
