import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path


from models.networks import SimpleMLP
from models.ddpm import DDPM2D
from data.toy_datasets import TwoClusterDataset
from training.trainer import DiffusionTrainer


def build_model(cfg: DictConfig):
    """Build model from config"""
    # Create network
    network = SimpleMLP(
        input_dim=cfg.model.network.input_dim,
        hidden_dim=cfg.model.network.hidden_dim,
        time_embed_dim=cfg.model.network.time_embed_dim,
        num_layers=cfg.model.network.num_layers,
    )

    # Create DDPM model
    model = DDPM2D(
        network=network,
        schedule_type=cfg.model.schedule.type,
        timesteps=cfg.model.schedule.timesteps,
        beta_start=cfg.model.schedule.beta_start,
        beta_end=cfg.model.schedule.beta_end,
        clip_denoised=cfg.model.sampling.clip_denoised,
    )

    return model


def build_dataset(cfg: DictConfig):
    """Build dataset from config"""
    if cfg.data.dataset_type == "TwoClusterDataset":
        dataset = TwoClusterDataset(
            n_points=cfg.data.n_points,
            cluster1_center=cfg.data.cluster1_center,
            cluster2_center=cfg.data.cluster2_center,
            cluster_std=cfg.data.cluster_std,
            seed=cfg.data.seed,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.dataset_type}")

    return dataset


def build_optimizer(model, cfg: DictConfig):
    """Build optimizer from config"""
    if cfg.training.optimizer.type == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=cfg.training.optimizer.betas,
            weight_decay=cfg.training.weight_decay,
        )
    elif cfg.training.optimizer.type == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=cfg.training.learning_rate,
            betas=cfg.training.optimizer.betas,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer type: {cfg.training.optimizer.type}")

    return optimizer


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function"""

    # Print config
    print("=" * 60)
    print("Configuration:")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    # Set seed
    torch.manual_seed(cfg.seed)

    # Determine device
    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = cfg.device

    print(f"\nUsing device: {device}")

    # Build dataset and dataloader
    print("\nBuilding dataset...")
    dataset = build_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
    )
    print(f"Dataset size: {len(dataset)}")

    # Build model
    print("\nBuilding model...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Build optimizer
    print("\nBuilding optimizer...")
    optimizer = build_optimizer(model, cfg)
    print(f"Optimizer: {cfg.training.optimizer.type}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    # Create trainer
    print("\nInitializing trainer...")
    trainer = DiffusionTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        log_dir=cfg.tensorboard.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        sample_dir=cfg.sample_dir,
    )

    # Train!
    print("\nStarting training...\n")
    trainer.train(
        epochs=cfg.training.epochs,
        sample_every=cfg.training.sample_every_n_epochs,
        save_every=cfg.training.save_every_n_epochs,
        vis_n_samples=cfg.training.vis_n_samples,
        vis_n_trajectories=cfg.training.vis_n_trajectories,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Checkpoints saved to: {cfg.checkpoint_dir}")
    print(f"Samples saved to: {cfg.sample_dir}")
    print(f"TensorBoard logs: {cfg.tensorboard.log_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
