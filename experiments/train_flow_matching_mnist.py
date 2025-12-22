import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet import UNet
from models.flow_matching_images import FlowMatchingImage
from data.image_datasets import MNISTDataset
from training.trainer import DiffusionTrainer


def build_model(cfg: DictConfig):
    if cfg.model.network.type == "UNet":
        network = UNet(
            in_channels=cfg.model.network.in_channels,
            out_channels=cfg.model.network.out_channels,
            base_channels=cfg.model.network.base_channels,
            channel_multipliers=cfg.model.network.channel_multipliers,
            num_res_block=cfg.model.network.num_res_blocks,
            time_emb_dim=cfg.model.network.time_emb_dim,
            use_attention=cfg.model.network.use_attention,
            dropout=cfg.model.network.dropout,
        )
    else:
        raise ValueError(f"Unknown network type: {cfg.model.network.type}")

    model = FlowMatchingImage(
        network=network, clip_samples=cfg.model.sampling.clip_samples
    )

    return model


def build_dataset(cfg: DictConfig):
    if cfg.data.dataset_type == "MNISTDataset":
        dataset = MNISTDataset(
            root=cfg.data.root,
            train=cfg.data.train,
            download=cfg.data.download,
            normalize=cfg.data.normalize,
        )
    else:
        raise ValueError(f"Unknown dataset type: {cfg.data.dataset_type}")

    return dataset


def build_optimizer(model, cfg: DictConfig):
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


def build_scheduler(optimizer, cfg: DictConfig, steps_per_epoch: int):
    if not cfg.training.scheduler.use_scheduler:
        return None

    if cfg.training.scheduler.type == "cosine":
        from torch.optim.lr_scheduler import CosineAnnealingLR

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg.training.epochs * steps_per_epoch,
            eta_min=cfg.training.learning_rate * 0.01,
        )
    else:
        raise ValueError(f"Unknown scheduler type: {cfg.training.scheduler.type}")

    return scheduler


@hydra.main(version_base=None, config_path="../configs", config_name="mnist_config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Training Flow Matching on MNIST")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    torch.manual_seed(cfg.seed)

    if cfg.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    else:
        device = cfg.device

    print(f"\nUsing device: {device}")

    print("\nBuilding dataset...")
    dataset = build_dataset(cfg)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", False),
    )
    print(f"Dataset size: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    print("\nBuilding Flow Matching model...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nBuilding optimizer...")
    optimizer = build_optimizer(model, cfg)
    print(f"Optimizer: {cfg.training.optimizer.type}")
    print(f"Learning rate: {cfg.training.learning_rate}")

    scheduler = build_scheduler(optimizer, cfg, len(dataloader))
    if scheduler is not None:
        print(f"LR Scheduler: {cfg.training.scheduler.type}")

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

    if scheduler is not None:
        trainer.scheduler = scheduler

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
