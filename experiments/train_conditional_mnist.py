import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.unet_conditional import ConditionalUNet
from models.flow_matching_conditional import ConditionalFlowMatching
from data.image_datasets import MNISTDataset
from training.trainer_conditional import ConditionalDiffusionTrainer


def build_model(cfg: DictConfig):
    network = ConditionalUNet(
        in_channels=cfg.model.network.in_channels,
        out_channels=cfg.model.network.out_channels,
        base_channels=cfg.model.network.base_channels,
        channel_multipliers=cfg.model.network.channel_multipliers,
        num_res_blocks=cfg.model.network.num_res_blocks,
        time_emb_dim=cfg.model.network.time_emb_dim,
        num_classes=cfg.model.network.num_classes,
        class_emb_dim=cfg.model.network.class_emb_dim,
        use_attention=cfg.model.network.use_attention,
        dropout=cfg.model.network.dropout,
    )

    model = ConditionalFlowMatching(
        network=network,
        clip_samples=cfg.model.sampling.clip_samples,
        unconditional_prob=cfg.model.unconditional_prob,
    )

    return model


@hydra.main(
    version_base=None, config_path="../configs", config_name="mnist_conditional_config"
)
def main(cfg: DictConfig):
    print("=" * 60)
    print("Training Conditional Flow Matching on MNIST")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    torch.manual_seed(cfg.seed)

    device = cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Dataset
    dataset = MNISTDataset(
        root=cfg.data.root,
        train=cfg.data.train,
        download=cfg.data.download,
        normalize=cfg.data.normalize,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.get("pin_memory", False),
    )
    print(f"Dataset size: {len(dataset)}, Batches: {len(dataloader)}")

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    trainer = ConditionalDiffusionTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        log_dir=cfg.tensorboard.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        sample_dir=cfg.sample_dir,
        num_classes=cfg.model.network.num_classes,
        guidance_scale=cfg.model.sampling.guidance_scale,
    )

    trainer.train(
        epochs=cfg.training.epochs,
        sample_every=cfg.training.sample_every_n_epochs,
        save_every=cfg.training.save_every_n_epochs,
        vis_n_samples=cfg.training.vis_n_samples,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
