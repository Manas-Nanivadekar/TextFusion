import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.transformer_d3pm import TransformerD3PM
from models.d3pm import D3PM
from data.text_datasets import ShakespeareDataset
from training.trainer_d3pm import D3PMTrainer


@hydra.main(
    version_base=None, config_path="../configs", config_name="d3pm_shakespeare_config"
)
def main(cfg: DictConfig):
    print("=" * 60)
    print("Training D3PM on Shakespeare")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    torch.manual_seed(cfg.seed)

    device = cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\nLoading dataset...")
    dataset = ShakespeareDataset(
        data_dir=cfg.data.data_dir,
        max_seq_len=cfg.data.max_seq_len,
        vocab_size=cfg.data.vocab_size,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    print(f"Batches per epoch: {len(dataloader)}")

    print("\nBuilding model...")
    network = TransformerD3PM(
        vocab_size=dataset.vocab_size,
        embed_dim=cfg.model.network.embed_dim,
        num_layers=cfg.model.network.num_layers,
        num_heads=cfg.model.network.num_heads,
        ff_dim=cfg.model.network.ff_dim,
        max_seq_len=cfg.data.max_seq_len + 2,
        dropout=cfg.model.network.dropout,
        time_emb_dim=cfg.model.network.time_emb_dim,
    )

    model = D3PM(
        network=network,
        mask_token_id=dataset.mask_token_id,
        pad_token_id=dataset.pad_token_id,
        schedule_type=cfg.model.schedule_type,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nSetting up optimizer...")
    optimizer = optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
        betas=cfg.training.optimizer.betas,
    )

    print("\nInitializing trainer...")
    trainer = D3PMTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        log_dir=cfg.tensorboard.log_dir,
        checkpoint_dir=cfg.checkpoint_dir,
        sample_dir=cfg.sample_dir,
        dataset=dataset,
    )

    print("\nStarting training...\n")
    trainer.train(
        epochs=cfg.training.epochs,
        sample_every=cfg.training.sample_every_n_epochs,
        save_every=cfg.training.save_every_n_epochs,
        vis_n_samples=cfg.training.vis_n_samples,
        vis_seq_len=cfg.training.vis_seq_len,
        vis_n_steps=cfg.training.vis_n_steps,
    )

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
