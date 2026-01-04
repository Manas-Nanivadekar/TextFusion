import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader

from models.transformer_d3pm import TransformerD3PM
from models.d3pm import D3PM
from data.code_tokenizer import CodeBPETokenizer
from data.code_datasets import PythonCodeDataset
from training.trainer_d3pm import D3PMTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="d3pm_code_config")
def main(cfg: DictConfig):
    print("=" * 60)
    print("Training D3PM on Code")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))

    torch.manual_seed(cfg.seed)
    device = cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu"
    print(f"\nUsing device: {device}")

    print("\nLoading tokenizer...")
    tokenizer = CodeBPETokenizer()
    tokenizer.load("data/code_tokenizer.json")
    print(f"Vocab size: {tokenizer.vocab_size}")

    print("\nLoading dataset...")
    dataset = PythonCodeDataset(
        data_source=cfg.data.data_source,
        tokenizer=tokenizer,
        max_seq_len=cfg.data.max_seq_len,
        max_files=cfg.data.get("max_files", None),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
    )

    print(f"Batches per epoch: {len(dataloader)}")

    print("\nBuilding model...")
    network = TransformerD3PM(
        vocab_size=tokenizer.vocab_size,
        embed_dim=cfg.model.network.embed_dim,
        num_layers=cfg.model.network.num_layers,
        num_heads=cfg.model.network.num_heads,
        ff_dim=cfg.model.network.ff_dim,
        max_seq_len=cfg.data.max_seq_len,
        dropout=cfg.model.network.dropout,
    )

    model = D3PM(
        network=network,
        mask_token_id=tokenizer.vocab["<MASK>"],
        pad_token_id=tokenizer.vocab["<PAD>"],
        schedule_type=cfg.model.schedule_type,
    )

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

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

    trainer.train(
        epochs=cfg.training.epochs,
        sample_every=cfg.training.sample_every_n_epochs,
        save_every=cfg.training.save_every_n_epochs,
    )


if __name__ == "__main__":
    main()
