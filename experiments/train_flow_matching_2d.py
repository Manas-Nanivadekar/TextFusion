import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from models.networks import SimpleMLP
from models.flow_matching import FlowMatching2D
from data.toy_datasets import TwoClusterDataset
from training.trainer import DiffusionTrainer


def build_model(cfg: DictConfig):
    network = SimpleMLP(
        input_dim=cfg.model.network.input_dim,
        hidden_dim=cfg.model.network.hidden_dim,
        time_embed_dim=cfg.model.network.time_embed_dim,
        num_layers=cfg.model.network.num_layers,
    )

    model = FlowMatching2D(
        network=network, clip_samples=cfg.model.sampling.clip_samples
    )

    return model


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    cfg.model = OmegaConf.load("configs/model/flow_matching_2d.yaml")

    print("=" * 60)
    print("Training Flow Matching on 2D Dataset")
    print("=" * 60)
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)

    torch.manual_seed(cfg.seed)

    device = cfg.device if cfg.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\nBuilding dataset...")
    dataset = TwoClusterDataset(
        n_points=cfg.data.n_points,
        cluster1_center=cfg.data.cluster1_center,
        cluster2_center=cfg.data.cluster2_center,
        cluster_std=cfg.data.cluster_std,
        seed=cfg.data.seed,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=cfg.data.shuffle,
        num_workers=cfg.data.num_workers,
    )
    print(f"Dataset size: {len(dataset)}")

    print("\nBuilding Flow Matching model...")
    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    print("\nBuilding optimizer...")
    optimizer = optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    print("\nInitializing trainer...")
    trainer = DiffusionTrainer(
        model=model,
        train_loader=dataloader,
        optimizer=optimizer,
        device=device,
        log_dir="outputs/logs_flow",
        checkpoint_dir="outputs/checkpoints_flow",
        sample_dir="outputs/samples_flow",
    )

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
    print(f"Checkpoints saved to: outputs/checkpoints_flow")
    print(f"Samples saved to: outputs/samples_flow")
    print(f"TensorBoard logs: outputs/logs_flow")
    print("=" * 60)


if __name__ == "__main__":
    main()
