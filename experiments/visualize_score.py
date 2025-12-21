import torch
from pathlib import Path

from models.networks import SimpleMLP
from models.ddpm import DDPM2D
from data.toy_datasets import TwoClusterDataset
from utils.visualization import visualize_score_field, visualize_denoising_process


def main():
    print("=" * 60)
    print("Visualizing Learned Score Field")
    print("=" * 60)

    checkpoint_path = "outputs/checkpoints/best.pt"

    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please train the model first!")
        return

    network = SimpleMLP(input_dim=2, hidden_dim=128, time_embed_dim=32, num_layers=3)
    model = DDPM2D(network, schedule_type="cosine", timesteps=1000)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['metrics']['loss']:.6f}")

    dataset = TwoClusterDataset(n_points=1000, seed=42)
    data = dataset.get_all_data()

    print("\n1. Visualizing score field...")
    visualize_score_field(
        model=model,
        t_values=[0.1, 0.3, 0.5, 0.7, 0.9],
        grid_size=30,
        data=data,
        save_path="outputs/score_field.png",
    )

    print("\n2. Visualizing denoising process...")
    visualize_denoising_process(
        model=model, n_samples=5, n_steps=50, save_path="outputs/denoising_process.png"
    )

    print("\n" + "=" * 60)
    print("Visualization complete!")
    print("Check outputs/score_field.png")
    print("Check outputs/denoising_process.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
