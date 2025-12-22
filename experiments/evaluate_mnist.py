import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models.unet import UNet
from models.flow_matching_images import FlowMatchingImage
from data.image_datasets import MNISTDataset
from utils.visualization import (
    visualize_mnist_samples,
    visualize_mnist_denoising,
    compare_sampling_methods,
    visualize_interpolation,
)


def evaluate_model(
    checkpoint_path: str = "outputs/checkpoints_mnist/best.pt", device: str = "cpu"
):
    """Comprehensive evaluation of trained model"""

    print("=" * 60)
    print("Evaluating MNIST Flow Matching Model")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    network = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_block=2,
        time_emb_dim=256,
        use_attention=True,
        dropout=0.1,
    )

    model = FlowMatchingImage(network, clip_samples=True)

    if Path(checkpoint_path).exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
        print(f"Training loss: {checkpoint['metrics']['loss']:.6f}")
    else:
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Using untrained model for demonstration")

    model = model.to(device)
    model.eval()

    output_dir = Path("outputs/evaluation_mnist")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n1. Generating sample grid...")
    samples, _ = model.sample(
        n_samples=64, image_shape=(1, 28, 28), n_steps=50, device=device
    )
    visualize_mnist_samples(
        samples.cpu(),
        save_path=output_dir / "samples_grid.png",
        title="Generated MNIST Samples (50 steps)",
    )

    # 2. Visualize denoising process
    print("\n2. Visualizing denoising process...")
    visualize_mnist_denoising(
        model=model,
        n_samples=8,
        n_steps=50,
        save_path=output_dir / "denoising_process.png",
        device=device,
    )

    print("\n3. Comparing sampling efficiency...")
    compare_sampling_methods(
        model=model,
        n_samples=64,
        step_counts=[5, 10, 25, 50],
        save_path=output_dir / "sampling_comparison.png",
        device=device,
    )

    print("\n4. Testing latent interpolation...")
    torch.manual_seed(42)
    start_noise = torch.randn(1, 1, 28, 28, device=device)
    end_noise = torch.randn(1, 1, 28, 28, device=device)

    visualize_interpolation(
        model=model,
        start_noise=start_noise,
        end_noise=end_noise,
        n_interp=8,
        n_steps=50,
        save_path=output_dir / "interpolation.png",
        device=device,
    )

    print("\n5. Computing sample statistics...")
    large_batch, _ = model.sample(
        n_samples=1000, image_shape=(1, 28, 28), n_steps=50, device=device
    )
    large_batch = large_batch.cpu()

    print(
        f"   Sample mean: {large_batch.mean():.4f} (should be ~0 for normalized data)"
    )
    print(f"   Sample std: {large_batch.std():.4f}")
    print(f"   Sample min: {large_batch.min():.4f}")
    print(f"   Sample max: {large_batch.max():.4f}")

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=str, default="outputs/checkpoints_mnist/best.pt"
    )
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    evaluate_model(checkpoint_path=args.checkpoint, device=args.device)
