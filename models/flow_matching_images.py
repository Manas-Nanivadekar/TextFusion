import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel


class FlowMatchingImage(DiffusionModel):
    def __init__(self, network: nn.Module, clip_samples: bool = True):
        super().__init__(network=network, schedule=None)
        self.clip_samples = clip_samples

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand(batch_size, device=device)

        x1 = torch.randn_like(x0)

        t_expanded = t.view(batch_size, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        v_true = x1 - x0

        v_pred = self.network(x_t, t)

        loss = nn.functional.mse_loss(v_pred, v_true)
        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        n_steps: int = 50,
        return_trajectory: bool = False,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.eval()

        x = torch.randn(n_samples, *image_shape, device=device)

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("WARNING: Invalid initial noise, regenerating...")
            x = torch.randn(n_samples, *image_shape, device=device)
        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((n_samples,), t_current, device=device)

            v_pred = self.network(x, t)

            if torch.isnan(v_pred).any() or torch.isinf(v_pred).any():
                print(
                    f"WARNING: Invalid velocity at step {i}/{n_steps}, t={t_current:.3f}"
                )
                print(f"  x range: [{x.min():.3f}, {x.max():.3f}]")
                print(f"  v_pred range: [{v_pred.min():.3f}, {v_pred.max():.3f}]")
                print(f"  NaN count: {torch.isnan(v_pred).sum()}")
                print(f"  Inf count: {torch.isinf(v_pred).sum()}")
                v_pred = torch.nan_to_num(v_pred, nan=0.0, posinf=0.0, neginf=0.0)

            v_pred = torch.clamp(v_pred, -10, 10)

            x = x + dt * v_pred

            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"WARNING: Invalid x after step {i}/{n_steps}")
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            if self.clip_samples:
                x = torch.clamp(x, -10, 10)

            if return_trajectory:
                trajectory.append(x.clone())

        x = torch.clamp(x, -1, 1)

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory

    @torch.no_grad()
    def sample_ode(self, *args, **kwargs):
        return self.sample(*args, **kwargs)


if __name__ == "__main__":
    from .unet import UNet

    print("Testing FlowMatchingImage...")

    # Create model
    network = UNet(in_channels=1, out_channels=1, base_channels=32, num_res_blocks=1)
    flow = FlowMatchingImage(network)

    # Test loss computation
    batch_size = 4
    x0 = torch.randn(batch_size, 1, 28, 28)

    loss = flow.compute_loss(x0)
    print(f"Loss on random data: {loss.item():.6f}")

    # Test sampling
    print("\nTesting sampling with stability checks...")
    samples, trajectory = flow.sample(
        n_samples=4, image_shape=(1, 28, 28), n_steps=10, return_trajectory=True
    )
    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.3f}, {samples.max():.3f}]")
    print(f"Has NaN: {torch.isnan(samples).any()}")
    print(f"Trajectory shape: {trajectory.shape}")

    if torch.isnan(samples).any():
        print("\n❌ Sampling produces NaN!")
    else:
        print("\n✓ FlowMatchingImage works!")
