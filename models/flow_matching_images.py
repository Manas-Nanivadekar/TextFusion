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

        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((n_samples,), t_current, device=device)

            v_pred = self.network(x, t)

            x = x + dt * v_pred

            if self.clip_samples:
                x = torch.clamp(x, -1, 1)

            if return_trajectory:
                trajectory.append(x.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory

    @torch.no_grad()
    def sample_ode(self, *args, **kwargs):
        """Alias for sample() (Flow is always ODE)"""
        return self.sample(*args, **kwargs)


if __name__ == "__main__":
    from .unet import UNet

    print("Testing FlowMatchingImage...")

    network = UNet(in_channels=1, out_channels=1, base_channels=32)
    flow = FlowMatchingImage(network)

    batch_size = 4
    x0 = torch.randn(batch_size, 1, 28, 28)

    loss = flow.compute_loss(x0)
    print(f"Loss on random data: {loss.item():.6f}")

    samples, trajectory = flow.sample(
        n_samples=4, image_shape=(1, 28, 28), n_steps=10, return_trajectory=True
    )
    print(f"\nSamples shape: {samples.shape}")
    print(f"Trajectory shape: {trajectory.shape}")

    print("\nFlowMatchingImage works!")
