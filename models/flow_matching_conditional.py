import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel


class ConditionalFlowMatching(DiffusionModel):
    def __init__(
        self,
        network: nn.Module,
        clip_samples: bool = True,
        unconditional_prob: float = 0.1,
    ):
        super().__init__(network=network, schedule=None)
        self.clip_samples = clip_samples
        self.unconditional_prob = unconditional_prob

    def compute_loss(self, x0: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand(batch_size, device=device)

        x1 = torch.randn_like(x0)

        t_expanded = t.view(batch_size, 1, 1, 1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        v_true = x1 - x0

        mask = torch.rand(batch_size, device=device) < self.unconditional_prob
        labels_masked = labels.clone()
        labels_masked[mask] = self.network.num_classes

        v_pred = self.network(x_t, t, labels_masked)

        loss = nn.functional.mse_loss(v_pred, v_true)

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        labels: Optional[torch.Tensor] = None,
        image_shape: Tuple[int, int, int] = (1, 28, 28),
        n_steps: int = 50,
        guidance_scale: float = 0.0,
        return_trajectory: bool = False,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.eval()

        x = torch.randn(n_samples, *image_shape, device=device)

        if labels is None:
            labels = torch.randint(
                0, self.network.num_classes, (n_samples,), device=device
            )
        else:
            labels = labels.to(device)

        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((n_samples,), t_current, device=device)

            v_cond = self.network(x, t, labels)

            if guidance_scale > 0:
                v_uncond = self.network(x, t, c=None)
                v_pred = v_uncond + guidance_scale * (v_cond - v_uncond)
            else:
                v_pred = v_cond

            v_pred = torch.clamp(v_pred, -10, 10)

            x = x + dt * v_pred

            if self.clip_samples:
                x = torch.clamp(x, -10, 10)

            if return_trajectory:
                trajectory.append(x.clone())

        x = torch.clamp(x, -1, 1)

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory


if __name__ == "__main__":
    from .unet_conditional import ConditionalUNet

    print("Testing Conditional Flow Matching...")

    network = ConditionalUNet(
        in_channels=1, out_channels=1, base_channels=32, num_classes=10
    )

    model = ConditionalFlowMatching(network, unconditional_prob=0.1)

    # Test loss
    x0 = torch.randn(4, 1, 28, 28)
    labels = torch.randint(0, 10, (4,))

    loss = model.compute_loss(x0, labels)
    print(f"Loss: {loss.item():.6f}")

    samples, _ = model.sample(
        n_samples=4, labels=torch.tensor([0, 1, 2, 3]), guidance_scale=3.0, n_steps=10
    )
    print(f"Samples shape: {samples.shape}")
    print(f"Samples range: [{samples.min():.2f}, {samples.max():.2f}]")

    print("\n Conditional Flow Matching works!")
