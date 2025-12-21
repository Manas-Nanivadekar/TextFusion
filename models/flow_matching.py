# models/flow_matching.py
import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel


class FlowMatching2D(DiffusionModel):
    def __init__(self, network: nn.Module, clip_samples: bool = True):
        super().__init__(network=network, schedule=None)
        self.clip_samples = clip_samples

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand(batch_size, device=device)

        x1 = torch.randn_like(x0)

        t_expanded = t.unsqueeze(1)
        x_t = (1 - t_expanded) * x0 + t_expanded * x1

        v_true = x1 - x0

        v_pred = self.network(x_t, t)

        loss = nn.functional.mse_loss(v_pred, v_true)

        loss = loss / x0.shape[1]

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps: int = 50,
        return_trajectory: bool = False,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self.eval()

        x = torch.randn(n_samples, 2, device=device)

        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        trajectory = [x.clone()] if return_trajectory else None

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((n_samples,), t_current, device=device)

            v_pred = self.network(x, t)

            # Euler step: x_{t+dt} = x_t + dt * v_θ(x_t, t)
            x = x + dt * v_pred

            if self.clip_samples:
                x = torch.clamp(x, -10, 10)

            if return_trajectory:
                trajectory.append(x.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory

    @torch.no_grad()
    def sample_ode(self, *args, **kwargs):
        return self.sample(*args, **kwargs)
