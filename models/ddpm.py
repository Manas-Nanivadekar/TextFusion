import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel
from training.schedulers import NoiseSchedule


class DDPM2D(DiffusionModel):
    def __init__(
        self,
        network: nn.Module,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        clip_denoised: bool = True,
    ):
        schedule = NoiseSchedule(
            schedule_type=schedule_type,
            timesteps=timesteps,
            beta_start=beta_start,
            beta_end=beta_end,
        )

        super().__init__(network=network, schedule=schedule)

        self.clip_denoised = clip_denoised
        self.timestamps = timesteps

    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        batch_size = x0.shape[0]
        device = x0.device

        t = torch.rand(batch_size, device=device)
        alpha_t, sigma_t = self.schedule.get_alpha_sigma(t)
        alpha_t = alpha_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)

        epsilon = torch.randn_like(x0)

        x_t = alpha_t * x0 + sigma_t * epsilon
        epsilon_pred = self.forward(x_t, t)
        loss = nn.functional.mse_loss(epsilon_pred, epsilon)

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        n_steps: Optional[int] = 50,
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

            t = torch.full((n_samples,), t_current, device=device)

            alpha_t, sigma_t = self.schedule.get_alpha_sigma(t)
            alpha_t = alpha_t.unsqueeze(1)
            sigma_t = sigma_t.unsqueeze(1)

            epsilon_pred = self.forward(x, t)

            x0_pred = (x - sigma_t * epsilon_pred) / alpha_t

            if self.clip_denoised:
                x0_pred = torch.clamp(x0_pred, -10, 10)

            if t_next > 0:
                t_next_tensor = torch.full((n_samples,), t_next, device=device)
                alpha_next, sigma_next = self.schedule.get_alpha_sigma(t_next_tensor)
                alpha_next = alpha_next.unsqueeze(1)
                sigma_next = sigma_next.unsqueeze(1)

                noise = torch.randn_like(x)
                x = alpha_next * x0_pred + sigma_next * noise
            else:
                x = x0_pred
            if return_trajectory:
                trajectory.append(x.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory
