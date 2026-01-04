"""
Denoising Diffusion Probabilistic Models (DDPM) for continuous data.

Implements the seminal diffusion model from:
Ho et al. "Denoising Diffusion Probabilistic Models" (2020)
https://arxiv.org/abs/2006.11239

Unlike D3PM which operates on discrete tokens, DDPM works on continuous data
(images, 2D points, etc.) by adding Gaussian noise. This implementation includes
both stochastic sampling (DDPM) and deterministic ODE sampling (DDIM-style).
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel
from training.schedulers import NoiseSchedule


class DDPM2D(DiffusionModel):
    """
    DDPM for continuous 2D data (extensible to higher dimensions).

    Forward process: Gradually add Gaussian noise according to schedule
    Reverse process: Iteratively denoise using learned noise prediction

    Training: Predict noise ε added at each timestep (ε-prediction)
    Sampling: Remove predicted noise iteratively (stochastic or deterministic)

    Args:
        network: Model that predicts noise ε given noisy input and timestep
        schedule_type: Noise schedule ('linear', 'cosine')
        timesteps: Number of discrete diffusion steps (typically 1000)
        beta_start: Minimum noise level (typical: 1e-4)
        beta_end: Maximum noise level (typical: 0.02)
        clip_denoised: Whether to clamp predictions for numerical stability
    """
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
        """
        DDPM training objective: predict noise added to clean data.

        Uses the simplified loss from Ho et al. (2020):
            L = E_t,ε [ ||ε - ε_θ(√ᾱ_t x₀ + √(1-ᾱ_t) ε, t)||² ]

        This is simpler than the full variational bound and works better empirically.

        Args:
            x0: (batch, *dims) clean data samples

        Returns:
            Scalar MSE loss between predicted and true noise
        """
        batch_size = x0.shape[0]
        device = x0.device

        # Sample random timesteps uniformly
        t = torch.rand(batch_size, device=device)
        alpha_t, sigma_t = self.schedule.get_alpha_sigma(t)
        alpha_t = alpha_t.unsqueeze(1)
        sigma_t = sigma_t.unsqueeze(1)

        # Sample noise and create noisy input: x_t = α_t x₀ + σ_t ε
        epsilon = torch.randn_like(x0)
        x_t = alpha_t * x0 + sigma_t * epsilon

        # Predict noise
        epsilon_pred = self.forward(x_t, t)

        # Simple MSE loss (simplified objective from Ho et al.)
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
        """
        Stochastic ancestral sampling (original DDPM sampling).

        Iteratively removes noise while adding controlled randomness to maintain
        the correct distribution. This is the original DDPM sampler from Ho et al.

        Process:
        1. Start from pure noise (t=1.0)
        2. Predict and remove noise to estimate x₀
        3. Re-noise to t_next level (except final step)
        4. Repeat until t=0

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of denoising steps (fewer = faster, lower quality)
            return_trajectory: If True, return all intermediate states
            device: Device for sampling

        Returns:
            samples: (n_samples, *dims) generated samples
            trajectory: (n_steps+1, n_samples, *dims) if return_trajectory, else None
        """
        self.eval()

        # Start from pure Gaussian noise (t=1.0)
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

            # Predict noise
            epsilon_pred = self.forward(x, t)

            # Estimate clean data: x₀ = (x_t - σ_t ε) / α_t
            x0_pred = (x - sigma_t * epsilon_pred) / alpha_t

            # Clamp for numerical stability (prevents runaway values)
            if self.clip_denoised:
                x0_pred = torch.clamp(x0_pred, -10, 10)

            # Re-noise to next timestep (stochastic sampling)
            if t_next > 0:
                t_next_tensor = torch.full((n_samples,), t_next, device=device)
                alpha_next, sigma_next = self.schedule.get_alpha_sigma(t_next_tensor)
                alpha_next = alpha_next.unsqueeze(1)
                sigma_next = sigma_next.unsqueeze(1)

                # Add noise scaled to next timestep
                noise = torch.randn_like(x)
                x = alpha_next * x0_pred + sigma_next * noise
            else:
                # Final step: no re-noising
                x = x0_pred

            if return_trajectory:
                trajectory.append(x.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory

    @torch.no_grad()
    def sample_ode(
        self,
        n_samples: int,
        n_steps: int = 50,
        return_trajectory: bool = False,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Deterministic ODE sampling (probability flow ODE).

        Samples from the same distribution as DDPM but deterministically by solving
        the probability flow ODE. Similar to DDIM (Song et al. 2020) but using
        continuous-time formulation.

        Benefits over stochastic sampling:
        - Deterministic (same noise → same output)
        - Can use fewer steps with less quality degradation
        - Enables latent space interpolation

        Solves: dx/dt = (dα/dt / α) x - (dσ/dt) ε_θ(x, t)

        Args:
            n_samples: Number of samples to generate
            n_steps: Number of integration steps
            return_trajectory: If True, return all intermediate states
            device: Device for sampling

        Returns:
            samples: (n_samples, *dims) generated samples
            trajectory: (n_steps+1, n_samples, *dims) if return_trajectory, else None
        """
        self.eval()

        x = torch.randn(n_samples, 2, device=device)

        timesteps = torch.linspace(1, 0, n_steps + 1, device=device)

        trajectory = [x.clone() if return_trajectory else None]

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]
            dt = t_next - t_current

            t = torch.full((n_samples,), t_current, device=device)

            alpha_t, sigma_t = self.schedule.get_alpha_sigma(t)
            alpha_t = alpha_t.unsqueeze(1)
            sigma_t = sigma_t.unsqueeze(1)

            epsilon_pred = self.forward(x, t)

            # Compute derivatives dα/dt and dσ/dt using finite differences
            if t_next > 0:
                t_next_tensor = torch.full((n_samples,), t_next, device=device)
                alpha_next, sigma_next = self.schedule.get_alpha_sigma(t_next_tensor)
                alpha_next = alpha_next.unsqueeze(1)
                sigma_next = sigma_next.unsqueeze(1)

                # Forward finite difference
                dalpha_dt = (alpha_next - alpha_t) / dt
                dsigma_dt = (sigma_next - sigma_t) / dt
            else:
                # Backward finite difference for final step
                eps = 0.001
                t_prev = torch.full((n_samples,), t_current - eps, device=device)
                alpha_prev, sigma_prev = self.schedule.get_alpha_sigma(t_prev)
                alpha_prev = alpha_prev.unsqueeze(1)
                sigma_prev = sigma_prev.unsqueeze(1)

                dalpha_dt = (alpha_t - alpha_prev) / eps
                dsigma_dt = (sigma_t - sigma_prev) / eps

            # Probability flow ODE: dx/dt = (dα/dt / α) x - (dσ/dt) ε
            dx_dt = (dalpha_dt / alpha_t) * x - dsigma_dt * epsilon_pred

            # Euler integration step
            x = x + dt * dx_dt

            if return_trajectory:
                trajectory.append(x.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return x, trajectory
