# training/schedulers.py
import torch
import math
from typing import Tuple


class NoiseSchedule:
    """Noise schedule for diffusion models"""

    def __init__(
        self,
        schedule_type: str = "linear",
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
    ):
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        t = torch.linspace(0, 1, timesteps)

        if schedule_type == "linear":
            beta = beta_start + (beta_end - beta_start) * t

        elif schedule_type == "cosine":
            # Improved cosine schedule from Nichol & Dhariwal (2021)
            # Paper: "Improved Denoising Diffusion Probabilistic Models"
            s = 0.008
            f_t = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f_t / f_t[0]

            # Compute beta from alpha_bar
            alpha_bar_prev = torch.cat([torch.tensor([1.0]), alpha_bar[:-1]])
            beta = 1 - (alpha_bar / alpha_bar_prev)
            beta = torch.clamp(beta, 0, 0.999)

        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")

        alpha = torch.sqrt(1 - beta)
        alpha_bar = torch.cumprod(alpha, dim=0)
        sigma = torch.sqrt(1 - alpha_bar**2)

        self.alpha_bar = alpha_bar
        self.sigma = sigma
        self.beta = beta

    def get_alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get α_t and σ_t for continuous time t ∈ [0,1]

        Args:
            t: Time steps, shape (batch_size,)

        Returns:
            alpha_t: Alpha values, shape (batch_size,)
            sigma_t: Sigma values, shape (batch_size,)
        """
        # Map continuous t to discrete indices
        idx = (t * (self.timesteps - 1)).long().clamp(0, self.timesteps - 1)
        return self.alpha_bar[idx], self.sigma[idx]

    def visualize(self):
        """Visualize the noise schedule"""
        import matplotlib.pyplot as plt

        t = torch.linspace(0, 1, self.timesteps)

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # Plot alpha_bar and sigma
        axes[0].plot(t.numpy(), self.alpha_bar.numpy(), label="α(t)", linewidth=2)
        axes[0].plot(t.numpy(), self.sigma.numpy(), label="σ(t)", linewidth=2)
        axes[0].plot(
            t.numpy(),
            (self.alpha_bar**2 + self.sigma**2).numpy(),
            "g--",
            label="α² + σ²",
            linewidth=1,
        )
        axes[0].set_xlabel("Time t")
        axes[0].set_ylabel("Value")
        axes[0].set_title(f"Noise Schedule ({self.schedule_type})")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Plot beta
        axes[1].plot(t.numpy(), self.beta.numpy(), linewidth=2, color="purple")
        axes[1].set_xlabel("Time t")
        axes[1].set_ylabel("β(t)")
        axes[1].set_title("Beta Schedule")
        axes[1].grid(True, alpha=0.3)

        # Plot signal-to-noise ratio
        snr = (self.alpha_bar / self.sigma) ** 2
        axes[2].plot(t.numpy(), snr.numpy(), linewidth=2, color="red")
        axes[2].set_xlabel("Time t")
        axes[2].set_ylabel("SNR")
        axes[2].set_title("Signal-to-Noise Ratio")
        axes[2].set_yscale("log")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig
