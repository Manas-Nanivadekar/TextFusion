from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional


class DiffusionModel(ABC, nn.Module):
    def __init__(self, network: nn.Module, schedule):
        super().__init__()
        self.network = network
        self.schedule = schedule

    @abstractmethod
    def compute_loss(self, x0: torch.Tensor) -> torch.Tensor:
        """
        Compute the training loss
        Args:
            x0: Clean data samples, shape (batch_size, data_dim)
        Returns:
            loss: Scalar loss value
        """
        pass

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        n_steps: Optional[int] = None,
        return_trajectory: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Generate samples from the model
        Args:
            n_samples: Number of samples to generate
            n_steps: Number of sampling steps (if None, use default)
            return_trajectory: Whether to return full denoising trajectory
        Returns:
            samples: Generated samples, shape (n_samples, data_dim)
            trajectory: Optional trajectory, shape (n_steps+1, n_samples, data_dim)
        """
        pass

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network

        Args:
            x: Input data, shape (batch_size, data_dim)
            t: Time steps, shape (batch_size,)

        Returns:
            output: Network prediction, shape (batch_size, data_dim)
        """
        return self.network(x, t)
