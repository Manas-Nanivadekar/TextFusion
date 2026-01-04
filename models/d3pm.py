"""
Discrete Diffusion Probabilistic Models (D3PM) for text generation.

Implements the masking-based corruption process from:
Austin et al. "Structured Denoising Diffusion Models in Discrete State Spaces" (2021)
https://arxiv.org/abs/2107.03006

Key differences from continuous diffusion (DDPM):
- Uses discrete masking instead of Gaussian noise
- Forward process replaces tokens with [MASK] token
- Reverse process unmasks iteratively based on model confidence
- Learns to predict original tokens directly (not noise)
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel
from .masking import MaskSchedule, apply_mask, unmask_top_k


class D3PM(DiffusionModel):
    """
    D3PM diffusion model for discrete token sequences (text, code, etc.).

    Forward process: Progressively mask tokens according to schedule
    Reverse process: Iteratively unmask using confidence-based selection

    Args:
        network: Model that predicts original tokens from masked input (e.g., Transformer)
        mask_token_id: Special token ID used for masking (analogous to noise in DDPM)
        pad_token_id: Padding token ID (excluded from masking and loss computation)
        schedule_type: Corruption schedule ('linear', 'cosine', 'sqrt')
    """
    def __init__(
        self,
        network: nn.Module,
        mask_token_id: int,
        pad_token_id: int,
        schedule_type: str = "linear",
    ):
        super().__init__(network=network, schedule=None)
        self.mask_token_id = mask_token_id
        self.pad_token_id = pad_token_id
        self.schedule = MaskSchedule(schedule_type)

    def compute_loss(self, tokens: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        D3PM training objective: predict original tokens from masked input.

        Loss computed only on masked positions (BERT-style) rather than all positions.
        This focuses gradient signal on the denoising task and is more sample-efficient
        than predicting all tokens.

        Args:
            tokens: (batch, seq_len) ground truth token IDs
            t: (batch,) corruption timesteps in [0,1]. If None, sampled uniformly.
               t=0: no masking, t=1: maximum masking per schedule

        Returns:
            Scalar loss averaged over masked positions only
        """
        batch_size = tokens.shape[0]
        device = tokens.device

        # Sample random timesteps uniformly (not importance-sampled for simplicity)
        if t is None:
            t = torch.rand(batch_size, device=device)

        # Apply forward corruption: mask tokens according to schedule
        masked_tokens, mask = apply_mask(
            tokens, t, self.mask_token_id, self.pad_token_id, self.schedule
        )

        padding_mask = tokens == self.pad_token_id

        # Predict original tokens from corrupted input
        logits = self.network(masked_tokens, t, padding_mask)

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = tokens.view(-1)
        mask_flat = mask.view(-1)

        loss_per_token = nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )

        # Loss only on masked positions - key difference from continuous diffusion
        # where loss is computed everywhere
        masked_loss = (loss_per_token * mask_flat.float()).sum()
        num_masked = mask_flat.sum()

        if num_masked > 0:
            loss = masked_loss / num_masked
        else:
            loss = masked_loss

        return loss

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        seq_len: int,
        n_steps: int = 50,
        return_trajectory: bool = False,
        device: str = "cpu",
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Iterative demasking sampling (reverse diffusion process).

        Starts from fully masked sequence and progressively unmasks tokens based on
        model confidence. Unlike autoregressive models, unmasking order is learned
        (high-confidence tokens unmasked first), enabling parallel generation.

        Sampling strategy:
        1. Start with all [MASK] tokens (t=1.0)
        2. At each step, predict all token logits
        3. Unmask top-k most confident predictions
        4. Repeat until fully unmasked (t=0.0)

        Args:
            n_samples: Number of sequences to generate
            seq_len: Length of sequences to generate
            n_steps: Number of denoising steps (more steps = higher quality, slower)
            return_trajectory: If True, return all intermediate states
            device: Device to run sampling on

        Returns:
            tokens: (n_samples, seq_len) generated sequences
            trajectory: (n_steps+1, n_samples, seq_len) if return_trajectory, else None
        """
        self.eval()

        # Start from fully masked sequence (t=1.0)
        tokens = torch.full(
            (n_samples, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )
        trajectory = [tokens.clone()] if return_trajectory else None

        # Reverse timesteps: 1.0 (fully masked) → 0.0 (fully unmasked)
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            # Compute how many tokens to unmask at this step
            # Based on difference in mask ratios between consecutive timesteps
            mask_ratio_current = self.schedule.get_mask_ratio(
                torch.tensor([t_current], device=device)
            ).item()
            mask_ratio_next = self.schedule.get_mask_ratio(
                torch.tensor([t_next], device=device)
            ).item()

            num_masked_current = int(seq_len * mask_ratio_current)
            num_masked_next = int(seq_len * mask_ratio_next)
            k = max(num_masked_current - num_masked_next, 1)

            # Predict token logits for current masked state
            t_batch = torch.full((n_samples,), t_current, device=device)
            logits = self.network(tokens, t_batch)

            # Unmask top-k most confident predictions
            # This implements the confidence-based ordering unique to D3PM
            tokens = unmask_top_k(tokens, logits, k, self.mask_token_id)

            if return_trajectory:
                trajectory.append(tokens.clone())

        self.train()

        if return_trajectory:
            trajectory = torch.stack(trajectory)

        return tokens, trajectory


if __name__ == "__main__":
    from .transformer_d3pm import TransformerD3PM

    print("Testing D3PM...")

    vocab_size = 5000
    network = TransformerD3PM(
        vocab_size=vocab_size, embed_dim=128, num_layers=4, num_heads=4
    )

    d3pm = D3PM(
        network=network, mask_token_id=1, pad_token_id=0, schedule_type="linear"
    )

    print(f"Model parameters: {sum(p.numel() for p in d3pm.parameters()):,}")

    print("\nTesting training loss...")
    tokens = torch.randint(5, vocab_size, (4, 32))
    loss = d3pm.compute_loss(tokens)
    print(f"Loss: {loss.item():.4f}")

    print("\nTesting sampling...")
    samples, trajectory = d3pm.sample(
        n_samples=2, seq_len=16, n_steps=10, return_trajectory=True
    )

    print(f"Samples shape: {samples.shape}")
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Sample tokens: {samples[0].tolist()}")

    num_masked = (trajectory == 1).sum(dim=2)
    print(f"\nMasked tokens per step:")
    for step in range(0, len(num_masked), 2):
        print(f"  Step {step}: {num_masked[step, 0].item()} masks")

    print("\nD3PM works!")
