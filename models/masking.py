"""
Masking schedules and utilities for D3PM discrete diffusion.

Implements various corruption schedules that control how aggressively tokens
are masked during the forward diffusion process. Analogous to noise schedules
in continuous diffusion (DDPM).
"""

import torch
import numpy as np
from typing import Tuple


class MaskSchedule:
    """
    Controls mask ratio as a function of diffusion timestep.

    Maps timestep t ∈ [0,1] to mask_ratio ∈ [0,1], where:
    - t=0: mask_ratio=0 (no masking, clean data)
    - t=1: mask_ratio=1 (fully masked)

    Available schedules:
    - 'linear': mask_ratio = t (uniform corruption)
    - 'cosine': mask_ratio = cos((1-t)π/2) (more aggressive early masking)
    - 'sqrt': mask_ratio = √t (less aggressive early masking)

    Choice of schedule affects learning dynamics and sample quality,
    similar to noise schedules in DDPM (Improved DDPM, Nichol & Dhariwal 2021).
    """

    def __init__(self, schedule_type: str = "linear"):
        self.schedule_type = schedule_type

    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute mask ratio for given timestep(s).

        Args:
            t: Timesteps in [0, 1], shape (batch,) or scalar

        Returns:
            Mask ratios in [0, 1], same shape as t
        """
        if self.schedule_type == "linear":
            return t
        elif self.schedule_type == "cosine":
            # Cosine schedule: more aggressive early in training
            # Inspired by cosine noise schedule from Improved DDPM
            return torch.cos((1 - t) * np.pi / 2)
        elif self.schedule_type == "sqrt":
            # Square root: less aggressive early, more gradual
            return torch.sqrt(t)
        else:
            raise ValueError(f"Unknown schedule: {self.schedule_type}")


def apply_mask(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    schedule: MaskSchedule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply random masking to tokens according to schedule (forward diffusion).

    Implements the corruption process q(x_t | x_0) in D3PM:
    - Randomly select mask_ratio * num_tokens positions
    - Replace selected tokens with [MASK] token
    - Never mask padding tokens

    Unlike BERT masking, positions are chosen uniformly at random (not biased
    toward any pattern). This is the discrete analog of adding Gaussian noise
    in continuous diffusion.

    Args:
        tokens: (batch, seq_len) original token IDs
        t: (batch,) corruption timesteps in [0, 1]
        mask_token_id: Token ID to use for masking
        pad_token_id: Padding token ID (never masked)
        schedule: MaskSchedule defining mask_ratio(t)

    Returns:
        masked_tokens: (batch, seq_len) tokens with masking applied
        mask: (batch, seq_len) boolean mask indicating which positions were masked
    """
    batch_size, seq_len = tokens.shape
    device = tokens.device

    mask_ratio = schedule.get_mask_ratio(t)

    is_not_pad = tokens != pad_token_id

    masked_tokens = tokens.clone()
    mask = torch.zeros_like(tokens, dtype=torch.bool)

    # Per-sample masking (each sample may have different sequence length due to padding)
    for i in range(batch_size):
        valid_position = torch.where(is_not_pad[i])[0]
        num_valid = len(valid_position)

        if num_valid == 0:
            continue

        num_to_mask = int(num_valid * mask_ratio[i].item())

        if num_to_mask == 0:
            continue

        # Random uniform selection of positions to mask
        # Note: uses randperm on smaller array for efficiency
        perm = torch.randperm(num_valid, device=device)
        position_to_mask = valid_position[perm[:num_to_mask]]

        masked_tokens[i, position_to_mask] = mask_token_id
        mask[i, position_to_mask] = True

    return masked_tokens, mask


def unmask_top_k(
    tokens: torch.Tensor, logits: torch.Tensor, k: int, mask_token_id: int
) -> torch.Tensor:
    """
    Unmask top-k most confident predictions (reverse diffusion step).

    Implements one step of the iterative sampling process p(x_{t-1} | x_t):
    1. Compute prediction confidence for each masked position
    2. Select k positions with highest confidence
    3. Replace [MASK] with predicted tokens at those positions

    This confidence-based ordering is a key innovation of D3PM - the model
    learns to generate high-confidence tokens first (e.g., function names)
    and low-confidence tokens last (e.g., variable names dependent on context).

    Unlike DDPM which samples stochastically, D3PM uses deterministic unmasking
    for simplicity. Stochastic variants exist but are less common.

    Args:
        tokens: (batch, seq_len) current state with [MASK] tokens
        logits: (batch, seq_len, vocab_size) model predictions
        k: Number of tokens to unmask in this step
        mask_token_id: Token ID representing [MASK]

    Returns:
        (batch, seq_len) tokens with k highest-confidence masks replaced
    """
    batch_size = tokens.shape[0]
    device = tokens.device

    # Compute prediction confidence (max softmax probability)
    probs = torch.softmax(logits, dim=-1)
    confidence, predicted_tokens = probs.max(dim=-1)

    # Only consider confidence at masked positions
    is_masked = tokens == mask_token_id
    confidence = confidence.masked_fill(~is_masked, -float("inf"))

    unmask = tokens.clone()

    # Per-sample top-k selection
    for i in range(batch_size):
        masked_positions = torch.where(is_masked[i])[0]
        if len(masked_positions) == 0:
            continue

        # Handle case where k > number of remaining masks
        num_unmask = min(k, len(masked_positions))
        _, top_k_idx = torch.topk(confidence[i, masked_positions], num_unmask)
        position_to_unmask = masked_positions[top_k_idx]

        # Replace [MASK] with predicted token at selected positions
        unmask[i, position_to_unmask] = predicted_tokens[i, position_to_unmask]

    return unmask


if __name__ == "__main__":
    print("Testing masking utilities...")

    # Test schedule
    schedule = MaskSchedule("linear")
    t = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    ratios = schedule.get_mask_ratio(t)
    print(f"Linear schedule: {ratios.numpy()}")

    # Test apply_mask
    tokens = torch.tensor(
        [[3, 100, 200, 300, 400, 4, 0, 0], [3, 150, 250, 350, 450, 550, 4, 0]]
    )

    t = torch.tensor([0.5, 0.5])
    masked, mask = apply_mask(
        tokens, t, mask_token_id=1, pad_token_id=0, schedule=schedule
    )

    print(f"\nOriginal: {tokens[0].numpy()}")
    print(f"Masked:   {masked[0].numpy()}")
    print(f"Mask:     {mask[0].numpy()}")

    print("\nMasking works!")
