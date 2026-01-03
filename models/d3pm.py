import torch
import torch.nn as nn
from typing import Tuple, Optional
from .base import DiffusionModel
from .masking import MaskSchedule, apply_mask, unmask_top_k


class D3PM(DiffusionModel):
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
        batch_size = tokens.shape[0]
        device = tokens.device

        if t is None:
            t = torch.rand(batch_size, device=device)

        masked_tokens, mask = apply_mask(
            tokens, t, self.mask_token_id, self.pad_token_id, self.schedule
        )

        padding_mask = tokens == self.pad_token_id

        logits = self.network(masked_tokens, t, padding_mask)

        logits_flat = logits.view(-1, logits.size(-1))
        targets_flat = tokens.view(-1)
        mask_flat = mask.view(-1)

        loss_per_token = nn.functional.cross_entropy(
            logits_flat, targets_flat, reduction="none"
        )

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
        self.eval()

        tokens = torch.full(
            (n_samples, seq_len), self.mask_token_id, dtype=torch.long, device=device
        )
        trajectory = [tokens.clone()] if return_trajectory else None
        timesteps = torch.linspace(1.0, 0.0, n_steps + 1, device=device)
        for i in range(n_steps):
            t_current = timesteps[i]
            t_next = timesteps[i + 1]

            mask_ratio_current = self.schedule.get_mask_ratio(
                torch.tensor([t_current], device=device)
            ).item()
            mask_ratio_next = self.schedule.get_mask_ratio(
                torch.tensor([t_next], device=device)
            ).item()

            num_masked_current = int(seq_len * mask_ratio_current)
            num_masked_next = int(seq_len * mask_ratio_next)
            k = max(num_masked_current - num_masked_next, 1)

            t_batch = torch.full((n_samples,), t_current, device=device)
            logits = self.network(tokens, t_batch)

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
