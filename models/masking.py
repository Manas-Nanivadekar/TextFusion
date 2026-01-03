import torch
import numpy as np
from typing import Tuple


class MaskSchedule:
    def __init__(self, schedule_type: str = "linear"):
        self.schedule_type = schedule_type

    def get_mask_ratio(self, t: torch.Tensor) -> torch.Tensor:
        if self.schedule_type == "linear":
            return t
        elif self.schedule_type == "cosine":
            return torch.cos((1 - t) * np.pi / 2)
        elif self.schedule_type == "sqrt":
            return torch.sqrt(t)
        else:
            raise ValueError(f"Unknown schediule: {self.schedule_type}")


def apply_mask(
    tokens: torch.Tensor,
    t: torch.Tensor,
    mask_token_id: int,
    pad_token_id: int,
    schedule: MaskSchedule,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = tokens.shape
    device = tokens.device

    mask_ratio = schedule.get_mask_ratio(t)

    is_not_pad = tokens != pad_token_id

    masked_tokens = tokens.clone()
    mask = torch.zeros_like(tokens, dtype=torch.bool)

    for i in range(batch_size):
        valid_position = torch.where(is_not_pad[i])[0]
        num_valid = len(valid_position)

        if num_valid == 0:
            continue

        num_to_mask = int(num_valid * mask_ratio[i].item())

        if num_to_mask == 0:
            continue

        perm = torch.randperm(num_to_mask, device=device)
        position_to_mask = valid_position[perm[:num_to_mask]]

        masked_tokens[i, position_to_mask] = mask_token_id
        mask[i, position_to_mask] = True

    return masked_tokens, mask


def unmask_top_k(
    tokens: torch.Tensor, logits: torch.Tensor, k: int, mask_token_id: int
) -> torch.Tensor:
    batch_size = tokens.shape[0]
    device = tokens.device

    probs = torch.softmax(logits, dim=-1)
    confidence, predicted_tokens = probs.max(dim=-1)

    is_masked = tokens == mask_token_id
    confidence = confidence.masked_fill(~is_masked, -float("inf"))

    unmask = tokens.clone()

    for i in range(batch_size):
        masked_positions = torch.where(is_masked[i])[0]
        if len(masked_positions) == 0:
            continue

        num_unmask = min(k, len(masked_positions))
        _, top_k_idx = torch.topk(confidence[i, masked_positions], num_unmask)
        position_to_unmask = masked_positions[top_k_idx]

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
