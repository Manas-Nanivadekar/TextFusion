"""
U-Net architecture for image diffusion models

Based on:
- Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation
- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Dhariwal & Nichol (2021): Diffusion Models Beat GANs on Image Synthesis
"""

import torch
import torch.nn as nn
import math
from typing import List


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal time embedding"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / max(half_dim - 1, 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -embeddings
        )
        embeddings = t[:, None].float() * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with BatchNorm (more stable than GroupNorm)"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_channels))

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),  # ← BatchNorm instead of GroupNorm
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),  # ← BatchNorm instead of GroupNorm
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.block1(x)
        time_emb = self.time_mlp(t_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.BatchNorm2d(channels)  # ← BatchNorm instead of GroupNorm
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)

        qkv = qkv.reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = (C // self.num_heads) ** -0.5
        attn = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale, dim=-1)

        h = torch.matmul(attn, v)
        h = h.permute(0, 1, 3, 2).reshape(B, C, H, W)
        h = self.proj(h)
        return x + h


class UNet(nn.Module):
    """U-Net with BatchNorm for stability"""

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,  # ← Fixed: plural
        time_emb_dim: int = 256,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        # Initial convolution
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        # Encoder
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels_block = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels_block, time_emb_dim, dropout
                    )
                )
                now_channels = out_channels_block
                channels.append(now_channels)

            if i != len(channel_multipliers) - 1:
                self.downs.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                channels.append(now_channels)

        # Bottleneck
        self.mid_block1 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )
        self.mid_attn = AttentionBlock(now_channels) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )

        # Decoder
        self.ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels_block = base_channels * mult

            for j in range(num_res_blocks + 1):
                self.ups.append(
                    ResidualBlock(
                        now_channels + channels.pop(),
                        out_channels_block,
                        time_emb_dim,
                        dropout,
                    )
                )
                now_channels = out_channels_block

            if i != 0:
                self.ups.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, kernel_size=4, stride=2, padding=1
                    )
                )

        # Final convolution
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(now_channels),  # ← BatchNorm instead of GroupNorm
            nn.SiLU(),
            nn.Conv2d(now_channels, self.out_channels, kernel_size=3, padding=1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Careful weight initialization"""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time embedding
        t_emb = self.time_embedding(t)

        # Initial conv
        h = self.init_conv(x)

        # Encoder
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            skips.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder
        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        # Final
        h = self.final_conv(h)

        return h


if __name__ == "__main__":
    print("Testing U-Net with BatchNorm...")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=2,
    )

    # Test batch_size=4
    x = torch.randn(4, 1, 28, 28)
    t = torch.rand(4)

    out = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Output range: [{out.min():.3f}, {out.max():.3f}]")
    print(f"Has NaN: {torch.isnan(out).any()}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    if torch.isnan(out).any():
        print("\n❌ Still has NaN!")
    else:
        print("\n✓ U-Net works with BatchNorm!")
