"""
U-Net architecture for image diffusion models

Based on:
- Ronneberger et al. (2015): U-Net: Convolutional Networks for Biomedical Image Segmentation
- Ho et al. (2020): Denoising Diffusion Probabilistic Models
- Dhariwal & Nichol (2021): Diffusion Models Beat GANs on Image Synthesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch_size,) time steps in [0, 1]
        Returns:
            embedding: (batch_size, dim)
        """
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) + -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
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
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
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
        """
        Args:
            x: (batch, in_channels, H, W)
            t_emb: (batch, time_emb_dim)
        Returns:
            out: (batch, out_channels, H, W)
        """
        h = self.block1(x)

        time_embd = self.time_mlp(t_emb)
        h = h + time_embd[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads

        self.norm = nn.GroupNorm(8, channels)
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
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        num_res_block: int = 2,
        time_emb_dim: int = 256,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )
        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels_blocks = base_channels * mult

            for _ in range(num_res_block):
                self.downs.append(
                    ResidualBlock(
                        now_channels, out_channels_blocks, time_emb_dim, dropout
                    )
                )
                now_channels = out_channels_blocks
                channels.append(now_channels)

            if i != len(channel_multipliers) - 1:
                self.downs.append(
                    nn.Conv2d(
                        now_channels, now_channels, kernel_size=3, stride=2, padding=1
                    )
                )
                channels.append(now_channels)

        self.mid_block1 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )
        self.mid_attn = AttentionBlock(now_channels) if use_attention else nn.Identity()
        self.mid_block2 = ResidualBlock(
            now_channels, now_channels, time_emb_dim, dropout
        )

        self.ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels_blocks = base_channels * mult

            for j in range(num_res_block + 1):
                self.ups.append(
                    ResidualBlock(
                        now_channels + channels.pop(),
                        out_channels_blocks,
                        time_emb_dim,
                        dropout,
                    )
                )
                now_channels = out_channels_blocks

            if i != 0:
                self.ups.append(
                    nn.ConvTranspose2d(
                        now_channels, now_channels, kernel_size=4, stride=2, padding=1
                    )
                )
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, self.out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.GroupNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.time_embedding(t)
        h = self.init_conv(x)
        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
            else:
                h = layer(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        for layer in self.ups:
            if isinstance(layer, ResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb)
            else:
                h = layer(h)

        h = self.final_conv(h)

        return h


if __name__ == "__main__":
    print("Testing U-Net")

    model = UNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_block=2,
    )

    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.randn(batch_size)

    out = model(x, t)

    print(f"Input shape: {x.shape}")
    print(f"Time shape: {t.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    assert out.shape == x.shape, "Output shape should match input shape"
    print("U-Net works!")
