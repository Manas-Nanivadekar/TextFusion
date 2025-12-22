import torch
import torch.nn as nn
from typing import List, Optional
from .unet import SinusoidalPositionEmbedding, AttentionBlock


class ConditionalResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        class_emb_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cond_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_emb_dim + class_emb_dim, out_channels)
        )

        self.block1 = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )

        self.block2 = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        )

        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(
        self, x: torch.Tensor, t_emb: torch.Tensor, c_emb: torch.Tensor
    ) -> torch.Tensor:
        h = self.block1(x)

        # Combine time and class embeddings
        cond_emb = torch.cat([t_emb, c_emb], dim=-1)
        cond_emb = self.cond_mlp(cond_emb)
        h = h + cond_emb[:, :, None, None]

        h = self.block2(h)

        return h + self.residual_conv(x)


class ConditionalUNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_multipliers: List[int] = [1, 2, 4],
        num_res_blocks: int = 2,
        time_emb_dim: int = 256,
        num_classes: int = 10,
        class_emb_dim: int = 128,
        use_attention: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_classes = num_classes

        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim * 4),
            nn.SiLU(),
            nn.Linear(time_emb_dim * 4, time_emb_dim),
        )

        self.class_embedding = nn.Embedding(num_classes + 1, class_emb_dim)

        self.init_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)

        self.downs = nn.ModuleList()
        channels = [base_channels]
        now_channels = base_channels

        for i, mult in enumerate(channel_multipliers):
            out_channels_block = base_channels * mult

            for _ in range(num_res_blocks):
                self.downs.append(
                    ConditionalResidualBlock(
                        now_channels,
                        out_channels_block,
                        time_emb_dim,
                        class_emb_dim,
                        dropout,
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

        self.mid_block1 = ConditionalResidualBlock(
            now_channels, now_channels, time_emb_dim, class_emb_dim, dropout
        )
        self.mid_attn = AttentionBlock(now_channels) if use_attention else nn.Identity()
        self.mid_block2 = ConditionalResidualBlock(
            now_channels, now_channels, time_emb_dim, class_emb_dim, dropout
        )

        self.ups = nn.ModuleList()

        for i, mult in reversed(list(enumerate(channel_multipliers))):
            out_channels_block = base_channels * mult

            for j in range(num_res_blocks + 1):
                self.ups.append(
                    ConditionalResidualBlock(
                        now_channels + channels.pop(),
                        out_channels_block,
                        time_emb_dim,
                        class_emb_dim,
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

        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(now_channels),
            nn.SiLU(),
            nn.Conv2d(now_channels, self.out_channels, kernel_size=3, padding=1),
        )

        self._init_weights()

    def _init_weights(self):
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
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, c: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size = x.shape[0]

        t_emb = self.time_embedding(t)

        if c is None:
            c = torch.full(
                (batch_size,), self.num_classes, device=x.device, dtype=torch.long
            )

        c_emb = self.class_embedding(c)

        h = self.init_conv(x)

        skips = [h]
        for layer in self.downs:
            if isinstance(layer, ConditionalResidualBlock):
                h = layer(h, t_emb, c_emb)
            else:
                h = layer(h)
            skips.append(h)

        h = self.mid_block1(h, t_emb, c_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, c_emb)

        for layer in self.ups:
            if isinstance(layer, ConditionalResidualBlock):
                h = torch.cat([h, skips.pop()], dim=1)
                h = layer(h, t_emb, c_emb)
            else:
                h = layer(h)

        h = self.final_conv(h)

        return h


if __name__ == "__main__":
    print("Testing Conditional U-Net...")

    model = ConditionalUNet(
        in_channels=1,
        out_channels=1,
        base_channels=64,
        channel_multipliers=[1, 2, 4],
        num_res_blocks=2,
        num_classes=10,
        class_emb_dim=128,
    )

    # Test
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    t = torch.rand(batch_size)
    c = torch.randint(0, 10, (batch_size,))

    out_cond = model(x, t, c)
    out_uncond = model(x, t, c=None)

    print(f"Input shape: {x.shape}")
    print(
        f"Conditional output: {out_cond.shape}, range=[{out_cond.min():.2f}, {out_cond.max():.2f}]"
    )
    print(
        f"Unconditional output: {out_uncond.shape}, range=[{out_uncond.min():.2f}, {out_uncond.max():.2f}]"
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\n✓ Conditional U-Net works!")
