import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerD3PM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 256,
        num_layers: int = 6,
        num_heads: int = 8,
        ff_dim: int = 1024,
        max_seq_len: int = 128,
        dropout: float = 0.1,
        time_emb_dim: int = 256,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)

        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self, tokens: torch.Tensor, t: torch.Tensor, padding_mask: torch.Tensor = None
    ) -> torch.Tensor:
        batch_size, seq_len = tokens.shape

        x = self.token_embedding(tokens)

        x = x + self.pos_embedding[:, :seq_len, :]

        t_emb = self.time_mlp(t)
        x = x + t_emb.unsqueeze(1)

        x = self.transformer(x, src_key_padding_mask=padding_mask)

        logits = self.output_proj(x)

        return logits


if __name__ == "__main__":
    print("Testing Transformer D3PM...")

    model = TransformerD3PM(vocab_size=5000, embed_dim=256, num_layers=6, num_heads=8)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    batch_size = 4
    seq_len = 32

    tokens = torch.randint(0, 5000, (batch_size, seq_len))
    t = torch.rand(batch_size)

    logits = model(tokens, t)

    print(f"\nInput tokens: {tokens.shape}")
    print(f"Time: {t.shape}")
    print(f"Output logits: {logits.shape}")
    print(f"Logits range: [{logits.min():.2f}, {logits.max():.2f}]")

    predicted_tokens = logits.argmax(dim=-1)
    print(f"Predicted tokens: {predicted_tokens.shape}")

    print("\nTransformer works!")
