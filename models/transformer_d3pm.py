"""
Transformer backbone for D3PM discrete diffusion models.

Implements time-conditioned transformer architecture for predicting original
tokens from masked sequences. Uses pre-LayerNorm architecture and sinusoidal
time embeddings following modern best practices.
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional encoding for continuous timesteps.

    Maps scalar timestep t ∈ [0,1] to high-dimensional vector using frequencies
    borrowed from the Transformer "Attention is All You Need" paper. Provides
    smooth interpolation across timesteps without learned parameters.

    Args:
        dim: Embedding dimension (must be even)
    """

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: (batch,) timesteps in [0, 1]

        Returns:
            (batch, dim) sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2

        # Compute frequencies: 1, 1/10000^(1/d), ..., 1/10000^((d-1)/d)
        # Same as original Transformer positional encoding
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb
        )

        # Scale timesteps by frequencies and apply sin/cos
        emb = t[:, None].float() * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb


class TransformerD3PM(nn.Module):
    """
    Time-conditioned Transformer for D3PM token prediction.

    Predicts original tokens from masked sequences conditioned on corruption timestep.
    Uses bidirectional self-attention (unlike autoregressive models) to leverage
    full context when denoising.

    Architecture choices:
    - Pre-LayerNorm (norm_first=True): More stable than post-norm for deep models
      See: "On Layer Normalization in the Transformer Architecture" (Xiong et al., 2020)
    - GELU activation: Standard for modern transformers (GPT, BERT)
    - Additive time conditioning: Time embedding added to all token embeddings

    Args:
        vocab_size: Total vocabulary size including special tokens (PAD, MASK, etc.)
        embed_dim: Token embedding dimension (typical: 256-512 for code, 768+ for text)
        num_layers: Number of transformer layers (6-12 for most tasks)
        num_heads: Number of attention heads (must divide embed_dim)
        ff_dim: Feedforward hidden dimension (typically 4x embed_dim)
        max_seq_len: Maximum sequence length for positional embeddings
        dropout: Dropout probability for regularization
        time_emb_dim: Dimension of sinusoidal time embeddings before projection
    """

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

        # Learned absolute positional embeddings (not sinusoidal)
        # Allows model to learn position-dependent masking patterns
        self.pos_embedding = nn.Parameter(torch.randn(1, max_seq_len, embed_dim))

        # Project continuous timestep to embedding space
        # MLP with expansion (4x) follows U-Net time conditioning design
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

        # Pre-LayerNorm transformer (norm_first=True)
        # More stable gradients than post-norm for deep networks
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

        # Output projection to vocab logits
        # LayerNorm before projection improves stability
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim), nn.Linear(embed_dim, vocab_size)
        )

        self._init_weights()

    def _init_weights(self):
        """
        Initialize weights following GPT-2 / BERT conventions.

        - Xavier for linear layers (maintains variance across layers)
        - Small Gaussian for embeddings (std=0.02 from BERT paper)
        """
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
        """
        Predict original tokens from masked input conditioned on timestep.

        Args:
            tokens: (batch, seq_len) token IDs (may contain MASK tokens)
            t: (batch,) corruption timesteps in [0, 1]
            padding_mask: (batch, seq_len) True for padding positions to ignore

        Returns:
            (batch, seq_len, vocab_size) logits over vocabulary for each position
        """
        batch_size, seq_len = tokens.shape

        # Token embeddings for input (including MASK tokens)
        x = self.token_embedding(tokens)

        # Add learned positional embeddings
        x = x + self.pos_embedding[:, :seq_len, :]

        # Add time conditioning to all positions
        # Broadcasts (batch, embed_dim) → (batch, 1, embed_dim) → (batch, seq_len, embed_dim)
        t_emb = self.time_mlp(t)
        x = x + t_emb.unsqueeze(1)

        # Bidirectional self-attention (not causal)
        # Padding positions masked out to prevent attention to padding
        x = self.transformer(x, src_key_padding_mask=padding_mask)

        # Project to vocabulary logits
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
