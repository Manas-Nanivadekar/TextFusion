import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class SimpleMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 2,
        hidden_dim: int = 128,
        time_embed_dim: int = 32,
        num_layers: int = 3,
        output_dim: Optional[int] = None,
    ):
        super().__init__()

        if output_dim is None:
            output_dim = input_dim

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.time_embed_dim = time_embed_dim
        self.output_dim = output_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        layers = []

        layers.extend([nn.Linear(input_dim + hidden_dim, hidden_dim), nn.SiLU()])

        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.SiLU()])

        layers.append(nn.Linear(hidden_dim, output_dim))

        self.network = nn.Sequential(*layers)

        self.register_buffer(
            "time_freqs",
            torch.exp(torch.linspace(0, -np.log(10000), time_embed_dim // 2)),
        )

    def get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        t = t.unsqueeze(-1)

        freqs = self.time_freqs.unsqueeze(0)
        args = t * freqs * 2 * np.pi

        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

        return embedding

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.get_time_embedding(t)
        t_emb = self.time_mlp(t_emb)

        h = torch.cat([x, t_emb], dim=-1)

        return self.network(h)
