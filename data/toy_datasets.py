import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple


class ToyDataset2D(Dataset):
    def __init__(self, n_points: int = 1000, seed: int = 42):
        self.n_points = n_points
        self.seed = seed
        torch.manual_seed(seed)
        self.data = self.generate_data()

    def generate_data(self) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]

    def get_all_data(self) -> torch.Tensor:
        return self.data


class TwoClusterDataset(ToyDataset2D):
    def __init__(
        self,
        n_points: int = 1000,
        cluster1_center: List[float] = [2.0, 2.0],
        cluster2_center: List[float] = [-2.0, -2.0],
        cluster_std: float = 0.3,
        seed: int = 42,
    ):
        self.cluster1_center = torch.tensor(cluster1_center, dtype=torch.float32)
        self.cluster2_center = torch.tensor(cluster2_center, dtype=torch.float32)
        self.cluster_std = cluster_std
        super().__init__(n_points=n_points, seed=seed)

    def generate_data(self) -> torch.Tensor:
        n_per_cluster = self.n_points // 2

        cluster1 = (
            torch.randn(n_per_cluster, 2) * self.cluster_std + self.cluster1_center
        )
        cluster2 = (
            torch.randn(n_per_cluster, 2) * self.cluster_std + self.cluster2_center
        )

        return torch.cat([cluster1, cluster2], dim=0)


class SwissRollDataset(ToyDataset2D):
    def __init__(self, n_points: int = 1000, noise: float = 0.1, seed: int = 42):
        self.noise = noise
        super().__init__(n_points=n_points, seed=seed)

    def generate_data(self) -> torch.Tensor:
        t = torch.linspace(0, 4 * np.pi, self.n_points)
        t = t + torch.randn_like(t) * 0.1

        x = t * torch.cos(t)
        y = t * torch.sin(t)

        data = torch.stack([x, y], dim=1)
        data = data + torch.randn_like(data) * self.noise

        data = (data - data.mean(dim=0)) / data.std(dim=0) * 2

        return data


class MoonsDataset(ToyDataset2D):
    def __init__(self, n_points: int = 1000, noise: float = 0.1, seed: int = 42):
        self.noise = noise
        super().__init__(n_points=n_points, seed=seed)

    def generate_data(self) -> torch.Tensor:
        n_per_moon = self.n_points // 2

        t1 = torch.linspace(0, np.pi, n_per_moon)
        x1 = torch.cos(t1)
        y1 = torch.sin(t1)
        moon1 = torch.stack([x1, y1], dim=1)

        t2 = torch.linspace(0, np.pi, n_per_moon)
        x2 = 1 - torch.cos(t2)
        y2 = 1 - torch.sin(t2) - 0.5
        moon2 = torch.stack([x2, y2], dim=1)

        data = torch.cat([moon1, moon2], dim=0)
        data = data + torch.randn_like(data) * self.noise

        return data
