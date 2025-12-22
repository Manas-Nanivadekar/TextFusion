import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from pathlib import Path
from typing import Optional, Tuple


class MNISTDataset(Dataset):
    def __init__(
        self,
        root: str = "data/mnist",
        train: bool = True,
        download: bool = True,
        normalize: bool = True,
    ):
        self.normalize = normalize

        if normalize:
            transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize([0.5], [0.5]),
                ]
            )
        else:
            transform = transforms.ToTensor()

        self.dataset = datasets.MNIST(
            root=root, train=train, transform=transform, download=download
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        return self.dataset[idx]

    def get_sample_batch(self, n: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
        indices = torch.randint(0, len(self), (n,))
        images = []
        labels = []

        for idx in indices:
            img, label = self[idx]
            images.append(img)
            labels.append(label)

        return torch.stack(images), torch.tensor(labels)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Testing MNIST Dataset...")

    dataset = MNISTDataset(train=True, download=True)

    print(f"Dataset size: {len(dataset)}")

    img, label = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Image range: [{img.min():.2f}, {img.max():.2f}]")
    print(f"Label: {label}")

    images, labels = dataset.get_sample_batch(n=16)

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    for idx, (ax, img, label) in enumerate(zip(axes.flat, images, labels)):
        img_display = (img.squeeze() + 1) / 2
        ax.imshow(img_display, cmap="gray")
        ax.set_title(f"Label: {label}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/mnist_samples.png", dpi=150)
    print("\nSaved sample visualization to outputs/mnist_samples.png")
    print("\nMNIST dataset works!")
