from __future__ import annotations

import torch
from torchvision import transforms


class AddGaussianNoise:
    def __init__(self, std: float = 0.03) -> None:
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.std <= 0:
            return tensor
        return torch.clamp(tensor + torch.randn_like(tensor) * self.std, 0.0, 1.0)


class MultiViewTransform:
    def __init__(self, transform, views: int) -> None:
        self.transform = transform
        self.views = views

    def __call__(self, image):
        return [self.transform(image) for _ in range(self.views)]


def build_transform(dataset: str, augmentation: str, views: int = 1, train: bool = True):
    steps = []
    if dataset in {"fashion_mnist", "mnist"}:
        steps.append(transforms.Grayscale(num_output_channels=3))
    steps.append(transforms.Resize((32, 32)))

    if train and augmentation in {"weak", "strong"}:
        steps.append(transforms.RandomCrop(32, padding=4, padding_mode="reflect"))
        if dataset not in {"svhn", "mnist"}:
            steps.append(transforms.RandomHorizontalFlip())
    if train and augmentation == "strong":
        steps.append(transforms.ColorJitter(brightness=0.25, contrast=0.25))

    steps.append(transforms.ToTensor())
    if train and augmentation == "strong":
        steps.append(AddGaussianNoise(std=0.03))
    steps.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))

    transform = transforms.Compose(steps)
    if train and views > 1:
        return MultiViewTransform(transform, views)
    return transform
