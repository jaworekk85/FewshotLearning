from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 10, normalization: str = "batchnorm") -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)
        if normalization not in {"batchnorm", "none", "groupnorm"}:
            raise ValueError("Unknown CNN normalization.")
        # Replace only after initializing all convolutions and linear layers.
        # Thus shared weights and subsequent RNG draws are identical across arms.
        if normalization != "batchnorm":
            for index, module in enumerate(self.features):
                if isinstance(module, nn.BatchNorm2d):
                    self.features[index] = nn.Identity() if normalization == "none" else nn.GroupNorm(8, module.num_features)

    def forward(self, x: torch.Tensor, normalize: bool = False):
        h = self.features(x).flatten(1)
        z = self.embedding(h)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z, self.head(z)


class GrayConvEncoder(nn.Module):
    """Small native 1x28x28 encoder; shared by every MNIST-family method."""
    def __init__(self, embedding_dim: int = 64, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.embedding = nn.Linear(32 * 4 * 4, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, normalize: bool = False):
        z = self.embedding(self.features(x).flatten(1))
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z, self.head(z)


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class SmallResNet(nn.Module):
    def __init__(self, embedding_dim: int = 128, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            BasicBlock(32, 32),
            BasicBlock(32, 64, stride=2),
            BasicBlock(64, 64),
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.embedding = nn.Linear(128, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, normalize: bool = False):
        h = self.blocks(self.stem(x)).flatten(1)
        z = self.embedding(h)
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z, self.head(z)


class CifarResNet18(nn.Module):
    """ResNet-18 from scratch, with a 3x3 stride-one stem for 32x32 images."""

    def __init__(self, embedding_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        from torchvision.models import resnet18

        self.encoder = resnet18(weights=None)
        self.encoder.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        nn.init.kaiming_normal_(self.encoder.conv1.weight, mode="fan_out", nonlinearity="relu")
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()
        self.embedding = nn.Identity() if embedding_dim == 512 else nn.Linear(512, embedding_dim)
        self.head = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor, normalize: bool = False):
        z = self.embedding(self.encoder(x))
        if normalize:
            z = F.normalize(z, p=2, dim=1)
        return z, self.head(z)


def build_model(name: str, embedding_dim: int = 128, num_classes: int = 10, normalization: str = "batchnorm") -> nn.Module:
    if normalization != "batchnorm" and name != "cnn":
        raise ValueError("Normalization ablations are currently implemented for cnn only.")
    if name == "cnn":
        return ConvEncoder(embedding_dim=embedding_dim, num_classes=num_classes, normalization=normalization)
    if name == "gray_cnn":
        return GrayConvEncoder(embedding_dim=embedding_dim, num_classes=num_classes)
    if name == "resnet":
        return SmallResNet(embedding_dim=embedding_dim, num_classes=num_classes)
    if name == "cifar_resnet18":
        return CifarResNet18(embedding_dim=embedding_dim, num_classes=num_classes)
    raise ValueError(f"Unknown architecture {name!r}.")


def shared_parameter_fingerprint(model: nn.Module) -> str:
    """Hash convolution/classification weights without normalization parameters."""
    import hashlib
    digest = hashlib.sha256()
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            for key, value in module.named_parameters(recurse=False):
                digest.update(f"{name}.{key}".encode())
                digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()
