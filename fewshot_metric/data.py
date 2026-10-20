from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .transforms import build_transform


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed + worker_id)


def _targets(dataset) -> np.ndarray:
    if hasattr(dataset, "targets"):
        labels = np.asarray(dataset.targets)
    elif hasattr(dataset, "labels"):
        labels = np.asarray(dataset.labels)
    else:
        raise TypeError(f"Cannot find labels for {type(dataset).__name__}.")

    labels = labels.astype(int)
    unique = sorted(np.unique(labels).tolist())
    if unique == list(range(1, 11)):
        labels = labels - 1
    return labels


def load_dataset(name: str, root: str | Path, split: str, transform):
    root = Path(root)
    if name == "cifar10":
        return datasets.CIFAR10(root=root, train=(split == "train"), download=True, transform=transform)
    if name == "fashion_mnist":
        return datasets.FashionMNIST(root=root, train=(split == "train"), download=True, transform=transform)
    if name == "mnist":
        return datasets.MNIST(root=root, train=(split == "train"), download=True, transform=transform)
    if name == "svhn":
        svhn_split = "train" if split == "train" else "test"
        return datasets.SVHN(root=root, split=svhn_split, download=True, transform=transform)
    if name == "stl10":
        stl_split = "train" if split == "train" else "test"
        return datasets.STL10(root=root, split=stl_split, download=True, transform=transform)
    raise ValueError(f"Unsupported dataset {name!r}.")


def balanced_splits(labels: np.ndarray, budget: int, val_per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for cls in sorted(np.unique(labels).tolist()):
        cls_indices = np.flatnonzero(labels == cls)
        rng.shuffle(cls_indices)
        needed = val_per_class + budget
        if len(cls_indices) < needed:
            raise ValueError(f"Class {cls} has {len(cls_indices)} samples, needs {needed}.")
        val_indices.extend(cls_indices[:val_per_class].tolist())
        train_indices.extend(cls_indices[val_per_class:needed].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def unlabeled_split(
    labels: np.ndarray,
    train_indices: list[int],
    val_indices: list[int],
    seed: int,
    limit: int | None = None,
) -> list[int]:
    used = set(train_indices) | set(val_indices)
    available = [idx for idx in range(len(labels)) if idx not in used]
    rng = np.random.default_rng(seed + 51_337)
    rng.shuffle(available)
    if limit is not None:
        available = available[:limit]
    return available


class LabelMappedSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int], labels: np.ndarray) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.labels = labels[self.indices].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, _ = self.dataset[self.indices[item]]
        return image, self.labels[item]


class UnlabeledSubset(Dataset):
    def __init__(self, dataset: Dataset, indices: list[int]) -> None:
        self.dataset = dataset
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        image, _ = self.dataset[self.indices[item]]
        return image


class TripletDataset(Dataset):
    def __init__(self, dataset: LabelMappedSubset, seed: int) -> None:
        self.dataset = dataset
        self.labels = np.asarray(dataset.labels)
        self.seed = seed
        self.epoch = 0
        self.by_class: dict[int, np.ndarray] = {}
        for cls in sorted(np.unique(self.labels).tolist()):
            self.by_class[int(cls)] = np.flatnonzero(self.labels == cls)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, item: int):
        label = int(self.labels[item])
        rng = np.random.default_rng(self.seed + self.epoch * 1_000_003 + item * 9_176)

        positives = self.by_class[label]
        if len(positives) > 1:
            pos_item = int(rng.choice(positives[positives != item]))
        else:
            pos_item = item

        negative_label = int(rng.choice([cls for cls in self.by_class if cls != label]))
        neg_item = int(rng.choice(self.by_class[negative_label]))

        anchor, _ = self.dataset[item]
        positive, _ = self.dataset[pos_item]
        negative, _ = self.dataset[neg_item]
        return anchor, positive, negative, label


def build_loaders(
    dataset_name: str,
    budget: int,
    method: dict,
    data_root: str | Path,
    seed: int,
    batch_size: int,
    val_per_class: int,
    num_workers: int = 0,
):
    plain_train = load_dataset(dataset_name, data_root, "train", transform=None)
    labels = _targets(plain_train)
    train_idx, val_idx = balanced_splits(labels, budget, val_per_class, seed)
    unlabeled_idx: list[int] = []
    use_unlabeled = method.get("unlabeled_objective", "none") != "none" and float(method.get("unlabeled_weight", 0.0)) > 0.0
    if use_unlabeled:
        unlabeled_idx = unlabeled_split(
            labels,
            train_indices=train_idx,
            val_indices=val_idx,
            seed=seed,
            limit=method.get("unlabeled_limit"),
        )

    train_transform = build_transform(
        dataset_name,
        augmentation=method["augmentation"],
        views=int(method.get("views", 1)),
        train=True,
    )
    unlabeled_transform = build_transform(
        dataset_name,
        augmentation=method.get("unlabeled_augmentation", "strong"),
        views=int(method.get("unlabeled_views", 2)),
        train=True,
    )
    eval_transform = build_transform(dataset_name, augmentation="none", views=1, train=False)

    train_base = load_dataset(dataset_name, data_root, "train", transform=train_transform)
    unlabeled_base = load_dataset(dataset_name, data_root, "train", transform=unlabeled_transform)
    eval_train_base = load_dataset(dataset_name, data_root, "train", transform=eval_transform)
    test_base = load_dataset(dataset_name, data_root, "test", transform=eval_transform)

    train_subset = LabelMappedSubset(train_base, train_idx, labels)
    unlabeled_subset = UnlabeledSubset(unlabeled_base, unlabeled_idx)
    val_subset = LabelMappedSubset(eval_train_base, val_idx, labels)

    test_labels = _targets(test_base)
    test_subset = LabelMappedSubset(test_base, list(range(len(test_base))), test_labels)

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset: Dataset = train_subset
    if method["objective"] == "triplet":
        train_dataset = TripletDataset(train_subset, seed=seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    unlabeled_loader = None
    if use_unlabeled and len(unlabeled_subset) > 0:
        unlabeled_loader = DataLoader(
            unlabeled_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=generator,
            drop_last=len(unlabeled_subset) >= batch_size,
        )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    split_info = {
        "labeled_train_size": len(train_idx),
        "validation_size": len(val_idx),
        "unlabeled_train_size": len(unlabeled_idx),
        "test_size": len(test_subset),
    }
    return train_loader, unlabeled_loader, val_loader, test_loader, split_info
