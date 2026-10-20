from __future__ import annotations

import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .config import RunSpec
from .data import TripletDataset, build_loaders
from .losses import supervised_contrastive_loss
from .models import build_model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _as_device_batch(batch, device: torch.device):
    if isinstance(batch, (list, tuple)):
        return [_as_device_batch(item, device) for item in batch]
    if torch.is_tensor(batch):
        return batch.to(device, non_blocking=True)
    return batch


def _ce_from_views(model: nn.Module, images, labels, normalize: bool):
    if isinstance(images, list):
        views = len(images)
        x = torch.cat(images, dim=0)
        y = labels.repeat(views)
    else:
        x = images
        y = labels
    _, logits = model(x, normalize=normalize)
    return F.cross_entropy(logits, y), logits, y


def _train_step(model: nn.Module, batch, method: dict[str, Any], loss_defaults: dict[str, float]):
    objective = method["objective"]
    normalize = bool(method["normalize_embedding"])

    if objective == "ce":
        images, labels = batch
        loss, logits, y = _ce_from_views(model, images, labels, normalize=normalize)
        return loss, logits, y

    if objective == "triplet":
        anchor, positive, negative, labels = batch
        x = torch.cat([anchor, positive, negative], dim=0)
        embeddings, logits_all = model(x, normalize=normalize)
        batch_size = labels.shape[0]
        za, zp, zn = embeddings.split(batch_size, dim=0)
        logits = logits_all[:batch_size]
        ce = F.cross_entropy(logits, labels)
        triplet = F.triplet_margin_loss(za, zp, zn, margin=float(loss_defaults["triplet_margin"]))
        loss = float(loss_defaults["ce_weight"]) * ce + float(loss_defaults["metric_weight"]) * triplet
        return loss, logits, labels

    if objective == "supcon_joint":
        images, labels = batch
        if not isinstance(images, list) or len(images) < 2:
            raise ValueError("SupCon-joint expects a multi-view transform.")
        views = len(images)
        x = torch.cat(images, dim=0)
        embeddings, logits = model(x, normalize=normalize)
        ce = F.cross_entropy(logits, labels.repeat(views))
        chunks = embeddings.split(labels.shape[0], dim=0)
        features = torch.stack(chunks, dim=1)
        supcon = supervised_contrastive_loss(
            features,
            labels,
            temperature=float(loss_defaults["supcon_temperature"]),
        )
        loss = float(loss_defaults["ce_weight"]) * ce + float(loss_defaults["metric_weight"]) * supcon
        return loss, logits, labels.repeat(views)

    raise ValueError(f"Unknown objective {objective!r}.")


def _unlabeled_step(model: nn.Module, images, method: dict[str, Any]):
    objective = method.get("unlabeled_objective", "none")
    normalize = bool(method["normalize_embedding"])
    if objective == "none":
        return None

    if not isinstance(images, list):
        images = [images]

    if objective == "pseudo_ce":
        threshold = float(method.get("pseudo_label_threshold", 0.95))
        with torch.no_grad():
            _, teacher_logits = model(images[0], normalize=normalize)
            probs = torch.softmax(teacher_logits, dim=1)
            confidence, pseudo_labels = probs.max(dim=1)
            mask = confidence >= threshold

        student_views = images[1:] if len(images) > 1 else images[:1]
        _, student_logits = model(torch.cat(student_views, dim=0), normalize=normalize)
        repeated_labels = pseudo_labels.repeat(len(student_views))
        repeated_mask = mask.repeat(len(student_views))
        if not bool(repeated_mask.any()):
            return student_logits.sum() * 0.0
        return F.cross_entropy(student_logits[repeated_mask], repeated_labels[repeated_mask])

    if objective == "consistency":
        if len(images) < 2:
            raise ValueError("consistency requires at least two unlabeled views.")
        with torch.no_grad():
            _, teacher_logits = model(images[0], normalize=normalize)
            target_probs = torch.softmax(teacher_logits, dim=1)
        losses = []
        for view in images[1:]:
            _, student_logits = model(view, normalize=normalize)
            losses.append(F.kl_div(F.log_softmax(student_logits, dim=1), target_probs, reduction="batchmean"))
        return torch.stack(losses).mean()

    if objective == "simclr":
        if len(images) < 2:
            raise ValueError("simclr requires at least two unlabeled views.")
        x = torch.cat(images, dim=0)
        embeddings, _ = model(x, normalize=normalize)
        batch_size = images[0].shape[0]
        chunks = embeddings.split(batch_size, dim=0)
        features = torch.stack(chunks, dim=1)
        labels = torch.arange(batch_size, device=embeddings.device)
        return supervised_contrastive_loss(features, labels)

    if objective == "pseudo_triplet":
        if len(images) < 2:
            raise ValueError("pseudo_triplet requires at least two unlabeled views.")
        threshold = float(method.get("pseudo_label_threshold", 0.95))
        margin = float(method.get("unlabeled_triplet_margin", method.get("triplet_margin", 0.2)))
        negative_strategy = method.get("unlabeled_negative_strategy", "pseudo_label_hard")

        anchor_view = images[0]
        positive_view = images[1]
        with torch.no_grad():
            _, teacher_logits = model(anchor_view, normalize=normalize)
            probs = torch.softmax(teacher_logits, dim=1)
            confidence, pseudo_labels = probs.max(dim=1)
            confident = confidence >= threshold

        embeddings, _ = model(torch.cat([anchor_view, positive_view], dim=0), normalize=normalize)
        batch_size = anchor_view.shape[0]
        z_anchor, z_positive = embeddings.split(batch_size, dim=0)
        if batch_size < 2:
            return z_anchor.sum() * 0.0

        not_self = ~torch.eye(batch_size, dtype=torch.bool, device=z_anchor.device)
        random_negative_indices = torch.randint(0, batch_size - 1, (batch_size,), device=z_anchor.device)
        random_negative_indices = random_negative_indices + (random_negative_indices >= torch.arange(batch_size, device=z_anchor.device)).long()

        if negative_strategy == "random":
            valid = torch.ones(batch_size, dtype=torch.bool, device=z_anchor.device)
            negative_indices = random_negative_indices
        else:
            label_mismatch = pseudo_labels[:, None] != pseudo_labels[None, :]
            confidence_pair = confident[:, None] & confident[None, :]
            candidate_mask = label_mismatch & confidence_pair & not_self
            has_pseudo_negative = candidate_mask.any(dim=1)

            with torch.no_grad():
                distances = torch.cdist(z_anchor.detach(), z_anchor.detach(), p=2)
                distances = distances.masked_fill(~candidate_mask, float("inf"))
                pseudo_negative_indices = distances.argmin(dim=1)

            if negative_strategy == "pseudo_label_hard":
                valid = has_pseudo_negative & confident
                negative_indices = pseudo_negative_indices
            elif negative_strategy == "pseudo_label_or_random":
                valid = torch.ones(batch_size, dtype=torch.bool, device=z_anchor.device)
                negative_indices = torch.where(has_pseudo_negative, pseudo_negative_indices, random_negative_indices)
            else:
                raise ValueError(f"Unknown unlabeled negative strategy {negative_strategy!r}.")

        if not bool(valid.any()):
            return z_anchor.sum() * 0.0

        return F.triplet_margin_loss(
            z_anchor[valid],
            z_positive[valid],
            z_anchor[negative_indices[valid]],
            margin=margin,
        )

    raise ValueError(f"Unknown unlabeled objective {objective!r}.")


@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, normalize: bool) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, logits = model(images, normalize=normalize)
        loss = F.cross_entropy(logits, labels)
        total_loss += float(loss.item()) * labels.shape[0]
        correct += int((logits.argmax(dim=1) == labels).sum().item())
        total += int(labels.shape[0])
    return {
        "loss": total_loss / max(total, 1),
        "accuracy": correct / max(total, 1),
        "total": total,
    }


def train_one_run(
    spec: RunSpec,
    data_root: str | Path,
    output_dir: str | Path,
    device_name: str = "auto",
    num_workers: int = 0,
) -> dict[str, Any]:
    seed_everything(spec.seed)
    protocol = spec.protocol
    method = spec.method
    optimizer_cfg = protocol["optimizer"]

    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    train_loader, unlabeled_loader, val_loader, test_loader, split_info = build_loaders(
        dataset_name=spec.dataset,
        budget=spec.budget,
        method=method,
        data_root=data_root,
        seed=spec.seed,
        batch_size=int(optimizer_cfg["batch_size"]),
        val_per_class=int(protocol["validation_per_class"]),
        num_workers=num_workers,
    )

    model = build_model(
        spec.architecture,
        embedding_dim=int(protocol["embedding_dim"]),
        num_classes=int(protocol["num_classes"]),
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(optimizer_cfg["learning_rate"]),
        weight_decay=float(optimizer_cfg["weight_decay"]),
    )

    updates = int(method["updates"])
    started = time.time()
    model.train()
    step = 0
    epoch = 0
    last_loss = float("nan")
    last_acc = float("nan")
    last_supervised_loss = float("nan")
    last_unlabeled_loss = float("nan")
    unlabeled_iter = iter(unlabeled_loader) if unlabeled_loader is not None else None
    while step < updates:
        if isinstance(train_loader.dataset, TripletDataset):
            train_loader.dataset.set_epoch(epoch)
        for batch in train_loader:
            batch = _as_device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            loss, logits, labels = _train_step(model, batch, method, protocol["loss_defaults"])
            supervised_loss = loss
            unlabeled_loss = None
            if unlabeled_loader is not None and unlabeled_iter is not None:
                try:
                    unlabeled_batch = next(unlabeled_iter)
                except StopIteration:
                    unlabeled_iter = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iter)
                unlabeled_batch = _as_device_batch(unlabeled_batch, device)
                unlabeled_loss = _unlabeled_step(model, unlabeled_batch, method)
                if unlabeled_loss is not None:
                    loss = loss + float(method["unlabeled_weight"]) * unlabeled_loss
            loss.backward()
            optimizer.step()

            last_loss = float(loss.item())
            last_supervised_loss = float(supervised_loss.item())
            if unlabeled_loss is not None:
                last_unlabeled_loss = float(unlabeled_loss.item())
            last_acc = float((logits.argmax(dim=1) == labels).float().mean().item())
            step += 1
            if step >= updates:
                break
        epoch += 1

    val = evaluate(model, val_loader, device=device, normalize=bool(method["normalize_embedding"]))
    test = evaluate(model, test_loader, device=device, normalize=bool(method["normalize_embedding"]))
    elapsed = time.time() - started

    result = {
        "dataset": spec.dataset,
        "architecture": spec.architecture,
        "budget": spec.budget,
        "method_id": spec.method_id,
        "method_label": method["paper_label"],
        "seed": spec.seed,
        "updates": updates,
        "objective": method["objective"],
        "views": int(method.get("views", 1)),
        "augmentation": method["augmentation"],
        "normalize_embedding": bool(method["normalize_embedding"]),
        "train_last_loss": last_loss,
        "train_last_supervised_loss": last_supervised_loss,
        "train_last_unlabeled_loss": last_unlabeled_loss,
        "train_last_accuracy": last_acc,
        "val_accuracy": val["accuracy"],
        "val_loss": val["loss"],
        "test_accuracy": test["accuracy"],
        "test_loss": test["loss"],
        "device": str(device),
        "elapsed_seconds": elapsed,
        "method_config_json": json.dumps(method, sort_keys=True),
        "optimizer_config_json": json.dumps(optimizer_cfg, sort_keys=True),
        "loss_config_json": json.dumps(protocol["loss_defaults"], sort_keys=True),
        "embedding_dim": int(protocol["embedding_dim"]),
        "validation_per_class": int(protocol["validation_per_class"]),
        **split_info,
    }
    save_run_outputs(result, output_dir, model)
    return result


def save_run_outputs(result: dict[str, Any], output_dir: str | Path, model: nn.Module) -> None:
    output_dir = Path(output_dir)
    run_dir = (
        output_dir
        / "runs"
        / result["dataset"]
        / result["architecture"]
        / f"N{result['budget']}"
        / result["method_id"]
        / f"seed{result['seed']}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(model.state_dict(), run_dir / "model.pt")

    csv_path = output_dir / "results.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(result)
