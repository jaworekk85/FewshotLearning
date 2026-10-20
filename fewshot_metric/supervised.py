"""CIFAR-10 supervised control, separate from the reconstructed paper protocol."""
from __future__ import annotations

import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from .data import LabelMappedSubset, balanced_splits, seed_worker
from .models import build_model
from .train import evaluate, seed_everything


def validate_config(config: dict) -> None:
    dataset_name = config.get("dataset", "cifar10")
    if dataset_name not in {"cifar10", "fashion_mnist", "mnist"}:
        raise ValueError("Unsupported supervised/pilot dataset.")
    gray = dataset_name in {"fashion_mnist", "mnist"}
    if gray and (config["architecture"] != "gray_cnn" or len(config["mean"]) != 1 or len(config["std"]) != 1):
        raise ValueError("MNIST-family pilots require gray_cnn and single-channel normalization.")
    if not gray and config["architecture"] == "gray_cnn":
        raise ValueError("gray_cnn requires a grayscale dataset.")
    if gray and config.get("reconstruction_pilot"):
        raise ValueError("The existing reconstruction decoder is CIFAR-only.")
    if not isinstance(config.get("labeled_drop_last", False), bool):
        raise ValueError("labeled_drop_last must be boolean.")
    if "fixed_split_manifest" in config:
        digest = config.get("fixed_split_manifest_sha256", "")
        if config.get("budget_per_class") == "all" or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("A fixed split requires a finite budget and a SHA256 snapshot.")
    normalization = config.get("normalization", "batchnorm")
    if normalization not in {"batchnorm", "none", "groupnorm"} or (normalization != "batchnorm" and config["architecture"] != "cnn"):
        raise ValueError("Unsupported normalization for this architecture.")
    for key in ("epochs", "batch_size", "embedding_dim"):
        if int(config[key]) <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= int(config["validation_per_class"]) < 5000:
        raise ValueError("validation_per_class must be between 0 and 4999.")
    budget = config["budget_per_class"]
    class_limit = 6000 if gray else 5000
    if budget != "all" and not 0 < int(budget) <= class_limit - int(config["validation_per_class"]):
        raise ValueError("Label budget exceeds the available images per class.")
    if config["selection"] not in {"last", "best_validation"}:
        raise ValueError("Unknown checkpoint selection rule.")
    if config["selection"] == "best_validation" and config["validation_per_class"] == 0:
        raise ValueError("best_validation needs a validation split.")
    if float(config["learning_rate"]) <= 0 or float(config["weight_decay"]) < 0:
        raise ValueError("Invalid learning rate or weight decay.")
    if not 0 <= float(config["momentum"]) < 1 or int(config["num_workers"]) < 0:
        raise ValueError("Invalid momentum or worker count.")
    if config.get("steps_per_epoch") is not None and int(config["steps_per_epoch"]) <= 0:
        raise ValueError("steps_per_epoch must be positive.")
    if config.get("pilot_mode", False):
        from .pilot import validate_pilot
        validate_pilot(config)
    if config.get("reconstruction_pilot", False):
        from .reconstruction import validate_reconstruction
        validate_reconstruction(config)


def cifar_transform(config: dict, train: bool):
    # Legacy name retained for CIFAR callers; MNIST-family policies are native grayscale.
    if config.get("dataset", "cifar10") in {"mnist", "fashion_mnist"}:
        from .pilot import view_transform
        return view_transform(config) if train else transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"]),
        ])
    steps = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] if train else []
    return transforms.Compose([
        *steps, transforms.ToTensor(), transforms.Normalize(config["mean"], config["std"]),
    ])


def source_dataset(config: dict, root: Path, train: bool, download: bool = False, transform=None):
    cls = {"cifar10": datasets.CIFAR10, "mnist": datasets.MNIST,
           "fashion_mnist": datasets.FashionMNIST}[config.get("dataset", "cifar10")]
    return cls(root=root, train=train, download=download, transform=transform)


def image_array(dataset):
    return dataset.data.numpy() if torch.is_tensor(dataset.data) else dataset.data


def build_datasets(config: dict, root: Path, download: bool):
    train_base = source_dataset(config, root, True, download, cifar_transform(config, True))
    labels = np.asarray(train_base.targets)
    budget = config["budget_per_class"]
    gray_all = budget == "all" and config.get("dataset", "cifar10") != "cifar10"
    if budget == "all" and not gray_all:
        budget = 5000 - int(config["validation_per_class"])
    train_labels = labels
    if "fixed_split_manifest" in config:
        from .fixed_splits import load_fixed_split
        splits, train_labels = load_fixed_split(config, labels)
        train_idx, val_idx = splits["train"], splits["validation"]
    elif gray_all:
        _, val_idx = balanced_splits(labels, 0, int(config["validation_per_class"]), int(config["seed"]))
        excluded = set(val_idx)
        train_idx = [i for i in range(len(labels)) if i not in excluded]
    else:
        train_idx, val_idx = balanced_splits(labels, int(budget), int(config["validation_per_class"]), int(config["seed"]))
    if config.get("dataset", "cifar10") != "cifar10":
        val_idx = sorted(val_idx)  # Identical validation ordering across budgets.
    eval_base = source_dataset(config, root, True, False, cifar_transform(config, False))
    if config.get("pilot_mode", False):
        from .pilot import labeled_dataset
        train = labeled_dataset(image_array(train_base), train_idx, train_labels, config)
    else:
        train = LabelMappedSubset(train_base, train_idx, train_labels)
    val = LabelMappedSubset(eval_base, val_idx, labels)
    # The official test split is instantiated only after training and selection.
    return train, val, {"train": train_idx, "validation": val_idx}


def synthetic_datasets():
    """Tiny deterministic, class-dependent images for pipeline checks only."""
    generator = torch.Generator().manual_seed(724)
    prototypes = torch.randn(10, 3, 32, 32, generator=generator)

    def make(size):
        labels = torch.arange(size) % 10
        images = prototypes[labels] + 0.05 * torch.randn(size, 3, 32, 32, generator=generator)
        return TensorDataset(images, labels)

    return make(40), make(20), make(20)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer, scheduler, device, steps: int, pilot=None) -> dict:
    model.train()
    iterator = iter(loader)
    total_loss = 0.0
    correct = total = 0
    extra_totals = {}
    for _ in range(steps):
        try:
            images, labels = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images, labels = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        if pilot is None:
            images, labels = images.to(device), labels.to(device)
            _, logits = model(images, normalize=False)
            loss = F.cross_entropy(logits, labels)
        else:
            loss, logits, labels, extra = pilot.step(model, images, labels, device)
            for key, value in extra.items():
                extra_totals[key] = extra_totals.get(key, 0) + value
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Non-finite training loss.")
        loss.backward()
        optimizer.step()
        if pilot is not None and hasattr(pilot, "after_optimizer_step"):
            for key, value in pilot.after_optimizer_step(model).items():
                extra_totals[key] = extra_totals.get(key, 0) + value
        scheduler.step()
        total_loss += float(loss.detach()) * labels.numel()
        correct += int((logits.detach().argmax(1) == labels).sum())
        total += labels.numel()
    for key in ("ce_loss", "triplet_l_loss", "triplet_u_loss", "pseudo_label_loss", "pseudo_triplet_loss", "reconstruction_loss", "kl_loss", "kl_weight", "contrastive_loss"):
        if key in extra_totals:
            extra_totals[key] /= steps
    attempted = extra_totals.get("pseudo_attempted", 0)
    if pilot is not None:
        extra_totals["pseudo_acceptance_rate"] = extra_totals["pseudo_accepted"] / attempted if attempted else 0.0
        confidence_sum = extra_totals.pop("pseudo_confidence_sum")
        extra_totals["pseudo_mean_confidence"] = confidence_sum / attempted if attempted else 0.0
        if "pseudo_triplet_valid" in extra_totals:
            count = extra_totals["pseudo_triplet_attempted"]
            extra_totals["pseudo_triplet_valid_rate"] = extra_totals["pseudo_triplet_valid"] / count if count else 0.0
        if "pseudo_weight_sum" in extra_totals:
            extra_totals["pseudo_mean_weight"] = extra_totals["pseudo_weight_sum"] / attempted if attempted else 0.0
    return {"loss": total_loss / total, "accuracy": correct / total,
            "processed_images": total + extra_totals.get("processed_unlabeled_views", 0), **extra_totals}


def rng_state(generator):
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(), "loader": generator.get_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng(state, generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    generator.set_state(state["loader"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def run_supervised(config: dict, run_dir: Path, data_root: Path, device_name: str,
                   download: bool = False, resume: bool = False, smoke: bool = False,
                   stop_after_epoch: int | None = None):
    validate_config(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_name == "auto" else torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment; install a compatible CUDA PyTorch build or use --device cpu.")
    if resume:
        if not (run_dir / "latest.pt").is_file():
            raise FileNotFoundError(f"No checkpoint to resume in {run_dir}.")
        if (run_dir / "metrics.json").exists():
            raise ValueError("This run is already complete; use a different run name for a new experiment.")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {run_dir}; use --resume or another --run-name.")
    seed_everything(int(config["seed"]))
    if smoke and "fixed_split_manifest" in config:
        raise ValueError("Fixed annotation snapshots must not be silently replaced by smoke splits.")
    if smoke:
        if config.get("pilot_mode", False):
            from .pilot import synthetic_datasets as synthetic_pilot_datasets
            train, val, test = synthetic_pilot_datasets(config)
        else:
            train, val, test = synthetic_datasets()
        splits = {"train": list(range(40)), "validation": list(range(40, 60))}
    else:
        train, val, splits = build_datasets(config, data_root, download)
        test = None
    # A reference epoch always uses the full available training pool, even for
    # small budgets. Thus label budgets share a fixed number of optimizer updates.
    pool_size = 50000 if config.get("dataset", "cifar10") == "cifar10" else 60000
    reference_size = len(train) if smoke else pool_size - 10 * int(config["validation_per_class"])
    steps = int(config.get("steps_per_epoch") or math.ceil(reference_size / int(config["batch_size"])))
    resolved = {**config, "steps_per_epoch": steps, "synthetic_smoke": smoke,
                "dataset": "synthetic" if smoke else config.get("dataset", "cifar10"), "device": str(device),
                "labeled_train_size": len(train), "validation_size": len(val),
                "validation_labels_used_for_selection": config["selection"] == "best_validation",
                "total_updates": int(config["epochs"]) * steps,
                "torch_version": str(torch.__version__)}
    generator = torch.Generator().manual_seed(int(config["seed"]))
    if config.get("labeled_drop_last") and len(train) < int(config["batch_size"]):
        raise ValueError("drop_last would discard the entire labeled pool.")
    train_loader = DataLoader(train, batch_size=int(config["batch_size"]), shuffle=True,
                              generator=generator, num_workers=int(config["num_workers"]),
                              worker_init_fn=seed_worker, pin_memory=device.type == "cuda",
                              drop_last=config.get("labeled_drop_last", False))
    val_loader = DataLoader(val, batch_size=256, shuffle=False, num_workers=int(config["num_workers"]))
    pilot = None
    if config.get("pilot_mode", False):
        from .pilot import PilotTraining
        if config.get("reconstruction_pilot", False):
            from .reconstruction import ReconstructionPilot
            pilot = ReconstructionPilot(config, train, splits)
        else:
            pilot = PilotTraining(config, train, splits)
        resolved.update(unlabeled_train_size=len(pilot.unlabeled) if pilot.unlabeled is not None else 0,
                        batchnorm_policy="joint_training_views", true_training_label_budget=len(train),
                        augmentation_rng="separate_keyed_labeled_unlabeled",
                        pseudo_labels_add_to_annotation_budget=False)
    model = build_model(config["architecture"], embedding_dim=int(config["embedding_dim"]),
                        normalization=config.get("normalization", "batchnorm")).to(device)
    if "normalization" in config:
        from .models import shared_parameter_fingerprint
        resolved.update(initial_shared_parameters_sha256=shared_parameter_fingerprint(model),
                        trainable_parameter_count=sum(p.numel() for p in model.parameters()),
                        inference_parameter_count=sum(p.numel() for p in model.parameters()),
                        batchnorm_policy="joint_training_views" if config["normalization"] == "batchnorm" else "not_applicable")
    if pilot is not None:
        from .pilot import model_fingerprint
        resolved["initial_model_sha256"] = model_fingerprint(model)
        if "fixed_split_manifest" in config:
            expected = config.get("fixed_split_source_initial_model_sha256")
            if expected is not None and resolved["initial_model_sha256"] != expected:
                raise ValueError("Fixed-subset control initialization differs from the source AL model.")
            resolved.update(label_budget_policy="frozen_active_final_subset",
                            final_class_counts=np.bincount(train.labels, minlength=10).tolist())
    if config.get("reconstruction_pilot", False):
        from .reconstruction import ReconstructionModel, auxiliary_metadata
        model = ReconstructionModel(model, config).to(device)
        resolved.update(auxiliary_metadata(model))
    optimizer = torch.optim.SGD(model.parameters(), lr=float(config["learning_rate"]),
                                momentum=float(config["momentum"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=resolved["total_updates"])
    start_epoch = 0
    best_accuracy = -1.0
    history = []
    elapsed_before = 0.0
    if resume:
        # Only load checkpoints created by this local experiment runner.
        checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False)
        previous = checkpoint["config"]
        comparable = lambda cfg: {key: value for key, value in cfg.items() if key not in {"device", "torch_version", "num_workers"}}
        if comparable(previous) != comparable(resolved):
            raise ValueError("Resume configuration differs from the checkpoint.")
        if checkpoint["splits"] != splits:
            raise ValueError("Resume dataset split differs from the checkpoint.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch, best_accuracy = checkpoint["epoch"], checkpoint["best_accuracy"]
        history, elapsed_before = checkpoint["history"], checkpoint["elapsed_seconds"]
        restore_rng(checkpoint["rng"], generator)
        if pilot is not None:
            pilot.load_state_dict(checkpoint["pilot_state"])
    if config.get("ema_teacher"):
        # Also needed when resuming a completed final epoch before test/export.
        pilot.ema.initialize(model)
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", resolved)
    write_json(run_dir / "splits.json", splits)
    started = time.perf_counter()
    print(f"device={device} labeled={len(train)} validation={len(val)} updates={resolved['total_updates']}", flush=True)
    for epoch in range(start_epoch, int(config["epochs"])):
        epoch_started = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        if pilot is not None:
            pilot.start_epoch(epoch)
            if config.get("neighbor_reference"):
                pilot.prepare_neighbor_epoch(model, device)
        trained = train_epoch(model, train_loader, optimizer, scheduler, device, steps, pilot=pilot)
        validated = evaluate(model, val_loader, device, normalize=False) if len(val) else None
        teacher_validated = pilot.evaluate_teacher(val_loader, device) if config.get("ema_teacher") else None
        row = {"epoch": epoch + 1, "updates": (epoch + 1) * steps, "learning_rate": lr,
               "train_loss": trained["loss"], "train_accuracy": trained["accuracy"],
               "processed_images": trained["processed_images"],
               "val_loss": validated["loss"] if validated else None,
               "val_accuracy": validated["accuracy"] if validated else None,
               "epoch_seconds": time.perf_counter() - epoch_started}
        if pilot is not None:
            row.update({key: value for key, value in trained.items() if key not in {"loss", "accuracy", "processed_images"}})
            row.update(pilot.finish_epoch())
        if teacher_validated is not None:
            row.update(teacher_val_accuracy=teacher_validated["accuracy"], teacher_val_loss=teacher_validated["loss"],
                       ema_validation_views=teacher_validated["total"])
        history.append(row)
        improved = validated is not None and validated["accuracy"] > best_accuracy
        if improved:
            best_accuracy = validated["accuracy"]
        checkpoint = {"epoch": epoch + 1, "model": model.state_dict(),
                      "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                      "config": resolved, "splits": splits, "best_accuracy": best_accuracy,
                      "history": history, "rng": rng_state(generator),
                      "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        if pilot is not None:
            checkpoint["pilot_state"] = pilot.state_dict()
        if improved and config["selection"] == "best_validation":
            torch.save({"epoch": epoch + 1, "model": model.state_dict()}, run_dir / "best.pending.pt")
            (run_dir / "best.pending.pt").replace(run_dir / "best.pt")
        # Commit the latest checkpoint after its selected best model is on disk.
        torch.save(checkpoint, run_dir / "latest.pending.pt")
        (run_dir / "latest.pending.pt").replace(run_dir / "latest.pt")
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        val_message = f" val={validated['accuracy']:.2%}" if validated else ""
        if pilot is not None and float(config.get("pseudo_label_weight", 0.0)) > 0:
            val_message += f" pseudo_accepted={trained['pseudo_acceptance_rate']:.1%}"
        print(f"epoch={epoch + 1}/{config['epochs']} train={trained['accuracy']:.2%}{val_message} seconds={row['epoch_seconds']:.1f}", flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < int(config["epochs"]):
            print("Stopped at an epoch boundary; resume with the same configuration and --resume.", flush=True)
            return None
    selected_epoch = int(config["epochs"])
    if config["selection"] == "best_validation":
        best = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(best["model"])
        selected_epoch = best["epoch"]
    if test is None:
        test = source_dataset(config, data_root, False, download, cifar_transform(config, False))
    test_loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=int(config["num_workers"]))
    tested = evaluate(model, test_loader, device, normalize=False)
    teacher_tested = pilot.evaluate_teacher(test_loader, device) if config.get("ema_teacher") else None
    result = {**resolved, "selected_epoch": selected_epoch, "selection": config["selection"],
              "val_accuracy": history[selected_epoch - 1]["val_accuracy"],
              "test_accuracy": tested["accuracy"], "test_loss": tested["loss"], "test_size": tested["total"],
              "processed_training_images": sum(row["processed_images"] for row in history),
              "elapsed_seconds": elapsed_before + time.perf_counter() - started,
              "status": "synthetic_smoke_only" if smoke else "completed"}
    if pilot is not None:
        for key in ("processed_labeled_views", "processed_unlabeled_views", "pseudo_accepted", "pseudo_attempted"):
            result[key] = sum(row[key] for row in history)
        result["pseudo_acceptance_rate"] = result["pseudo_accepted"] / result["pseudo_attempted"] if result["pseudo_attempted"] else 0.0
        result["accepted_pseudo_class_histogram"] = [sum(row[f"pseudo_class_{i}"] for row in history) for i in range(10)]
        if "pseudo_triplet_valid" in history[0]:
            for key in ("pseudo_triplet_valid", "pseudo_triplet_attempted", "pseudo_triplet_labeled_positives",
                        "pseudo_triplet_labeled_negatives", "pseudo_triplet_pseudo_positives", "pseudo_triplet_pseudo_negatives"):
                result[key] = sum(row[key] for row in history)
            count = result["pseudo_triplet_attempted"]
            result["pseudo_triplet_valid_rate"] = result["pseudo_triplet_valid"] / count if count else 0.0
        for key in ("pseudo_triplet_active", "pseudo_triplet_semihard", "pseudo_weight_sum", "triplet_l_valid", "triplet_l_active", "triplet_l_semihard", "triplet_u_pairs", "triplet_u_active", "neighbor_reference_views", "neighbor_query_views", "neighbor_reference_seconds", "neighbor_eligible", "neighbor_reweighted", "neighbor_js_sum", "neighbor_raw_factor_sum", "pseudo_base_weight_sum"):
            if key in history[0]:
                result[key] = sum(row[key] for row in history)
        if "pseudo_weight_sum" in result:
            result["pseudo_mean_weight"] = result["pseudo_weight_sum"] / result["pseudo_attempted"]
        for key in history[0]:
            if key.startswith("metric_"):
                result[key] = sum(row[key] for row in history)
            if key.startswith("ema_") or (key.startswith("contrastive_") and key != "contrastive_loss"):
                result[key] = sum(row[key] for row in history)
        if "contrastive_loss" in history[-1]:
            result["final_contrastive_loss"] = history[-1]["contrastive_loss"]
        if config.get("reconstruction_pilot", False):
            for key in ("processed_decoder_views",):
                result[key] = sum(row[key] for row in history)
            result["final_reconstruction_loss"] = history[-1]["reconstruction_loss"]
            result["final_kl_loss"] = history[-1]["kl_loss"]
    if teacher_tested is not None:
        result.update(teacher_test_accuracy=teacher_tested["accuracy"], teacher_test_loss=teacher_tested["loss"],
                      teacher_val_accuracy=history[-1]["teacher_val_accuracy"], ema_test_views=teacher_tested["total"],
                      teacher_buffer_policy="copy_student_buffers_after_each_optimizer_step",
                      primary_evaluation="student_last", teacher_trainable_parameters=0,
                      teacher_parameter_count=sum(p.numel() for p in pilot.ema.model.parameters()))
        torch.save(pilot.ema.model.state_dict(), run_dir/"teacher_model.pt")
    torch.save((model.backbone if config.get("reconstruction_pilot", False) else model).state_dict(), run_dir / "model.pt")
    write_json(run_dir / "metrics.json", result)
    print(json.dumps(result, indent=2), flush=True)
    return result
