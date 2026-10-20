"""Budgeted active learning: selection sees images and predictions, never labels."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets

from .data import balanced_splits
from .models import build_model
from .pilot import (KeyedViews, PilotTraining, labeled_dataset, model_fingerprint,
                    synthetic_datasets, view_transform)
from .active_selection import NEW_STRATEGIES, select_prepared_queries, validate_settings
from .supervised import (cifar_transform, restore_rng, rng_state, train_epoch,
                         validate_config, write_json)
from .train import evaluate, seed_everything


class LabelOracle:
    def __init__(self, truth, eligible, budget):
        self._truth = np.asarray(truth)
        self.eligible = set(eligible)
        self.budget = int(budget)
        self.known = {}

    def reveal(self, indices):
        indices = [int(i) for i in indices]
        if len(set(indices)) != len(indices) or any(i in self.known or i not in self.eligible for i in indices):
            raise ValueError("Oracle request includes duplicate, previously labeled or excluded images.")
        if len(self.known) + len(indices) > self.budget:
            raise ValueError("Annotation budget exceeded.")
        self.known.update({i: int(self._truth[i]) for i in indices})

    def visible_labels(self):
        labels = np.full(len(self._truth), -1, dtype=int)
        for i, label in self.known.items():
            labels[i] = label
        return labels


class ImageIndices(Dataset):
    def __init__(self, images, indices, transform):
        self.images, self.indices, self.transform = images, list(indices), transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[self.indices[index]]))


@torch.no_grad()
def encode_images(model, images, indices, transform, device, normalize_features=True):
    model.eval()
    features, probabilities = [], []
    for batch in DataLoader(ImageIndices(images, indices, transform), batch_size=256, shuffle=False):
        z, logits = model(batch.to(device), normalize=False)
        features.append((F.normalize(z, dim=1) if normalize_features else z).cpu())
        probabilities.append(logits.softmax(1).cpu())
    return torch.cat(features), torch.cat(probabilities)


@torch.no_grad()
def encode_acquisition_views(model, images, indices, config, device, epoch, base_probabilities):
    """Extra eval-only weak views, with RNG isolated from all training streams."""
    model.eval()
    views = [base_probabilities]
    for view in range(1, int(config.get("active_scoring_views", 1))):
        seed = int(config["seed"]) + 97_009 + view * 100_003
        dataset = KeyedViews(images, indices, view_transform(config), seed)
        dataset.set_epoch(epoch)
        loader_rng = torch.Generator().manual_seed(seed + epoch)
        batches = []
        for batch in DataLoader(dataset, batch_size=256, shuffle=False, generator=loader_rng):
            _, logits = model(batch.to(device), normalize=False)
            batches.append(logits.softmax(1).cpu())
        views.append(torch.cat(batches))
    return torch.stack(views)


def select_queries(probabilities, features, known_features, count, strategy, generator):
    """Returns candidate positions; this API has no labels or oracle argument."""
    if count <= 0 or count > len(probabilities):
        raise ValueError("Invalid acquisition batch size.")
    if strategy == "random":
        return torch.randperm(len(probabilities), generator=generator)[:count]
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(1)
    ordered = torch.argsort(entropy, descending=True, stable=True)
    if strategy == "entropy":
        return ordered[:count]
    if strategy != "entropy_diverse":
        raise ValueError("Unknown acquisition strategy.")
    # Pilot heuristic, not BADGE: entropy shortlist + farthest-first cosine coverage.
    shortlist = ordered[:min(count * 10, len(ordered))]
    z = F.normalize(features[shortlist], dim=1)
    if len(known_features):
        distance = (1 - z @ F.normalize(known_features, dim=1).T).clamp_min(0).min(1).values
    else:
        distance = torch.ones(len(shortlist))
    selected = []
    for _ in range(count):
        position = int(distance.argmax())
        selected.append(position)
        distance = torch.minimum(distance, (1 - z @ z[position]).clamp_min(0))
        distance[selected] = -1
    return shortlist[torch.tensor(selected)]


def validate_active(config):
    validate_config(config)
    if config.get("active_strategy") not in {"random", "entropy", "entropy_diverse"} | NEW_STRATEGIES:
        raise ValueError("Unknown active strategy.")
    version = config.get("active_selection_version", 1)
    if version not in {1, 2}:
        raise ValueError("Unknown acquisition protocol version.")
    if config["active_strategy"] in NEW_STRATEGIES and version != 2:
        raise ValueError("Prepared acquisition rules require active_selection_version=2.")
    views = config.get("active_scoring_views", 1)
    if isinstance(views, bool) or not isinstance(views, int) or views < 1:
        raise ValueError("active_scoring_views must be a positive integer.")
    if version == 1 and (views != 1 or config.get("active_selection_settings")):
        raise ValueError("Extra scoring views/settings require acquisition version 2.")
    if config["active_strategy"] == "view_disagreement" and views < 2:
        raise ValueError("View disagreement requires at least two acquisition views.")
    validate_settings(config.get("active_selection_settings", {}))
    rounds = config["active_rounds"]
    if not rounds or rounds != sorted(set(rounds)) or rounds[0] < 1 or rounds[-1] >= config["epochs"]:
        raise ValueError("Acquisition rounds must be distinct epoch boundaries before the final epoch.")
    final = 10 * int(config["budget_per_class"]) + len(rounds) * int(config["active_query_size"])
    if final != int(config["active_final_budget"]) or int(config["active_query_size"]) <= 0:
        raise ValueError("Acquisition schedule does not match final label budget.")
    if int(config["active_candidate_limit"]) < int(config["active_query_size"]):
        raise ValueError("Candidate limit is smaller than query batch.")
    if config["selection"] != "last":
        raise ValueError("Active pilot uses the last checkpoint.")


def run_active(config, run_dir, data_root, device_name="cpu", resume=False, stop_after_epoch=None, smoke=False):
    validate_active(config)
    run_dir, data_root = Path(run_dir), Path(data_root)
    if resume and not (run_dir / "latest.pt").exists():
        raise FileNotFoundError("Missing active checkpoint.")
    if not resume and run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {run_dir}.")
    if (run_dir / "metrics.json").exists():
        raise ValueError("Active run is already complete.")
    seed_everything(int(config["seed"]))
    device = torch.device(device_name)
    if smoke:
        original, val, test = synthetic_datasets(config)
        images = original.images
        truth = np.arange(len(images)) % 10
        initial, validation = balanced_splits(truth, int(config["budget_per_class"]), 2, int(config["seed"]))
        # Keep evaluation aligned with actual validation indices in this fixture.
        eval_images = ImageIndices(images, validation, cifar_transform(config, False))
        val = torch.utils.data.TensorDataset(torch.stack([eval_images[i] for i in range(len(eval_images))]), torch.tensor(truth[validation]))
    else:
        base = datasets.CIFAR10(data_root, train=True, download=False)
        images, truth = base.data, np.asarray(base.targets)
        initial, validation = balanced_splits(truth, int(config["budget_per_class"]), int(config["validation_per_class"]), int(config["seed"]))
        eval_images = ImageIndices(images, validation, cifar_transform(config, False))
        val = torch.utils.data.TensorDataset(torch.stack([eval_images[i] for i in range(len(eval_images))]), torch.tensor(truth[validation]))
        test = None
    eligible = sorted(set(range(len(images))) - set(validation))
    oracle = LabelOracle(truth, eligible, config["active_final_budget"])
    generator = torch.Generator().manual_seed(int(config["seed"]))
    candidate_rng = torch.Generator().manual_seed(int(config["seed"]) + 85_001)
    selection_rng = torch.Generator().manual_seed(int(config["seed"]) + 85_003)
    checkpoint = torch.load(run_dir / "latest.pt", map_location="cpu", weights_only=False) if resume else None
    oracle.reveal(checkpoint["known_indices"] if checkpoint else initial)
    events = checkpoint["events"] if checkpoint else []
    def make_training(previous_state=None):
        train = labeled_dataset(images, list(oracle.known), oracle.visible_labels(), config)
        splits = {"train": list(oracle.known), "validation": validation}
        pilot = PilotTraining(config, train, splits)
        if previous_state is not None:
            pilot.load_state_dict(previous_state)
        loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator, num_workers=0)
        return pilot, loader
    pilot, loader = make_training()
    model = build_model(config["architecture"], embedding_dim=config["embedding_dim"],
                        normalization=config.get("normalization", "batchnorm")).to(device)
    steps = int(config.get("steps_per_epoch") or (len(eligible) + config["batch_size"] - 1) // config["batch_size"])
    resolved = {**config, "dataset": "synthetic" if smoke else "cifar10", "synthetic_smoke": smoke,
                "steps_per_epoch": steps, "total_updates": steps * config["epochs"],
                "initial_model_sha256": model_fingerprint(model), "device": str(device),
                "validation_size": len(validation), "selection": "last", "validation_labels_used_for_selection": False,
                "initial_label_budget": len(initial), "initial_train_indices": initial,
                "pseudo_labels_add_to_annotation_budget": False}
    optimizer = torch.optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=config["momentum"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=resolved["total_updates"])
    history, start_epoch, elapsed_before, scoring_views = [], 0, 0.0, 0
    if checkpoint:
        if checkpoint["config"] != resolved or checkpoint["validation_indices"] != validation:
            raise ValueError("Active resume configuration or validation differs.")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        pilot.load_state_dict(checkpoint["pilot_state"])
        restore_rng(checkpoint["rng"], generator)
        candidate_rng.set_state(checkpoint["candidate_rng"])
        selection_rng.set_state(checkpoint["selection_rng"])
        history, start_epoch = checkpoint["history"], checkpoint["epoch"]
        events, elapsed_before, scoring_views = checkpoint["events"], checkpoint["elapsed_seconds"], checkpoint["scoring_views"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", resolved)
    val_loader = DataLoader(val, batch_size=256, shuffle=False)
    started = time.perf_counter()
    print(f"active={config['active_strategy']} initial_labels={len(initial)} final_labels={config['active_final_budget']}", flush=True)
    for epoch in range(start_epoch, config["epochs"]):
        tick = time.perf_counter()
        acquisition_seconds = 0.0
        used_labels = len(oracle.known)
        pilot.start_epoch(epoch)
        if config.get("neighbor_reference"):
            pilot.prepare_neighbor_epoch(model, device)
        trained = train_epoch(model, loader, optimizer, scheduler, device, steps, pilot)
        validated = evaluate(model, val_loader, device, normalize=False)
        soft = pilot.finish_epoch()
        if epoch + 1 in config["active_rounds"]:
            query_tick = time.perf_counter()
            pool = sorted(set(eligible) - set(oracle.known))
            limit = min(config["active_candidate_limit"], len(pool))
            candidates = [pool[i] for i in torch.randperm(len(pool), generator=candidate_rng)[:limit].tolist()]
            all_indices = candidates + list(oracle.known)
            prepared = config.get("active_selection_version", 1) == 2
            z, probs = encode_images(model, images, all_indices, cifar_transform(config, False), device,
                                     normalize_features=not prepared)
            scoring_views += len(all_indices)
            diagnostics = None
            if prepared:
                predictions = encode_acquisition_views(model, images, candidates, config, device, epoch,
                                                         probs[:limit])
                extra_views = (len(predictions) - 1) * limit
                scoring_views += extra_views
                if config["active_strategy"] in NEW_STRATEGIES:
                    chosen, diagnostics = select_prepared_queries(
                        probs[:limit], z[:limit], z[limit:], list(oracle.known.values()),
                        config["active_query_size"], config["active_strategy"], selection_rng,
                        view_probabilities=predictions, settings=config.get("active_selection_settings", {}))
                else:
                    chosen = select_queries(probs[:limit], z[:limit], z[limit:], config["active_query_size"],
                                            config["active_strategy"], selection_rng)
                    diagnostics = {"strategy": config["active_strategy"]}
                diagnostics.update(protocol_version=2, candidate_views=len(predictions),
                                   scoring_forward_views=len(all_indices) + extra_views)
                acquisition_seconds = time.perf_counter() - query_tick
            else:
                chosen = select_queries(probs[:limit], z[:limit], z[limit:], config["active_query_size"], config["active_strategy"], selection_rng)
            queries = [candidates[i] for i in chosen.tolist()]
            # Only now are hidden labels revealed; no candidate label was used above.
            oracle.reveal(queries)
            events.append({"after_epoch": epoch + 1, "indices": queries, "labels": [oracle.known[i] for i in queries],
                           "total_labels": len(oracle.known), "candidate_indices_sha256": __import__('hashlib').sha256(np.asarray(candidates, dtype=np.int64).tobytes()).hexdigest()})
            if diagnostics is not None:
                events[-1]["acquisition_diagnostics"] = diagnostics
            pilot, loader = make_training(pilot.state_dict())
        row = {"epoch": epoch + 1, "updates": (epoch + 1) * steps,
               "train_label_budget": used_labels, "label_budget_after_acquisition": len(oracle.known),
               "train_loss": trained["loss"], "train_accuracy": trained["accuracy"],
               "val_accuracy": validated["accuracy"], "val_loss": validated["loss"],
               "epoch_seconds": time.perf_counter() - tick,
               **{k: v for k, v in trained.items() if k not in {"loss", "accuracy"}}, **soft}
        if config.get("active_selection_version", 1) == 2:
            row["acquisition_seconds"] = acquisition_seconds
        history.append(row)
        state = {"config": resolved, "epoch": epoch + 1, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "history": history,
                 "rng": rng_state(generator), "pilot_state": pilot.state_dict(), "known_indices": list(oracle.known),
                 "validation_indices": validation, "events": events, "candidate_rng": candidate_rng.get_state(),
                 "selection_rng": selection_rng.get_state(), "scoring_views": scoring_views,
                 "elapsed_seconds": elapsed_before + time.perf_counter() - started}
        torch.save(state, run_dir / "latest.pending.pt")
        (run_dir / "latest.pending.pt").replace(run_dir / "latest.pt")
        write_json(run_dir / "splits.json", {"train": list(oracle.known), "validation": validation})
        write_json(run_dir / "acquisitions.json", events)
        with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(f"epoch={epoch+1}/{config['epochs']} labels={used_labels}->{len(oracle.known)} val={validated['accuracy']:.2%} seconds={row['epoch_seconds']:.1f}", flush=True)
        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch and epoch + 1 < config["epochs"]:
            return None
    if len(oracle.known) != config["active_final_budget"]:
        raise ValueError("Final annotation budget was not reached.")
    if test is None:
        test = datasets.CIFAR10(data_root, train=False, download=False, transform=cifar_transform(config, False))
    tested = evaluate(model, DataLoader(test, batch_size=256), device, normalize=False)
    result = {**resolved, "status": "synthetic_smoke_only" if smoke else "completed",
              "val_accuracy": history[-1]["val_accuracy"], "test_accuracy": tested["accuracy"],
              "test_size": tested["total"], "selected_epoch": config["epochs"],
              "labeled_train_size": len(oracle.known), "true_training_label_budget": len(oracle.known),
              "final_class_counts": np.bincount(list(oracle.known.values()), minlength=10).tolist(),
              "scoring_forward_views": scoring_views, "elapsed_seconds": elapsed_before + time.perf_counter() - started,
              **{key: sum(r[key] for r in history) for key in ["processed_labeled_views", "processed_unlabeled_views", "pseudo_accepted", "pseudo_attempted"]}}
    for key in ("pseudo_weight_sum", "triplet_u_pairs", "triplet_u_active",
                "neighbor_reference_views", "neighbor_query_views", "neighbor_reference_seconds",
                "neighbor_eligible", "neighbor_reweighted", "neighbor_js_sum", "neighbor_raw_factor_sum", "pseudo_base_weight_sum",
                "triplet_l_valid", "triplet_l_active", "triplet_l_semihard",
                "pseudo_triplet_valid", "pseudo_triplet_attempted", "pseudo_triplet_active", "pseudo_triplet_semihard",
                "pseudo_triplet_labeled_positives", "pseudo_triplet_labeled_negatives",
                "pseudo_triplet_pseudo_positives", "pseudo_triplet_pseudo_negatives"):
        if any(key in row for row in history):
            result[key] = sum(row.get(key, 0) for row in history)
    for key in history[0]:
        if key.startswith("metric_"):
            result[key] = sum(row[key] for row in history)
    torch.save(model.state_dict(), run_dir / "model.pt")
    write_json(run_dir / "metrics.json", result)
    return result
