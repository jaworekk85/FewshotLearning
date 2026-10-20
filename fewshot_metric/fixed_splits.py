"""Frozen annotation sets selected by completed AL runs, with provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def validate_indices(values, population_size, name):
    if not isinstance(values, list) or any(isinstance(i, bool) or not isinstance(i, int) for i in values):
        raise ValueError(f"{name} must be a list of integer indices.")
    if len(set(values)) != len(values) or any(i < 0 or i >= population_size for i in values):
        raise ValueError(f"{name} contains duplicates or out-of-range indices.")


def load_fixed_split(config, labels):
    """Validate the snapshot before constructing datasets or exposing labels."""
    path = Path(config["fixed_split_manifest"])
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != config["fixed_split_manifest_sha256"]:
        raise ValueError("Fixed split manifest hash changed.")
    manifest = json.loads(data)
    size = len(labels)
    if manifest.get("dataset") != "cifar10" or manifest.get("population_size") != size:
        raise ValueError("Fixed split dataset or population differs.")
    train, val = manifest["train"], manifest["validation"]
    validate_indices(train, size, "train")
    validate_indices(val, size, "validation")
    if set(train) & set(val):
        raise ValueError("Fixed train and validation overlap.")
    if len(train) != 10 * int(config["budget_per_class"]) or len(val) != 10 * int(config["validation_per_class"]):
        raise ValueError("Fixed split annotation budget differs.")
    known = manifest["known_labels"]
    if (not isinstance(known, list) or len(known) != len(train)
            or any(isinstance(y, bool) or not isinstance(y, int) or not 0 <= y < 10 for y in known)):
        raise ValueError("Invalid revealed label snapshot.")
    if not np.array_equal(np.asarray(known), np.asarray(labels)[train]):
        raise ValueError("Previously revealed labels differ from the dataset.")
    visible = np.full(size, -1, dtype=int)
    visible[train] = known
    return {"train": train, "validation": val}, visible


def manifest_from_active_run(run_dir, labels, strategy):
    """Freeze only labels that this run already obtained, never select anew."""
    root = Path(run_dir)
    contents = {name: (root/name).read_bytes() for name in
                ("config.json", "metrics.json", "splits.json", "acquisitions.json")}
    cfg, metrics, splits, events = (json.loads(contents[name]) for name in contents)
    if (metrics.get("status") != "completed" or metrics.get("synthetic_smoke")
            or cfg.get("dataset") != "cifar10" or cfg.get("active_strategy") != strategy
            or cfg.get("method") != "ce_2v_pseudo_soft" or cfg.get("epochs") != 20
            or cfg.get("seed") != 0 or metrics.get("true_training_label_budget") != 500):
        raise ValueError("Expected a completed real seed-0 soft-SSL AL run with 500 final labels.")
    initial = cfg["initial_train_indices"]
    acquired = [i for event in events for i in event["indices"]]
    validate_indices(initial + acquired, len(labels), "annotation history")
    if len(initial) != 300 or len(acquired) != 200 or set(initial + acquired) != set(splits["train"]):
        raise ValueError("Final split differs from actually acquired annotations.")
    validate_indices(splits["train"], len(labels), "train")
    validate_indices(splits["validation"], len(labels), "validation")
    if len(splits["validation"]) != 1000 or set(splits["train"]) & set(splits["validation"]):
        raise ValueError("Invalid original validation split.")
    for event in events:
        if len(event["indices"]) != len(event["labels"]):
            raise ValueError("Acquisition labels are not aligned.")
        if any(int(labels[i]) != y for i, y in zip(event["indices"], event["labels"])):
            raise ValueError("Acquisition labels differ from previously revealed annotations.")
    train = sorted(splits["train"])
    return {"dataset": "cifar10", "population_size": len(labels), "train": train,
            "validation": splits["validation"], "known_labels": [int(labels[i]) for i in train],
            "source_run": str(root), "selection_strategy": strategy,
            "source_artifact_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
            "source_initial_model_sha256": cfg["initial_model_sha256"],
            "training_order_policy": "sorted_indices_then_seeded_shuffle"}
