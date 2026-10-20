from __future__ import annotations

import json
import re
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


METHOD_DEFAULTS: dict[str, Any] = {
    "paper_label": None,
    "objective": "ce",
    "updates": 500,
    "views": 1,
    "augmentation": "none",
    "normalize_embedding": False,
    "unlabeled_objective": "none",
    "unlabeled_augmentation": "strong",
    "unlabeled_views": 2,
    "unlabeled_weight": 0.0,
    "unlabeled_limit": None,
    "pseudo_label_threshold": 0.95,
    "unlabeled_triplet_margin": 0.2,
    "unlabeled_negative_strategy": "pseudo_label_hard",
    "grid": "custom",
}

VALID_OBJECTIVES = {"ce", "triplet", "supcon_joint"}
VALID_AUGMENTATIONS = {"none", "weak", "strong"}
VALID_UNLABELED_OBJECTIVES = {"none", "pseudo_ce", "consistency", "simclr", "pseudo_triplet"}
VALID_UNLABELED_NEGATIVE_STRATEGIES = {"random", "pseudo_label_hard", "pseudo_label_or_random"}


@dataclass(frozen=True)
class RunSpec:
    dataset: str
    architecture: str
    budget: int
    method_id: str
    seed: int
    method: dict[str, Any]
    protocol: dict[str, Any]


def load_protocol(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9_+-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "custom"


def normalize_method(method: dict[str, Any], method_id: str | None = None) -> dict[str, Any]:
    normalized = deepcopy(METHOD_DEFAULTS)
    normalized.update(deepcopy(method))
    normalized["updates"] = int(normalized["updates"])
    normalized["views"] = int(normalized["views"])
    normalized["normalize_embedding"] = bool(normalized["normalize_embedding"])
    normalized["unlabeled_views"] = int(normalized["unlabeled_views"])
    normalized["unlabeled_weight"] = float(normalized["unlabeled_weight"])
    normalized["pseudo_label_threshold"] = float(normalized["pseudo_label_threshold"])
    normalized["unlabeled_triplet_margin"] = float(normalized["unlabeled_triplet_margin"])
    if normalized["unlabeled_limit"] is not None:
        normalized["unlabeled_limit"] = int(normalized["unlabeled_limit"])
    if normalized["paper_label"] is None:
        normalized["paper_label"] = method_id or describe_method(normalized)
    validate_method(normalized)
    return normalized


def validate_method(method: dict[str, Any]) -> None:
    if method["objective"] not in VALID_OBJECTIVES:
        raise ValueError(f"Unknown objective {method['objective']!r}. Valid: {sorted(VALID_OBJECTIVES)}.")
    if method["augmentation"] not in VALID_AUGMENTATIONS:
        raise ValueError(f"Unknown augmentation {method['augmentation']!r}. Valid: {sorted(VALID_AUGMENTATIONS)}.")
    if method["unlabeled_objective"] not in VALID_UNLABELED_OBJECTIVES:
        raise ValueError(
            f"Unknown unlabeled objective {method['unlabeled_objective']!r}. "
            f"Valid: {sorted(VALID_UNLABELED_OBJECTIVES)}."
        )
    if method["unlabeled_negative_strategy"] not in VALID_UNLABELED_NEGATIVE_STRATEGIES:
        raise ValueError(
            f"Unknown unlabeled negative strategy {method['unlabeled_negative_strategy']!r}. "
            f"Valid: {sorted(VALID_UNLABELED_NEGATIVE_STRATEGIES)}."
        )
    if method["unlabeled_augmentation"] not in VALID_AUGMENTATIONS:
        raise ValueError(
            f"Unknown unlabeled augmentation {method['unlabeled_augmentation']!r}. "
            f"Valid: {sorted(VALID_AUGMENTATIONS)}."
        )
    if int(method["updates"]) <= 0:
        raise ValueError("updates must be positive.")
    if int(method["views"]) <= 0:
        raise ValueError("views must be positive.")
    if method["objective"] == "supcon_joint" and int(method["views"]) < 2:
        raise ValueError("supcon_joint requires at least two views.")
    if method["unlabeled_objective"] in {"consistency", "simclr", "pseudo_triplet"} and int(method["unlabeled_views"]) < 2:
        raise ValueError(f"{method['unlabeled_objective']} requires at least two unlabeled views.")
    if float(method["unlabeled_weight"]) < 0:
        raise ValueError("unlabeled_weight must be non-negative.")
    if method["unlabeled_limit"] is not None and int(method["unlabeled_limit"]) < 0:
        raise ValueError("unlabeled_limit must be non-negative or null.")


def describe_method(method: dict[str, Any]) -> str:
    parts = [
        str(method["objective"]),
        str(method["augmentation"]),
        f"{int(method['views'])}v",
        f"{int(method['updates'])}u",
    ]
    if bool(method["normalize_embedding"]):
        parts.append("l2")
    if method.get("unlabeled_objective", "none") != "none":
        parts.append(f"u_{method['unlabeled_objective']}")
    return slugify("_".join(parts))


def compose_method(
    protocol: dict[str, Any],
    base_method_id: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    overrides = {key: value for key, value in (overrides or {}).items() if value is not None}
    run_name = overrides.pop("run_name", None)
    if base_method_id is not None:
        if base_method_id not in protocol["methods"]:
            raise ValueError(f"Unknown method preset {base_method_id!r}.")
        method = normalize_method(protocol["methods"][base_method_id], method_id=base_method_id)
        method_id = base_method_id
    else:
        method = normalize_method({}, method_id="custom")
        method_id = "custom"

    method.update(deepcopy(overrides))
    method = normalize_method(method)
    if overrides:
        method_id = slugify(run_name or describe_method(method))
        method["paper_label"] = method_id
        method["grid"] = "custom"
    elif run_name:
        method_id = slugify(run_name)
        method["paper_label"] = method_id
    return method_id, method


def override_protocol(protocol: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    updated = deepcopy(protocol)
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in {"learning_rate", "batch_size", "weight_decay"}:
            updated["optimizer"][key] = value
        elif key in {"embedding_dim", "validation_per_class", "num_classes"}:
            updated[key] = value
        elif key in {"ce_weight", "metric_weight", "triplet_margin", "supcon_temperature"}:
            updated["loss_defaults"][key] = value
        else:
            raise ValueError(f"Unsupported protocol override {key!r}.")
    return updated


def make_run_spec(
    protocol: dict[str, Any],
    dataset: str,
    architecture: str,
    budget: int,
    method_id: str,
    seed: int,
    method: dict[str, Any] | None = None,
) -> RunSpec:
    if dataset not in protocol["datasets"]:
        raise ValueError(f"Unknown dataset {dataset!r}.")
    if architecture not in protocol["architectures"]:
        raise ValueError(f"Unknown architecture {architecture!r}.")
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if method is None and method_id not in protocol["methods"]:
        raise ValueError(f"Unknown method {method_id!r}.")
    if method is None:
        method = protocol["methods"][method_id]

    return RunSpec(
        dataset=dataset,
        architecture=architecture,
        budget=budget,
        method_id=method_id,
        seed=seed,
        method=normalize_method(method, method_id=method_id),
        protocol=protocol,
    )
