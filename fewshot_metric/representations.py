"""Reusable validation-only representation audit for any pilot checkpoint."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torchvision import datasets

from .active import encode_images
from .models import build_model
from .supervised import cifar_transform, image_array, source_dataset, write_json


def representation_metrics(train_z, train_labels, val_z, val_labels, neighbors=(1, 5, 15)):
    """Cosine neighborhood and scatter diagnostics in the original embedding."""
    train_z, val_z = F.normalize(train_z, dim=1), F.normalize(val_z, dim=1)
    train_labels, val_labels = torch.as_tensor(train_labels), torch.as_tensor(val_labels)
    classes = int(max(train_labels.max(), val_labels.max())) + 1
    similarity = val_z @ train_z.T
    ranked = similarity.argsort(dim=1, descending=True, stable=True)
    result = {}
    for k in neighbors:
        k = min(k, len(train_z))
        labels = train_labels[ranked[:, :k]]
        prediction = F.one_hot(labels, num_classes=classes).sum(1).argmax(1)
        result[f"knn_{k}_val_accuracy"] = float((prediction == val_labels).float().mean())
    vv = val_z @ val_z.T
    vv.fill_diagonal_(-float("inf"))
    k = min(10, len(val_z) - 1)
    nearest = vv.topk(k, dim=1).indices
    result["val_neighbor_purity_10"] = float((val_labels[nearest] == val_labels[:, None]).float().mean())
    centered = val_z - val_z.mean(0)
    total = float(centered.square().sum())
    within = sum(float((val_z[val_labels == c] - val_z[val_labels == c].mean(0)).square().sum()) for c in val_labels.unique())
    result["within_over_total_scatter"] = within / total if total else 0.0
    eigen = torch.linalg.svdvals(centered).square()
    distribution = eigen / eigen.sum().clamp_min(1e-12)
    result["effective_rank"] = float((-(distribution * distribution.clamp_min(1e-12).log()).sum()).exp())
    result["normalized_feature_variance"] = float(centered.square().mean())
    return result


def analyze_run(run_dir, output_dir, data_root="data", tsne=True, *, reference_indices=None,
                validation_indices=None, model_file="model.pt", pca_plot=True):
    """Exports embeddings, PCA/t-SNE, kNN, linear probe and metric diagnostics.

    Labels from U/test are never inspected. Probe fits only the run's true L.
    Plots and measurements use the same held-out validation images per seed.
    """
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir, output_dir = Path(run_dir), Path(output_dir)
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("synthetic_smoke"):
        raise ValueError("Synthetic fixtures are not representation research results.")
    splits = json.loads((run_dir / "splits.json").read_text())
    train_indices = list(splits["train"] if reference_indices is None else reference_indices)
    val_indices = list(splits["validation"] if validation_indices is None else validation_indices)
    for indices, allowed in ((train_indices, splits["train"]), (val_indices, splits["validation"])):
        if not indices or len(indices) != len(set(indices)) or not set(indices) <= set(allowed):
            raise ValueError("Audit indices must be unique subsets of the model's corresponding source split")
    if set(train_indices) & set(val_indices):
        raise ValueError("Probe reference and validation overlap")
    if model_file not in {"model.pt", "teacher_model.pt"}:
        raise ValueError("Unknown representation checkpoint")
    model_path = run_dir / model_file
    source_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    signature = {"checkpoint_sha256": source_sha, "splits_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
                 "tsne": bool(tsne), "audit_version": 1}
    if config.get("dataset", "cifar10") != "cifar10" or not pca_plot:
        signature.update(dataset=config.get("dataset", "cifar10"), pca_plot=bool(pca_plot),
                         audit_version=2, config_sha256=hashlib.sha256((run_dir/"config.json").read_bytes()).hexdigest())
    if reference_indices is not None or validation_indices is not None or model_file != "model.pt":
        signature.update(audit_version=2, reference_indices=train_indices, validation_indices=val_indices,
                         model_file=model_file, config_sha256=hashlib.sha256((run_dir/"config.json").read_bytes()).hexdigest())
    destination = output_dir / (run_dir.name + ("__teacher" if model_file == "teacher_model.pt" else ""))
    if (destination / "metrics.json").exists():
        existing = json.loads((destination / "metrics.json").read_text())
        if existing.get("signature") != signature:
            raise ValueError(f"Representation result differs from current checkpoint: {destination}")
        if "normalization" not in existing:
            existing["normalization"] = config.get("normalization", "batchnorm")
            write_json(destination / "metrics.json", existing)
        return existing
    base = source_dataset(config, Path(data_root), True)
    model = build_model(config["architecture"], embedding_dim=config["embedding_dim"],
                        normalization=config.get("normalization", "batchnorm"))
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    transform = cifar_transform(config, False)
    z, probs = encode_images(model, image_array(base), train_indices + val_indices, transform, torch.device("cpu"))
    n = len(train_indices)
    train_labels = np.asarray([base.targets[i] for i in train_indices])
    val_labels = np.asarray([base.targets[i] for i in val_indices])
    train_z, val_z = z[:n], z[n:]
    result = {"run": run_dir.name, "signature": signature, "evaluation_split": "validation",
              "normalization": config.get("normalization", "batchnorm"),
              "labeled_reference_size": n, "validation_size": len(val_labels),
              "encoder_features": "l2_normalized_embedding", "probe_training_labels": n,
              **representation_metrics(train_z, train_labels, val_z, val_labels)}
    if config.get("dataset", "cifar10") != "cifar10":
        result["dataset"] = config["dataset"]
    if signature["audit_version"] == 2:
        result.update(run=destination.name, source_training_label_budget=len(splits["train"]), checkpoint=model_file)
    scaler = StandardScaler().fit(train_z.numpy())
    probe = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(scaler.transform(train_z.numpy()), train_labels)
    result["linear_probe_val_accuracy"] = float(probe.score(scaler.transform(val_z.numpy()), val_labels))
    result["linear_probe_iterations"] = int(probe.n_iter_.max())
    result["original_head_val_accuracy"] = float((probs[n:].argmax(1).numpy() == val_labels).mean())
    result["val_silhouette_cosine"] = float(silhouette_score(val_z.numpy(), val_labels, metric="cosine"))
    # Measure violations under known validation classes without training on them.
    rng = np.random.default_rng(0)
    losses = []
    for i, label in enumerate(val_labels):
        positive = np.flatnonzero((val_labels == label) & (np.arange(len(val_labels)) != i))
        negative = np.flatnonzero(val_labels != label)
        if len(positive) and len(negative):
            p, neg = int(rng.choice(positive)), int(rng.choice(negative))
            losses.append(max(float((val_z[i] - val_z[p]).norm() - (val_z[i] - val_z[neg]).norm()) + 0.2, 0))
    result["val_random_triplet_active_fraction"] = float(np.mean(np.asarray(losses) > 0))
    result["val_random_triplet_mean_loss"] = float(np.mean(losses))
    destination.mkdir(parents=True, exist_ok=True)
    coordinates = {"pca": PCA(n_components=2, svd_solver="full").fit_transform(val_z.numpy())} if pca_plot else {}
    if tsne:
        reduced = PCA(n_components=min(30, val_z.shape[1], len(val_z) - 1), svd_solver="full").fit_transform(val_z.numpy())
        coordinates["tsne"] = TSNE(n_components=2, perplexity=min(30, (len(val_z)-1)/3), random_state=0,
                                    init="pca", learning_rate="auto", max_iter=1000).fit_transform(reduced)
    names = base.classes
    for method, xy in coordinates.items():
        fig, ax = plt.subplots(figsize=(8, 6))
        for cls, name in enumerate(names):
            selected = val_labels == cls
            ax.scatter(xy[selected, 0], xy[selected, 1], s=12, alpha=0.65, label=name)
        ax.set_title(f"{run_dir.name}\n{method.upper()} - validation; geometry is diagnostic")
        ax.set_xlabel(f"{method.upper()} 1")
        ax.set_ylabel(f"{method.upper()} 2")
        ax.legend(fontsize=8, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(destination / f"{method}.png", dpi=180)
        fig.savefig(destination / f"{method}.svg")
        plt.close(fig)
    np.savez_compressed(destination / "embeddings.npz", train=train_z.numpy(), validation=val_z.numpy(),
                        train_labels=train_labels, validation_labels=val_labels,
                        train_indices=train_indices, validation_indices=val_indices, **coordinates)
    write_json(destination / "metrics.json", result)
    return result


def analyze_many(runs, output_dir, data_root="data", tsne=True):
    rows = []
    for run in runs:
        print(f"Representation audit: {run}", flush=True)
        rows.append(analyze_run(run, output_dir, data_root, tsne))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    keys = sorted(set().union(*(row.keys() for row in rows)) - {"signature"})
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows({k: row.get(k) for k in keys} for row in rows)
    write_json(output_dir / "comparison.json", rows)
    return rows
