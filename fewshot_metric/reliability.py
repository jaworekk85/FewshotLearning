"""Post-hoc validation diagnostics; never acquires labels or updates a model."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets

from .active_selection import entropy, js_divergence, neighbor_probabilities
from .models import build_model
from .pilot import KeyedViews, view_transform
from .supervised import cifar_transform, write_json


def calibration(probabilities, labels, bins=15):
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    confidence, prediction = p.max(1), p.argmax(1)
    correct = prediction == y
    assignments = np.minimum((confidence * bins).astype(int), bins - 1)
    reliability = []
    ece = 0.0
    for b in range(bins):
        selected = assignments == b
        n = int(selected.sum())
        acc = float(correct[selected].mean()) if n else None
        conf = float(confidence[selected].mean()) if n else None
        if n:
            ece += n / len(y) * abs(acc - conf)
        reliability.append(dict(lower=b / bins, upper=(b + 1) / bins, count=n,
                                accuracy=acc, confidence=conf))
    thresholds = []
    for threshold in (0.7, 0.8, 0.9, 0.95, 0.99):
        selected = confidence >= threshold
        thresholds.append(dict(threshold=threshold, count=int(selected.sum()),
                               wrong=int((selected & ~correct).sum()),
                               accuracy=float(correct[selected].mean()) if selected.any() else None,
                               coverage=float(selected.mean())))
    return dict(accuracy=float(correct.mean()), mean_confidence=float(confidence.mean()),
                nll=float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
                brier_sum_classes=float(((p - np.eye(p.shape[1])[y]) ** 2).sum(1).mean()),
                ece_15=float(ece), bins=reliability, thresholds=thresholds)


def hypothetical_weights(confidence, config, final_history):
    """Transfer saved next-epoch rule to final-model validation predictions."""
    if not config.get("pseudo_label_weight", 0):
        return None, {"rule": "not_applicable"}
    if config.get("pseudo_weighting", "hard") != "epoch_gaussian":
        threshold = float(config.get("pseudo_label_threshold", 0.95))
        return (confidence >= threshold).astype(float), {"rule": "hard", "threshold": threshold}
    center = float(final_history["pseudo_weight_center_next_epoch"])
    std = float(final_history["pseudo_weight_std_next_epoch"])
    if not np.isfinite(center) or not np.isfinite(std) or std <= 0:
        raise ValueError("Invalid saved Gaussian weight parameters")
    return np.exp(-np.minimum(confidence - center, 0) ** 2 / (2 * std ** 2)), {
        "rule": "epoch_gaussian", "center": center, "std": std,
        "interpretation": "hypothetical next-epoch weights, final checkpoint in eval mode"}


def weight_metrics(probabilities, labels, weights, config):
    if weights is None:
        return None
    correct = probabilities.argmax(1) == labels
    total = float(weights.sum())
    # Soft targets are sharpened, not their one-hot argmax approximation.
    if config.get("pseudo_target", "hard") == "soft":
        q = probabilities.astype(np.float64).clip(1e-30) ** (1 / float(config.get("pseudo_temperature", .5)))
        q /= q.sum(1, keepdims=True)
    else:
        q = np.eye(probabilities.shape[1])[probabilities.argmax(1)]
    return dict(mean_weight=float(weights.mean()),
                mean_weight_correct=float(weights[correct].mean()) if correct.any() else None,
                mean_weight_wrong=float(weights[~correct].mean()) if (~correct).any() else None,
                wrong_argmax_weight_fraction=float(weights[~correct].sum() / total) if total else None,
                target_mass_outside_true_class=float((weights * (1 - q[np.arange(len(labels)), labels])).sum() / total) if total else None,
                high_weight_wrong=int(((weights >= .9) & ~correct).sum()))


def detection_metrics(wrong, scores):
    from sklearn.metrics import average_precision_score, roc_auc_score
    wrong = np.asarray(wrong, dtype=bool)
    result = {}
    for name, score in scores.items():
        order = np.argsort(-np.asarray(score), kind="stable")
        row = dict(auroc=float(roc_auc_score(wrong, score)) if wrong.any() and not wrong.all() else None,
                   average_precision=float(average_precision_score(wrong, score)) if wrong.any() else None,
                   random_precision=float(wrong.mean()))
        for fraction in (.05, .1, .2):
            k = max(1, int(np.ceil(len(wrong) * fraction)))
            row[f"precision_top_{int(100 * fraction)}pct"] = float(wrong[order[:k]].mean())
        result[name] = row
    return result


def _infer(model, images, indices, transform, seed):
    loader = DataLoader(KeyedViews(images, indices, transform, seed), batch_size=128,
                        shuffle=False, num_workers=0, generator=torch.Generator().manual_seed(seed))
    features, logits = [], []
    model.eval()
    with torch.inference_mode():
        for x in loader:
            z, output = model(x)
            features.append(F.normalize(z, dim=1))
            logits.append(output)
    return torch.cat(features), torch.cat(logits)


def analyze_run(run_dir, output_dir, data_root="data", weak_views=4):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    source_files = ("model.pt", "config.json", "splits.json", "history.csv")
    signature = {name: hashlib.sha256((run_dir / name).read_bytes()).hexdigest() for name in source_files}
    signature.update(audit_version=1, weak_views=weak_views, augmentation_seed=710003,
                     neighbor_k=15, neighbor_temperature=.2)
    destination = output_dir / run_dir.name
    if (destination / "metrics.json").exists():
        existing = json.loads((destination / "metrics.json").read_text())
        if existing["signature"] != signature:
            raise ValueError(f"Diagnostic cache mismatch: {destination}")
        return existing
    config = json.loads((run_dir / "config.json").read_text())
    splits = json.loads((run_dir / "splits.json").read_text())
    if config.get("synthetic_smoke"):
        raise ValueError("Smoke runs are not research results")
    train, val = splits["train"], splits["validation"]
    if set(train) & set(val) or len(set(train)) != len(train) or len(set(val)) != len(val):
        raise ValueError("Training and validation indices must be unique and disjoint")
    with (run_dir / "history.csv").open(newline="") as handle:
        history = list(csv.DictReader(handle))
    base = datasets.CIFAR10(data_root, train=True, download=False)
    # Only the true labeled reference and held-out validation labels are read.
    train_y = np.array([base.targets[i] for i in train])
    y = np.array([base.targets[i] for i in val])
    with torch.random.fork_rng(devices=[]):
        model = build_model(config["architecture"], embedding_dim=config["embedding_dim"],
                            normalization=config.get("normalization", "batchnorm"))
    model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cpu", weights_only=True))
    features, logits = _infer(model, base.data, train + val, cifar_transform(config, False), 710003)
    z, p = features[len(train):], logits[len(train):].softmax(1)
    weak = []
    for view in range(weak_views):
        _, output = _infer(model, base.data, val, view_transform(config), 710003 + 13007 * (view + 1))
        weak.append(output.softmax(1))
    views = torch.stack(weak)
    q = neighbor_probabilities(z, features[:len(train)], torch.tensor(train_y), p.shape[1])
    scores = {"uncertainty": (1 - p.max(1).values).numpy(), "entropy": entropy(p).numpy(),
              "neighbor_js": js_divergence(p, q).numpy(),
              "weak_view_js": (entropy(views.mean(0)) - entropy(views).mean(0)).clamp_min(0).numpy()}
    probabilities = p.numpy()
    wrong = probabilities.argmax(1) != y
    weights, rule = hypothetical_weights(probabilities.max(1), config, history[-1])
    first_weak = views[0].numpy()
    weak_weights, _ = hypothetical_weights(first_weak.max(1), config, history[-1])
    similarity = z @ z.T
    similarity.fill_diagonal_(-torch.inf)
    nearest = similarity.topk(min(10, len(val) - 1), dim=1).indices.numpy()
    neighbor_errors = wrong[nearest].mean(1)
    result = dict(run=run_dir.name, signature=signature, seed=config["seed"], method=config["method"],
                  epochs=config["epochs"], evaluation_split="validation_only", labeled_reference_size=len(train),
                  validation_size=len(val), clean=calibration(probabilities, y),
                  weak=calibration(first_weak, y), weighting=rule,
                  clean_weights=weight_metrics(probabilities, y, weights, config),
                  weak_weights=weight_metrics(first_weak, y, weak_weights, config),
                  detection=detection_metrics(wrong, scores),
                  wrong_neighbor_fraction_for_wrong=float(neighbor_errors[wrong].mean()) if wrong.any() else None,
                  wrong_neighbor_fraction_for_correct=float(neighbor_errors[~wrong].mean()) if (~wrong).any() else None)
    result["per_class"] = []
    for cls, name in enumerate(base.classes):
        selected = y == cls
        result["per_class"].append(dict(name=name, count=int(selected.sum()),
            accuracy=float((~wrong[selected]).mean()) if selected.any() else None,
            confidently_wrong=int((selected & wrong & (probabilities.max(1) >= .95)).sum())))
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "predictions.npz", validation_indices=val, labels=y,
                        probabilities=probabilities, logits=logits[len(train):].numpy(), features=z.numpy(),
                        weak_probabilities=views.numpy(), neighbor_probabilities=q.numpy(),
                        weights=weights if weights is not None else np.array([]), **scores)
    _plot(result, probabilities, y, z.numpy(), destination)
    write_json(destination / "metrics.json", result)
    return result


def _plot(result, p, y, z, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    bins = [b for b in result["clean"]["bins"] if b["count"]]
    axes[0].plot([0, 1], [0, 1], "k--", linewidth=1)
    axes[0].plot([b["confidence"] for b in bins], [b["accuracy"] for b in bins], "o-")
    axes[0].set(xlabel="Confidence", ylabel="Accuracy", xlim=(0, 1), ylim=(0, 1), title="Calibration (15 bins)")
    wrong = p.argmax(1) != y
    axes[1].hist([p.max(1)[~wrong], p.max(1)[wrong]], bins=np.linspace(0, 1, 21), label=["Correct", "Wrong"])
    axes[1].legend()
    axes[1].set(xlabel="Confidence", ylabel="Images", title="Validation predictions")
    xy = PCA(n_components=2, svd_solver="full").fit_transform(z)
    for mask, color, label in ((~wrong, "steelblue", "Correct"), (wrong, "tomato", "Wrong")):
        axes[2].scatter(*xy[mask].T, c=color, s=9, alpha=.5, label=label)
    axes[2].set(title="PCA: errors in representation", xlabel="PC1", ylabel="PC2")
    axes[2].legend()
    fig.suptitle(result["run"], fontsize=10)
    fig.tight_layout()
    fig.savefig(destination / "diagnostics.png", dpi=160)
    fig.savefig(destination / "diagnostics.svg")
    plt.close(fig)


def write_report(rows, output_dir):
    output_dir = Path(output_dir)
    lines = ["# Pseudo-label reliability audit", "",
        "Validation only; no model updates, test inspection, calibration fitting or acquisition.",
        "Native weights below are hypothetical transfers of the saved next-epoch rule to final-model predictions in eval mode, not historical training weights.",
        "Weak diagnostics use the first of four fixed random crop/flip views. Error rankings use clean predictions and four weak views; all scores are label-free, with true validation labels used only for evaluation.",
        "ECE uses 15 equal-width confidence bins; Brier sums squared errors across all classes. AUROC measures error detection. Rankings and comparisons are exploratory, not independently validated acquisition methods.", "",
        "| Run | Val % | ECE pp | >=.95 count/wrong | Weak >=.95 count/wrong | Wrong weight mass % (weak) |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        high = next(t for t in r["clean"]["thresholds"] if t["threshold"] == .95)
        whigh = next(t for t in r["weak"]["thresholds"] if t["threshold"] == .95)
        wm = r["weak_weights"]
        mass = f'{100 * wm["wrong_argmax_weight_fraction"]:.1f}' if wm and wm["wrong_argmax_weight_fraction"] is not None else "n/a"
        lines.append(f'| {r["run"]} | {100*r["clean"]["accuracy"]:.2f} | {100*r["clean"]["ece_15"]:.2f} | {high["count"]}/{high["wrong"]} | {whigh["count"]}/{whigh["wrong"]} | {mass} |')
    lines += ["", "## Label-free error detection", "", "| Run / score | AUROC | Error precision in top 10% | Random precision |", "|---|---:|---:|---:|"]
    for r in rows:
        for name, d in r["detection"].items():
            auc = f'{d["auroc"]:.3f}' if d["auroc"] is not None else "n/a"
            lines.append(f'| {r["run"]} / {name} | {auc} | {100*d["precision_top_10pct"]:.1f}% | {100*d["random_precision"]:.1f}% |')
    lines += ["", "Each model directory contains full metrics (including per-class errors, weight mass and neighbor clustering), predictions.npz and diagnostics.png/svg.",
              "Neighbor reference uses only the run's true labeled training subset (k=15, cosine softmax temperature=0.2). Validation-to-validation neighborhoods (k=10) describe error clustering only.",
              "20/40/80-epoch checkpoints are separate runs with different cosine horizons, not snapshots of one trajectory. Seeds have different validation subsets; compare paired methods within seed.",
              "For soft targets, wrong-argmax weight mass is a proxy; target_mass_outside_true_class reports actual sharpened target mass assigned to incorrect classes. Neither is a measured training gradient or causal harm estimate."]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(output_dir / "comparison.json", rows)
