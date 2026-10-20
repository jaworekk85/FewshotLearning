"""Post-training-only plots of all CIFAR training images; U truth is for colors only."""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from threadpoolctl import threadpool_limits
from torchvision.datasets import CIFAR10

from fewshot_metric.active import encode_images
from fewshot_metric.models import build_model
from fewshot_metric.supervised import cifar_transform


DEFAULT_RUNS = [
    "outputs/pilot_validation_study/cnn_N50_seed0_ce_2v_pseudo_soft_80epochs",
    "outputs/supervised/cnn_pilot_full_seed0",
]
REFERENCE = "outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft/splits.json"


def analyze(run_dir, output, data, reference, max_iter=1000):
    run_dir, output = Path(run_dir), Path(output)
    destination = output / run_dir.name
    destination.mkdir(parents=True, exist_ok=True)
    config_path, checkpoint = run_dir / "config.json", run_dir / "model.pt"
    config = json.loads(config_path.read_text())
    if config.get("synthetic_smoke"):
        raise ValueError("Only real CIFAR models are eligible")
    total = len(data.data)
    labeled = np.asarray(reference["train"], dtype=int)
    validation = np.asarray(reference["validation"], dtype=int)
    if len(set(labeled) | set(validation)) != len(labeled) + len(validation):
        raise ValueError("Labeled and validation indices overlap or contain duplicates")
    role = np.full(total, 1, dtype=np.uint8)  # 0=L, 1=U, 2=validation under common SSL protocol
    role[labeled], role[validation] = 0, 2
    actual_splits = json.loads((run_dir / "splits.json").read_text())
    if set(actual_splits["validation"]) != set(validation):
        raise ValueError("Models must have the same held-out validation set")
    if set(actual_splits["train"]) & set(validation):
        raise ValueError("Model training labels overlap the validation set")
    signature = {
        "version": 1, "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        "image_count": total, "order": "original CIFAR training index, ascending", "max_iter": max_iter,
        "tsne": {"perplexity": 30, "init": "pca", "learning_rate": "auto", "random_state": 0,
                 "method": "barnes_hut", "angle": .5, "n_jobs": 1, "pca_dimensions": 30},
    }
    signature_path = destination / "signature.json"
    if signature_path.exists():
        if json.loads(signature_path.read_text()) != signature:
            raise ValueError(f"Cached source/projection settings differ: {destination}")
    else:
        signature_path.write_text(json.dumps(signature, indent=2) + "\n")
    features_path = destination / "frozen_features.npz"
    started = time.monotonic()
    if features_path.exists():
        with np.load(features_path) as archive:
            features, probabilities = archive["features"], archive["probabilities"]
    else:
        print(f"{run_dir.name}: encoding all {total} images (eval only)", flush=True)
        model = build_model(config["architecture"], embedding_dim=config["embedding_dim"],
                            normalization=config.get("normalization", "batchnorm"))
        model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        z, p = encode_images(model, data.data, range(total), cifar_transform(config, False), torch.device("cpu"))
        features, probabilities = z.numpy(), p.numpy()
        np.savez_compressed(features_path, features=features, probabilities=probabilities,
                            image_indices=np.arange(total), common_ssl_role=role)
        del model, z, p
    print(f"{run_dir.name}: features ready in {time.monotonic()-started:.1f}s", flush=True)
    coordinates_path = destination / "coordinates.npz"
    projection_threads = torch.get_num_threads()
    if coordinates_path.exists():
        with np.load(coordinates_path) as archive:
            coordinates = {name: archive[name] for name in ("pca", "tsne")}
        old_report = destination / "report.json"
        projection_threads = json.loads(old_report.read_text()).get("projection_cpu_threads", 1) if old_report.exists() else 1
    else:
        pca = PCA(n_components=30, svd_solver="full", random_state=0)
        reduced = pca.fit_transform(features)
        print(f"{run_dir.name}: t-SNE of all {total} points; no labels passed to projection", flush=True)
        projector = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
                         random_state=0, max_iter=max_iter, n_jobs=1, verbose=2)
        coordinates = {"pca": reduced[:, :2], "tsne": projector.fit_transform(reduced)}
        np.savez_compressed(coordinates_path, **coordinates, image_indices=np.arange(total))
    # Ground truth for U is read only after the label-free projection, for diagnostic colors.
    truth = np.asarray(data.targets)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    source_labels = len(actual_splits["train"])
    method_title = "Full supervision" if source_labels == 49000 else config.get("method", "CNN")
    title = f"{method_title}, {config['epochs']} epochs, {source_labels:,} training labels"
    handles = [Line2D([], [], marker="o", linestyle="none", color=plt.get_cmap("tab10")(i), label=name)
               for i, name in enumerate(data.classes)]
    rng = np.random.default_rng(0)
    shuffled = rng.permutation(total)  # Avoid hiding one class under another via plotting order.
    u_title = "Unlabeled reference U (48,500)"
    if set(np.flatnonzero(role == 1)) <= set(actual_splits["train"]):
        u_title = "Remaining reference pool (48,500)\nLabels used by this fully supervised CNN"
    panels = [("All 50,000 images", np.ones(total, dtype=bool)),
              ("Labeled reference L (500)", role == 0),
              (u_title, role == 1),
              ("Held-out validation (1,000)", role == 2)]
    for method, xy in coordinates.items():
        fig, axes = plt.subplots(2, 2, figsize=(13, 11), sharex=True, sharey=True)
        for ax, (name, mask) in zip(axes.flat, panels):
            selected = shuffled[mask[shuffled]]
            ax.scatter(xy[selected, 0], xy[selected, 1], c=truth[selected], cmap="tab10",
                       vmin=0, vmax=9, s=2 if len(selected)>1000 else 10,
                       alpha=.45 if len(selected)>1000 else .8, linewidths=0, rasterized=True)
            ax.set_title(name)
            ax.set_xticks([])
            ax.set_yticks([])
        fig.suptitle(title + f"\n{method.upper()}: one joint projection; colors = true classes (diagnostic only)")
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=10)
        fig.tight_layout(rect=(0, .055, 1, .94))
        fig.savefig(destination / f"{method}_full_pool.png", dpi=170)
        fig.savefig(destination / f"{method}_full_pool.svg")
        plt.close(fig)
    row = {"run": str(run_dir), "signature": signature, "source_training_labels": len(actual_splits["train"]),
           "common_role_counts": {"labeled": int((role==0).sum()), "unlabeled": int((role==1).sum()),
                                  "validation": int((role==2).sum())},
           "total_seconds": time.monotonic()-started,
           "projection_cpu_threads": projection_threads,
           "diagnostic_only": True, "truth_used_for_projection_or_fitting": False,
           "note": "All 50k CIFAR training-pool images, including 1k held-out validation; no official test images. Common SSL roles are display references. The full-supervised CNN saw labels for all 49k non-validation images, including the common U reference. Never use U truth from this diagnostic for training or classifier selection."}
    (destination / "report.json").write_text(json.dumps(row, indent=2) + "\n")
    print(f"{run_dir.name}: plots saved; total {row['total_seconds']:.1f}s", flush=True)
    return row


def main():
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", default=DEFAULT_RUNS)
    parser.add_argument("--output", default="outputs/representation_full_pool")
    parser.add_argument("--data", default="data")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    Path(args.output).mkdir(parents=True, exist_ok=True)
    (Path(args.output) / "process.json").write_text(json.dumps({
        "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_threads": args.threads, "runs": args.runs, "argv": sys.argv,
    }, indent=2) + "\n")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output).resolve() / ".mplconfig"))
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    reference = json.loads(Path(REFERENCE).read_text())
    data = CIFAR10(args.data, train=True, download=False)
    with threadpool_limits(limits=args.threads):
        rows = [analyze(run, args.output, data, reference, args.max_iter) for run in args.runs]
    out = Path(args.output)
    (out / "comparison.json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Full-pool representation diagnostic", "", rows[0]["note"], "",
             "Each model has a single label-free PCA/t-SNE projection fitted to all 50,000 images. Four panels show all images and their common L/U/validation subsets in identical coordinates. Coordinate systems differ between models and from earlier validation-only projections. Colors use ground truth only after projection; no model/head fitting or label-budget change.", ""]
    for row in rows:
        name = Path(row["run"]).name
        lines += [f"## {name}", "", f"![t-SNE full pool]({name}/tsne_full_pool.png)", ""]
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
