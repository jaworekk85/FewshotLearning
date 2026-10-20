"""Measure synthetic training throughput; estimates exclude data loading and validation."""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--architecture", default="cifar_resnet18", choices=["cnn", "resnet", "cifar_resnet18"])
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.steps <= 0 or args.warmup < 0 or args.epochs <= 0:
        parser.error("Batch size, epochs and steps must be positive; warmup must be non-negative.")
    if args.embedding_dim is not None and args.embedding_dim <= 0:
        parser.error("Embedding dimension must be positive.")
    import torch
    from torch.nn import functional as F
    from fewshot_metric.models import build_model
    from fewshot_metric.train import seed_everything

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("This PyTorch environment does not have access to CUDA.")
    seed_everything(0)
    embedding_dim = args.embedding_dim or (512 if args.architecture == "cifar_resnet18" else 128)
    model = build_model(args.architecture, embedding_dim=embedding_dim).to(device).train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    images = torch.randn(args.batch_size, 3, 32, 32, device=device)
    labels = torch.arange(args.batch_size, device=device) % 10
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    times = []
    for step in range(args.warmup + args.steps):
        if device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        _, logits = model(images)
        F.cross_entropy(logits, labels).backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if step >= args.warmup:
            times.append(elapsed)
    median = statistics.median(times)
    reference_updates = args.epochs * math.ceil(49000 / args.batch_size)
    result = {
        "device": str(device), "torch_version": str(torch.__version__),
        "gpu_name": torch.cuda.get_device_name() if device.type == "cuda" else None,
        "cpu_threads": torch.get_num_threads(), "precision": "float32",
        "architecture": args.architecture, "embedding_dim": embedding_dim,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "batch_size": args.batch_size, "warmup_steps": args.warmup, "measured_steps": args.steps,
        "median_step_seconds": median, "step_seconds": times,
        "training_images_per_second": args.batch_size / median,
        "reference_epochs": args.epochs, "reference_training_images": 49000,
        "reference_updates": reference_updates,
        "estimated_training_hours_excluding_io_validation": median * reference_updates / 3600,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else None,
        "research_accuracy_measured": False,
    }
    print(json.dumps(result, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
