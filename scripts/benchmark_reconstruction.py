"""Short synthetic timing for rough AE/VAE ETA, not an accuracy experiment."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewshot_metric.models import build_model
from fewshot_metric.reconstruction import ReconstructionModel, auxiliary_metadata, reconstruction_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output", default="outputs/pilot_reconstruction/benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())
    measurements = []
    for variant in ("ce_u_control", "dae_pseudo", "dvae_pseudo"):
        torch.manual_seed(0)
        config = reconstruction_config(base, variant)
        model = ReconstructionModel(build_model("cnn", embedding_dim=128), config)
        parameter_counts = {k: v for k, v in auxiliary_metadata(model).items() if "parameter_count" in k}
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        inputs = torch.randn(512, 3, 32, 32)
        labels = (torch.arange(512) % 10).long()
        target = torch.rand(128, 3, 32, 32)
        latent_rng = torch.Generator().manual_seed(60409)
        timings = []
        for step in range(args.steps+2):
            started = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            features, logits = model(inputs)
            # Force supervised-like signal through all views for a stable compute estimate.
            loss = F.cross_entropy(logits, labels)
            if model.decoder is not None:
                reconstruction, kl = model.decode(features[-128:], latent_rng)
                loss = loss + F.mse_loss(reconstruction, target) + 0.01 * kl
            loss.backward()
            optimizer.step()
            if step >= 2:
                timings.append(time.perf_counter()-started)
        measurements.append({"variant": variant, "median_step_seconds": statistics.median(timings),
                             "step_seconds": timings, **parameter_counts})
    reference = measurements[0]["median_step_seconds"]
    for row in measurements:
        row["relative_to_no_decoder"] = row["median_step_seconds"] / reference
    result = {"synthetic_timing_only": True, "cpu_threads": args.threads,
              "encoder_views_per_step": 512, "decoder_views_per_step_if_present": 128,
              "note": "Short resident-tensor benchmark; concurrent training and thread count limit ETA precision.",
              "measurements": measurements}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
