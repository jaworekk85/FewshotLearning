from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fewshot_metric.config import compose_method, load_protocol, make_run_spec, override_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one low-label label-only experiment.")
    parser.add_argument("--config", default="configs/v4_protocol.json")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--architecture", required=True, choices=["cnn", "resnet"])
    parser.add_argument("--budget", required=True, type=int)
    parser.add_argument("--method", help="Preset method id from the config. Omit to compose a method from defaults.")
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--run-name", help="Stable id used in output paths/results for a custom composed method.")
    parser.add_argument("--objective", choices=["ce", "triplet", "supcon_joint"])
    parser.add_argument("--updates", type=int)
    parser.add_argument("--views", type=int)
    parser.add_argument("--augmentation", choices=["none", "weak", "strong"])
    parser.add_argument("--normalize-embedding", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--unlabeled-objective", choices=["none", "pseudo_ce", "consistency", "simclr", "pseudo_triplet"])
    parser.add_argument("--unlabeled-augmentation", choices=["none", "weak", "strong"])
    parser.add_argument("--unlabeled-views", type=int)
    parser.add_argument("--unlabeled-weight", type=float)
    parser.add_argument("--unlabeled-limit", type=int)
    parser.add_argument("--pseudo-label-threshold", type=float)
    parser.add_argument("--unlabeled-triplet-margin", type=float)
    parser.add_argument(
        "--unlabeled-negative-strategy",
        choices=["random", "pseudo_label_hard", "pseudo_label_or_random"],
    )
    parser.add_argument("--ce-weight", type=float)
    parser.add_argument("--metric-weight", type=float)
    parser.add_argument("--triplet-margin", type=float)
    parser.add_argument("--supcon-temperature", type=float)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--print-spec", action="store_true", help="Print the resolved run spec and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    protocol = override_protocol(
        protocol,
        {
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "weight_decay": args.weight_decay,
            "embedding_dim": args.embedding_dim,
            "validation_per_class": args.validation_per_class,
            "ce_weight": args.ce_weight,
            "metric_weight": args.metric_weight,
            "triplet_margin": args.triplet_margin,
            "supcon_temperature": args.supcon_temperature,
        },
    )
    method_id, method = compose_method(
        protocol,
        base_method_id=args.method,
        overrides={
            "run_name": args.run_name,
            "objective": args.objective,
            "updates": args.updates,
            "views": args.views,
            "augmentation": args.augmentation,
            "normalize_embedding": args.normalize_embedding,
            "unlabeled_objective": args.unlabeled_objective,
            "unlabeled_augmentation": args.unlabeled_augmentation,
            "unlabeled_views": args.unlabeled_views,
            "unlabeled_weight": args.unlabeled_weight,
            "unlabeled_limit": args.unlabeled_limit,
            "pseudo_label_threshold": args.pseudo_label_threshold,
            "unlabeled_triplet_margin": args.unlabeled_triplet_margin,
            "unlabeled_negative_strategy": args.unlabeled_negative_strategy,
        },
    )
    spec = make_run_spec(
        protocol=protocol,
        dataset=args.dataset,
        architecture=args.architecture,
        budget=args.budget,
        method_id=method_id,
        seed=args.seed,
        method=method,
    )
    if args.print_spec:
        print(
            json.dumps(
                {
                    "dataset": spec.dataset,
                    "architecture": spec.architecture,
                    "budget": spec.budget,
                    "seed": spec.seed,
                    "method_id": spec.method_id,
                    "method": spec.method,
                    "optimizer": spec.protocol["optimizer"],
                    "loss_defaults": spec.protocol["loss_defaults"],
                    "embedding_dim": spec.protocol["embedding_dim"],
                    "validation_per_class": spec.protocol["validation_per_class"],
                },
                indent=2,
            )
        )
        return

    from fewshot_metric.train import train_one_run

    result = train_one_run(
        spec,
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
        device_name=args.device,
        num_workers=args.num_workers,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
