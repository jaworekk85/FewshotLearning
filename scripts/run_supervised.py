from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description="CIFAR-10 full-label and low-label supervised control.")
    parser.add_argument("--config", default="configs/cifar10_supervised.json")
    parser.add_argument("--budget", help="Labeled images per class, or all.")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--method", help="Paired pilot method, e.g. ce_2v or ce_2v_pseudo (uses pilot mode).")
    parser.add_argument("--pseudo-label-threshold", type=float)
    parser.add_argument("--unlabeled-batch-size", type=int)
    parser.add_argument("--triplet-l-weight", type=float)
    parser.add_argument("--triplet-u-weight", type=float)
    parser.add_argument("--pseudo-label-weight", type=float)
    parser.add_argument("--pseudo-triplet-weight", type=float)
    parser.add_argument("--pseudo-triplet-pool", choices=["labeled_and_pseudo", "pseudo_only"])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps-per-epoch", type=int, help="Fixed update block shared across label budgets; defaults to one full-pool epoch.")
    parser.add_argument("--validation-per-class", type=int)
    parser.add_argument("--selection", choices=["last", "best_validation"])
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--normalization", choices=["batchnorm", "none", "groupnorm"])
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/supervised")
    parser.add_argument("--run-name")
    parser.add_argument("--download", action="store_true", help="Allow torchvision to download CIFAR-10 if absent.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int, help="Stop at an epoch boundary without evaluating test data.")
    parser.add_argument("--smoke", action="store_true", help="Two short blocks on synthetic images; no CIFAR download or research result.")
    parser.add_argument("--print-spec", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.method is not None:
        from fewshot_metric.pilot import select_method
        config = select_method(config, args.method)
    for key in ("seed", "epochs", "batch_size", "steps_per_epoch", "validation_per_class", "selection", "num_workers", "normalization",
                "pseudo_label_threshold", "unlabeled_batch_size", "triplet_l_weight", "triplet_u_weight", "pseudo_label_weight",
                "pseudo_triplet_weight", "pseudo_triplet_pool"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if args.budget is not None:
        config["budget_per_class"] = "all" if args.budget == "all" else int(args.budget)
    if args.smoke:
        config.update(epochs=2, steps_per_epoch=2, batch_size=8, num_workers=0)
        if config.get("pilot_mode", False):
            config["unlabeled_batch_size"] = 8
    if args.stop_after_epoch is not None and args.stop_after_epoch <= 0:
        parser.error("--stop-after-epoch must be positive.")
    name = args.run_name or f"{'synthetic_smoke' if args.smoke else 'cifar10'}_N{config['budget_per_class']}_seed{config['seed']}"
    if args.run_name is None and config.get("pilot_mode", False):
        name += "_" + config.get("method", "custom_pilot")
    if name in {".", ".."} or Path(name).name != name or "/" in name or "\\" in name:
        parser.error("--run-name must be a directory name, without path separators.")
    run_dir = Path(args.output_dir) / name
    if args.print_spec:
        print(json.dumps({"config": config, "output": str(run_dir), "synthetic_smoke": args.smoke}, indent=2))
        return
    from fewshot_metric.supervised import run_supervised

    run_supervised(config, run_dir, Path(args.data_root), args.device,
                   download=args.download, resume=args.resume, smoke=args.smoke,
                   stop_after_epoch=args.stop_after_epoch)


if __name__ == "__main__":
    main()
