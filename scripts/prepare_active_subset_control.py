"""Freeze two already acquired AL annotation sets; never starts training."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fewshot_metric.fixed_splits import manifest_from_active_run
from fewshot_metric.pilot import select_method


def prepare(data_root="data", output_dir="configs/active_subset_control",
            plan_path="configs/cifar10_pilot_active_subset_control.json"):
    from torchvision.datasets import CIFAR10
    labels = CIFAR10(data_root, train=True, download=False).targets
    base = select_method(json.loads(Path("configs/cifar10_pilot_ssl.json").read_text()), "ce_2v_pseudo_soft")
    sources = {s: Path("outputs/pilot_validation_study")/f"cnn_N30to500_seed0_soft_active_{s}"
               for s in ("random", "badge")}
    manifests = {s: manifest_from_active_run(p, labels, s) for s, p in sources.items()}
    if manifests["random"]["validation"] != manifests["badge"]["validation"]:
        raise ValueError("Source validation indices or order differ.")
    configs = {s: json.loads((p/"config.json").read_text()) for s, p in sources.items()}
    for strategy, cfg in configs.items():
        for key in ("architecture", "embedding_dim", "batch_size", "learning_rate", "momentum", "weight_decay",
                    "num_workers", "mean", "std", "pseudo_label_threshold", "pseudo_weighting", "pseudo_target",
                    "pseudo_temperature", "labeled_views", "unlabeled_batch_size", "triplet_l_weight", "triplet_u_weight"):
            if cfg[key] != base[key]:
                raise ValueError(f"Source {strategy} training recipe differs: {key}")
    if (configs["random"]["initial_train_indices"] != configs["badge"]["initial_train_indices"]
            or configs["random"]["initial_model_sha256"] != configs["badge"]["initial_model_sha256"]):
        raise ValueError("Source initial annotations or initialization differ.")
    output_dir = Path(output_dir)
    snapshots, experiments = {}, []
    for strategy, manifest in manifests.items():
        path = output_dir/f"{strategy}_final500.json"
        text = json.dumps(manifest, indent=2, allow_nan=False) + "\n"
        snapshots[path] = text
        experiments.append({"name": f"cnn_N500_seed0_soft_frozen_{strategy}", "method": "ce_2v_pseudo_soft", "seed": 0,
            "overrides": {"epochs": 20, "steps_per_epoch": 383,
                "fixed_split_manifest": path.as_posix(),
                "fixed_split_manifest_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "fixed_subset_strategy": strategy, "fixed_subset_source_run": sources[strategy].as_posix(),
                "fixed_split_source_initial_model_sha256": manifest["source_initial_model_sha256"]}})
    plan = {"manual_start_only": True, "base_config": "configs/cifar10_pilot_ssl.json", "seed": 0,
            "wait_for": "outputs/pilot_validation_study/results.json",
            "wait_error_log": "outputs/pilot_validation_study/queue_mixed_20261017.stderr.log",
            "wait_timeout_hours": 48, "preceding_summary": True, "expected_preceding_runs": 16,
            "output_dir": "outputs/pilot_active_subset_control", "representation_output": "outputs/representations",
            "analysis_python": ".venv-analysis/Scripts/python.exe",
            "diagnose_previous_runs": [p.as_posix() for p in sources.values()],
            "custom_report_script": "scripts/summarize_active_subset_control.py",
            "custom_report_plan": Path(plan_path).as_posix(),
            "custom_report_output": "outputs/pilot_active_subset_control_analysis",
            "experiments": experiments}
    snapshots[Path(plan_path)] = json.dumps(plan, indent=2, allow_nan=False) + "\n"
    # Refuse changed snapshots before writing any file; repeat calls are idempotent.
    for path, text in snapshots.items():
        if path.exists() and path.read_bytes() != text.encode("utf-8"):
            raise FileExistsError(f"Refusing to overwrite a different frozen snapshot: {path}")
    for path, text in snapshots.items():
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode("utf-8"))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    plan = prepare(args.data_root)
    print(f"Prepared {len(plan['experiments'])} exact frozen-subset controls. Training has not started.")
