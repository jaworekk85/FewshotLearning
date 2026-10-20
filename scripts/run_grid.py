from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fewshot_metric.config import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build or execute the v3/v4 experiment grid.")
    parser.add_argument("--config", default="configs/v4_protocol.json")
    parser.add_argument("--grid", choices=["primary", "v4", "all"], default="all")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--execute", action="store_true", help="Run commands instead of only writing a manifest.")
    parser.add_argument("--manifest", default="outputs/grid_manifest.csv")
    return parser.parse_args()


def iter_runs(protocol: dict, grid: str):
    for method_id, method in protocol["methods"].items():
        if grid != "all" and method["grid"] != grid:
            continue
        seed_key = "primary_seeds" if method["grid"] == "primary" else "v4_seeds"
        for dataset in protocol["datasets"]:
            for architecture in protocol["architectures"]:
                for budget in protocol["budgets_per_class"]:
                    for seed in protocol[seed_key]:
                        yield {
                            "dataset": dataset,
                            "architecture": architecture,
                            "budget": budget,
                            "method": method_id,
                            "seed": seed,
                            "grid": method["grid"],
                        }


def command_for(row: dict, args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "scripts/run_experiment.py",
        "--config",
        args.config,
        "--dataset",
        row["dataset"],
        "--architecture",
        row["architecture"],
        "--budget",
        str(row["budget"]),
        "--method",
        row["method"],
        "--seed",
        str(row["seed"]),
        "--data-root",
        args.data_root,
        "--output-dir",
        args.output_dir,
        "--device",
        args.device,
        "--num-workers",
        str(args.num_workers),
    ]


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    runs = list(iter_runs(protocol, args.grid))
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "architecture", "budget", "method", "seed", "grid"])
        writer.writeheader()
        writer.writerows(runs)

    print(f"Wrote {len(runs)} planned runs to {manifest_path}.")
    if not args.execute:
        print("Dry run only. Add --execute to run the manifest.")
        print("First command:")
        print(" ".join(command_for(runs[0], args)) if runs else "No runs selected.")
        return

    for i, row in enumerate(runs, start=1):
        print(f"[{i}/{len(runs)}] {row}")
        subprocess.run(command_for(row, args), check=True)


if __name__ == "__main__":
    main()
