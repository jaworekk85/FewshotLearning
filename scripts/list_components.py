from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fewshot_metric.config import load_protocol


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List available experiment components and method presets.")
    parser.add_argument("--config", default="configs/v4_protocol.json")
    return parser.parse_args()


def print_mapping(title: str, values: dict) -> None:
    print(f"\n{title}")
    for key, description in values.items():
        print(f"  {key}: {description}")


def main() -> None:
    args = parse_args()
    protocol = load_protocol(args.config)
    components = protocol.get("component_space", {})

    print_mapping("Datasets", components.get("datasets", {}))
    print_mapping("Architectures", components.get("architectures", {}))
    print_mapping("Objectives", components.get("objectives", {}))
    print_mapping("Unlabeled objectives", components.get("unlabeled_objectives", {}))
    print_mapping("Unlabeled negative strategies", components.get("unlabeled_negative_strategies", {}))
    print_mapping("Augmentations", components.get("augmentations", {}))

    print("\nExample compositions")
    for name, example in components.get("example_compositions", {}).items():
        base = example.get("base_method", "custom")
        overrides = ", ".join(f"{key}={value}" for key, value in example.get("overrides", {}).items())
        print(f"  {name}: base={base}, {overrides}")

    print("\nMethod presets")
    for method_id, method in protocol["methods"].items():
        bits = [
            f"objective={method['objective']}",
            f"updates={method['updates']}",
            f"views={method['views']}",
            f"augmentation={method['augmentation']}",
            f"l2={method['normalize_embedding']}",
            f"unlabeled={method.get('unlabeled_objective', 'none')}",
            f"grid={method['grid']}",
        ]
        print(f"  {method_id}: " + ", ".join(bits))


if __name__ == "__main__":
    main()
