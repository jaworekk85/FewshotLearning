from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


KEY = ("dataset", "architecture", "budget", "seed")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze paired low-label experiment results.")
    parser.add_argument("--results", default="outputs/results.csv")
    parser.add_argument("--out", default="outputs/analysis")
    parser.add_argument("--bootstrap-resamples", type=int, default=100000)
    parser.add_argument("--analysis-seed", type=int, default=12345)
    parser.add_argument("--compare", nargs=2, metavar=("METHOD_A", "METHOD_B"), action="append")
    return parser.parse_args()


def load_rows(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["budget"] = int(row["budget"])
        row["seed"] = int(row["seed"])
        row["test_accuracy"] = float(row["test_accuracy"])
    return rows


def pooled_means(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["method_id"], row["budget"])].append(row["test_accuracy"])

    out = []
    for (method_id, budget), values in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "method_id": method_id,
                "budget": budget,
                "n": len(arr),
                "mean_accuracy": float(arr.mean()),
                "mean_accuracy_percent": float(arr.mean() * 100.0),
            }
        )
    return out


def exact_sign_p(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    tail = min(wins, losses)
    prob = sum(math.comb(n, k) for k in range(tail + 1)) / (2**n)
    return min(1.0, 2.0 * prob)


def paired_comparison(rows: list[dict], method_a: str, method_b: str, resamples: int, seed: int) -> list[dict]:
    by_method = defaultdict(dict)
    for row in rows:
        key = tuple(row[field] for field in KEY)
        by_method[row["method_id"]][key] = row["test_accuracy"]

    rng = np.random.default_rng(seed)
    out = []
    budgets = sorted({row["budget"] for row in rows})
    for budget in budgets:
        diffs = []
        for key, a_value in by_method[method_a].items():
            if key[2] != budget or key not in by_method[method_b]:
                continue
            diffs.append(a_value - by_method[method_b][key])
        arr = np.asarray(diffs, dtype=float)
        if arr.size == 0:
            continue
        wins = int((arr > 0).sum())
        losses = int((arr < 0).sum())
        if arr.size == 1:
            ci_low = ci_high = float(arr[0])
        else:
            samples = rng.choice(arr, size=(resamples, arr.size), replace=True).mean(axis=1)
            ci_low, ci_high = np.percentile(samples, [2.5, 97.5]).tolist()
        out.append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "budget": budget,
                "pairs": int(arr.size),
                "mean_diff": float(arr.mean()),
                "mean_diff_percent_points": float(arr.mean() * 100.0),
                "ci_low_percent_points": float(ci_low * 100.0),
                "ci_high_percent_points": float(ci_high * 100.0),
                "wins": wins,
                "losses": losses,
                "ties": int((arr == 0).sum()),
                "sign_test_p": exact_sign_p(wins, losses),
            }
        )
    return out


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.results)
    out_dir = Path(args.out)
    write_csv(out_dir / "pooled_means.csv", pooled_means(rows))

    comparisons = args.compare or [
        ("ce_aug_4v_l2_1000", "ce_aug_4v_l2_500"),
        ("ce_aug_4v_l2_1000", "triplet_l2_500"),
        ("ce_2000", "triplet_l2_500"),
    ]
    for method_a, method_b in comparisons:
        comparison_rows = paired_comparison(
            rows,
            method_a,
            method_b,
            resamples=args.bootstrap_resamples,
            seed=args.analysis_seed,
        )
        write_csv(out_dir / f"{method_a}_minus_{method_b}.csv", comparison_rows)
    print(f"Wrote analysis tables to {out_dir}.")


if __name__ == "__main__":
    main()
