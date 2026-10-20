"""Matched-exposure comparison of four new triplet miners and reused baselines."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_next_pilot import build_jobs
from fewshot_metric.supervised import write_json


def summarize(plan_path, output):
    plan = json.loads(Path(plan_path).read_text())
    rows = []
    for job in build_jobs(plan):
        run = Path(job.get("reuse_run", str(Path(plan["output_dir"]) / job["name"])))
        row = dict(run=job["name"], source=str(run), reused="reuse_run" in job, status="pending")
        if (run / "metrics.json").exists():
            r = json.loads((run / "metrics.json").read_text())
            if r.get("status") != "completed" or any(r.get(k) != v for k, v in job["config"].items()):
                raise ValueError(f"Results differ from plan: {run}")
            splits = json.loads((run / "splits.json").read_text())
            if len(splits["train"]) != 500 or set(splits["train"]) & set(splits["validation"]):
                raise ValueError("Incorrect label budget or validation overlap")
            row.update(status="completed", val_accuracy=r["val_accuracy"], test_accuracy=r["test_accuracy"],
                       updates=r["total_updates"], labeled_views=r["processed_labeled_views"],
                       unlabeled_views=r["processed_unlabeled_views"], elapsed_minutes=r["elapsed_seconds"]/60,
                       initial_model_sha256=r["initial_model_sha256"],
                       split_sha256=hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
                       class_triplet_valid=r.get("pseudo_triplet_valid", 0),
                       class_triplet_attempted=r.get("pseudo_triplet_attempted", 0),
                       class_triplet_active=r.get("pseudo_triplet_active", 0),
                       **{k: v for k, v in r.items() if k.startswith("metric_")})
            valid = row["class_triplet_valid"]
            if "metric_anchor_l" in r and r["metric_anchor_l"] + r["metric_anchor_u"] != valid:
                raise ValueError("Triplet anchor counters disagree")
            row["class_triplet_coverage"] = valid / row["class_triplet_attempted"] if row["class_triplet_attempted"] else None
        elif (run / "history.csv").exists():
            with (run / "history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            if history:
                row.update(status="in_progress", epoch=int(history[-1]["epoch"]), val_accuracy=float(history[-1]["val_accuracy"]))
        rows.append(row)
    done = [r for r in rows if r["status"] == "completed"]
    for key in ("updates", "labeled_views", "unlabeled_views", "initial_model_sha256", "split_sha256"):
        if len({r[key] for r in done}) > 1:
            raise ValueError(f"Unmatched comparison: {key}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", rows)
    lines = ["# Distance-selected triplet pilot", "", "Four new training runs; two completed baselines reused.",
        "Same CNN/init, seed0, true L=500, validation split, optimizer, 20 epochs and 7660 updates; same labeled/unlabeled encoder views. Mining uses detached existing batch tensors; no extra encoder passes. Runtime includes mining and is reported separately.",
        "New miners share a 3-epoch class-triplet warmup. Their class term has at most 128 attempted anchors per step, a fixed denominator of 128 and confidence-product weights. Existing L semi-hard and U instance losses remain. Each triplet component has coefficient 1/30.",
        "B vs C changes only the mining distance; B vs D changes only the instance-negative exclusion. A vs B changes the allowed source pattern. Comparisons with the old mixed baseline also change the warmup and confidence weighting, so do not isolate a single cause.", "",
        "| Run | Status | Val % | Test % | L / U encoder views | Valid class triplets / attempts | Minutes |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] == "completed":
            label = "reused" if r["reused"] else "completed"
            lines.append(f'| {r["run"]} | {label} | {100*r["val_accuracy"]:.2f} | {100*r["test_accuracy"]:.2f} | {r["labeled_views"]} / {r["unlabeled_views"]} | {r["class_triplet_valid"]} / {r["class_triplet_attempted"]} | {r["elapsed_minutes"]:.1f} |')
        elif r["status"] == "in_progress":
            lines.append(f'| {r["run"]} | epoch {r["epoch"]}/20 | {100*r["val_accuracy"]:.2f} | — | — | — | — |')
        else:
            lines.append(f'| {r["run"]} | pending | — | — | — | — | — |')
    lines += ["", "Pixel distance is RGB distance between existing weak crop/flip views, not between canonical unaugmented images. Loss distance always uses normalized embeddings, with strong positives and weak negatives.",
              "D excludes confident same-pseudo-class pairs only from the instance-negative pool and averages over remaining pairs; this selection uses pseudo-labels, while A/B/C retain a fully unsupervised instance component.",
              "One exploratory seed; no claim of guaranteed improvement, statistical confirmation, or scientific novelty. Checkpoints and reusable representation audits are retained."]
    (output / "comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/cifar10_pilot_mined_triplets.json")
    parser.add_argument("--output", default="outputs/pilot_mined_triplets_analysis")
    args = parser.parse_args()
    summarize(args.plan, args.output)
