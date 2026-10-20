"""Compare four new neighbor experiments against two existing baselines."""
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
        run = Path(job["reuse_run"]) if "reuse_run" in job else Path(plan["output_dir"]) / job["name"]
        path = run / "metrics.json"
        row = dict(run=job["name"], kind=job["kind"],
                   weighting=job["config"].get("neighbor_pseudo_weighting", False),
                   reused="reuse_run" in job, source_run=str(run),
                   strategy=job["config"].get("active_strategy", "fixed"), status="pending")
        if path.exists():
            result = json.loads(path.read_text())
            if result.get("status") != "completed" or any(result.get(k) != v for k, v in job["config"].items()):
                raise ValueError(f"Result does not match plan: {run}")
            splits = json.loads((run / "splits.json").read_text())
            with (run / "history.csv").open(newline="") as handle:
                history = list(csv.DictReader(handle))
            if len(history) != job["config"]["epochs"] or len(splits["train"]) != 500:
                raise ValueError("Incomplete training or annotation budget")
            if job["config"].get("neighbor_reference"):
                original = sum(float(h["pseudo_base_weight_sum"]) for h in history)
                effective = sum(float(h["pseudo_weight_sum"]) for h in history)
                if abs(original - effective) > 1e-6 * max(original, 1):
                    raise ValueError("Pseudo-weight mass was not preserved")
            row.update(status="completed", val_accuracy=result["val_accuracy"], test_accuracy=result["test_accuracy"],
                updates=result["total_updates"], labeled_views=result["processed_labeled_views"],
                unlabeled_views=result["processed_unlabeled_views"],
                reference_views=result.get("neighbor_reference_views", 0), query_views=result.get("neighbor_query_views", 0),
                acquisition_views=result.get("scoring_forward_views", 0),
                initial_model_sha256=result["initial_model_sha256"],
                validation_sha256=hashlib.sha256(json.dumps(sorted(splits["validation"])).encode()).hexdigest(),
                elapsed_minutes=result["elapsed_seconds"] / 60)
        elif (run / "history.csv").exists():
            row["status"] = "in_progress"
        rows.append(row)
    complete = [r for r in rows if r["status"] == "completed"]
    for key in ("initial_model_sha256", "validation_sha256"):
        if len({r[key] for r in complete}) > 1:
            raise ValueError(f"Experiment does not share {key}")
    for kind in ("fixed", "active"):
        paired = [r for r in complete if r["kind"] == kind]
        for key in ("updates", "labeled_views", "unlabeled_views", "acquisition_views"):
            if len({r[key] for r in paired}) > 1:
                raise ValueError(f"Unequal exposure in {kind}: {key}")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "comparison.json", rows)
    lines = ["# Neighbor reliability pilot", "", "Prepared experiment; pending rows are not results.",
        "Two existing baselines are reused; four new training runs. Same CNN, initialization, validation, 20 epochs / 7660 updates. Weight mass preserved per batch where neighbor weighting is enabled.",
        "Extra eval-only reference/query forwards are computed only for neighbor weighting. Their cost is reported separately, not artificially added to baselines.",
        "Compare fixed arms with each other, and active arms with each other. Active arms share the annotation schedule but receive fewer labeled training views than fixed arms.", "",
        "| Run | Status | Val % | Test % | Gradient L / U views | Extra eval reference / U / acquisition | Minutes |", "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"] == "completed":
            status = "reused baseline" if r["reused"] else "completed"
            lines.append(f'| {r["run"]} | {status} | {100*r["val_accuracy"]:.2f} | {100*r["test_accuracy"]:.2f} | {r["labeled_views"]} / {r["unlabeled_views"]} | {r["reference_views"]} / {r["query_views"]} / {r["acquisition_views"]} | {r["elapsed_minutes"]:.1f} |')
        else:
            lines.append(f'| {r["run"]} | {r["status"]} | — | — | — | — | — |')
    lines += ["", "Single-seed exploratory study selected after validation diagnostics; no confirmatory statistical claims.",
              "Ranking errors is not evidence that acquiring those labels improves training. Test is evaluated only after each final checkpoint."]
    (output / "comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/cifar10_pilot_neighbor_reliability.json")
    parser.add_argument("--output", default="outputs/pilot_neighbor_reliability_analysis")
    args = parser.parse_args()
    summarize(args.plan, args.output)
