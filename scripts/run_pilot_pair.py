"""Run paired pilot methods sequentially, resuming unfinished runs."""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/cifar10_pilot_ssl.json")
    parser.add_argument("--budget", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", default=["ce_2v", "ce_2v_pseudo"])
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="outputs/pilot")
    parser.add_argument("--queue-tag", default="", help="Suffix for queue and summary filenames.")
    parser.add_argument("--wait-for", help="Wait for a preceding queue's completed summary before training.")
    parser.add_argument("--wait-status", help="Abort waiting if this preceding queue reports a failed job.")
    parser.add_argument("--wait-timeout-hours", type=float, default=6.0)
    args = parser.parse_args()
    if args.queue_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.queue_tag):
        parser.error("--queue-tag accepts only letters, digits, hyphens and underscores.")
    if args.wait_timeout_hours <= 0:
        parser.error("--wait-timeout-hours must be positive.")
    from fewshot_metric.pilot import select_method
    from fewshot_metric.supervised import run_supervised, validate_config, write_json

    base = json.loads(Path(args.config).read_text(encoding="utf-8"))
    base.update(budget_per_class=args.budget, seed=args.seed)
    configs = [select_method(base, name) for name in args.methods]
    for config in configs:
        validate_config(config)
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    jobs = [{"method": config["method"], "status": "pending", "run_name":
             f"{config['architecture']}_N{args.budget}_seed{args.seed}_{config['method']}"} for config in configs]
    suffix = "_" + args.queue_tag if args.queue_tag else ""
    status_path = root / f"queue_N{args.budget}_seed{args.seed}{suffix}.json"
    state = {"budget_per_class": args.budget, "seed": args.seed, "jobs": jobs}
    write_json(status_path, state)
    if args.wait_for:
        state["waiting_for"] = args.wait_for
        write_json(status_path, state)
        print(f"Waiting for preceding experiments: {args.wait_for}", flush=True)
        deadline = time.monotonic() + args.wait_timeout_hours * 3600
        while not Path(args.wait_for).exists():
            failure = None
            if args.wait_status and Path(args.wait_status).exists():
                try:
                    preceding = json.loads(Path(args.wait_status).read_text(encoding="utf-8"))
                    if any(job.get("status") == "failed" for job in preceding.get("jobs", [])):
                        failure = "Preceding queue failed; triplet training has not started."
                except json.JSONDecodeError:
                    pass  # The preceding process may be replacing its status file.
            if time.monotonic() >= deadline:
                failure = "Timed out waiting for preceding experiments."
            if failure:
                state["error"] = failure
                write_json(status_path, state)
                raise RuntimeError(failure)
            time.sleep(5)
        state.pop("waiting_for")
        write_json(status_path, state)
    results = []
    for config, job in zip(configs, jobs):
        path = root / job["run_name"]
        try:
            if (path / "metrics.json").exists():
                result = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
                if any(result.get(key) != value for key, value in config.items()):
                    raise ValueError(f"Existing result configuration differs: {path}.")
            else:
                job["status"] = "running"
                write_json(status_path, state)
                result = run_supervised(config, path, Path(args.data_root), args.device,
                                        resume=(path / "latest.pt").exists())
            job.update(status="completed", val_accuracy=result["val_accuracy"], test_accuracy=result["test_accuracy"])
            results.append(result)
            write_json(status_path, state)
        except Exception as error:
            job.update(status="failed", error=str(error))
            write_json(status_path, state)
            raise
    write_json(root / f"paired_results_N{args.budget}_seed{args.seed}{suffix}.json", results)


if __name__ == "__main__":
    main()
