"""Sequential next-stage pilot: representation audit, mining, soft PL, active learning."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def build_jobs(plan):
    from fewshot_metric.pilot import select_method
    from fewshot_metric.supervised import validate_config
    from fewshot_metric.active import validate_active
    base = json.loads(Path(plan["base_config"]).read_text())
    base["seed"] = plan["seed"]
    jobs = []
    for method in plan.get("training_methods", []):
        config = select_method(base, method)
        validate_config(config)
        jobs.append({"kind": "fixed", "name": f"cnn_N{config['budget_per_class']}_seed{plan['seed']}_{method}", "config": config})
    active = plan.get("active_learning", {})
    for strategy in active.get("strategies", []):
        config = select_method(base, "ce_2v_pseudo")
        config.update({k: v for k, v in active.items() if k != "strategies"})
        config["active_strategy"] = strategy
        validate_active(config)
        initial = 10 * int(config["budget_per_class"])
        final = config["active_final_budget"]
        name = f"cnn_N{initial // 10}to{final}_seed{plan['seed']}_active_{strategy}"
        jobs.append({"kind": "active", "name": name, "config": config})
    for entry in plan.get("reconstruction_variants", []):
        from fewshot_metric.reconstruction import reconstruction_config
        config = reconstruction_config(base, entry["variant"])
        config.update(entry.get("overrides", {}))
        validate_config(config)
        name = f"cnn_N{config['budget_per_class']}_seed{plan['seed']}_reconstruction_{entry.get('name', entry['variant'])}"
        jobs.append({"kind": "fixed", "name": name, "config": config})
    for entry in plan.get("method_variants", []):
        config = select_method(base, entry["method"])
        config.update(entry.get("overrides", {}))
        validate_config(config)
        name = f"cnn_N{config['budget_per_class']}_seed{plan['seed']}_{entry['method']}_{entry['name']}"
        jobs.append({"kind": "fixed", "name": name, "config": config})
    for entry in plan.get("experiments", []):
        config = select_method({**base, "seed": entry.get("seed", plan["seed"])}, entry["method"])
        config.update(entry.get("overrides", {}))
        kind = entry.get("kind", "fixed")
        if kind not in {"fixed", "active"}:
            raise ValueError("Unknown experiment kind.")
        (validate_active if kind == "active" else validate_config)(config)
        name = entry["name"]
        if Path(name).name != name or name in {".", ".."}:
            raise ValueError("Invalid experiment name.")
        job = {"kind": kind, "name": name, "config": config}
        if "reuse_run" in entry:
            job["reuse_run"] = entry["reuse_run"]
        jobs.append(job)
    if len({job["name"] for job in jobs}) != len(jobs):
        raise ValueError("Duplicate experiment names.")
    return jobs


def atomic_json(path, value):
    pending = path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    pending.replace(path)


def run_queue(plan, device="cpu", start_prepared=False):
    if plan.get("manual_start_only", False) and not start_prepared:
        raise ValueError("This plan is prepared only. Inspect --print-plan; launch explicitly with --start-prepared.")
    from fewshot_metric.supervised import run_supervised
    from fewshot_metric.active import run_active
    jobs = build_jobs(plan)
    root = Path(plan["output_dir"])
    root.mkdir(parents=True, exist_ok=True)
    # A live advisory lock is automatically released by the OS after a restart.
    lock = (root / "queue.lock").open("a+b")
    lock.seek(0, 2)
    if not lock.tell():
        lock.write(b"0")
        lock.flush()
    lock.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    state_path = root / "queue.json"
    stages = [{"name": "representations_existing", "status": "pending"}]
    stages += [{"name": job["name"], "status": "pending", "kind": job["kind"]} for job in jobs]
    state = {"status": "waiting", "pid": os.getpid(), "waiting_for": plan["wait_for"], "jobs": stages, "plan": plan}
    def save():
        atomic_json(state_path, state)
    save()
    current = None
    try:
        deadline = time.monotonic() + plan["wait_timeout_hours"] * 3600
        print(f"Waiting for preceding experiments: {plan['wait_for']}", flush=True)
        while not Path(plan["wait_for"]).exists():
            error_log = Path(plan["wait_error_log"])
            if error_log.exists() and "Traceback (most recent call last)" in error_log.read_text(errors="replace"):
                raise RuntimeError("Preceding experiments failed; this series has not started.")
            if time.monotonic() > deadline:
                raise TimeoutError("Timed out waiting for preceding experiments; rerun this queue to resume.")
            time.sleep(5)
        preceding = json.loads(Path(plan["wait_for"]).read_text())
        if plan.get("preceding_summary", False):
            if not isinstance(preceding, list) or len(preceding) != plan["expected_preceding_runs"] or any(row.get("status") != "completed" for row in preceding):
                raise ValueError("Preceding series summary is incomplete.")
        elif preceding.get("status") != "completed" or preceding.get("selected_epoch") != 40:
            raise ValueError("Preceding CE40 run is not a completed real 40-epoch experiment.")
        state.update(status="running")
        state.pop("waiting_for", None)
        runs = list(plan["diagnose_previous_runs"])
        analysis_python = plan.get("analysis_python", sys.executable)
        if not Path(analysis_python).is_file():
            analysis_python = sys.executable
        def diagnostics():
            environment = os.environ.copy()
            environment["MPLCONFIGDIR"] = str((Path(plan["representation_output"]) / ".mplconfig").resolve())
            command = [analysis_python, "-u", "scripts/analyze_representations.py", *runs,
                       "--output-dir", plan["representation_output"]]
            subprocess.run(command, check=True, env=environment)
            if "study_summary_output" in plan:
                subprocess.run([analysis_python, "scripts/summarize_validation_study.py",
                                "--plan", plan["study_plan_path"], "--output", plan["study_summary_output"]],
                               check=True, env=environment)
            if "custom_report_script" in plan:
                subprocess.run([analysis_python, plan["custom_report_script"],
                                "--plan", plan["custom_report_plan"], "--output", plan["custom_report_output"]],
                               check=True, env=environment)
        current = stages[0]
        current["status"] = "running"
        save()
        diagnostics()
        current["status"] = "completed"
        save()
        results = []
        for job, stage in zip(jobs, stages[1:]):
            current = stage
            run_dir = Path(job["reuse_run"]) if "reuse_run" in job else root / job["name"]
            config = job["config"]
            atomic_json(root / f"{job['name']}.config.json", config)
            current["status"] = "running"
            save()
            print(f"Starting {job['name']}", flush=True)
            if "reuse_run" in job and not (run_dir / "metrics.json").exists():
                raise FileNotFoundError(f"Requested reused result is missing: {run_dir}")
            if (run_dir / "metrics.json").exists():
                result = json.loads((run_dir / "metrics.json").read_text())
                if result.get("status") != "completed" or any(result.get(k) != v for k, v in config.items()):
                    raise ValueError(f"Existing result differs: {run_dir}")
            else:
                resume = (run_dir / "latest.pt").exists()
                result = (run_active if job["kind"] == "active" else run_supervised)(
                    config, run_dir, Path("data"), device, resume=resume)
            current.update(status="analyzing", val_accuracy=result["val_accuracy"], test_accuracy=result["test_accuracy"])
            save()
            if str(run_dir) not in runs:
                runs.append(str(run_dir))
            diagnostics()
            results.append(result)
            current["status"] = "completed"
            save()
        atomic_json(root / "results.json", results)
        state["status"] = "completed"
        save()
    except Exception as error:
        state.update(status="failed", error=str(error))
        if current is not None:
            current.update(status="failed", error=str(error))
        save()
        raise
    finally:
        lock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/cifar10_pilot_next.json")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--print-plan", action="store_true")
    parser.add_argument("--start-prepared", action="store_true", help="Explicitly launch a manual-start-only plan.")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = json.loads(Path(args.plan).read_text())
    if args.print_plan:
        print(json.dumps(build_jobs(plan), indent=2))
    else:
        run_queue(plan, args.device, start_prepared=args.start_prepared)
