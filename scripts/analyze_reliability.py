"""Inference-only diagnostics of calibration, pseudo-label weights and error rankings."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*")
    parser.add_argument("--plan", help="JSON with an explicit list of runs")
    parser.add_argument("--output-dir", default="outputs/reliability")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()
    runs = args.runs + (json.loads(Path(args.plan).read_text())["runs"] if args.plan else [])
    if not runs:
        parser.error("Provide runs or --plan")
    os.environ.setdefault("MPLCONFIGDIR", str(Path(args.output_dir).resolve() / ".mplconfig"))
    import torch
    torch.set_num_threads(6)
    from fewshot_metric.reliability import analyze_run, write_report
    rows = []
    for run in runs:
        print(f"Reliability audit: {run}", flush=True)
        rows.append(analyze_run(run, args.output_dir, args.data_root))
        write_report(rows, args.output_dir)
    print(f"Completed {len(rows)} models: {args.output_dir}/report.md", flush=True)
