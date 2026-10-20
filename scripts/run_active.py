"""Run one controlled active-learning pilot with a simulated annotation oracle."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default="outputs/pilot_next")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        parser.error("Invalid run name.")
    from fewshot_metric.active import run_active
    run_active(json.loads(Path(args.config).read_text()), Path(args.output_dir)/args.run_name,
               Path(args.data_root), args.device, resume=args.resume)
