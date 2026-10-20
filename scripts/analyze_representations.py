"""Validation representation audit; supports current and future pilot runs."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--output-dir", default="outputs/representations")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--no-tsne", action="store_true")
    args = parser.parse_args()
    from fewshot_metric.representations import analyze_many
    analyze_many(args.runs, args.output_dir, args.data_root, tsne=not args.no_tsne)
    from fewshot_metric.reconstruction import export_reconstruction_examples
    for run in args.runs:
        export_reconstruction_examples(run, Path(args.output_dir)/Path(run).name, args.data_root)
    from fewshot_metric.comparison import write_comparison
    write_comparison()
