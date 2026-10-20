"""Write one comparison table for all completed and running pilot families."""
import sys
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from fewshot_metric.comparison import write_comparison

if __name__ == "__main__":
    rows=write_comparison()
    print(f"Updated outputs/pilot_comparison.md, .csv, .json ({len(rows)} runs)")
