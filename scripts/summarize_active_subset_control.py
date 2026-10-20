"""Compare exact AL-selected annotation sets with labels available from start."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def summarize(plan, output):
    from scripts.run_next_pilot import build_jobs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows, completed = [], []
    for job in build_jobs(plan):
        cfg = job["config"]
        path = Path(cfg["fixed_split_manifest"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != cfg["fixed_split_manifest_sha256"]:
            raise ValueError("Frozen manifest changed.")
        manifest = json.loads(path.read_text())
        source = Path(cfg["fixed_subset_source_run"])
        for name, expected in manifest["source_artifact_sha256"].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"Source artifact changed: {source/name}")
        online = json.loads((source/"metrics.json").read_text())
        directory = Path(plan["output_dir"])/job["name"]
        row = {"strategy": cfg["fixed_subset_strategy"], "source_run": str(source), "frozen_run": str(directory),
               "status": "pending", "online_val": online["val_accuracy"], "online_test": online["test_accuracy"],
               "online_labeled_views": online["processed_labeled_views"], "online_unlabeled_views": online["processed_unlabeled_views"],
               "final_class_counts": online["final_class_counts"], "frozen_manifest_sha256": cfg["fixed_split_manifest_sha256"]}
        if (directory/"metrics.json").exists():
            m = json.loads((directory/"metrics.json").read_text())
            splits = json.loads((directory/"splits.json").read_text())
            if (m.get("status") != "completed" or any(m.get(k) != v for k, v in cfg.items())
                    or splits != {"train": manifest["train"], "validation": manifest["validation"]}
                    or m["initial_model_sha256"] != online["initial_model_sha256"]):
                raise ValueError("Frozen-subset result does not match its annotation snapshot or initialization.")
            row.update(status="completed", frozen_val=m["val_accuracy"], frozen_test=m["test_accuracy"],
                       frozen_labeled_views=m["processed_labeled_views"], frozen_unlabeled_views=m["processed_unlabeled_views"],
                       frozen_updates=m["total_updates"],
                       frozen_minus_online_test=m["test_accuracy"] - online["test_accuracy"])
            completed.append(m)
        rows.append(row)
    if len(completed) == 2:
        for key in ("processed_labeled_views", "processed_unlabeled_views", "total_updates", "initial_model_sha256"):
            if completed[0][key] != completed[1][key]:
                raise ValueError(f"Frozen arms differ in controlled quantity: {key}")
    def pct(value):
        return "—" if value is None else f"{100*value:.2f}%"
    text = ["# Jakość końcowego zbioru AL: identyczny trening od początku", "",
            "Dwa ukończone zbiory AL: po 500 prawdziwych etykiet, bez dodatkowych oznaczeń.",
            "Nowe modele: miękkie cele i wagi pseudoetykiet, CNN, seed 0, 20 epok × 383 aktualizacje.",
            "Całe 500 etykiet dostępne od pierwszej aktualizacji. Indeksy L kanonicznie posortowane,",
            "następnie zwykły shuffle. Oryginalna walidacja AL jest wspólna i niezmieniona.", "",
            "| Końcowy zbiór wybrany przez | AL online: wal. | AL online: test | Od początku: wal. | Od początku: test | Status nowego runu |",
            "| --- | ---: | ---: | ---: | ---: | --- |"]
    for r in rows:
        text.append(f"| {r['strategy']} | {pct(r['online_val'])} | {pct(r['online_test'])} | {pct(r.get('frozen_val'))} | {pct(r.get('frozen_test'))} | {r['status']} |")
    if len(completed) == 2:
        delta = rows[1]["frozen_test"] - rows[0]["frozen_test"]
        text += ["", f"BADGE minus random przy identycznym treningu od początku: **{100*delta:+.2f} pp test**."]
    text += ["", "| Zbiór | Widoki L: online | Widoki L: od początku | Widoki U: online | Widoki U: od początku |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for r in rows:
        text.append(f"| {r['strategy']} | {r['online_labeled_views']} | {r.get('frozen_labeled_views', '—')} | {r['online_unlabeled_views']} | {r.get('frozen_unlabeled_views', '—')} |")
    text += ["", "Wniosek o jakości wyboru opieramy na dwóch nowych runach z identycznym harmonogramem.",
             "Online kontra od początku zmienia łącznie moment dostępu do etykiet, ich historię ekspozycji,",
             "liczbę wejść L (online miało niepełne minibatch'e) i skład U w czasie. Nie izoluje samego czasu ujawnienia.",
             "To pojedynczy seed i dwa konkretne zbiory; brak oszacowania zmienności wyboru danych.",
             "Nie wybieramy na nowo 500 obrazów, nie równoważymy klas ukrytymi etykietami, nie używamy trójek.",
             "Oryginalny model AL służył do wyboru danych; jego wagi nie są przenoszone do nowych modeli.",
             "80 epok i wynik 65,34% dotyczą osobnego harmonogramu i zbioru, nie są kontrolą kosztu tych runów.", ""]
    def save(name, value):
        p = output/name
        pending = p.with_suffix(p.suffix+".pending")
        pending.write_text(value, encoding="utf-8")
        pending.replace(p)
    save("comparison.md", "\n".join(text))
    save("comparison.json", json.dumps(rows, indent=2, allow_nan=False))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/cifar10_pilot_active_subset_control.json")
    parser.add_argument("--output", default="outputs/pilot_active_subset_control_analysis")
    args = parser.parse_args()
    rows = summarize(json.loads(Path(args.plan).read_text()), args.output)
    print(f"Frozen subset report: {sum(r['status']=='completed' for r in rows)}/{len(rows)} completed")
