"""Report replication, duration, factorial controls and cached representations."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def statistics_summary(values):
    values = list(values)
    n = len(values)
    if not n:
        return {"n": 0, "mean": None, "sd": None, "ci95": None}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if n > 1 else None
    interval = None
    if n > 1:
        from scipy.stats import t
        half_width = float(t.ppf(0.975, n - 1)) * sd / math.sqrt(n)
        interval = [mean - half_width, mean + half_width]
    return {"n": n, "mean": mean, "sd": sd, "ci95": interval}


def paired_effect(first, second):
    """Second minus first, paired by split and initialization, not test images."""
    matched = sorted(set(first) & set(second))
    for seed in matched:
        if first[seed]["pairing_signature"] != second[seed]["pairing_signature"]:
            raise ValueError(f"Unpaired training split, validation split or initialization for seed {seed}.")
    return {"seeds": matched, **{split: statistics_summary(
        second[s][f"{split}_accuracy"] - first[s][f"{split}_accuracy"] for s in matched)
        for split in ("val", "test")}}


def save(path, value):
    path = Path(path)
    pending = path.with_suffix(path.suffix + ".pending")
    pending.write_text(value, encoding="utf-8")
    pending.replace(path)


def write_report(plan, output, representation_root="outputs/representations", plots=True):
    from scripts.run_next_pilot import build_jobs
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    entries = {e["name"]: e for e in plan["experiments"]}
    rows, groups = [], {}
    for job in build_jobs(plan):
        entry = entries[job["name"]]
        directory = Path(job.get("reuse_run", str(Path(plan["output_dir"])/job["name"])))
        row = {"run": job["name"], "family": entry["family"], "method": job["config"]["method"],
               "seed": job["config"]["seed"], "epochs": job["config"]["epochs"],
               "active_strategy": job["config"].get("active_strategy"), "source": str(directory),
               "status": "pending"}
        if (directory/"metrics.json").exists():
            metrics = json.loads((directory/"metrics.json").read_text())
            if metrics.get("status") != "completed" or metrics.get("synthetic_smoke"):
                raise ValueError("Study reports accept completed real results only.")
            if any(metrics.get(k) != v for k, v in job["config"].items()):
                raise ValueError(f"Study result differs from planned config: {directory}")
            splits = json.loads((directory/"splits.json").read_text())
            paired = {k: sorted(splits[k]) for k in ("train", "validation")}
            paired["initial_model_sha256"] = metrics["initial_model_sha256"]
            row.update(status="completed", val_accuracy=metrics["val_accuracy"], test_accuracy=metrics["test_accuracy"],
                       pairing_signature=hashlib.sha256(json.dumps(paired, sort_keys=True).encode()).hexdigest())
            if entry["family"] == "replication":
                groups.setdefault(row["method"], {})[row["seed"]] = row
        elif (directory/"history.csv").exists():
            history = list(csv.DictReader((directory/"history.csv").open(encoding="utf-8")))
            row["status"] = "running"
            if history:
                row.update(epoch_completed=int(history[-1]["epoch"]), val_accuracy=float(history[-1]["val_accuracy"]))
        rows.append(row)
    summaries = {method: {split: statistics_summary(row[f"{split}_accuracy"] for row in seeds.values())
                          for split in ("val", "test")} for method, seeds in groups.items()}
    methods = ["ce_2v", "ce_2v_pseudo", "ce_2v_pseudo_soft"]
    effects = {}
    for a, b in zip(methods, methods[1:]):
        effects[f"{b} minus {a}"] = paired_effect(groups.get(a, {}), groups.get(b, {}))
    effects["ce_2v_pseudo_soft minus ce_2v"] = paired_effect(groups.get(methods[0], {}), groups.get(methods[2], {}))
    representations = []
    focus = ["ce_2v", "ce_2v_triplet_l_semihard", "ce_2v_pseudo", "ce_2v_pseudo_random_triplet",
             "ce_2v_pseudo_guided_triplet", "ce_2v_pseudo_semihard_triplet", "ce_2v_pseudo_soft"]
    for method in focus:
        p = Path(representation_root)/f"cnn_N50_seed0_{method}"/"metrics.json"
        if p.exists():
            m = json.loads(p.read_text())
            representations.append({"method": method, "source": str(p), **{k: m[k] for k in (
                "original_head_val_accuracy", "knn_5_val_accuracy", "linear_probe_val_accuracy",
                "val_neighbor_purity_10", "val_silhouette_cosine", "within_over_total_scatter")}})
    result = {"planned_runs": len(rows), "new_runs": sum("reuse_run" not in j for j in build_jobs(plan)),
              "runs": rows, "replication": summaries, "paired_effects": effects, "representations": representations}
    save(output/"study.json", json.dumps(result, indent=2, allow_nan=False))
    def pct(value):
        return "—" if value is None else f"{100*value:.2f}%"
    def interval(value):
        return "—" if value is None else f"[{100*value[0]:.2f}; {100*value[1]:.2f}] pp"
    text = ["# Weryfikacja pilota: powtarzalność, czas treningu, reprezentacje i kombinacje", "",
            "Plan z góry ustalony; seedy 0,1,2. Trzy powtórzenia to wstępna ocena stabilności.",
            "Średnie i SD liczymy po seedach. Przedziały 95% są t-Studenta, eksploracyjne,",
            "przy małym n bardzo niepewne; nie dowodzą przewagi po przeszukaniu wielu wariantów.",
            "Dla n=1 nie wyznaczamy SD ani CI. Test nie służy do wyboru wag i harmonogramu.", "",
            "| Metoda | Seedy ukończone | Średnia walidacja | Średni test | SD test | 95% CI średniej test |",
            "| --- | ---: | ---: | ---: | ---: | --- |"]
    for method, summary in summaries.items():
        s = summary["test"]
        ci = "—" if s["ci95"] is None else f"[{100*s['ci95'][0]:.2f}; {100*s['ci95'][1]:.2f}]%"
        text.append(f"| {method} | {s['n']} | {pct(summary['val']['mean'])} | {pct(s['mean'])} | {pct(s['sd'])} | {ci} |")
    text += ["", "Różnice sparowane: drugi wariant minus pierwszy, wspólne seedy i hashe inicjalizacji/podziałów.", "",
             "| Porównanie | Wspólne seedy | Różnica test | 95% CI różnicy |",
             "| --- | --- | ---: | --- |"]
    for name, effect in effects.items():
        delta = effect["test"]
        text.append(f"| {name} | {effect['seeds']} | {pct(delta['mean']).replace('%', ' pp')} | {interval(delta['ci95'])} |")
    text += ["", "Geometria reprezentacji: te same 1000 obrazów walidacji, seed 0, normalizacja L2.",
             "Linear probe i kNN korzystają wyłącznie z prawdziwego L. Sąsiedztwa/purity mierzymy w 128D, nie na t-SNE.", "",
             "| Metoda | Głowica wal. | kNN5 wal. | Linear probe wal. | Purity10 | Silhouette | Within/total |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for r in representations:
        text.append(f"| {r['method']} | {pct(r['original_head_val_accuracy'])} | {pct(r['knn_5_val_accuracy'])} | {pct(r['linear_probe_val_accuracy'])} | {pct(r['val_neighbor_purity_10'])} | {r['val_silhouette_cosine']:.4f} | {r['within_over_total_scatter']:.4f} |")
    text += ["", "Trójki klasowe semi-hard poprawiły lokalne purity i silhouette względem twardych pseudoetykiet,",
             "ale nie wszystkie miary zmieniły się zgodnie (within/total). Samo uporządkowanie geometrii",
             "nie gwarantuje lepszej klasyfikacji. Trójki instancji mogą rozsuwać obrazy tej samej klasy.",
             "To obserwacje jednego seeda; t-SNE jest ilustracją, a nie dowodem separacji klas.", "",
             "| Run | Rodzina | Status | Walidacja | Test |", "| --- | --- | --- | ---: | ---: |"]
    for r in rows:
        text.append(f"| {r['run']} | {r['family']} | {r['status']} | {pct(r.get('val_accuracy'))} | {pct(r.get('test_accuracy'))} |")
    text += ["", "Porównanie AL: losowanie i BADGE mają ten sam harmonogram 300→500, ze składnikiem trójek i bez niego.",
             "Stałe 500 od początku ma inny dostęp do etykiet; CE ma mniej treningowych wejść niż SSL.",
             "40/80 epok to dodatkowy koszt treningu i osobny eksperyment, nie identyczny budżet obliczeniowy.", ""]
    save(output/"study.md", "\n".join(text))
    if representations:
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(representations[0]))
        writer.writeheader()
        writer.writerows(representations)
        save(output/"representations.csv", stream.getvalue())
    if plots and representations:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        short = [r["method"].replace("ce_2v", "CE").replace("pseudo", "PL") for r in representations]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        for ax, key, title in zip(axes, ("knn_5_val_accuracy", "linear_probe_val_accuracy", "val_neighbor_purity_10"),
                                  ("kNN5 validation", "Linear probe validation", "Neighbor purity (10)")):
            ax.barh(short, [100*r[key] for r in representations])
            ax.set_xlabel("%")
            ax.set_title(title)
            ax.set_xlim(0, 100)
        axes[0].invert_yaxis()
        fig.suptitle("CIFAR-10: same validation images, seed 0; geometry measured in 128D")
        fig.tight_layout()
        fig.savefig(output/"representations.png", dpi=180)
        fig.savefig(output/"representations.svg")
        plt.close(fig)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", default="configs/cifar10_pilot_validation_study.json")
    parser.add_argument("--output", default="outputs/pilot_validation_study_analysis")
    args = parser.parse_args()
    result = write_report(json.loads(Path(args.plan).read_text()), args.output)
    print(f"Study report updated: {args.output}; {sum(r['status']=='completed' for r in result['runs'])}/{result['planned_runs']} completed")
