"""Derived diagnostics of confidently wrong predictions; no model inference."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fewshot_metric.reliability import detection_metrics
from fewshot_metric.supervised import write_json


def summarize(output_dir):
    root = Path(output_dir)
    rows = json.loads((root / "comparison.json").read_text())
    findings = []
    for row in rows:
        with np.load(root / row["run"] / "predictions.npz") as data:
            p, y = data["probabilities"], data["labels"]
            wrong = p.argmax(1) != y
            selected = p.max(1) >= .95
            scores = {k: data[k][selected] for k in row["detection"]}
            conditional = detection_metrics(wrong[selected], scores) if selected.any() else {}
            confusion = np.zeros((p.shape[1], p.shape[1]), dtype=int)
            np.add.at(confusion, (y, p.argmax(1)), 1)
            offdiag = confusion.copy()
            np.fill_diagonal(offdiag, 0)
            order = np.argsort(-offdiag.ravel(), kind="stable")[:5]
            pairs = [dict(true_class=int(i // p.shape[1]), predicted_class=int(i % p.shape[1]),
                          count=int(offdiag.ravel()[i])) for i in order if offdiag.ravel()[i]]
        findings.append(dict(run=row["run"], confident_count=int(selected.sum()),
                             confident_wrong=int(wrong[selected].sum()),
                             confident_error_detection=conditional, confusion=confusion.tolist(),
                             most_frequent_errors=pairs))
    write_json(root / "confident_errors.json", findings)
    lines = ["# Detecting confidently wrong validation predictions", "",
             "Restriction: clean confidence >= 0.95. Scores and cutoffs are fixed in advance; this is exploratory analysis, not a trained selector. Top 10% refers to this restricted subset.", "",
             "| Model / score | Confident count / wrong | AUROC within confident subset | Wrong in top 10% | Random precision |", "|---|---:|---:|---:|---:|"]
    for row in findings:
        for name, d in row["confident_error_detection"].items():
            auc = f'{d["auroc"]:.3f}' if d["auroc"] is not None else "n/a"
            lines.append(f'| {row["run"]} / {name} | {row["confident_count"]}/{row["confident_wrong"]} | {auc} | {100*d["precision_top_10pct"]:.1f}% | {100*d["random_precision"]:.1f}% |')
    (root / "confident_errors.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    soft_runs = ["cnn_N50_seed0_ce_2v_pseudo_soft", "cnn_N50_seed1_ce_2v_pseudo_soft",
                 "cnn_N50_seed2_ce_2v_pseudo_soft"]
    selected_rows = [r for name in soft_runs for r in findings if r["run"] == name]
    summary = ["# Wyniki diagnostyki wiarygodności", "",
        f"Przeanalizowano {len(rows)} końcowych modeli na walidacji; bez treningu i bez użycia testu.", "",
        "## Miękkie pseudoetykiety: 20 epok, trzy seedy", "",
        "Tabela dotyczy obrazów bez augmentacji z pewnością predykcji ≥95%.",
        "Ranking: JS między predykcją modelu a rozkładem klas 15 oznaczonych sąsiadów.",
        "Wybieramy 10% najwyższych wyników rankingu spośród tych pewnych predykcji.", "",
        "| Seed | Pewne predykcje / błędne | Błędy przy losowaniu | Błędy w wybranych 10%: sąsiedzi | Błędy w wybranych 10%: niepewność |",
        "|---|---:|---:|---:|---:|"]
    for i, r in enumerate(selected_rows):
        d = r["confident_error_detection"]
        summary.append(f'| {soft_runs.index(r["run"])} | {r["confident_count"]} / {r["confident_wrong"]} | {100*d["neighbor_js"]["random_precision"]:.1f}% | {100*d["neighbor_js"]["precision_top_10pct"]:.1f}% | {100*d["uncertainty"]["precision_top_10pct"]:.1f}% |')
    if len(selected_rows) == 3:
        summary += ["", "Model bywa bardzo pewny mimo błędnej odpowiedzi. W tej grupie niezgodność",
        "z oznaczonymi sąsiadami wzbogaca ranking w błędy we wszystkich trzech seedach.",
        "Na całej walidacji sąsiedzki JS ma niższe AUROC niż zwykła niepewność: sygnał jest",
        "interesujący warunkowo, jako uzupełnienie pewności, a nie uniwersalny zamiennik."]
    r0 = next((r for r in rows if r["run"] == soft_runs[0]), None)
    if r0:
        high = next(t for t in r0["weak"]["thresholds"] if t["threshold"] == .95)
        weights = r0["weak_weights"]
        summary += ["", f'Soft seed0: po słabej augmentacji {high["wrong"]} błędów wśród {high["count"]} predykcji z pewnością ≥95%.',
            f'Hipotetyczne wagi: błędne argmax odpowiadają za {100*weights["wrong_argmax_weight_fraction"]:.1f}% sumy wag; średnia waga błędnych {weights["mean_weight_wrong"]:.3f}, poprawnych {weights["mean_weight_correct"]:.3f}.',
            "To opis końcowego modelu w eval, nie pomiar błędów podczas treningu.",
            f'Błędny obraz ma średnio {100*r0["wrong_neighbor_fraction_for_wrong"]:.1f}% błędnych sąsiadów walidacyjnych, poprawny {100*r0["wrong_neighbor_fraction_for_correct"]:.1f}% (10 sąsiadów w reprezentacji).']
    summary += ["",
        "Hipoteza do kolejnego eksperymentu: oceniać zaufanie do pseudoetykiety także przez",
        "zgodność z oznaczonymi sąsiadami, szczególnie gdy pewność modelu jest wysoka.",
        "Można to następnie sprawdzić jako ważenie straty lub sygnał zapytania o etykietę.",
        "Nie wykazaliśmy jeszcze poprawy uczenia ani nowości naukowej takiej metody.", "",
        "Pełne wyniki: [report.md](report.md), [confident_errors.md](confident_errors.md).",
        "Protokół i ograniczenia: [PILOT_RELIABILITY.md](../../PILOT_RELIABILITY.md)."]
    (root / "wnioski.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    return findings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/reliability")
    args = parser.parse_args()
    summarize(args.output_dir)
