"""One auditable comparison of all pilot families, preserving budget differences."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


LABELS = {
    "ce_2v": "CE", "ce_2v_u_control": "CE + wspólny forward U (kontrola BN)",
    "ce_2v_triplet_l": "CE + trójki L", "ce_2v_triplet_u": "CE + trójki U",
    "ce_2v_triplet_lu": "CE + trójki L/U", "ce_2v_pseudo": "CE + pseudoetykiety",
    "ce_2v_pseudo_random_triplet": "CE + pseudo + trójki instancji",
    "ce_2v_pseudo_guided_triplet": "CE + pseudo + trójki klasowe",
    "ce_2v_triplet_l_semihard": "CE + trójki L semi-hard",
    "ce_2v_pseudo_semihard_triplet": "CE + pseudo + trójki semi-hard",
    "ce_2v_pseudo_soft_weight": "CE + miękkie wagi pseudoetykiet",
    "ce_2v_pseudo_soft": "CE + miękkie cele i wagi pseudoetykiet",
    "ce_2v_pseudo_soft_instance_triplet": "CE + miękkie pseudo + trójki instancji batch-all",
    "ce_2v_pseudo_soft_mixed_triplet": "CE + miękkie pseudo + trójki L/pseudo/U",
    "ce_2v_pseudo_soft_mined_triplet": "CE + soft pseudo + mined triplets",
    "dae": "CE + AE", "dae_pseudo": "CE + AE + pseudoetykiety",
    "dvae": "CE + VAE", "dvae_pseudo": "CE + VAE + pseudoetykiety",
}


def collect_comparison(roots):
    rows = []
    for root in roots:
        for directory in sorted(Path(root).glob("cnn_*")):
            if not directory.is_dir() or not (directory/"config.json").exists():
                continue
            try:
                config = json.loads((directory/"config.json").read_text())
                marker = directory/"run_status.json"
                if marker.exists() and json.loads(marker.read_text()).get("status") == "superseded":
                    continue
                if config.get("synthetic_smoke"):
                    continue
                metrics = json.loads((directory/"metrics.json").read_text()) if (directory/"metrics.json").exists() else {}
                history = []
                if (directory/"history.csv").exists():
                    with (directory/"history.csv").open(encoding="utf-8", newline="") as handle:
                        history = list(csv.DictReader(handle))
                splits = json.loads((directory/"splits.json").read_text()) if (directory/"splits.json").exists() else {}
            except (OSError, json.JSONDecodeError):
                continue  # A running worker may be replacing an artifact.
            active = "active_strategy" in config
            planned = int(config["epochs"])
            done = planned if metrics else (int(history[-1]["epoch"]) if history else 0)
            def count(key):
                if metrics:
                    return metrics.get(key)
                return sum(int(float(row.get(key, 0))) for row in history)
            labeled = count("processed_labeled_views")
            unlabeled = count("processed_unlabeled_views")
            initial = config.get("initial_label_budget", config.get("labeled_train_size", 10*int(config["budget_per_class"])))
            final = config.get("active_final_budget", config.get("labeled_train_size", initial))
            val_accuracy = metrics.get("val_accuracy", float(history[-1]["val_accuracy"]) if history else None)
            method = config.get("reconstruction_variant", config.get("method", "ce_2v"))
            label = f"{LABELS.get(method, method)} + AL {config['active_strategy']}" if active else LABELS.get(method, method)
            if config.get("neighbor_reference"):
                label += "; neighbor weighting" if config.get("neighbor_pseudo_weighting") else "; neighbor eval control"
            if "metric_triplet_design" in config:
                label += f"; {config['metric_triplet_design']} / {config['metric_triplet_distance']}"
                if config.get("metric_triplet_mask_same_pseudo"):
                    label += "; instance negative mask"
            if config.get("ema_teacher"):
                label += "; EMA targets"
            if config.get("contrastive_weight",0):
                label += "; mixed contrastive" + (" + pseudo pairs" if config.get("contrastive_pseudo_pairs") else "")
            if method == "ce_2v_pseudo_soft" and config.get("pseudo_temperature",.5) != .5:
                label += f"; T={config['pseudo_temperature']}"
            if "fixed_subset_strategy" in config:
                label += f"; końcowe AL {config['fixed_subset_strategy']} od początku"
            def signature(indices):
                return hashlib.sha256(json.dumps(sorted(indices)).encode()).hexdigest() if indices else None
            rows.append({"run": directory.name, "variant": label, "status": "completed" if metrics else "in_progress",
                         "seed": config["seed"], "dataset": config.get("dataset", "cifar10"),
                         "normalization": config.get("normalization", "batchnorm"),
                         "budget_schedule": "gradual" if active else "fixed",
                         "fixed_subset_strategy": config.get("fixed_subset_strategy"),
                         "fixed_split_manifest_sha256": config.get("fixed_split_manifest_sha256"),
                         "initial_label_budget": initial, "final_label_budget": final,
                         "current_label_budget": len(splits.get("train", [])),
                         "validation_labels": config.get("validation_size", 1000),
                         "epochs_planned": planned, "epochs_completed": done,
                         "updates_completed": metrics.get("total_updates", int(history[-1]["updates"]) if history else 0),
                         "labeled_encoder_views": labeled, "unlabeled_encoder_views": unlabeled,
                         "total_encoder_views": labeled+unlabeled if labeled is not None and unlabeled is not None else None,
                         "decoder_views": metrics.get("processed_decoder_views", count("processed_decoder_views") or 0),
                         "acquisition_scoring_views": metrics.get("scoring_forward_views"),
                         "neighbor_reference_views": count("neighbor_reference_views"),
                         "neighbor_query_views": count("neighbor_query_views"),
                         "teacher_training_views": count("ema_training_views"),
                         "teacher_val_accuracy": metrics.get("teacher_val_accuracy"),
                         "teacher_test_accuracy": metrics.get("teacher_test_accuracy"),
                         "val_accuracy": val_accuracy, "test_accuracy": metrics.get("test_accuracy"),
                         "training_parameters": metrics.get("trainable_parameter_count", config.get("trainable_parameter_count", 111498)),
                         "inference_parameters": metrics.get("inference_parameter_count", config.get("inference_parameter_count", 111498)),
                         "elapsed_minutes": round(metrics["elapsed_seconds"]/60, 2) if metrics else None,
                         "initial_model_sha256": config.get("initial_model_sha256"),
                         "initial_shared_parameters_sha256": config.get("initial_shared_parameters_sha256"),
                         "validation_indices_sha256": signature(splits.get("validation", [])),
                         "final_train_indices_sha256": signature(splits.get("train", [])),
                         "input_policy": config.get("reconstruction_input_policy", "existing_weak_strong_views" if (unlabeled or active) else "labeled_weak_views"),
                         "source": str(directory/"metrics.json") if metrics else str(directory/"history.csv")})
    return rows


def write_comparison(roots=None, output="outputs/pilot_comparison"):
    roots = roots or ["outputs/pilot", "outputs/pilot_next", "outputs/pilot_reconstruction", "outputs/pilot_normalization",
                      "outputs/pilot_active_prepared", "outputs/pilot_validation_study", "outputs/pilot_active_subset_control",
                      "outputs/pilot_neighbor_reliability", "outputs/pilot_mined_triplets", "outputs/pilot_teacher_contrastive"]
    rows = collect_comparison(roots)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    def save(suffix, text):
        path = output.with_suffix(suffix)
        pending = path.with_suffix(path.suffix+".pending")
        pending.write_text(text, encoding="utf-8")
        pending.replace(path)
    save(".json", json.dumps(rows, indent=2, allow_nan=False))
    if rows:
        import io
        stream = io.StringIO(newline="")
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        save(".csv", stream.getvalue())
    seeds=", ".join(str(seed) for seed in sorted({row["seed"] for row in rows}))
    text = ["# Wspólne porównanie pilota CIFAR-10", "",
            f"Wszystkie rodziny w jednym zestawieniu. Seedy: {seeds}; budżet walidacji jest zapisany w CSV/JSON.",
            "L/U to rzeczywiste wejścia encodera, a nie liczba relacji lub pseudoetykiet.", "",
            "| Wariant | Normalizacja | Budżet etykiet | Epoki | Kroki | Widoki L | Widoki U | Dekoder | Czas min | Wal. | Test |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def number(value):
        return f"{value:,}".replace(",", " ") if value is not None else "—"
    def accuracy(value):
        return f"{100*value:.2f}%" if value is not None else "—"
    for row in rows:
        budget = str(row["final_label_budget"]) if row["budget_schedule"] == "fixed" else f"{row['initial_label_budget']} → {row['final_label_budget']}"
        epochs = f"{row['epochs_completed']}/{row['epochs_planned']}"
        minutes = f"{row['elapsed_minutes']:.1f}" if row['elapsed_minutes'] is not None else "—"
        text.append(f"| {row['variant']} | {row['normalization']} | {budget} | {epochs} | {number(row['updates_completed'])} | {number(row['labeled_encoder_views'])} | {number(row['unlabeled_encoder_views'])} | {number(row['decoder_views'])} | {minutes} | {accuracy(row['val_accuracy'])} | {accuracy(row['test_accuracy'])} |")
    text += ["", "AE/VAE współdzielą wejścia encodera z istniejącymi wariantami SSL, ale mają dodatkowy koszt dekodera.",
             "CE i trójki wyłącznie L mają mniej wejść; CE40 jest osobną kontrolą dłuższego treningu.",
             "Active learning ma inny harmonogram dostępu do etykiet. Jego strategie porównujemy przede wszystkim z AL random; rzeczywiste wejścia L mogą różnić się od stałego budżetu przez mniejsze końcowe minibatchy.",
             "Równe epoki oznaczają wspólną liczbę aktualizacji; nie gwarantują równych FLOPs.",
             "CSV/JSON zawierają ponadto parametry, czas, forwardy selekcji, hash inicjalizacji i podziałów danych.", ""]
    save(".md", "\n".join(text))
    return rows
