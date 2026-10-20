"""Train-only selection of nonlinear heads on previously exported frozen embeddings."""
import argparse
import hashlib
import json
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


def fit_heads(train, labels):
    """No validation data or labels enter fitting or hyperparameter selection."""
    heads = {
        "linear": make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=1000, random_state=0)),
        "rbf_svm": GridSearchCV(
            SVC(kernel="rbf"), {"C": [1, 10, 100], "gamma": [.1, 1, 10]},
            cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=0),
            scoring="accuracy", n_jobs=1, refit=True, error_score="raise"),
        "mlp64": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(64,), solver="lbfgs", alpha=1., max_iter=1500,
            max_fun=20000, random_state=0)),
    }
    diagnostics = {}
    for name, head in heads.items():
        started = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            head.fit(train, labels)
        diagnostics[name] = {
            "fit_seconds": time.monotonic() - started,
            "warnings": [str(w.message) for w in caught],
        }
        if name == "rbf_svm":
            diagnostics[name].update(cv_accuracy=float(head.best_score_),
                                     selected_params=head.best_params_,
                                     support_vectors=int(head.best_estimator_.n_support_.sum()))
        if name == "mlp64":
            diagnostics[name].update(iterations=int(head[-1].n_iter_), loss=float(head[-1].loss_))
    return heads, diagnostics


def analyze(source="outputs/representation_reference", output="outputs/frozen_classifiers"):
    source, output = Path(source), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((source / "comparison.json").read_text(encoding="utf-8"))
    results, common = [], None
    with threadpool_limits(limits=1):
        for row in rows:
            path = source / row["run"] / "embeddings.npz"
            with np.load(path) as data:
                train, validation = data["train"], data["validation"]
                labels, truth = data["train_labels"], data["validation_labels"]
                indices = {k: data[k].copy() for k in (
                    "train_indices", "validation_indices", "train_labels", "validation_labels")}
            if set(indices["train_indices"]) & set(indices["validation_indices"]):
                raise ValueError("Training and validation references overlap")
            if common is None:
                common = indices
            elif any(not np.array_equal(common[k], v) for k, v in indices.items()):
                raise ValueError("All heads must use identical ordered labeled/validation references")
            print(f'Fitting heads: {row["title"]}', flush=True)
            heads, diagnostics = fit_heads(train, labels)
            result = {"run": row["run"], "title": row["title"],
                      "source_training_label_budget": row["source_training_label_budget"],
                      "embedding_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "encoder_signature": row["signature"],
                      "training_examples": len(labels), "validation_examples": len(truth),
                      "original_head_val_accuracy": row["original_head_val_accuracy"],
                      "knn_15_val_accuracy": row["knn_15_val_accuracy"], "heads": {}}
            predictions = {}
            for name, head in heads.items():
                predictions[name] = head.predict(validation)
                result["heads"][name] = {
                    **diagnostics[name],
                    "train_accuracy": float(np.mean(head.predict(train) == labels)),
                    "val_accuracy": float(np.mean(predictions[name] == truth)),
                }
            np.savez_compressed(output / f'{row["run"]}_predictions.npz', **indices, **predictions)
            results.append(result)
            print({name: round(100 * r["val_accuracy"], 2) for name, r in result["heads"].items()}, flush=True)
    payload = {
        "protocol_version": 1,
        "protocol": "Frozen normalized 128D embeddings. Fit heads on common 500 true labeled images only. RBF SVM: 3-fold stratified train-only CV, C=[1,10,100], gamma=[0.1,1,10]. Linear: StandardScaler + C=1 logistic regression. MLP: StandardScaler + 64 hidden units, lbfgs, alpha=1, seed=0, max_iter=1500, fixed settings. One CPU thread. No validation fitting, no test/U labels, no CNN training or forwards. Validation was inspected previously: exploratory diagnostic, not independent confirmation. Full encoder used 49k labels and is only a diagnostic reference.",
        "cv_limitation": "The encoder already used all 500 reference labels during training, including head-CV held-out folds. CV only selects a head conditional on this fixed encoder; it is not an independent generalization estimate for the entire pipeline. Ties use GridSearchCV's first parameter combination. Final validation images were excluded from encoder and head training.",
        "results": results,
    }
    (output / "report.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    lines = ["# Frozen classifiers", "", payload["protocol"], "", payload["cv_limitation"], "",
             "Validation accuracy (%):", "",
             "| Encoder | Source labels | Original head | kNN15 | Linear | RBF SVM | MLP64 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for r in results:
        values = [r["original_head_val_accuracy"], r["knn_15_val_accuracy"]]
        values += [r["heads"][h]["val_accuracy"] for h in ("linear", "rbf_svm", "mlp64")]
        lines.append(f'| {r["title"]} | {r["source_training_label_budget"]} | ' + " | ".join(f"{100*v:.2f}" for v in values) + " |")
    lines += ["", "Training fit and train-only CV diagnostics:", "",
              "| Encoder | Linear train % | SVM train % | SVM CV % | Selected C, gamma | MLP train % |",
              "|---|---:|---:|---:|---|---:|"]
    for r in results:
        h = r["heads"]
        lines.append(f'| {r["title"]} | {100*h["linear"]["train_accuracy"]:.2f} | {100*h["rbf_svm"]["train_accuracy"]:.2f} | {100*h["rbf_svm"]["cv_accuracy"]:.2f} | {h["rbf_svm"]["selected_params"]} | {100*h["mlp64"]["train_accuracy"]:.2f} |')
    for r in results:
        for name, metrics in r["heads"].items():
            for warning in metrics["warnings"]:
                lines += ["", f'Fit warning ({r["title"]}, {name}): ' + warning.replace("\n", " ")]
    lines += ["", "These heads do not establish the maximum accuracy achievable from an embedding. RBF and MLP costs include additional supervised fitting; this is a frozen-representation diagnostic, not an equal-compute replacement for the original training comparison."]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="outputs/representation_reference")
    parser.add_argument("--output", default="outputs/frozen_classifiers")
    args = parser.parse_args()
    analyze(args.source, args.output)
