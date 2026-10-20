"""Common held-out images and common 500-label probes, including a full-label CNN."""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BASE_RUNS = [
    ("CE, 500 labels", "outputs/pilot/cnn_N50_seed0_ce_2v"),
    ("Soft SSL, 20 epochs", "outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft"),
    ("Soft SSL, 80 epochs", "outputs/pilot_validation_study/cnn_N50_seed0_ce_2v_pseudo_soft_80epochs"),
    ("Mined mixed triplets", "outputs/pilot_mined_triplets/cnn_N50_seed0_mined_mixed_embedding"),
    ("Full supervised, 49k labels", "outputs/supervised/cnn_pilot_full_seed0"),
]


def compare(output="outputs/representation_reference", extra_runs=()):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(output).resolve()/".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from fewshot_metric.representations import analyze_run
    from fewshot_metric.supervised import write_json
    torch.set_num_threads(6)
    reference = json.loads(Path("outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft/splits.json").read_text())
    rows = []
    aliases={"cnn_N50_seed0_temperature1":"Soft SSL, T=1",
             "cnn_N50_seed0_teacher_temperature05":"EMA targets, T=0.5",
             "cnn_N50_seed0_teacher_temperature1":"EMA targets, T=1",
             "cnn_N50_seed0_contrastive_labeled_instance":"Contrastive L + instances",
             "cnn_N50_seed0_contrastive_with_pseudo_pairs":"Contrastive L + instances + pseudo"}
    sources = BASE_RUNS + [(aliases.get(Path(p).name,Path(p).name), p) for p in extra_runs]
    for title, path in sources:
        files = ["model.pt"] + (["teacher_model.pt"] if (Path(path)/"teacher_model.pt").exists() else [])
        for filename in files:
            print(f"Common-reference audit: {title} / {filename}", flush=True)
            row = analyze_run(path, output, reference_indices=reference["train"],
                              validation_indices=reference["validation"], model_file=filename)
            suffix = " (teacher)" if filename != "model.pt" else (" (student)" if len(files)>1 else "")
            row = {**row, "title": title + suffix}
            rows.append(row)
    out = Path(output)
    write_json(out/"comparison.json", rows)
    lines = ["# Common-reference representation comparison", "",
        "The same 1,000 held-out validation images in the same order and the same 500 labeled probe/kNN references for every model. Source training labels differ: the full supervised encoder/head saw 49,000 labels, so it is a diagnostic reference, not a matched-label competitor.",
        "Linear probes fit only those 500 true training labels; no validation fitting. No U/test labels used. Independent PCA/t-SNE coordinate systems; plot distances across panels are not comparable.", "",
        "| Model | Source labels | Head val % | kNN15 % | Probe % | Neighbor purity % | Silhouette |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f'| {r["title"]} | {r["source_training_label_budget"]} | {100*r["original_head_val_accuracy"]:.2f} | {100*r["knn_15_val_accuracy"]:.2f} | {100*r["linear_probe_val_accuracy"]:.2f} | {100*r["val_neighbor_purity_10"]:.2f} | {r["val_silhouette_cosine"]:.4f} |')
    lines += ["", "Silhouette, purity and scatter use the original normalized 128D embedding. Plots are exploratory; compact-looking clusters do not establish better generalization. The 80-epoch run uses a different optimization horizon. The full supervised run also has a different labeled-only training-view protocol; its gap is not attributed causally to annotation count alone."]
    (out/"comparison.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    for method in ("pca", "tsne"):
        columns = min(3, len(rows))
        nrows = (len(rows)+columns-1)//columns
        fig, axes = plt.subplots(nrows, columns, figsize=(5*columns,4*nrows), squeeze=False)
        for ax, r in zip(axes.flat, rows):
            with np.load(out/r["run"]/"embeddings.npz") as data:
                xy, labels = data[method], data["validation_labels"]
            ax.scatter(xy[:,0],xy[:,1],c=labels,cmap="tab10",vmin=0,vmax=9,s=8,alpha=.7)
            ax.set_title(r["title"],fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        for ax in list(axes.flat)[len(rows):]:ax.set_visible(False)
        fig.suptitle(f"{method.upper()} — same validation images; colors = true classes",fontsize=12)
        from matplotlib.lines import Line2D
        names=["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
        handles=[Line2D([],[],marker="o",linestyle="none",color=plt.get_cmap("tab10")(i),label=name) for i,name in enumerate(names)]
        fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=9)
        fig.tight_layout(rect=(0,.07,1,.96))
        fig.savefig(out/f"{method}_comparison.png",dpi=170)
        fig.savefig(out/f"{method}_comparison.svg")
        plt.close(fig)
    return rows


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="outputs/representation_reference")
    parser.add_argument("runs",nargs="*")
    args=parser.parse_args()
    compare(args.output,args.runs)
