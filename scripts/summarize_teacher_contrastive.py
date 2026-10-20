"""Five new pilots: target temperature, EMA teacher, and mixed contrastive pairs."""
import argparse
import json
import sys
import hashlib
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.run_next_pilot import build_jobs
from fewshot_metric.supervised import write_json


def summarize(plan_path, output, representations=True):
    plan=json.loads(Path(plan_path).read_text())
    rows=[]; completed_paths=[]
    for job in build_jobs(plan):
        run=Path(job.get("reuse_run",str(Path(plan["output_dir"])/job["name"])))
        row=dict(run=job["name"],source=str(run),reused="reuse_run" in job,status="pending")
        if (run/"metrics.json").exists():
            r=json.loads((run/"metrics.json").read_text())
            if r.get("status")!="completed" or any(r.get(k)!=v for k,v in job["config"].items()):
                raise ValueError(f"Result differs from plan: {run}")
            splits=json.loads((run/"splits.json").read_text())
            if len(splits["train"])!=500 or set(splits["train"])&set(splits["validation"]):
                raise ValueError("Annotation budget or validation overlap")
            row.update(status="completed",val_accuracy=r["val_accuracy"],test_accuracy=r["test_accuracy"],
                teacher_val_accuracy=r.get("teacher_val_accuracy"),teacher_test_accuracy=r.get("teacher_test_accuracy"),
                updates=r["total_updates"],labeled_views=r["processed_labeled_views"],unlabeled_views=r["processed_unlabeled_views"],
                teacher_training_views=r.get("ema_training_views",0),teacher_validation_views=r.get("ema_validation_views",0),
                teacher_test_views=r.get("ema_test_views",0),elapsed_minutes=r["elapsed_seconds"]/60,
                initial_model_sha256=r["initial_model_sha256"],
                split_sha256=hashlib.sha256(json.dumps(splits,sort_keys=True).encode()).hexdigest(),
                contrastive_counts={k:v for k,v in r.items() if k.startswith("contrastive_") and k.endswith("pairs")})
            if r.get("ema_teacher"):
                expected=(r["epochs"]-r["ema_teacher_warmup_epochs"])*r["steps_per_epoch"]*r["unlabeled_batch_size"]
                if r["ema_training_views"]!=expected or r["ema_updates"]!=r["total_updates"]:
                    raise ValueError("EMA forward/update counters disagree")
                if not (run/"teacher_model.pt").exists():raise ValueError("Teacher checkpoint missing")
            if "reuse_run" not in job:completed_paths.append(str(run))
        rows.append(row)
    done=[r for r in rows if r["status"]=="completed"]
    for key in ("updates","labeled_views","unlabeled_views","initial_model_sha256","split_sha256"):
        if len({r[key] for r in done})>1:raise ValueError(f"Unmatched {key}")
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    write_json(out/"comparison.json",rows)
    lines=["# Temperature, teacher and contrastive pilot","",
        "Five new training runs, one existing baseline. Same CNN and trainable parameters, true L=500, split/init, 20 epochs and 7660 SGD updates. Primary evaluation is always the student at the last epoch. Teacher scores are secondary, without selecting the better model.",
        "EMA has additional eval-only weak-view forwards starting at epoch4; validation/test forwards and runtime are also reported. Equal updates and labels do not imply equal compute. No dummy forward baselines are trained.",
        "Contrastive variants use the same normalized 128D embedding, with no extra projection head or encoder passes. Their only difference is whether confident pseudo-label class pairs (U-U and L-U) enter the positive graph. CE and soft pseudo-label losses remain enabled.","",
        "| Run | Status | Student val % | Student test % | EMA val / test % | Extra teacher train / val / test views | Minutes |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["status"]=="completed":
            teacher=f'{100*r["teacher_val_accuracy"]:.2f} / {100*r["teacher_test_accuracy"]:.2f}' if r["teacher_test_accuracy"] is not None else "—"
            status="reused" if r["reused"] else "completed"
            lines.append(f'| {r["run"]} | {status} | {100*r["val_accuracy"]:.2f} | {100*r["test_accuracy"]:.2f} | {teacher} | {r["teacher_training_views"]} / {r["teacher_validation_views"]} / {r["teacher_test_views"]} | {r["elapsed_minutes"]:.1f} |')
        else:lines.append(f'| {r["run"]} | pending | — | — | — | — | — |')
    lines += ["","One exploratory seed; no guarantee of accuracy improvement. Parameters frozen before new runs. The full-label model is a representation reference with a different annotation budget.",
              "Common-image, common-500-label-reference representation tables and PCA/t-SNE: outputs/representation_reference/comparison.md and *_comparison.png/svg."]
    (out/"comparison.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    if representations:
        from scripts.compare_reference_representations import compare
        compare(extra_runs=completed_paths)
    return rows


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default="configs/cifar10_pilot_teacher_contrastive.json")
    parser.add_argument("--output",default="outputs/pilot_teacher_contrastive_analysis")
    parser.add_argument("--no-representations",action="store_true")
    args=parser.parse_args()
    summarize(args.plan,args.output,not args.no_representations)
