"""Small resumable MNIST-family SSL study; no CIFAR outputs are modified."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_suffix(".pending.json")
    pending.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    pending.replace(path)


def build_jobs(plan):
    from fewshot_metric.pilot import select_method
    from fewshot_metric.supervised import validate_config
    presets={"ce":"ce_2v","hard":"ce_2v_pseudo","soft":"ce_2v_pseudo_soft",
             "soft_triplet":"ce_2v_pseudo_soft_mined_triplet","soft_contrastive":"ce_2v_pseudo_soft"}
    jobs=[]
    dataset=plan["dataset"]
    if dataset not in {"mnist","fashion_mnist"} or plan["base_config"]["dataset"]!=dataset:
        raise ValueError("This runner requires a consistent MNIST-family dataset.")
    for seed in plan["seeds"]:
        for budget in plan["budgets_per_class"]:
            for variant in plan["methods"]:
                config=select_method({**plan["base_config"],"seed":seed,"budget_per_class":budget},presets[variant])
                if variant=="soft_contrastive":
                    config.update(contrastive_weight=.05,contrastive_temperature=.2,
                                  contrastive_pseudo_pairs=True,contrastive_pseudo_threshold=.95,
                                  contrastive_warmup_epochs=3)
                validate_config(config)
                jobs.append({"name":f"{dataset}_N{budget}_seed{seed}_{variant}","variant":variant,"config":config})
        if plan.get("full_reference"):
            config=select_method({**plan["base_config"],"seed":seed,"budget_per_class":"all",
                "epochs":plan["full_reference"]["epochs"],
                "steps_per_epoch":math.ceil((60000-10*plan["base_config"]["validation_per_class"])/plan["base_config"]["batch_size"])},"ce_2v")
            validate_config(config)
            jobs.append({"name":f"{dataset}_all_seed{seed}_ce","variant":"full_reference","config":config})
    if len({j["name"] for j in jobs})!=len(jobs):raise ValueError("Duplicate runs in plan.")
    return jobs


def benchmark(plan,steps=12):
    import torch
    from torch.utils.data import DataLoader
    from fewshot_metric.models import build_model
    from fewshot_metric.pilot import PilotTraining
    from fewshot_metric.supervised import build_datasets,train_epoch
    from fewshot_metric.train import seed_everything
    torch.set_num_threads(plan["torch_threads"])
    rows=[]
    for job in [j for j in build_jobs(plan) if j["config"]["budget_per_class"]==plan["budgets_per_class"][0]]:
        config=job["config"];seed_everything(config["seed"])
        train,_,splits=build_datasets(config,Path(plan["data_root"]),False)
        loader=DataLoader(train,batch_size=config["batch_size"],shuffle=True,drop_last=True,
                          generator=torch.Generator().manual_seed(config["seed"]))
        pilot=PilotTraining(config,train,splits)
        model=build_model(config["architecture"],embedding_dim=config["embedding_dim"])
        optimizer=torch.optim.SGD(model.parameters(),lr=config["learning_rate"],momentum=.9)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=steps+2)
        pilot.start_epoch(3)  # Include the active triplet/contrastive work in the estimate.
        train_epoch(model,loader,optimizer,scheduler,torch.device("cpu"),2,pilot)
        tick=time.perf_counter()
        train_epoch(model,loader,optimizer,scheduler,torch.device("cpu"),steps,pilot)
        seconds=(time.perf_counter()-tick)/steps
        rows.append({"variant":job["variant"],"seconds_per_step":seconds,
                     "estimated_training_seconds":seconds*config["epochs"]*config["steps_per_epoch"]})
        print(json.dumps(rows[-1]),flush=True)
    value={"purpose":"Timing only; no official test evaluation or research results",
           "device":"cpu","torch_threads":plan["torch_threads"],"steps_timed":steps,"methods":rows}
    atomic_json(Path(plan["output_dir"])/"benchmark.json",value)
    return value


def audit_job(job,plan):
    from fewshot_metric.representations import analyze_run
    root=Path(plan["output_dir"])
    run=root/job["name"]
    audits=[]
    if job["variant"]=="full_reference":
        for budget in plan["budgets_per_class"]:
            ce=root/f"{plan['dataset']}_N{budget}_seed{job['config']['seed']}_ce"
            splits=json.loads((ce/"splits.json").read_text())
            audits.append(analyze_run(run,root/"representations"/f"N{budget}",plan["data_root"],
                tsne=plan["representation_tsne"],reference_indices=splits["train"],
                validation_indices=splits["validation"],pca_plot=False))
    else:
        audits.append(analyze_run(run,root/"representations"/f"N{job['config']['budget_per_class']}",
            plan["data_root"],tsne=plan["representation_tsne"],pca_plot=False))
    return audits


def summarize(plan,jobs):
    root=Path(plan["output_dir"])
    rows=[]
    for job in jobs:
        path=root/job["name"]/"metrics.json"
        if not path.exists():continue
        m=json.loads(path.read_text())
        r={"run":job["name"],"variant":job["variant"],"dataset":m["dataset"],"seed":m["seed"],
           "budget_per_class":m["budget_per_class"],"training_labels":m["true_training_label_budget"],
           "val_accuracy":m["val_accuracy"],"test_accuracy":m["test_accuracy"],
           "epochs":m["epochs"],"updates":m["total_updates"],
           "labeled_views":m["processed_labeled_views"],"unlabeled_views":m["processed_unlabeled_views"],
           "elapsed_seconds":m["elapsed_seconds"],"initial_model_sha256":m["initial_model_sha256"]}
        if job["variant"]!="full_reference":
            audit=root/"representations"/f"N{m['budget_per_class']}"/job["name"]/"metrics.json"
            if audit.exists():
                a=json.loads(audit.read_text())
                for key in ["linear_probe_val_accuracy","knn_15_val_accuracy","val_neighbor_purity_10","val_silhouette_cosine"]:r[key]=a[key]
        rows.append(r)
    atomic_json(root/"comparison.json",rows)
    if rows:
        keys=list(dict.fromkeys(k for r in rows for k in r))
        with (root/"comparison.csv").open("w",newline="",encoding="utf-8-sig") as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)
        plot_summary(plan,rows)
    return rows


def plot_summary(plan,rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    root=Path(plan["output_dir"]);plots=root/"plots";plots.mkdir(exist_ok=True)
    labels={"ce":"CE","hard":"Twarde PL","soft":"Miękkie PL","soft_triplet":"Miękkie PL + trójki B",
            "soft_contrastive":"Miękkie PL + contrastive"}
    colors={k:plt.get_cmap("tab10")(i) for i,k in enumerate(labels)}
    for key,title,filename in [("test_accuracy","Dokładność testowa [%]","accuracy_vs_labels"),
                               ("linear_probe_val_accuracy","Liniowy probe: walidacja [%]","probe_vs_labels"),
                               ("val_neighbor_purity_10","Czystość 10 sąsiadów: walidacja [%]","purity_vs_labels")]:
        fig,ax=plt.subplots(figsize=(8,4.7))
        for variant in plan["methods"]:
            selected=[r for r in rows if r["variant"]==variant and key in r]
            budgets=sorted({r["budget_per_class"] for r in selected})
            if not budgets:continue
            means=[100*np.mean([r[key] for r in selected if r["budget_per_class"]==b]) for b in budgets]
            ax.plot(budgets,means,"o-",label=labels[variant],color=colors[variant])
        ref=[r for r in rows if r["variant"]=="full_reference"]
        if ref and key=="test_accuracy":
            ax.axhline(100*np.mean([r[key] for r in ref]),color="gray",ls="--",label="Pełny nadzór: 59 000 etykiet, inny koszt")
        ax.set_xticks(sorted(plan["budgets_per_class"]));ax.set_xlabel("Prawdziwe etykiety na klasę (10 klas)")
        ax.set_ylabel(title);ax.set_title(plan["dataset"]+"; pilot, nie pełna reprodukcja literatury")
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(plots/(filename+".png"),dpi=180);fig.savefig(plots/(filename+".svg"));plt.close(fig)
    # Separate panels per budget, identical held-out images; projections have independent axes.
    for budget in sorted(plan["budgets_per_class"]):
        available=[r for r in rows if r["budget_per_class"]==budget and r["seed"]==plan["seeds"][0]]
        if not available:continue
        fig,axes=plt.subplots(2,3,figsize=(10,6.6))
        archive_root=root/"representations"/f"N{budget}"
        available += [r for r in rows if r["variant"]=="full_reference" and r["seed"]==plan["seeds"][0]]
        for ax in axes.flat:ax.set_axis_off()
        for ax,r in zip(axes.flat,available):
            file=archive_root/r["run"]/"embeddings.npz"
            if not file.exists():continue
            with np.load(file) as a:
                if "tsne" not in a:continue
                ax.scatter(a["tsne"][:,0],a["tsne"][:,1],c=a["validation_labels"],cmap="tab10",vmin=0,vmax=9,s=4,alpha=.65)
            ax.set_title(labels.get(r["variant"],"Pełny nadzór: 59k"),fontsize=9)
        from matplotlib.lines import Line2D
        names=(["T-shirt","trousers","pullover","dress","coat","sandal","shirt","sneaker","bag","ankle boot"]
               if plan["dataset"]=="fashion_mnist" else [str(i) for i in range(10)])
        handles=[Line2D([],[],marker="o",linestyle="none",markersize=4,color=plt.get_cmap("tab10")(i),label=n) for i,n in enumerate(names)]
        fig.legend(handles=handles,loc="lower center",ncol=5,fontsize=8)
        fig.suptitle(f"{plan['dataset']}: {budget} etykiet/klasę; wspólna walidacja, osobne t-SNE",fontsize=11)
        fig.tight_layout(rect=(0,.07,1,.96));fig.savefig(plots/f"tsne_N{budget}.png",dpi=180);plt.close(fig)


def run_queue(plan,device="cpu",max_jobs=None):
    import torch
    from fewshot_metric.supervised import run_supervised
    jobs=build_jobs(plan);root=Path(plan["output_dir"]);root.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR",str((root/".mplconfig").resolve()))
    torch.set_num_threads(plan["torch_threads"])
    lock=(root/"queue.lock").open("a+b");lock.seek(0,2)
    if not lock.tell():lock.write(b"0");lock.flush()
    lock.seek(0)
    if os.name=="nt":
        import msvcrt
        msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    else:
        import fcntl
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state={"status":"running","pid":os.getpid(),"started_utc":datetime.now(timezone.utc).isoformat(),
           "plan":plan,"jobs":[{"name":j["name"],"variant":j["variant"],"status":"pending"} for j in jobs]}
    fingerprint=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    old=root/"queue.json"
    if old.exists() and json.loads(old.read_text())["plan_sha256"]!=fingerprint:
        raise ValueError("Plan differs from the existing queue; use a new output directory.")
    state["plan_sha256"]=fingerprint
    source_files=[Path(__file__),ROOT/"configs/fashion_mnist_ssl.json",*sorted((ROOT/"fewshot_metric").glob("*.py"))]
    atomic_json(root/"source_manifest.json",{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    def save():atomic_json(root/"queue.json",state)
    save();current=None;executed=0
    try:
        for job,stage in zip(jobs,state["jobs"]):
            current=stage;run=root/job["name"]
            stage["status"]="training";save()
            metrics=run/"metrics.json"
            if metrics.exists():
                m=json.loads(metrics.read_text())
                if m.get("status")!="completed" or any(m.get(k)!=v for k,v in job["config"].items()):
                    raise ValueError(f"Existing result/configuration mismatch: {run}")
            else:
                if max_jobs is not None and executed>=max_jobs:
                    stage["status"]="pending";state["status"]="stopped_at_job_limit";save();break
                print(f"Starting {job['name']}",flush=True)
                m=run_supervised(job["config"],run,Path(plan["data_root"]),device,resume=(run/"latest.pt").exists())
                executed+=1
            stage.update(status="analyzing",val_accuracy=m["val_accuracy"],test_accuracy=m["test_accuracy"],
                         elapsed_seconds=m["elapsed_seconds"]);save()
            audit_job(job,plan)
            stage["status"]="completed";save()
            rows=summarize(plan,jobs)
            # Paired methods at a fixed budget must share init and labeled exposure.
            for budget in plan["budgets_per_class"]:
                paired=[r for r in rows if r["budget_per_class"]==budget and r["seed"]==job["config"]["seed"]]
                assert len({r["initial_model_sha256"] for r in paired})<=1
                assert len({r["labeled_views"] for r in paired})<=1
                assert len({r["updates"] for r in paired})<=1
        else:state["status"]="completed";save()
    except Exception as e:
        state.update(status="failed",error=str(e))
        if current:current["status"]="failed"
        save();raise
    finally:lock.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan",default="configs/fashion_mnist_ssl.json")
    parser.add_argument("--device",default="cpu",choices=["cpu","cuda","auto"])
    parser.add_argument("--print-plan",action="store_true")
    parser.add_argument("--benchmark",action="store_true")
    parser.add_argument("--max-jobs",type=int)
    args=parser.parse_args();os.chdir(ROOT)
    plan=json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if args.print_plan:print(json.dumps(build_jobs(plan),indent=2))
    elif args.benchmark:benchmark(plan)
    else:run_queue(plan,args.device,args.max_jobs)
