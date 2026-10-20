"""Export completed research metadata for GitHub, without data, weights or caches."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
RESULTS=ROOT/"results"
REPORTS=ROOT/"reports"


def read(path):return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def export():
    written=set()
    def copy(source,destination):
        if not source.is_file():raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True,exist_ok=True)
        if source.suffix in {".json",".csv",".md"}:
            destination.write_bytes(source.read_bytes().replace(b"\r\n",b"\n").replace(b"\r",b""))
        else:shutil.copyfile(source,destination)
        written.add(destination)

    def write(path,value):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+"\n").encode("utf-8"))
        written.add(path)

    cifar=read(ROOT/"outputs/pilot_comparison.json")
    fashion=read(ROOT/"outputs/fashion_mnist_ssl/comparison.json")
    sources=[(ROOT/r["source"]).parent for r in cifar]
    sources += [ROOT/"outputs/supervised"/name for name in
                ("cnn_pilot_N10_seed0","cnn_pilot_N50_seed0","cnn_pilot_full_seed0")]
    sources += [ROOT/"outputs/fashion_mnist_ssl"/r["run"] for r in fashion]
    if len(sources)!=74 or len(set(sources))!=74:raise ValueError("Unexpected or duplicate run registry.")
    registry=[]
    for run in sources:
        metrics,config=read(run/"metrics.json"),read(run/"config.json")
        if metrics["status"]!="completed" or config.get("synthetic_smoke"):
            raise ValueError(f"Not a completed real run: {run}")
        destination=RESULTS/"runs"/run.relative_to(ROOT/"outputs")
        for name in ("config.json","metrics.json","splits.json","history.csv"):
            copy(run/name,destination/name)
        registry.append({"run":run.name,"dataset":metrics["dataset"],"source":run.relative_to(ROOT).as_posix(),
                         "metadata":destination.relative_to(ROOT).as_posix(),"test_accuracy":metrics["test_accuracy"]})
    write(RESULTS/"runs.json",registry)
    for family,source in [("cifar10",ROOT/"outputs/pilot_comparison"),
                          ("fashion_mnist",ROOT/"outputs/fashion_mnist_ssl/comparison")]:
        for suffix in (".json",".csv"):
            copy(source.with_suffix(suffix),RESULTS/family/("comparison"+suffix))
    allowed={p.name for p in sources}
    for directory in [ROOT/"outputs/representations",ROOT/"outputs/representation_reference"]:
        for source in directory.glob("*/metrics.json"):
            if source.parent.name.removesuffix("__teacher") in allowed:
                copy(source,RESULTS/"representations"/directory.name/source.parent.name/"metrics.json")
    for source in (ROOT/"outputs/fashion_mnist_ssl/representations").glob("N*/*/metrics.json"):
        copy(source,RESULTS/"representations/fashion_mnist"/source.relative_to(ROOT/"outputs/fashion_mnist_ssl/representations"))
    for relative in ("representation_reference/comparison.json","frozen_classifiers/report.json","reliability/comparison.json"):
        copy(ROOT/"outputs"/relative,RESULTS/"analyses"/relative)
    for source in (ROOT/"outputs/fashion_mnist_ssl/plots").glob("*.png"):
        copy(source,RESULTS/"figures"/("fashion_"+source.name))
    for name in ("cnn_architecture.png","validation_tsne.png","three_seeds.png","training_horizon.png"):
        copy(ROOT/"outputs/research_report/assets"/name,RESULTS/"figures"/("cifar_"+name))
    copy(ROOT/"outputs/research_report/Raport_badan_CIFAR10_2026-10-20.pdf",REPORTS/"Raport_badan_CIFAR10_2026-10-20.pdf")
    copy(ROOT/"outputs/research_report/katalog_wynikow.csv",RESULTS/"cifar10/all_runs.csv")
    packages={}
    for name in ("torch","torchvision","numpy","pandas","scipy","scikit-learn","matplotlib","PyMuPDF"):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    write(RESULTS/"environment.json",{"description":"Environment at handoff export; per-run versions are in run configs/metrics",
          "python":platform.python_version(),"system":platform.system(),"packages":packages})
    write(RESULTS/"manifest.json",{"snapshot_date":"2026-10-20","completed_runs":len(registry),
          "scope":"58 CIFAR-10 and 16 Fashion-MNIST real completed trainings; metadata only, no checkpoint weights",
          "source_files":{p.relative_to(ROOT).as_posix():sha(p) for p in
             [*sorted((ROOT/"fewshot_metric").glob("*.py")),*sorted((ROOT/"scripts").glob("*.py")),
              *sorted((ROOT/"configs").glob("*.json"))]},
          "files":[{"path":p.relative_to(ROOT).as_posix(),"bytes":p.stat().st_size,"sha256":sha(p)} for p in sorted(written)]})
    check()


def check():
    manifest=read(RESULTS/"manifest.json")
    registry=read(RESULTS/"runs.json")
    assert len(registry)==manifest["completed_runs"]==74
    assert len({r["source"] for r in registry})==74
    for record in manifest["files"]:
        path=(ROOT/record["path"]).resolve()
        if not path.is_relative_to(RESULTS.resolve()) and not path.is_relative_to(REPORTS.resolve()):
            raise ValueError("Snapshot path outside results/reports")
        assert path.is_file() and path.stat().st_size==record["bytes"] and sha(path)==record["sha256"],path
        assert path.suffix not in {".pt",".pth",".npz",".log"},path
        assert path.stat().st_size<10*1024*1024,path
    for row in registry:
        root=ROOT/row["metadata"];metrics=read(root/"metrics.json");config=read(root/"config.json")
        splits=read(root/"splits.json")
        assert metrics["status"]=="completed" and not config.get("synthetic_smoke")
        assert metrics["test_accuracy"]==row["test_accuracy"]
        assert not set(splits["train"])&set(splits["validation"])
    for relative,digest in manifest["source_files"].items():
        assert sha(ROOT/relative)==digest,f"Code changed since export: {relative}"
    size=sum(r["bytes"] for r in manifest["files"])
    print(f"Verified {len(registry)} real runs, {len(manifest['files'])} files, {size/1024/1024:.2f} MiB, source and artifact hashes.")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check",action="store_true",help="Validate the portable snapshot without local outputs/data")
    args=parser.parse_args()
    check() if args.check else export()
