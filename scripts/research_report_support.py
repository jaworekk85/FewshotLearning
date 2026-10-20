"""Build a Polish, evidence-linked PDF report from completed local research runs."""

from __future__ import annotations

import csv

import hashlib

import html

import json

import os

import shutil

import sys

from datetime import datetime, timezone

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

OUT = ROOT / "outputs/research_report"

REPORT_BASENAME = "Raport_badan_CIFAR10_2026-10-20"

os.environ.setdefault("MPLCONFIGDIR", str(OUT / ".mplconfig"))

import fitz

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib import font_manager

import numpy as np

REFERENCES = [
    ("Sohn i in. (2020)", "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence", "https://arxiv.org/abs/2001.07685"),
    ("Chen i in. (2023)", "SoftMatch: Addressing the Quantity-Quality Trade-off in Semi-supervised Learning", "https://arxiv.org/html/2301.10921v2"),
    ("Berthelot i in. (2019)", "MixMatch: A Holistic Approach to Semi-Supervised Learning", "https://arxiv.org/abs/1905.02249"),
    ("Berthelot i in. (2020)", "ReMixMatch: Semi-Supervised Learning with Distribution Alignment and Augmentation Anchoring", "https://arxiv.org/abs/1911.09785"),
    ("Tarvainen i Valpola (2017)", "Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results", "https://arxiv.org/abs/1703.01780"),
    ("Schroff, Kalenichenko i Philbin (2015)", "FaceNet: A Unified Embedding for Face Recognition and Clustering", "https://arxiv.org/abs/1503.03832"),
    ("Khosla i in. (2020)", "Supervised Contrastive Learning", "https://arxiv.org/abs/2004.11362"),
    ("Chen i in. (2020)", "A Simple Framework for Contrastive Learning of Visual Representations", "https://arxiv.org/abs/2002.05709"),
    ("Li, Xiong i Hoi (2021)", "CoMatch: Semi-supervised Learning with Contrastive Graph Regularization", "https://arxiv.org/abs/2011.11183"),
    ("Zheng i in. (2022)", "SimMatch: Semi-supervised Learning with Similarity Matching", "https://arxiv.org/abs/2203.06915"),
    ("Ash i in. (2020)", "Deep Batch Active Learning by Diverse, Uncertain Gradient Lower Bounds (BADGE)", "https://arxiv.org/abs/1906.03671"),
    ("Song, Berthelot i Rostamizadeh (2019)", "Combining MixMatch and Active Learning for Better Accuracy with Fewer Labels", "https://arxiv.org/abs/1912.00594"),
    ("Vincent i in. (2008)", "Extracting and Composing Robust Features with Denoising Autoencoders", "https://icml.cc/Conferences/2008/papers/592.pdf"),
    ("Kingma i Welling (2014)", "Auto-Encoding Variational Bayes", "https://arxiv.org/abs/1312.6114"),
    ("Kingma i in. (2014)", "Semi-Supervised Learning with Deep Generative Models", "https://arxiv.org/abs/1406.5298"),
    ("Wattenberg, Viégas i Johnson (2016)", "How to Use t-SNE Effectively", "https://distill.pub/2016/misread-tsne/"),
    ("Munjal i in. (2022)", "Towards Robust and Reproducible Active Learning Using Neural Networks", "https://arxiv.org/abs/2002.09564"),
    ("Kod autorów FixMatch", "Referencyjna implementacja i ustawienia treningu", "https://github.com/google-research/fixmatch/blob/master/fixmatch.py"),
    ("Kod SoftMatch / USB", "Wyznaczanie wag oraz twardych lub miękkich celów", "https://github.com/microsoft/Semi-supervised-learning/tree/main/semilearn/algorithms/softmatch"),
    ("Rasmus i in. (2015)", "Semi-Supervised Learning with Ladder Networks", "https://arxiv.org/abs/1507.02672"),
    ("Lee i in. (2022)", "Contrastive Regularization for Semi-Supervised Learning", "https://arxiv.org/abs/2201.06247"),
]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def esc(value):
    return html.escape(str(value))

def num(value, digits=2):
    return f"{float(value):.{digits}f}".replace(".", ",")

def pct(value):
    return "—" if value is None else num(100 * float(value))

def integer(value):
    return f"{int(value):,}".replace(",", " ")

def p(text, cls=""):
    return f'<p class="{cls}">{text}</p>'

def note(text):
    return f'<div class="note">{text}</div>'

def table(headers, rows, cls=""):
    body = "".join('<tr class="%s">%s</tr>' % ("alt" if i % 2 else "", "".join(f"<td>{v}</td>" for v in row))
                   for i, row in enumerate(rows))
    return '<table class="%s"><thead><tr>%s</tr></thead><tbody>%s</tbody></table>' % (
        cls, "".join(f"<th>{v}</th>" for v in headers), body)

def image(name, caption, width=500):
    return f'<p class="figure"><img src="assets/{name}" width="{width}"></p><p class="caption">{caption}</p>'

def load_data():
    rows = read(ROOT / "outputs/pilot_comparison.json")
    assert len(rows) == 55 and len({r["run"] for r in rows}) == len(rows)
    records, sources = [], []
    for i, row in enumerate(rows, 1):
        source = ROOT / row["source"]
        config_path = source.parent / "config.json"
        metrics, config = read(source), read(config_path)
        assert row["status"] == metrics["status"] == "completed"
        assert not config.get("synthetic_smoke") and metrics["dataset"] == "cifar10"
        assert abs(metrics["test_accuracy"] - row["test_accuracy"]) < 1e-12
        assert abs(metrics["val_accuracy"] - row["val_accuracy"]) < 1e-12
        row["report_id"] = f"R{i:02d}"
        records.append({"report_id": row["report_id"], "comparison": row, "config": config,
                        "metrics": metrics, "source": source.relative_to(ROOT).as_posix()})
        sources.extend([source, config_path])
    supervised = []
    for i, run in enumerate(("cnn_pilot_N10_seed0", "cnn_pilot_N50_seed0", "cnn_pilot_full_seed0"), 1):
        folder = ROOT / "outputs/supervised" / run
        metrics, config = read(folder / "metrics.json"), read(folder / "config.json")
        with (folder / "history.csv").open(encoding="utf-8", newline="") as stream:
            history = list(csv.DictReader(stream))
        last = history[-1]
        val = float(last.get("val_accuracy", last.get("validation_accuracy", "nan")))
        assert np.isfinite(val)
        supervised.append({"report_id": f"S{i:02d}", "run": run, "config": config, "metrics": metrics,
                           "val_accuracy": val, "source": (folder / "metrics.json").relative_to(ROOT).as_posix()})
        sources.extend([folder / "metrics.json", folder / "config.json", folder / "history.csv"])
    reps = read(ROOT / "outputs/representation_reference/comparison.json")
    # Validate that every old diagnostic used the same seed0 labeled references and validation order.
    canonical = read(ROOT / "outputs/pilot_next/cnn_N50_seed0_ce_2v_pseudo_soft/splits.json")
    extra_reps = {}
    for row in rows:
        if row["seed"] != 0 or row["budget_schedule"] != "fixed":
            continue
        folder = (ROOT / row["source"]).parent
        splits = read(folder / "splits.json")
        audit = ROOT / "outputs/representations" / row["run"] / "metrics.json"
        if audit.exists() and splits["train"] == canonical["train"] and splits["validation"] == canonical["validation"]:
            d = read(audit)
            assert hashlib.sha256((folder / "model.pt").read_bytes()).hexdigest() == d["signature"]["checkpoint_sha256"]
            extra_reps[row["run"]] = d
            sources.append(audit)
    for relative in ("outputs/pilot_comparison.json", "outputs/representation_reference/comparison.json",
                     "outputs/frozen_classifiers/report.json", "outputs/reliability/comparison.json"):
        sources.append(ROOT / relative)
    return {"rows": rows, "records": records, "supervised": supervised, "reps": reps,
            "extra_reps": extra_reps, "sources": sources,
            "frozen": read(ROOT / "outputs/frozen_classifiers/report.json"),
            "reliability": read(ROOT / "outputs/reliability/comparison.json")}

def result_table(data, run_names, titles=None, delta=None):
    lookup = {r["run"]: r for r in data["rows"]}
    values = []
    for i, name in enumerate(run_names):
        r = lookup[name]
        cells = [r["report_id"], esc(titles[i] if titles else r["variant"]), pct(r["val_accuracy"]), pct(r["test_accuracy"])]
        if delta is not None:
            cells.append(num(100*(r["test_accuracy"] - lookup[delta]["test_accuracy"])))
        values.append(cells)
    headers = ["ID", "Wariant", "Wal. [%]", "Test [%]"] + (["Δ test [p.p.]"] if delta else [])
    return table(headers, values)

CSS = """
@font-face {font-family:Report;src:url(DejaVuSans.ttf);}
@font-face {font-family:Report;src:url(DejaVuSans-Bold.ttf);font-weight:bold;}
@font-face {font-family:Report;src:url(DejaVuSans-Oblique.ttf);font-style:italic;}
body {font-family:Report;font-size:9.7pt;line-height:1.32;color:#203345;margin:0;}
h1 {font-size:19pt;line-height:1.12;color:#14394c;margin:0 0 13pt 0;}
h2 {font-size:11.5pt;line-height:1.2;color:#147b84;margin:12pt 0 5pt;}
p {margin:6pt 0;}
.lead {font-size:15pt;line-height:1.3;color:#147b84;margin:12pt 0 17pt;}
.kicker {font-size:8.6pt;letter-spacing:.7pt;color:#5f7584;margin:7pt 0;}
table {border-collapse:collapse;width:100%;font-size:8.2pt;line-height:1.23;margin:8pt 0;}
td,th {padding:5pt 5pt;border-bottom:.5pt solid #cedbe1;vertical-align:top;}
th {background-color:#173d50;color:#ffffff;font-weight:bold;text-align:left;}
tr.alt td {background-color:#f0f5f7;}
.compact {font-size:7.7pt;}
.compact td,.compact th {padding:4pt;}
.registry {font-size:7.9pt;}
.registry td,.registry th {padding:5pt 4pt;}
.note {background-color:#e9f3f3;border-left:3pt solid #16858b;padding:9pt;margin:10pt 0;font-size:9.3pt;}
.small {font-size:8.5pt;line-height:1.3;color:#4e6473;}
.tiny {font-size:7.2pt;color:#526a78;}
.caption {font-size:8pt;line-height:1.25;color:#526977;margin:3pt 0 8pt;}
.figure {margin:6pt 0 2pt;}
.reference {font-size:9pt;line-height:1.28;margin:0 0 12pt;}
a {color:#096e89;}
.toc {font-size:8.8pt;}
.toc td {padding:3pt;}
"""

def write_attachments(data):
    exact = {
        "report_date": "2026-10-20", "main_completed_runs": len(data["records"]),
        "earlier_supervised_runs": len(data["supervised"]),
        "scope": "Completed non-synthetic local CIFAR-10 runs. Not a reproduction of original student results.",
        "main_runs": data["records"], "supervised": data["supervised"],
        "common_reference_representation_metrics": data["reps"],
        "additional_matched_representation_metrics": data["extra_reps"],
        "frozen_classifiers": data["frozen"], "reliability_audit": data["reliability"],
        "references": [{"id": i, "authors": a, "title": t, "url": u} for i,(a,t,u) in enumerate(REFERENCES,1)],
    }
    (OUT / "konfiguracje_i_wyniki.json").write_text(json.dumps(exact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    columns = ["report_id", "run", "variant", "seed", "epochs", "initial_labels", "final_labels", "validation_labels",
               "normalization", "val_accuracy", "test_accuracy", "updates", "labeled_views", "unlabeled_views",
               "decoder_views", "scoring_eval_views", "teacher_training_eval_views", "neighbor_reference_eval_views",
               "neighbor_query_eval_views", "teacher_validation_eval_views", "teacher_test_eval_views", "training_parameters",
               "inference_parameters", "elapsed_minutes", "source"]
    with (OUT / "katalog_wynikow.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for record in data["records"]:
            r, m = record["comparison"], record["metrics"]
            writer.writerow(dict(zip(columns, [record["report_id"], r["run"], r["variant"], r["seed"], r["epochs_completed"],
                r["initial_label_budget"], r["final_label_budget"], r["validation_labels"], r["normalization"],
                r["val_accuracy"], r["test_accuracy"], r["updates_completed"], r["labeled_encoder_views"],
                r["unlabeled_encoder_views"], r["decoder_views"], m.get("scoring_forward_views",0), m.get("ema_training_views",0),
                m.get("neighbor_reference_views",0), m.get("neighbor_query_views",0), m.get("ema_validation_views",0), m.get("ema_test_views",0),
                r["training_parameters"], r["inference_parameters"], r["elapsed_minutes"], record["source"]], strict=True)))
        for s in data["supervised"]:
            m,c = s["metrics"],s["config"]
            writer.writerow({"report_id": s["report_id"], "run": s["run"], "variant": "Supervised, one view",
                "seed": c["seed"], "epochs": c["epochs"], "initial_labels": m["labeled_train_size"],
                "final_labels": m["labeled_train_size"], "validation_labels": m["validation_size"],
                "normalization": c.get("normalization","batchnorm"), "val_accuracy": s["val_accuracy"],
                "test_accuracy": m["test_accuracy"], "updates": m["total_updates"],
                "labeled_views": m["processed_training_images"], "unlabeled_views": 0, "decoder_views": 0,
                "scoring_eval_views": 0, "teacher_training_eval_views": 0, "neighbor_reference_eval_views": 0,
                "neighbor_query_eval_views": 0, "teacher_validation_eval_views": 0, "teacher_test_eval_views": 0, "training_parameters": 111498,
                "inference_parameters": 111498, "elapsed_minutes": m["elapsed_seconds"]/60, "source": s["source"]})
    sources = data["sources"] + list((ROOT / "docs/archive").glob("PILOT*.md")) + list((ROOT / "fewshot_metric").glob("*.py")) + [Path(__file__)]
    if (ROOT / "access_v4.pdf").is_file():
        sources.append(ROOT / "access_v4.pdf")
    for name in ("konfiguracje_i_wyniki.json", "katalog_wynikow.csv"):
        sources.append(OUT / name)
    manifest = {"report_date": "2026-10-20", "built_utc": datetime.now(timezone.utc).isoformat(), "files": []}
    for path in sorted(set(sources)):
        manifest["files"].append({"path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                                  "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT / "manifest_zrodel.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+"\n",encoding="utf-8")

def render_report(sections):
    fonts = Path(font_manager.findfont("DejaVu Sans")).parent
    archive = fitz.Archive()
    archive.add(str(fonts)); archive.add(str(OUT))
    paper = fitz.paper_rect("a4")
    content = fitz.Rect(43, 62, paper.width-43, paper.height-51)
    raw = OUT / "_report_raw.pdf"
    writer = fitz.DocumentWriter(str(raw))
    page_count = 0
    for section in sections:
        section["first_page"] = page_count
        story = fitz.Story(section["html"], user_css=CSS, archive=archive)
        more, page_in_section = 1, 0
        while more:
            page_in_section += 1
            if page_in_section > 5:
                raise RuntimeError(f"Excessive pagination or unplaceable element: {section['title']}")
            dev = writer.begin_page(paper)
            more, filled = story.place(content)
            story.draw(dev); writer.end_page(); page_count += 1
        section["page_count"] = page_in_section
        print(f"{section['title']}: {page_in_section} page(s)", flush=True)
    writer.close()
    del writer
    document = fitz.open(raw)
    toc_page = document.new_page(pno=1, width=paper.width, height=paper.height)
    toc_rows = [[esc(s["title"]), str(s["first_page"]+2)] for s in sections[1:]]
    toc_html = "<h1>Spis treści</h1>" + p("Najpierw wyniki i interpretacja; pełny katalog treningów oraz źródła znajdują się na końcu.","small") + table(["Rozdział", "Strona"], toc_rows, "toc")
    spare, scale = toc_page.insert_htmlbox(content, toc_html, css=CSS, archive=archive)
    if spare < 0 or scale < .9:
        raise RuntimeError(f"Table of contents does not fit readably: scale={scale}")
    toc = [[1,"Streszczenie",1],[1,"Spis treści",2]]
    toc.extend([1,s["title"],s["first_page"]+2] for s in sections[1:])
    document.set_toc(toc)
    footer_font = str(fonts / "DejaVuSans.ttf")
    for i, page in enumerate(document):
        page.insert_font(fontname="ReportFooter", fontfile=footer_font)
        if i:
            page.insert_text((43,36),"CIFAR-10 • badania pilotażowe • 20.10.2026",fontname="ReportFooter",fontsize=7.5,color=(.34,.43,.49))
            page.draw_line((43,46),(paper.width-43,46),color=(.78,.84,.87),width=.5)
        page.insert_text((43,paper.height-22),"FewshotLearning • raport roboczy",fontname="ReportFooter",fontsize=7.5,color=(.34,.43,.49))
        page.insert_textbox(fitz.Rect(465,paper.height-31,paper.width-43,paper.height-13),
                            f"{i+1} / {len(document)}",fontname="ReportFooter",fontsize=8,align=2,color=(.34,.43,.49))
        for number,(_,_,url) in enumerate(REFERENCES,1):
            for term in (f"[{number}]",url):
                for rect in page.search_for(term):
                    page.insert_link({"kind":fitz.LINK_URI,"from":rect,"uri":url})
    toc_page = document[1]
    for s in sections[1:]:
        for rect in toc_page.search_for(s["title"]):
            toc_page.insert_link({"kind":fitz.LINK_GOTO,"from":rect,"page":s["first_page"]+1,"to":fitz.Point(0,0)})
    for name in ("konfiguracje_i_wyniki.json", "katalog_wynikow.csv", "manifest_zrodel.json"):
        document.embfile_add(name,(OUT/name).read_bytes(),filename=name,desc=name)
    document.set_metadata({"title":"Raport z badań pilotażowych — CIFAR-10",
                           "subject":"Uczenie półnadzorowane, reprezentacje, active learning; 58 treningów",
                           "author":"Opracowanie na podstawie repozytorium FewshotLearning",
                           "keywords":"CIFAR-10, SSL, pseudo-labeling, triplet, active learning, raport",
                           "creator":"scripts/build_research_overview.py; PyMuPDF"})
    document.subset_fonts()
    destination = OUT / (REPORT_BASENAME + ".pdf")
    document.save(destination,garbage=4,deflate=True)
    document.close()
    # Portable editable source using local assets and system font fallback in a browser.
    browser_css = CSS + "@media print{section{break-before:page;} @page{size:A4;margin:16mm;}} body{max-width:850px;margin:30px auto;font-family:'DejaVu Sans','Segoe UI',sans-serif;} img{max-width:100%;height:auto;}"
    html_source = '<!doctype html><html lang="pl"><meta charset="utf-8"><title>Raport badań CIFAR-10</title><style>'+browser_css+'</style><body>'
    html_source += "".join('<section>'+s["html"]+'</section>' for s in sections) + "</body></html>"
    (OUT / (REPORT_BASENAME + ".html")).write_text(html_source,encoding="utf-8")
    (OUT / "layout.json").write_text(json.dumps(sections,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return destination

def verify_report(path, sections):
    document = fitz.open(path)
    text = "\n".join(p.get_text() for p in document)
    for term in ("Miękkie", "65,34", "79,75", "60,96", "R55", "S03", "SoftMatch", "500", "48 500"):
        if term not in text:
            raise AssertionError(f"Missing report content: {term}")
    assert document.embfile_count() == 3
    exact = json.loads(document.embfile_get("konfiguracje_i_wyniki.json"))
    assert len(exact["main_runs"]) == 55 and len(exact["supervised"]) == 3
    # Identify clipping, empty pages and unexpected font substitution characters.
    outside, empty = [], []
    for i,page in enumerate(document):
        if not page.get_text().strip(): empty.append(i+1)
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0: continue
            for line in block["lines"]:
                for span in line["spans"]:
                    x0,y0,x1,y1=span["bbox"]
                    if x0 < 30 or x1 > page.rect.width-27 or y0 < 15 or y1 > page.rect.height-10:
                        outside.append({"page":i+1,"text":span["text"],"bbox":span["bbox"]})
    validation = {"pages":len(document),"bytes":path.stat().st_size,"embedded_attachments":document.embfile_names(),
                  "bookmarks":len(document.get_toc()),"empty_pages":empty,"out_of_bounds_text":outside,
                  "unicode_replacement_characters":text.count("\ufffd"),
                  "main_run_count":len(exact["main_runs"]),"supervised_run_count":len(exact["supervised"])}
    (OUT / "validation.json").write_text(json.dumps(validation,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if empty or outside or validation["unicode_replacement_characters"]:
        raise AssertionError(f"PDF layout validation failed: {json.dumps(validation,ensure_ascii=True)}")
    previews=OUT/"preview";previews.mkdir(exist_ok=True)
    for i,page in enumerate(document):
        page.get_pixmap(matrix=fitz.Matrix(.75,.75)).save(previews/f"page_{i+1:02d}.png")
    from PIL import Image,ImageOps,ImageDraw
    thumbnails=[]
    for i in range(len(document)):
        thumb=Image.open(previews/f"page_{i+1:02d}.png").convert("RGB")
        thumb.thumbnail((210,290))
        tile=Image.new("RGB",(230,320),"#e4eaef");tile.paste(thumb,((230-thumb.width)//2,10))
        ImageDraw.Draw(tile).text((10,302),f"{i+1}",fill="#123344")
        thumbnails.append(tile)
    columns=5
    contact=Image.new("RGB",(columns*230,((len(thumbnails)+columns-1)//columns)*320),"white")
    for i,tile in enumerate(thumbnails):contact.paste(tile,((i%columns)*230,(i//columns)*320))
    contact.save(previews/"contact_sheet.png")
    (OUT / "extracted_text.txt").write_text(text,encoding="utf-8")
    print(json.dumps(validation,ensure_ascii=True,indent=2),flush=True)
    document.close()
