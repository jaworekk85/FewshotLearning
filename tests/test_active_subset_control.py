import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch

from fewshot_metric.fixed_splits import load_fixed_split, manifest_from_active_run
from fewshot_metric.pilot import PilotTraining, select_method
from fewshot_metric.supervised import build_datasets, run_supervised
from scripts.run_next_pilot import build_jobs


class FakeCIFAR:
    """In-memory dataset fixture; no download and no real result artifacts."""
    def __init__(self, root, train=True, download=False, transform=None):
        self.data = np.random.default_rng(724).integers(0, 256, (120 if train else 20, 32, 32, 3), dtype=np.uint8)
        self.targets = (np.arange(len(self.data)) % 10).tolist()
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        image = Image.fromarray(self.data[index])
        return (self.transform(image) if self.transform else image), self.targets[index]


class ActiveSubsetControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def fixture(self, root):
        manifest = {"dataset": "cifar10", "population_size": 120,
                    "train": [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], "validation": list(range(30, 50)),
                    "known_labels": [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]}
        p = root/"manifest.json"
        p.write_text(json.dumps(manifest))
        cfg = select_method({**self.base, "budget_per_class": 1, "validation_per_class": 2,
                             "epochs": 2, "steps_per_epoch": 2, "batch_size": 8,
                             "unlabeled_batch_size": 8, "embedding_dim": 16,
                             "fixed_split_manifest": str(p),
                             "fixed_split_manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}, "ce_2v_pseudo_soft")
        return cfg, manifest

    def test_snapshot_preserves_indices_and_exposes_only_previously_known_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, manifest = self.fixture(Path(tmp))
            labels = np.arange(120) % 10
            splits, visible = load_fixed_split(cfg, labels)
            self.assertEqual(splits, {k:manifest[k] for k in ("train", "validation")})
            self.assertEqual(int((visible >= 0).sum()), 10)
            self.assertTrue((visible[manifest["validation"]] == -1).all())
            self.assertEqual(np.bincount(visible[visible >= 0], minlength=10).tolist(), [2,1,1,1,1,1,1,1,1,0])

    def test_invalid_indices_overlap_annotation_budget_labels_and_mutation_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg, original = self.fixture(Path(tmp))
            p = Path(cfg["fixed_split_manifest"])
            changes = [dict(train=[0]*10), dict(train=[999]+original["train"][1:]),
                       dict(validation=[0]+original["validation"][1:]), dict(train=original["train"][:-1]),
                       dict(known_labels=[9]*10), dict(population_size=119)]
            for changed in changes:
                with self.subTest(changed=changed):
                    p.write_text(json.dumps({**original, **changed}))
                    new_cfg = {**cfg, "fixed_split_manifest_sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
                    with self.assertRaises(ValueError):
                        load_fixed_split(new_cfg, np.arange(120)%10)
            p.write_text(json.dumps(original)+" ")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                load_fixed_split(cfg, np.arange(120)%10)

    def test_dataset_builder_does_not_rebalance_or_resample_and_excludes_L_and_val_from_U(self):
        with tempfile.TemporaryDirectory() as tmp, patch("fewshot_metric.supervised.datasets.CIFAR10", FakeCIFAR):
            root = Path(tmp)
            cfg, manifest = self.fixture(root)
            train, val, splits = build_datasets(cfg, root, False)
            self.assertEqual(train.indices, manifest["train"])
            self.assertEqual(train.labels, manifest["known_labels"])
            self.assertEqual(len(val), 20)
            pilot = PilotTraining(cfg, train, splits)
            self.assertIsNone(pilot.unlabeled.labels)
            self.assertEqual(set(pilot.unlabeled.indices), set(range(120))-set(manifest["train"])-set(manifest["validation"]))

    def test_frozen_ssl_resume_preserves_exact_weights_splits_and_exposures(self):
        with tempfile.TemporaryDirectory() as tmp, patch("fewshot_metric.supervised.datasets.CIFAR10", FakeCIFAR), contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            cfg, manifest = self.fixture(root)
            full = run_supervised(cfg, root/"full", root, "cpu")
            run_supervised(cfg, root/"resume", root, "cpu", stop_after_epoch=1)
            self.assertFalse((root/"resume/metrics.json").exists())
            resumed = run_supervised(cfg, root/"resume", root, "cpu", resume=True)
            for key, value in torch.load(root/"full/model.pt", weights_only=True).items():
                torch.testing.assert_close(value, torch.load(root/"resume/model.pt", weights_only=True)[key], rtol=0, atol=0)
            self.assertEqual(full["processed_labeled_views"], resumed["processed_labeled_views"])
            self.assertEqual(full["processed_unlabeled_views"], resumed["processed_unlabeled_views"])
            self.assertEqual(full["initial_model_sha256"], resumed["initial_model_sha256"])
            self.assertEqual(json.loads((root/"full/splits.json").read_text()), {k:manifest[k] for k in ("train", "validation")})
            self.assertEqual(full["final_class_counts"], [2,1,1,1,1,1,1,1,1,0])

    def test_initialization_mismatch_and_silent_smoke_replacement_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch("fewshot_metric.supervised.datasets.CIFAR10", FakeCIFAR):
            root = Path(tmp)
            cfg, _ = self.fixture(root)
            with self.assertRaisesRegex(ValueError, "initialization differs"):
                run_supervised({**cfg, "fixed_split_source_initial_model_sha256": "different"}, root/"bad", root, "cpu")
            self.assertFalse((root/"bad").exists())
            with self.assertRaisesRegex(ValueError, "smoke splits"):
                run_supervised(cfg, root/"smoke", root, "cpu", smoke=True)

    def test_builder_refuses_labels_not_obtained_by_source_active_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = np.arange(2000)%10
            cfg = {"dataset":"cifar10", "active_strategy":"badge", "method":"ce_2v_pseudo_soft", "epochs":20,
                   "seed":0, "initial_train_indices":list(range(300)), "initial_model_sha256":"same"}
            events = [{"indices":list(range(300,500)), "labels":labels[300:500].tolist()}]
            files = {"config.json":cfg, "metrics.json":{"status":"completed","true_training_label_budget":500},
                     "splits.json":{"train":list(reversed(range(500))),"validation":list(range(500,1500))},
                     "acquisitions.json":events}
            for name, value in files.items():
                (root/name).write_text(json.dumps(value))
            m = manifest_from_active_run(root, labels, "badge")
            self.assertEqual(m["train"],list(range(500)))
            self.assertEqual(len(m["known_labels"]),500)
            events[0]["indices"][-1]=1600
            (root/"acquisitions.json").write_text(json.dumps(events))
            with self.assertRaisesRegex(ValueError, "actually acquired"):
                manifest_from_active_run(root, labels, "badge")

    def test_real_plan_uses_exact_final_sets_and_same_training_recipe(self):
        plan = json.loads(Path("configs/cifar10_pilot_active_subset_control.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs),2)
        for job in jobs:
            cfg = job["config"]
            manifest = json.loads(Path(cfg["fixed_split_manifest"]).read_text())
            source = Path(cfg["fixed_subset_source_run"])
            if not (source/"splits.json").exists():
                source = Path("results/runs")/source.relative_to("outputs")
            original = json.loads((source/"splits.json").read_text())
            self.assertEqual(set(manifest["train"]),set(original["train"]))
            self.assertEqual(manifest["validation"],original["validation"])
            self.assertEqual(len(manifest["train"]),500)
            self.assertEqual(cfg["epochs"],20)
            self.assertEqual(cfg["steps_per_epoch"],383)
            self.assertEqual(cfg["method"],"ce_2v_pseudo_soft")
            self.assertNotIn("active_strategy",cfg)
        ignored = {"fixed_split_manifest","fixed_split_manifest_sha256","fixed_subset_strategy","fixed_subset_source_run"}
        self.assertEqual({k:v for k,v in jobs[0]["config"].items() if k not in ignored},
                         {k:v for k,v in jobs[1]["config"].items() if k not in ignored})


if __name__ == "__main__":
    unittest.main()
