import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fewshot_metric.active import run_active
from fewshot_metric.comparison import collect_comparison
from fewshot_metric.pilot import PilotTraining, select_method, synthetic_datasets, unlabeled_triplet_loss
from fewshot_metric.supervised import run_supervised
from scripts.run_next_pilot import build_jobs
from scripts.summarize_validation_study import paired_effect, statistics_summary


class ValidationStudyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def test_batch_all_triplets_equal_explicit_instance_pairs_and_have_gradients(self):
        first = torch.tensor([[1., 0.], [0., 1.], [0.8, 0.2]], requires_grad=True)
        second = torch.tensor([[0.7, 0.3], [0.2, 0.8], [0.9, 0.1]], requires_grad=True)
        expected = torch.stack([((first[i]-second[i]).norm() - (first[i]-second[j]).norm() + 0.2).relu()
                                for i in range(3) for j in range(3) if i != j]).mean()
        stats = {}
        rng = torch.Generator().manual_seed(0)
        state = rng.get_state().clone()
        actual = unlabeled_triplet_loss(first, second, rng, 0.2, mining="batch_all", stats=stats)
        torch.testing.assert_close(actual, expected)
        self.assertEqual(stats["triplet_u_pairs"], 6)
        torch.testing.assert_close(state, rng.get_state(), rtol=0, atol=0)
        actual.backward()
        self.assertTrue(torch.isfinite(first.grad).all())
        self.assertGreater(float(second.grad.abs().sum()), 0)

    def test_single_image_has_no_negatives_and_finite_zero_gradient(self):
        a, b = torch.randn(1, 3, requires_grad=True), torch.randn(1, 3, requires_grad=True)
        stats = {}
        loss = unlabeled_triplet_loss(a, b, torch.Generator(), 0.2, "batch_all", stats)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(stats["triplet_u_pairs"], 0)
        loss.backward()
        self.assertEqual(float(a.grad.abs().sum() + b.grad.abs().sum()), 0)

    def test_mixed_triplets_use_all_three_objectives_and_keep_total_weight_point_one(self):
        cfg = select_method({**self.base, "unlabeled_batch_size": 8, "pseudo_label_threshold": 0.0},
                            "ce_2v_pseudo_soft_mixed_triplet")
        self.assertAlmostEqual(sum(cfg[k] for k in ["triplet_l_weight", "triplet_u_weight", "pseudo_triplet_weight"]), 0.1)
        train, _, _ = synthetic_datasets(cfg)
        pilot = PilotTraining(cfg, train, {"train": list(range(40)), "validation": list(range(40, 60))})
        self.assertIsNone(pilot.unlabeled.labels)
        pilot.start_epoch(0)
        images, labels = next(iter(DataLoader(train, batch_size=10)))
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3*32*32, 8)
                self.head = torch.nn.Linear(8, 10)
                self.calls = 0
            def forward(self, images, normalize=False):
                self.calls += 1
                z = self.encoder(images.flatten(1))
                return z, self.head(z)
        model = Model()
        loss, _, _, m = pilot.step(model, images, labels, torch.device("cpu"))
        self.assertEqual(model.calls, 1)
        self.assertGreater(m["triplet_l_valid"], 0)
        self.assertGreater(m["pseudo_triplet_valid"], 0)
        self.assertEqual(m["triplet_u_pairs"], 56)
        expected = m["ce_loss"] + m["pseudo_label_loss"] + (m["triplet_l_loss"] + m["triplet_u_loss"] + m["pseudo_triplet_loss"])/30
        self.assertAlmostEqual(float(loss.detach()), expected, places=5)
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0)

    def test_superseded_partial_run_is_retained_but_excluded_from_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)/"cnn_superseded"
            d.mkdir()
            (d/"config.json").write_text(json.dumps(self.base))
            (d/"run_status.json").write_text(json.dumps({"status": "superseded"}))
            (d/"latest.pt").write_bytes(b"preserved")
            self.assertEqual(collect_comparison([tmp]), [])
            self.assertEqual((d/"latest.pt").read_bytes(), b"preserved")

    def test_combined_soft_and_instance_loss_reuses_views_without_candidate_labels(self):
        cfg = select_method({**self.base, "unlabeled_batch_size": 8}, "ce_2v_pseudo_soft_instance_triplet")
        train, _, _ = synthetic_datasets(cfg)
        pilot = PilotTraining(cfg, train, {"train": list(range(40)), "validation": list(range(40, 60))})
        self.assertIsNone(pilot.unlabeled.labels)
        pilot.start_epoch(0)
        images, labels = next(iter(DataLoader(train, batch_size=10)))
        class CountedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3*32*32, 8)
                self.head = torch.nn.Linear(8, 10)
                self.calls = []
            def forward(self, images, normalize=False):
                self.calls.append(len(images))
                z = self.encoder(images.flatten(1))
                return z, self.head(z)
        model = CountedModel()
        loss, _, _, metrics = pilot.step(model, images, labels, torch.device("cpu"))
        self.assertEqual(model.calls, [36])
        self.assertEqual(metrics["processed_labeled_views"], 20)
        self.assertEqual(metrics["processed_unlabeled_views"], 16)
        self.assertEqual(metrics["triplet_u_pairs"], 56)
        self.assertAlmostEqual(float(loss.detach()), metrics["ce_loss"] + metrics["pseudo_label_loss"] + 0.1*metrics["triplet_u_loss"], places=5)
        loss.backward()
        self.assertGreater(float(model.encoder.weight.grad.abs().sum()), 0)

    def test_zero_triplet_weight_exactly_reproduces_soft_baseline(self):
        base = {**self.base, "epochs": 2, "steps_per_epoch": 1, "batch_size": 8,
                "unlabeled_batch_size": 8, "embedding_dim": 16}
        cfg = select_method(base, "ce_2v_pseudo_soft_instance_triplet")
        cfg["triplet_u_weight"] = 0.0
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            a = run_supervised(select_method(base, "ce_2v_pseudo_soft"), root/"base", root, "cpu", smoke=True)
            b = run_supervised(cfg, root/"zero", root, "cpu", smoke=True)
            self.assertEqual(a["initial_model_sha256"], b["initial_model_sha256"])
            for key, value in torch.load(root/"base/model.pt", weights_only=True).items():
                torch.testing.assert_close(value, torch.load(root/"zero/model.pt", weights_only=True)[key], rtol=0, atol=0)

    def test_soft_triplets_with_active_learning_resume_exactly(self):
        for strategy in ("random", "badge"):
            cfg = select_method({**self.base, "epochs": 3, "steps_per_epoch": 1,
                "batch_size": 8, "unlabeled_batch_size": 8, "embedding_dim": 16,
                "budget_per_class": 1, "active_final_budget": 14, "active_query_size": 2,
                "active_rounds": [1, 2], "active_candidate_limit": 16,
                "active_strategy": strategy, "active_selection_version": 2, "active_scoring_views": 1},
                "ce_2v_pseudo_soft_mixed_triplet")
            cfg["pseudo_label_threshold"] = 0.0
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                full = run_active(cfg, root/"full", root, smoke=True)
                run_active(cfg, root/"resume", root, smoke=True, stop_after_epoch=1)
                resumed = run_active(cfg, root/"resume", root, smoke=True, resume=True)
                self.assertEqual(full["triplet_u_pairs"], 3*8*7)
                self.assertEqual(full["triplet_u_pairs"], resumed["triplet_u_pairs"])
                self.assertEqual(full["pseudo_weight_sum"], resumed["pseudo_weight_sum"])
                self.assertEqual((root/"full/acquisitions.json").read_text(), (root/"resume/acquisitions.json").read_text())
                for key, value in torch.load(root/"full/model.pt", weights_only=True).items():
                    torch.testing.assert_close(value, torch.load(root/"resume/model.pt", weights_only=True)[key], rtol=0, atol=0)

    def test_fixed_instance_triplets_count_every_negative_and_resume(self):
        cfg = select_method({**self.base, "epochs": 2, "steps_per_epoch": 1, "batch_size": 8,
                             "unlabeled_batch_size": 8, "embedding_dim": 16}, "ce_2v_pseudo_soft_mixed_triplet")
        cfg["pseudo_label_threshold"] = 0.0
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full = run_supervised(cfg, root/"full", root, "cpu", smoke=True)
            run_supervised(cfg, root/"resume", root, "cpu", smoke=True, stop_after_epoch=1)
            resumed = run_supervised(cfg, root/"resume", root, "cpu", smoke=True, resume=True)
            self.assertEqual(full["triplet_u_pairs"], 2*8*7)
            self.assertEqual(full["triplet_u_pairs"], resumed["triplet_u_pairs"])
            for key, value in torch.load(root/"full/model.pt", weights_only=True).items():
                torch.testing.assert_close(value, torch.load(root/"resume/model.pt", weights_only=True)[key], rtol=0, atol=0)

    def test_study_has_three_seeds_two_durations_and_complete_active_factorial(self):
        plan = json.loads(Path("configs/cifar10_pilot_validation_study.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs), 16)
        self.assertEqual(sum("reuse_run" not in j for j in jobs), 13)
        for job in jobs:
            if "reuse_run" in job:
                source = Path(job["reuse_run"])
                if not (source/"metrics.json").exists():
                    source = Path("results/runs")/source.relative_to("outputs")
                metrics = json.loads((source/"metrics.json").read_text())
                self.assertTrue(all(metrics.get(k) == v for k, v in job["config"].items()))
        methods = ("ce_2v", "ce_2v_pseudo", "ce_2v_pseudo_soft")
        for method in methods:
            seeds = {j["config"]["seed"] for j in jobs if j["kind"] == "fixed" and j["config"]["method"] == method and j["config"]["epochs"] == 20}
            self.assertEqual(seeds, {0, 1, 2})
        active = [j["config"] for j in jobs if j["kind"] == "active"]
        self.assertEqual({(c["active_strategy"], c["triplet_u_weight"]) for c in active},
                         {("random", 0.0), ("random", 1/30), ("badge", 0.0), ("badge", 1/30)})
        self.assertEqual({j["config"]["epochs"] for j in jobs}, {20, 40, 80})

    def test_paired_statistics_use_seed_differences_and_reject_mismatched_splits(self):
        first = {s: {"pairing_signature": str(s), "val_accuracy": v, "test_accuracy": v} for s, v in enumerate([0.4, 0.5, 0.6])}
        second = {s: {**r, "val_accuracy": r["val_accuracy"]+0.02, "test_accuracy": r["test_accuracy"]+0.02} for s, r in first.items()}
        result = paired_effect(first, second)
        self.assertEqual(result["test"]["n"], 3)
        self.assertAlmostEqual(result["test"]["mean"], 0.02)
        self.assertAlmostEqual(result["test"]["ci95"][0], 0.02)
        self.assertIsNone(statistics_summary([0.5])["ci95"])
        second[0]["pairing_signature"] = "different"
        with self.assertRaises(ValueError):
            paired_effect(first, second)


if __name__ == "__main__":
    unittest.main()
