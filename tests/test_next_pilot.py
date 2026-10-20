import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fewshot_metric.active import LabelOracle, run_active, select_queries, validate_active
from fewshot_metric.pilot import EpochPseudoState, labeled_triplet_loss, pseudo_triplet_loss, select_method
from fewshot_metric.representations import representation_metrics
from fewshot_metric.supervised import run_supervised


class NextPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def test_semihard_mining_selects_active_negative_when_random_negative_is_easy(self):
        # Anchor class 0, one easy and one semi-hard class-1 negative.
        def at(angle):
            return [np.cos(angle), np.sin(angle)]
        l1 = torch.tensor([at(0.2), at(2.0), at(0.3)], dtype=torch.float32, requires_grad=True)
        l2 = l1.detach().clone().requires_grad_()
        u1 = torch.tensor([at(0)], dtype=torch.float32, requires_grad=True)
        u2 = u1.detach().clone().requires_grad_()
        loss, counts = pseudo_triplet_loss(l1, l2, u1, u2, torch.tensor([0, 1, 1]),
                                          torch.tensor([0]), torch.tensor([True]),
                                          torch.Generator().manual_seed(0), 0.2, mining="semihard")
        self.assertEqual(counts["pseudo_triplet_semihard"], 1)
        self.assertEqual(counts["pseudo_triplet_active"], 1)
        self.assertGreater(float(loss.detach()), 0)
        loss.backward()
        self.assertGreater(float(l1.grad[2].abs().sum()), 0)
        self.assertEqual(float(l1.grad[1].abs().sum()), 0)

    def test_soft_weights_update_only_at_epoch_boundary_and_targets_are_detached(self):
        state = EpochPseudoState({"pseudo_target": "soft", "pseudo_temperature": 0.5})
        weak = torch.tensor([[3.0, 0.0], [1.0, 0.0]], requires_grad=True)
        strong = torch.zeros(2, 2, requires_grad=True)
        loss, _, _, _, weights = state.loss(weak, strong)
        self.assertEqual(weights.tolist(), [1.0, 0.0])
        self.assertFalse(state.ready)
        loss.backward()
        self.assertIsNone(weak.grad)
        self.assertEqual(float(strong.grad[1].abs().sum()), 0)
        state.finish_epoch()
        self.assertTrue(state.ready)
        state.reset()
        _, _, _, _, weights = state.loss(weak, strong)
        self.assertGreater(float(weights[1]), 0)
        self.assertLess(float(weights[1]), 1)

    def test_labeled_semihard_triplets_track_violations_and_safe_single_class_batch(self):
        for labels in (torch.tensor([0, 0, 1, 1]), torch.zeros(4, dtype=torch.long)):
            first = torch.nn.functional.normalize(torch.randn(4, 8), dim=1).requires_grad_()
            second = torch.nn.functional.normalize(torch.randn(4, 8), dim=1).requires_grad_()
            stats = {}
            loss = labeled_triplet_loss(first, second, labels, torch.Generator().manual_seed(0), 0.2,
                                        mining="semihard", stats=stats)
            self.assertTrue(torch.isfinite(loss))
            self.assertLessEqual(stats["triplet_l_active"], stats["triplet_l_valid"])
            if labels.unique().numel() == 1:
                self.assertEqual(stats["triplet_l_valid"], 0)
            loss.backward()

    def test_next_queue_has_all_requested_families_and_valid_fixed_label_budget(self):
        from scripts.run_next_pilot import build_jobs
        plan = json.loads(Path("configs/cifar10_pilot_next.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs), 7)
        self.assertEqual(len({job["name"] for job in jobs}), len(jobs))
        for job in jobs:
            cfg = job["config"]
            if job["kind"] == "active":
                validate_active(cfg)
                self.assertEqual(cfg["active_final_budget"], 500)
                self.assertEqual(cfg["budget_per_class"], 30)
            else:
                self.assertEqual(cfg["budget_per_class"], 50)

    def test_soft_resume_is_exact(self):
        cfg = select_method({**self.base, "epochs": 2, "steps_per_epoch": 2,
                             "batch_size": 8, "unlabeled_batch_size": 8, "embedding_dim": 16}, "ce_2v_pseudo_soft")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full = run_supervised(cfg, root/"full", root, "cpu", smoke=True)
            run_supervised(cfg, root/"resume", root, "cpu", smoke=True, stop_after_epoch=1)
            resumed = run_supervised(cfg, root/"resume", root, "cpu", smoke=True, resume=True)
            for k, v in torch.load(root/"full/model.pt", weights_only=True).items():
                torch.testing.assert_close(v, torch.load(root/"resume/model.pt", weights_only=True)[k], rtol=0, atol=0)
            self.assertEqual(full["pseudo_weight_sum"], resumed["pseudo_weight_sum"])
            self.assertGreater(full["pseudo_weight_sum"], 0)

    def test_new_fixed_variants_run_and_keep_initialization_paired(self):
        methods = ("ce_2v_triplet_l_semihard", "ce_2v_pseudo_semihard_triplet", "ce_2v_pseudo_soft_weight")
        fingerprints = []
        for method in methods:
            cfg = select_method({**self.base, "epochs": 1, "steps_per_epoch": 2,
                                 "batch_size": 8, "unlabeled_batch_size": 8,
                                 "pseudo_label_threshold": 0.0, "embedding_dim": 16}, method)
            with self.subTest(method=method), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                result = run_supervised(cfg, Path(tmp)/"run", Path(tmp), "cpu", smoke=True)
                self.assertEqual(result["processed_labeled_views"], 32)
                fingerprints.append(result["initial_model_sha256"])
                if "semihard_triplet" in method:
                    self.assertGreater(result["pseudo_triplet_valid"], 0)
                    self.assertLessEqual(result["pseudo_triplet_active"], result["pseudo_triplet_valid"])
        self.assertEqual(len(set(fingerprints)), 1)

    def test_oracle_enforces_budget_disjointness_and_hides_unqueried_labels(self):
        oracle = LabelOracle([0, 1, 2, 3, 4], [0, 1, 2, 3], 2)
        oracle.reveal([0])
        self.assertEqual(oracle.visible_labels().tolist(), [0, -1, -1, -1, -1])
        for query in ([0], [4], [1, 2], [1, 1]):
            with self.assertRaises(ValueError):
                oracle.reveal(query)
        oracle.reveal([2])
        self.assertEqual(oracle.visible_labels().tolist(), [0, -1, 2, -1, -1])

    def test_uncertainty_and_diversity_query_without_duplicate_images(self):
        probs = torch.tensor([[0.5, 0.5], [0.51, 0.49], [0.6, 0.4], [0.99, 0.01]])
        features = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.]])
        rng = torch.Generator().manual_seed(0)
        self.assertEqual(select_queries(probs, features, features[:0], 2, "entropy", rng).tolist(), [0, 1])
        diverse = select_queries(probs, features, features[:1], 2, "entropy_diverse", rng)
        self.assertEqual(len(set(diverse.tolist())), 2)
        self.assertIn(3, diverse.tolist())

    def test_active_resume_after_acquisition_preserves_queries_and_weights(self):
        for strategy in ("random", "entropy", "entropy_diverse"):
            cfg = select_method({**self.base, "epochs": 3, "steps_per_epoch": 1,
                                 "batch_size": 8, "unlabeled_batch_size": 8, "embedding_dim": 16,
                                 "budget_per_class": 1, "active_final_budget": 14,
                                 "active_rounds": [1, 2], "active_query_size": 2,
                                 "active_candidate_limit": 16, "active_strategy": strategy}, "ce_2v_pseudo")
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                full = run_active(cfg, root/"full", root, smoke=True)
                run_active(cfg, root/"resume", root, smoke=True, stop_after_epoch=1)
                self.assertFalse((root/"resume/metrics.json").exists())
                resumed = run_active(cfg, root/"resume", root, smoke=True, resume=True)
                self.assertEqual(full["true_training_label_budget"], 14)
                self.assertEqual(sum(full["final_class_counts"]), 14)
                self.assertEqual(full["processed_labeled_views"], resumed["processed_labeled_views"])
                self.assertEqual((root/"full/acquisitions.json").read_text(), (root/"resume/acquisitions.json").read_text())
                for k, v in torch.load(root/"full/model.pt", weights_only=True).items():
                    torch.testing.assert_close(v, torch.load(root/"resume/model.pt", weights_only=True)[k], rtol=0, atol=0)
                splits = json.loads((root/"full/splits.json").read_text())
                self.assertFalse(set(splits["train"]) & set(splits["validation"]))
                checkpoint = torch.load(root/"full/latest.pt", weights_only=False)
                self.assertEqual([r["train_label_budget"] for r in checkpoint["history"]], [10, 12, 14])

    def test_representation_metrics_detect_separable_classes(self):
        labels = torch.tensor([0, 0, 1, 1])
        features = torch.tensor([[1., 0.], [0.99, 0.01], [0., 1.], [0.01, 0.99]])
        metrics = representation_metrics(features, labels, features, labels, neighbors=(1,))
        self.assertEqual(metrics["knn_1_val_accuracy"], 1)
        self.assertLess(metrics["within_over_total_scatter"], 0.01)
        self.assertGreater(metrics["normalized_feature_variance"], 0)


if __name__ == "__main__":
    unittest.main()
