import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fewshot_metric.pilot import (METHODS, PilotTraining, labeled_triplet_loss,
                                  pseudo_label_loss, pseudo_triplet_indices,
                                  pseudo_triplet_loss, select_method, synthetic_datasets)
from fewshot_metric.supervised import run_supervised, validate_config


class PilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def test_pseudo_mask_uses_full_batch_denominator_and_detached_targets(self):
        weak = torch.tensor([[10.0, 0.0], [0.0, 0.0]], requires_grad=True)
        strong = torch.zeros(2, 2, requires_grad=True)
        loss, mask, _, _ = pseudo_label_loss(weak, strong, 0.95)
        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(float(loss.detach()), math.log(2) / 2, places=6)
        loss.backward()
        self.assertIsNone(weak.grad)
        self.assertGreater(float(strong.grad[0].abs().sum()), 0)
        self.assertEqual(float(strong.grad[1].abs().sum()), 0)

    def test_no_accepted_pseudo_labels_has_zero_finite_gradient(self):
        weak = torch.zeros(3, 10, requires_grad=True)
        strong = torch.randn(3, 10, requires_grad=True)
        loss, mask, _, _ = pseudo_label_loss(weak, strong, 0.95)
        self.assertFalse(bool(mask.any()))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(strong.grad.abs().sum()), 0)

    def test_unlabeled_access_does_not_change_labeled_views_or_global_rng(self):
        train_a, _, _ = synthetic_datasets(self.config)
        train_b, _, _ = synthetic_datasets(self.config)
        splits = {"train": list(range(40)), "validation": list(range(40, 60))}
        pilot = PilotTraining(self.config, train_a, splits)
        self.assertIsNone(pilot.unlabeled.labels)
        train_a.set_epoch(3)
        train_b.set_epoch(3)
        for item in (0, 2, 0, 5):
            expected, label = train_a[item]
            before = torch.get_rng_state().clone()
            pilot.unlabeled[item]
            torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
            actual, other_label = train_b[item]
            self.assertEqual(label, other_label)
            for x, y in zip(expected, actual):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
        self.assertFalse(set(pilot.unlabeled.indices) & set(splits["train"] + splits["validation"]))

    def test_all_losses_reuse_one_encoder_forward_and_count_views(self):
        config = select_method({**self.config, "unlabeled_batch_size": 8,
                                "pseudo_label_threshold": 0.0}, "ce_2v_pseudo_guided_triplet")
        config.update(triplet_l_weight=1.0, triplet_u_weight=1.0)
        train, _, _ = synthetic_datasets(config)
        pilot = PilotTraining(config, train, {"train": list(range(40)), "validation": list(range(40, 60))})
        pilot.start_epoch(0)
        images, labels = next(iter(DataLoader(train, batch_size=10)))

        class CountedModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = torch.nn.Linear(3 * 32 * 32, 8)
                self.head = torch.nn.Linear(8, 10)
                self.calls = []

            def forward(self, images, normalize=False):
                self.calls.append(images.shape[0])
                features = self.encoder(images.flatten(1))
                return features, self.head(features)

        model = CountedModel()
        loss, logits, targets, metrics = pilot.step(model, images, labels, torch.device("cpu"))
        loss.backward()
        self.assertEqual(model.calls, [36])
        self.assertEqual(logits.shape[0], targets.numel())
        self.assertEqual(metrics["processed_labeled_views"], 20)
        self.assertEqual(metrics["processed_unlabeled_views"], 16)
        self.assertEqual(metrics["pseudo_triplet_attempted"], 8)
        self.assertGreater(metrics["pseudo_triplet_valid"], 0)
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_labeled_triplets_handle_batch_without_class_negatives(self):
        first = torch.randn(4, 8, requires_grad=True)
        loss = labeled_triplet_loss(first, first + 0.1, torch.zeros(4, dtype=torch.long),
                                    torch.Generator().manual_seed(1), margin=0.2)
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertEqual(float(first.grad.abs().sum()), 0)

    def test_pseudo_triplets_use_distinct_confident_images_and_class_labels(self):
        labels = torch.tensor([0, 1])
        predictions = torch.tensor([0, 0, 1, 1, 2])
        accepted = torch.tensor([True, True, True, True, False])
        combined = torch.cat([labels, predictions])
        for pool in ("labeled_and_pseudo", "pseudo_only"):
            anchors, positives, negatives = pseudo_triplet_indices(
                labels, predictions, accepted, torch.Generator().manual_seed(4), pool)
            self.assertEqual(anchors.tolist(), [2, 3, 4, 5])
            self.assertTrue(bool((anchors != positives).all()))
            self.assertTrue(bool((combined[anchors] == combined[positives]).all()))
            self.assertTrue(bool((combined[anchors] != combined[negatives]).all()))
            self.assertNotIn(6, positives.tolist() + negatives.tolist())
            if pool == "pseudo_only":
                self.assertTrue(bool((positives >= 2).all()))
                self.assertTrue(bool((negatives >= 2).all()))

    def test_known_labels_supply_missing_pseudo_triplet_positives(self):
        labels = torch.tensor([0, 1])
        predictions = torch.tensor([0, 1])
        accepted = torch.ones(2, dtype=torch.bool)
        anchors, positives, _ = pseudo_triplet_indices(
            labels, predictions, accepted, torch.Generator().manual_seed(2))
        self.assertEqual(anchors.tolist(), [2, 3])
        self.assertEqual(positives.tolist(), [0, 1])
        anchors, _, _ = pseudo_triplet_indices(
            labels, predictions, accepted, torch.Generator().manual_seed(2), "pseudo_only")
        self.assertEqual(anchors.numel(), 0)

    def test_pseudo_triplet_denominator_includes_rejected_anchors(self):
        features = [torch.ones(2, 8, requires_grad=True) for _ in range(4)]
        loss, counts = pseudo_triplet_loss(
            *features, torch.tensor([0, 1]), torch.tensor([0, 0]),
            torch.tensor([True, False]), torch.Generator().manual_seed(0), 0.2)
        self.assertAlmostEqual(float(loss.detach()), 0.1, places=6)
        self.assertEqual(counts["pseudo_triplet_valid"], 1)
        self.assertEqual(counts["pseudo_triplet_attempted"], 2)
        self.assertEqual(counts["pseudo_triplet_labeled_positives"], 1)
        loss.backward()
        for feature in features:
            if feature.grad is not None:
                self.assertTrue(bool(torch.isfinite(feature.grad).all()))

    def test_pseudo_triplets_without_negatives_skip_safely(self):
        features = [torch.randn(2, 8, requires_grad=True) for _ in range(4)]
        loss, counts = pseudo_triplet_loss(
            *features, torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long),
            torch.ones(2, dtype=torch.bool), torch.Generator().manual_seed(0), 0.2)
        self.assertEqual(float(loss.detach()), 0)
        self.assertEqual(counts["pseudo_triplet_valid"], 0)
        loss.backward()
        self.assertEqual(float(features[0].grad.abs().sum()), 0)

    def test_presets_and_invalid_parameters(self):
        for name in METHODS:
            validate_config(select_method(self.config, name))
        for changed in ({"pseudo_label_threshold": 1.1}, {"pseudo_label_weight": -1},
                        {"pseudo_triplet_weight": -1}, {"pseudo_triplet_pool": "unknown"},
                        {"unlabeled_batch_size": 0}, {"num_workers": 1}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config({**self.config, **changed})

    def test_different_methods_start_from_identical_random_weights(self):
        config = {**self.config, "epochs": 1, "steps_per_epoch": 1,
                  "batch_size": 8, "unlabeled_batch_size": 8, "embedding_dim": 32}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            ce = run_supervised(select_method(config, "ce_2v"), root / "ce", root, "cpu", smoke=True)
            pseudo = run_supervised(select_method(config, "ce_2v_pseudo"), root / "pseudo", root, "cpu", smoke=True)
            self.assertEqual(ce["initial_model_sha256"], pseudo["initial_model_sha256"])
            self.assertEqual(ce["processed_labeled_views"], pseudo["processed_labeled_views"])

    def test_combined_resume_preserves_weights_and_exposure_counters(self):
        self.check_resume("ce_2v_triplet_lu_pseudo")

    def test_guided_resume_preserves_mining_and_exposure_counters(self):
        self.check_resume("ce_2v_pseudo_guided_triplet")

    def check_resume(self, method):
        config = select_method({**self.config, "epochs": 2, "steps_per_epoch": 2,
                                "batch_size": 8, "unlabeled_batch_size": 8,
                                "pseudo_label_threshold": 0.0, "embedding_dim": 32},
                               method)
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            full = run_supervised(config, root / "full", root, "cpu", smoke=True)
            run_supervised(config, root / "resumed", root, "cpu", smoke=True, stop_after_epoch=1)
            self.assertFalse((root / "resumed" / "metrics.json").exists())
            resumed = run_supervised(config, root / "resumed", root, "cpu", smoke=True, resume=True)
            full_weights = torch.load(root / "full" / "model.pt", weights_only=True)
            resumed_weights = torch.load(root / "resumed" / "model.pt", weights_only=True)
            for name, value in full_weights.items():
                torch.testing.assert_close(value, resumed_weights[name], rtol=0, atol=0)
            self.assertEqual(full["initial_model_sha256"], resumed["initial_model_sha256"])
            self.assertEqual(full["processed_labeled_views"], 64)
            self.assertEqual(full["processed_unlabeled_views"], 64)
            self.assertEqual(full["pseudo_accepted"], 32)
            self.assertEqual(sum(full["accepted_pseudo_class_histogram"]), 32)
            self.assertFalse(full["pseudo_labels_add_to_annotation_budget"])
            if method == "ce_2v_pseudo_guided_triplet":
                self.assertEqual(full["pseudo_triplet_attempted"], 32)
                self.assertEqual(full["pseudo_triplet_valid"], resumed["pseudo_triplet_valid"])
                self.assertGreater(full["pseudo_triplet_valid"], 0)


if __name__ == "__main__":
    unittest.main()
