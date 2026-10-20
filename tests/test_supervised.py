import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from fewshot_metric.data import balanced_splits
from fewshot_metric.models import CifarResNet18
from fewshot_metric.supervised import run_supervised, validate_config


class SupervisedControlTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.config = json.loads(Path("configs/cifar10_supervised.json").read_text())

    def test_budgeted_splits_are_disjoint_and_nested(self):
        labels = np.repeat(np.arange(10), 150)
        small, val = balanced_splits(labels, 5, 10, seed=7)
        large, val_again = balanced_splits(labels, 50, 10, seed=7)
        self.assertEqual(set(val), set(val_again))
        self.assertTrue(set(small) <= set(large))
        self.assertFalse(set(large) & set(val))
        self.assertEqual(np.bincount(labels[small]).tolist(), [5] * 10)

    def test_invalid_budgets_and_selection_are_rejected(self):
        for changed in ({"budget_per_class": 5000}, {"budget_per_class": 0},
                        {"validation_per_class": 0, "selection": "best_validation"},
                        {"steps_per_epoch": 0}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                validate_config({**self.config, **changed})

    def test_resnet_accepts_cifar_and_backpropagates(self):
        model = CifarResNet18(embedding_dim=512)
        features, logits = model(torch.randn(2, 3, 32, 32), normalize=True)
        self.assertEqual(tuple(features.shape), (2, 512))
        self.assertEqual(tuple(logits.shape), (2, 10))
        torch.testing.assert_close(features.norm(dim=1), torch.ones(2))
        logits.square().mean().backward()
        self.assertIsNotNone(model.encoder.conv1.weight.grad)

    def test_resume_matches_uninterrupted_training_and_no_early_test_metrics(self):
        config = {**self.config, "architecture": "cnn", "embedding_dim": 32,
                  "epochs": 2, "steps_per_epoch": 3, "batch_size": 10,
                  "learning_rate": 0.02}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            run_supervised(config, root / "full", root, "cpu", smoke=True)
            paused = run_supervised(config, root / "resumed", root, "cpu", smoke=True, stop_after_epoch=1)
            self.assertIsNone(paused)
            self.assertFalse((root / "resumed" / "metrics.json").exists())
            resumed = run_supervised(config, root / "resumed", root, "cpu", smoke=True, resume=True)
            self.assertEqual(resumed["status"], "synthetic_smoke_only")
            full_weights = torch.load(root / "full" / "model.pt", weights_only=True)
            resumed_weights = torch.load(root / "resumed" / "model.pt", weights_only=True)
            for name, value in full_weights.items():
                torch.testing.assert_close(value, resumed_weights[name], rtol=0, atol=0)
            with self.assertRaises(FileExistsError):
                run_supervised(config, root / "full", root, "cpu", smoke=True)

    def test_training_can_learn_class_dependent_synthetic_images(self):
        config = {**self.config, "architecture": "cnn", "embedding_dim": 32,
                  "epochs": 5, "steps_per_epoch": 10, "batch_size": 20,
                  "learning_rate": 0.05}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            run_supervised(config, Path(folder) / "learn", Path(folder), "cpu", smoke=True)
            import csv
            with (Path(folder) / "learn" / "history.csv").open() as handle:
                history = list(csv.DictReader(handle))
            self.assertLess(float(history[-1]["train_loss"]), float(history[0]["train_loss"]))
            self.assertGreater(float(history[-1]["train_accuracy"]), 0.7)

    def test_best_validation_selects_the_recorded_checkpoint(self):
        config = {**self.config, "architecture": "cnn", "embedding_dim": 32,
                  "epochs": 2, "steps_per_epoch": 3, "batch_size": 10,
                  "selection": "best_validation"}
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            result = run_supervised(config, root / "best", root, "cpu", smoke=True)
            best = torch.load(root / "best" / "best.pt", weights_only=True)
            weights = torch.load(root / "best" / "model.pt", weights_only=True)
            self.assertEqual(result["selected_epoch"], best["epoch"])
            for name, value in best["model"].items():
                torch.testing.assert_close(value, weights[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
