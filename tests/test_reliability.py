import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fewshot_metric.reliability import (analyze_run, calibration, detection_metrics,
                                       hypothetical_weights, weight_metrics)
from fewshot_metric.models import build_model


class ReliabilityTests(unittest.TestCase):
    def test_perfect_and_overconfident_calibration_includes_probability_one(self):
        perfect = calibration(np.eye(2), [0, 1])
        self.assertEqual(perfect["ece_15"], 0)
        self.assertEqual(perfect["brier_sum_classes"], 0)
        self.assertEqual(perfect["bins"][-1]["count"], 2)
        wrong = calibration(np.array([[.1, .9], [.9, .1]]), [0, 1])
        self.assertAlmostEqual(wrong["ece_15"], .9)
        self.assertAlmostEqual(wrong["brier_sum_classes"], 1.62)

    def test_gaussian_weight_and_soft_target_mass(self):
        config = dict(pseudo_label_weight=1, pseudo_weighting="epoch_gaussian",
                      pseudo_target="soft", pseudo_temperature=.5)
        w, rule = hypothetical_weights(np.array([.5, .8, .9]), config,
            dict(pseudo_weight_center_next_epoch=.8, pseudo_weight_std_next_epoch=.1))
        np.testing.assert_allclose(w, [np.exp(-4.5), 1, 1])
        self.assertEqual(rule["center"], .8)
        result = weight_metrics(np.array([[.8, .2]]), np.array([0]), np.ones(1), config)
        self.assertAlmostEqual(result["target_mass_outside_true_class"], 1 / 17)
        self.assertEqual(result["wrong_argmax_weight_fraction"], 0)

    def test_hard_rule_empty_acceptance_and_ce(self):
        w, _ = hypothetical_weights(np.array([.5]), dict(pseudo_label_weight=1), {})
        self.assertEqual(w.tolist(), [0])
        self.assertIsNone(weight_metrics(np.array([[.5, .5]]), np.array([0]), w, {})["wrong_argmax_weight_fraction"])
        self.assertIsNone(hypothetical_weights(np.array([.5]), {}, {})[0])

    def test_detection_direction_and_single_class(self):
        scores = dict(perfect=np.array([0, 0, 1, 1]))
        result = detection_metrics([False, False, True, True], scores)["perfect"]
        self.assertEqual(result["auroc"], 1)
        self.assertEqual(result["precision_top_10pct"], 1)
        self.assertIsNone(detection_metrics([False] * 4, scores)["perfect"]["auroc"])

    def test_export_is_deterministic_preserves_rng_and_reads_no_hidden_labels(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(3)
        # Index 35 is hidden U: reading that label fails the test.
        class GuardedLabels:
            def __getitem__(self, i):
                if i >= 30:
                    raise AssertionError("Hidden label read")
                return i % 10
        base = SimpleNamespace(data=rng.integers(0, 256, (40, 32, 32, 3), dtype=np.uint8),
                               targets=GuardedLabels(), classes=[str(i) for i in range(10)])
        config = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())
        config.update(embedding_dim=16, synthetic_smoke=False, method="ce_2v", pseudo_label_weight=0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "fixture"
            run.mkdir()
            (run / "config.json").write_text(json.dumps(config))
            (run / "splits.json").write_text(json.dumps(dict(train=list(range(10)), validation=list(range(10, 30)))))
            (run / "history.csv").write_text("epoch,val_accuracy\n1,0.1\n")
            model = build_model("cnn", embedding_dim=16)
            torch.save(model.state_dict(), run / "model.pt")
            before = torch.get_rng_state().clone()
            with patch.dict(os.environ, {"MPLCONFIGDIR": str(root / "mpl")}), patch("fewshot_metric.reliability.datasets.CIFAR10", return_value=base) as loader:
                result = analyze_run(run, root / "audit", weak_views=2)
                cached = analyze_run(run, root / "audit", weak_views=2)
                self.assertEqual(loader.call_count, 1)
                self.assertEqual(result, cached)
                with self.assertRaisesRegex(ValueError, "cache mismatch"):
                    analyze_run(run, root / "audit", weak_views=3)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            with np.load(root / "audit/fixture/predictions.npz") as exported:
                self.assertEqual(exported["validation_indices"].tolist(), list(range(10, 30)))
                self.assertEqual(exported["weak_probabilities"].shape, (2, 20, 10))
            self.assertTrue((root / "audit/fixture/diagnostics.png").exists())


if __name__ == "__main__":
    unittest.main()
