import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch

from fewshot_metric.active import run_active
from fewshot_metric.active_selection import select_prepared_queries
from fewshot_metric.neighbor_reliability import agreement_scores, isolated_eval, reweight
from fewshot_metric.pilot import select_method
from fewshot_metric.supervised import run_supervised, validate_config
from scripts.run_next_pilot import build_jobs, run_queue


class NeighborReliabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def tiny(self, active=False):
        cfg = select_method({**self.base, "epochs": 3, "steps_per_epoch": 2,
             "embedding_dim": 16, "batch_size": 8, "unlabeled_batch_size": 8,
             "budget_per_class": 1, "neighbor_reference": True, "neighbor_pseudo_weighting": True,
             "neighbor_warmup_epochs": 0, "neighbor_confidence_threshold": 0.0}, "ce_2v_pseudo_soft")
        if active:
            cfg.update(active_final_budget=14, active_rounds=[1, 2], active_query_size=2,
                       active_candidate_limit=16, active_strategy="confident_neighbor",
                       active_selection_version=2, active_scoring_views=1)
        return cfg

    def assert_models_equal(self, first, second):
        for key, value in torch.load(first, weights_only=True).items():
            torch.testing.assert_close(value, torch.load(second, weights_only=True)[key], rtol=0, atol=0)

    def test_weighting_is_detached_preserves_mass_and_only_reduces_confident_disagreement(self):
        base = torch.tensor([1., 1., .5], requires_grad=True)
        confidence = torch.tensor([.99, .99, .6])
        js = torch.tensor([.5, 0., .6], requires_grad=True)
        cfg = dict(neighbor_pseudo_weighting=True, neighbor_warmup_epochs=3)
        weights, counts = reweight(base, confidence, js, cfg, 3)
        self.assertFalse(weights.requires_grad)
        torch.testing.assert_close(weights.sum(), base.sum())
        self.assertLess(weights[0], weights[1])
        self.assertAlmostEqual(float(weights[1] / weights[2]), 2)
        self.assertEqual(counts["neighbor_reweighted"], 2)
        warmup, _ = reweight(base, confidence, js, cfg, 2)
        torch.testing.assert_close(warmup, base)
        zeros, _ = reweight(torch.zeros(3), confidence, js, cfg, 3)
        self.assertEqual(float(zeros.sum()), 0)

    def test_eval_restores_modes_rng_and_batchnorm_even_after_error(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(3), torch.nn.Dropout(.5))
        model.train()
        model[1].eval()
        before_rng = torch.get_rng_state().clone()
        before_bn = model[0].running_mean.clone()
        with self.assertRaises(RuntimeError):
            with isolated_eval(model):
                self.assertFalse(model.training)
                model(torch.ones(4, 3))
                torch.rand(3)
                raise RuntimeError("fixture")
        self.assertTrue(model.training)
        self.assertFalse(model[1].training)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(torch.equal(before_bn, model[0].running_mean))

    def test_reference_has_only_known_labels_and_disagreement_direction(self):
        scores = agreement_scores(torch.tensor([[.99, .01], [.01, .99]]), torch.ones(2, 3),
                                  torch.ones(1, 3), torch.tensor([0]), {})
        self.assertLess(scores[0], scores[1])
        with self.assertRaises(ValueError):
            agreement_scores(torch.tensor([[.99, .01]]), torch.ones(1, 3),
                             torch.ones(1, 3), torch.tensor([-1]), {})

    def test_acquisition_spends_half_on_confident_disagreement_then_entropy_and_falls_back(self):
        p = torch.tensor([[.01, .99], [.99, .01], [.5, .5], [.6, .4]])
        z = torch.ones(4, 3)
        args = (p, z, z[:1], [0], 2, "confident_neighbor", torch.Generator().manual_seed(0))
        chosen, diagnostics = select_prepared_queries(*args)
        self.assertEqual(chosen.tolist(), [0, 2])
        self.assertEqual([s["selection_component"] for s in diagnostics["selected"]], ["confident_neighbor", "entropy"])
        chosen, _ = select_prepared_queries(*args, settings={"confident_threshold": 1.0})
        self.assertEqual(chosen.tolist(), [2, 3])

    def test_four_new_runs_two_reused_baselines_and_guard(self):
        plan = json.loads(Path("configs/cifar10_pilot_neighbor_reliability.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs), 6)
        self.assertEqual(sum("reuse_run" in j for j in jobs), 2)
        self.assertEqual(sum(j["kind"] == "active" for j in jobs), 4)
        self.assertEqual(sum(j["config"].get("neighbor_pseudo_weighting", False) for j in jobs), 3)
        for job in jobs:
            if not job["config"].get("neighbor_pseudo_weighting"):
                self.assertFalse(job["config"].get("neighbor_reference", False))
        for j in jobs:
            self.assertEqual(j["config"]["epochs"], 20)
            self.assertEqual(j["config"]["steps_per_epoch"], 383)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "unstarted"
            with self.assertRaisesRegex(ValueError, "prepared only"):
                run_queue({**plan, "output_dir": str(path)})
            self.assertFalse(path.exists())
        with self.assertRaises(ValueError):
            validate_config({**jobs[1]["config"], "neighbor_k": 0})
        with self.assertRaises(ValueError):
            validate_config({**jobs[0]["config"], "neighbor_reference": False, "neighbor_pseudo_weighting": True})

    def test_eval_control_is_bitwise_identical_to_existing_soft_ssl(self):
        for active in (False, True):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                cfg = self.tiny(active)
                cfg["neighbor_pseudo_weighting"] = False
                legacy = {k: v for k, v in cfg.items() if not k.startswith("neighbor_")}
                runner = run_active if active else run_supervised
                runner(legacy, root/"old", root, "cpu", smoke=True)
                result = runner(cfg, root/"control", root, "cpu", smoke=True)
                self.assert_models_equal(root/"old/model.pt", root/"control/model.pt")
                self.assertGreater(result["neighbor_query_views"], 0)
                if active:
                    self.assertEqual((root/"old/acquisitions.json").read_text(), (root/"control/acquisitions.json").read_text())

    def test_enabled_weighting_resumes_exactly_for_static_and_acquiring_reference(self):
        for active in (False, True):
            with self.subTest(active=active), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                cfg = self.tiny(active)
                runner = run_active if active else run_supervised
                full = runner(cfg, root/"full", root, "cpu", smoke=True)
                runner(cfg, root/"resume", root, "cpu", smoke=True, stop_after_epoch=1)
                self.assertFalse((root/"resume/metrics.json").exists())
                resumed = runner(cfg, root/"resume", root, "cpu", smoke=True, resume=True)
                self.assert_models_equal(root/"full/model.pt", root/"resume/model.pt")
                self.assertEqual(full["neighbor_reference_views"], resumed["neighbor_reference_views"])
                self.assertEqual(full["neighbor_query_views"], resumed["neighbor_query_views"])
                self.assertGreater(full["neighbor_reweighted"], 0)
                self.assertAlmostEqual(full["pseudo_base_weight_sum"], full["pseudo_weight_sum"], places=4)
                if active:
                    self.assertEqual(full["neighbor_reference_views"], 10 + 12 + 14)
                    self.assertEqual((root/"full/acquisitions.json").read_text(), (root/"resume/acquisitions.json").read_text())


if __name__ == "__main__":
    unittest.main()
