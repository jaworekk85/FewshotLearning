import contextlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F

from fewshot_metric.active import run_active, validate_active
from fewshot_metric.active_selection import (
    NEW_STRATEGIES, badge_queries, badge_squared_distances, local_density,
    select_prepared_queries,
)
from fewshot_metric.pilot import select_method
from scripts.run_next_pilot import build_jobs, run_queue


class PreparedActiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def choose(self, probabilities, features, strategy, *, labels=None, known=None, count=1, **kwargs):
        known = features[:0] if known is None else known
        return select_prepared_queries(probabilities, features, known, labels, count, strategy,
                                        torch.Generator().manual_seed(7), **kwargs)

    def test_badge_distances_equal_explicit_last_layer_gradients(self):
        p = torch.tensor([[0.7, 0.3], [0.1, 0.9], [0.5, 0.5]])
        z = torch.tensor([[1., 0.], [0., 3.], [2., 1.]])
        residual = p - F.one_hot(p.argmax(1), 2)
        gradients = (residual[:, :, None] * z[:, None, :]).flatten(1)
        for center in range(3):
            distances, norms = badge_squared_distances(p, z, center)
            torch.testing.assert_close(distances, (gradients - gradients[center]).square().sum(1))
            torch.testing.assert_close(norms, gradients.square().sum(1))
        chosen, _ = badge_queries(p, z, 3, torch.Generator().manual_seed(0))
        self.assertEqual(int(chosen[0]), int(gradients.square().sum(1).argmax()))
        self.assertEqual(len(set(chosen.tolist())), 3)

    def test_badge_handles_identical_zero_gradients_without_duplicates(self):
        p = torch.tensor([[1., 0.]] * 6)
        chosen, _ = self.choose(p, torch.zeros(6, 3), "badge", count=6)
        self.assertEqual(len(set(chosen.tolist())), 6)

    def test_density_rejects_isolated_point_at_equal_entropy_and_excludes_self(self):
        z = torch.tensor([[1., 0.], [1., 0.], [-1., 0.]])
        density = local_density(z, neighbors=1, chunk_size=2)
        self.assertEqual(density[:2].tolist(), [1., 1.])
        self.assertLess(float(density[2]), 1e-6)
        chosen, _ = self.choose(torch.full((3, 2), 0.5), z, "entropy_density",
                               settings={"density_neighbors": 1})
        self.assertEqual(chosen.tolist(), [0])

    def test_neighbor_disagreement_finds_confident_wrong_prediction(self):
        p = torch.tensor([[0.01, 0.99], [0.5, 0.5], [0.99, 0.01]])
        z = torch.tensor([[1., 0.]] * 3)
        chosen, diagnostics = self.choose(p, z, "neighbor_disagreement_density", known=z[:1], labels=[0])
        self.assertEqual(chosen.tolist(), [0])
        self.assertEqual(diagnostics["selected"][0]["neighbor_class"], 0)
        self.assertEqual(diagnostics["selected"][0]["predicted_class"], 1)
        with self.assertRaises(ValueError):
            self.choose(p, z, "neighbor_disagreement_density", known=z[:1], labels=[-1])
        self.assertNotIn("candidate_labels", inspect.signature(select_prepared_queries).parameters)

    def test_view_disagreement_distinguishes_instability_from_high_entropy(self):
        p = torch.tensor([[0.5, 0.5], [0.99, 0.01]])
        views = torch.stack([p, torch.tensor([[0.5, 0.5], [0.01, 0.99]])])
        chosen, _ = self.choose(p, torch.eye(2), "view_disagreement", view_probabilities=views)
        self.assertEqual(chosen.tolist(), [1])
        with self.assertRaises(ValueError):
            self.choose(p, torch.eye(2), "view_disagreement", view_probabilities=views[:1])

    def test_mixture_preserves_ranked_half_and_draws_other_half_without_duplicates(self):
        p = torch.stack([torch.tensor([v, 1-v]) for v in torch.linspace(0.01, 0.99, 12)])
        z = torch.tensor([[1., 0.]] * 12)
        common = dict(known=z[:1], labels=[0], count=4)
        ranked, _ = self.choose(p, z, "neighbor_disagreement_density", **common)
        mixed, _ = self.choose(p, z, "neighbor_disagreement_density_mix", **common)
        self.assertEqual(mixed[:2].tolist(), ranked[:2].tolist())
        self.assertEqual(len(set(mixed.tolist())), 4)

    def test_plan_is_complete_valid_and_cannot_start_without_explicit_flag(self):
        plan = json.loads(Path("configs/cifar10_pilot_active_prepared.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs), 5)
        self.assertEqual({j["config"]["active_strategy"] for j in jobs}, NEW_STRATEGIES - {"confident_neighbor"})
        for job in jobs:
            validate_active(job["config"])
            self.assertEqual(job["config"]["active_scoring_views"], 4)
            self.assertEqual(job["config"]["epochs"], 20)
            self.assertEqual(job["config"]["active_final_budget"], 500)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/"unstarted"
            with self.assertRaisesRegex(ValueError, "prepared only"):
                run_queue({**plan, "output_dir": str(output)})
            self.assertFalse(output.exists())

    def tiny_config(self, strategy, prepared=True):
        cfg = select_method({**self.base, "epochs": 3, "steps_per_epoch": 1,
                             "batch_size": 8, "unlabeled_batch_size": 8, "embedding_dim": 16,
                             "budget_per_class": 1, "active_final_budget": 14,
                             "active_rounds": [1, 2], "active_query_size": 2,
                             "active_candidate_limit": 16, "active_strategy": strategy}, "ce_2v_pseudo")
        if prepared:
            cfg.update(active_selection_version=2, active_scoring_views=4)
        return cfg

    def test_extra_scoring_views_do_not_change_legacy_random_training_or_batchnorm(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            old = run_active(self.tiny_config("random", prepared=False), root/"old", root, smoke=True)
            new = run_active(self.tiny_config("random"), root/"new", root, smoke=True)
            for key, value in torch.load(root/"old/model.pt", weights_only=True).items():
                torch.testing.assert_close(value, torch.load(root/"new/model.pt", weights_only=True)[key], rtol=0, atol=0)
            self.assertGreater(new["scoring_forward_views"], old["scoring_forward_views"])
            first = json.loads((root/"old/acquisitions.json").read_text())
            second = json.loads((root/"new/acquisitions.json").read_text())
            self.assertEqual([e["indices"] for e in first], [e["indices"] for e in second])
            self.assertEqual(old["processed_labeled_views"], new["processed_labeled_views"])

    def test_all_prepared_rules_resume_exactly_and_record_label_free_diagnostics(self):
        for strategy in sorted(NEW_STRATEGIES):
            with self.subTest(strategy=strategy), tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
                root = Path(tmp)
                cfg = self.tiny_config(strategy)
                full = run_active(cfg, root/"full", root, smoke=True)
                run_active(cfg, root/"resume", root, smoke=True, stop_after_epoch=1)
                self.assertFalse((root/"resume/metrics.json").exists())
                resumed = run_active(cfg, root/"resume", root, smoke=True, resume=True)
                self.assertEqual(full["true_training_label_budget"], 14)
                self.assertEqual(full["scoring_forward_views"], resumed["scoring_forward_views"])
                self.assertEqual(full["processed_labeled_views"], resumed["processed_labeled_views"])
                self.assertEqual((root/"full/acquisitions.json").read_text(), (root/"resume/acquisitions.json").read_text())
                for key, value in torch.load(root/"full/model.pt", weights_only=True).items():
                    torch.testing.assert_close(value, torch.load(root/"resume/model.pt", weights_only=True)[key], rtol=0, atol=0)
                event = json.loads((root/"full/acquisitions.json").read_text())[0]
                diagnostics = event["acquisition_diagnostics"]
                self.assertEqual(diagnostics["candidate_views"], 4)
                self.assertEqual(len(diagnostics["selected"]), 2)


if __name__ == "__main__":
    unittest.main()
