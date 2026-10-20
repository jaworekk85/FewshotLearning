import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from fewshot_metric.models import build_model
from fewshot_metric.pilot import PilotTraining, model_fingerprint, select_method, synthetic_datasets
from fewshot_metric.reconstruction import (ReconstructionModel, ReconstructionPilot, ReconstructionViews,
                                           kl_to_standard_normal, reconstruction_config)
from fewshot_metric.supervised import run_supervised, validate_config


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def test_all_variants_have_identical_views_single_encoder_pass_and_common_initialization(self):
        seen, fingerprints = [], []
        for variant in ("ce_u_control", "dae", "dae_pseudo", "dvae", "dvae_pseudo"):
            config = reconstruction_config({**self.base, "batch_size": 8, "unlabeled_batch_size": 8,
                                             "embedding_dim": 16}, variant)
            validate_config(config)
            torch.manual_seed(0)
            backbone = build_model("cnn", embedding_dim=16)
            fingerprints.append(model_fingerprint(backbone))
            before = torch.get_rng_state().clone()
            model = ReconstructionModel(backbone, config)
            torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)
            train, _, _ = synthetic_datasets(config)
            pilot = ReconstructionPilot(config, train, {"train": list(range(40)), "validation": list(range(40,60))})
            pilot.start_epoch(0)
            images, labels = next(iter(DataLoader(train, batch_size=8)))
            calls = []
            hook = backbone.register_forward_pre_hook(lambda module, args: calls.append(args[0].detach().clone()))
            loss, _, _, metrics = pilot.step(model, images, labels, torch.device("cpu"))
            hook.remove()
            loss.backward()
            self.assertEqual(len(calls), 1)
            self.assertEqual(len(calls[0]), 32)
            seen.append(calls[0])
            self.assertEqual(metrics["processed_labeled_views"], 16)
            self.assertEqual(metrics["processed_unlabeled_views"], 16)
            self.assertEqual(metrics["processed_decoder_views"], 0 if variant == "ce_u_control" else 8)
            self.assertTrue(torch.isfinite(loss))
            if model.decoder is not None:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.decoder.parameters()), 0)
            model.eval()
            with torch.no_grad():
                a = model(images[0])[1]
                b = model(images[0])[1]
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.assertEqual(len(set(fingerprints)), 1)
        for batch in seen[1:]:
            torch.testing.assert_close(batch, seen[0], rtol=0, atol=0)

    def test_reconstruction_inputs_match_original_pilot_exactly_across_epochs_and_visits(self):
        config = select_method({**self.base, "batch_size":8,"unlabeled_batch_size":8}, "ce_2v_pseudo")
        train_a, _, _ = synthetic_datasets(config)
        train_b, _, _ = synthetic_datasets(config)
        splits = {"train":list(range(40)),"validation":list(range(40,60))}
        original = PilotTraining(config, train_a, splits)
        reconstructed = ReconstructionPilot(reconstruction_config(config,"dae_pseudo"),train_b,splits)
        for epoch in (0,1,3):
            original.start_epoch(epoch)
            reconstructed.start_epoch(epoch)
            for _ in range(3):
                a, b = next(original.iterator), next(reconstructed.iterator)
                for x,y in zip(a,b[:2]):
                    torch.testing.assert_close(x,y,rtol=0,atol=0)
                self.assertTrue(bool(((b[2]>=0)&(b[2]<=1)).all()))

    def test_zero_auxiliary_weight_reproduces_original_pseudo_training(self):
        base = select_method({**self.base,"epochs":1,"steps_per_epoch":2,"batch_size":8,
                              "unlabeled_batch_size":8,"embedding_dim":16},"ce_2v_pseudo")
        ae = reconstruction_config(base,"dae_pseudo")
        ae.update(reconstruction_weight=0,reconstruction_kl_weight=0)
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            run_supervised(base,root/"original",root,"cpu",smoke=True)
            run_supervised(ae,root/"ae",root,"cpu",smoke=True)
            original=torch.load(root/"original/model.pt",weights_only=True)
            for k,v in torch.load(root/"ae/model.pt",weights_only=True).items():
                torch.testing.assert_close(v,original[k],rtol=0,atol=0)

    def test_gaussian_kl_zero_for_standard_normal_and_positive_for_shift(self):
        mean = torch.zeros(2, 8, requires_grad=True)
        logvar = torch.zeros_like(mean, requires_grad=True)
        self.assertEqual(float(kl_to_standard_normal(mean, logvar).detach()), 0)
        shifted = kl_to_standard_normal(mean + 1, logvar)
        self.assertAlmostEqual(float(shifted.detach()), 0.5)
        shifted.backward()
        self.assertGreater(float(mean.grad.abs().sum()), 0)

    def test_reconstruction_queue_reuses_existing_baselines_and_has_five_common_protocol_runs(self):
        from scripts.run_next_pilot import build_jobs
        plan = json.loads(Path("configs/cifar10_pilot_reconstruction.json").read_text())
        jobs = build_jobs(plan)
        self.assertEqual(len(jobs), 5)
        self.assertEqual(len({j["name"] for j in jobs}), 5)
        self.assertEqual([j["config"]["epochs"] for j in jobs], [20]*5)
        self.assertEqual(jobs[0]["config"]["method"],"ce_2v_u_control")
        self.assertFalse(jobs[0]["config"].get("reconstruction_pilot",False))
        for job in jobs:
            cfg = job["config"]
            self.assertEqual(cfg["budget_per_class"], 50)
            self.assertNotIn("reconstruction_mask_fraction",cfg)
            self.assertTrue(cfg["use_unlabeled_control"] or cfg["pseudo_label_weight"] > 0)

    def test_vae_resume_preserves_decoder_rng_and_export_is_plain_classifier(self):
        config = reconstruction_config({**self.base, "epochs": 2, "steps_per_epoch": 2,
                                         "batch_size": 8, "unlabeled_batch_size": 8,
                                         "embedding_dim": 16}, "dvae_pseudo")
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            full = run_supervised(config, root/"full", root, "cpu", smoke=True)
            run_supervised(config, root/"resume", root, "cpu", smoke=True, stop_after_epoch=1)
            self.assertFalse((root/"resume/metrics.json").exists())
            resumed = run_supervised(config, root/"resume", root, "cpu", smoke=True, resume=True)
            a = torch.load(root/"full/latest.pt", weights_only=False)
            b = torch.load(root/"resume/latest.pt", weights_only=False)
            for name, value in a["model"].items():
                torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
            exported = torch.load(root/"full/model.pt", weights_only=True)
            model = build_model("cnn", embedding_dim=16)
            model.load_state_dict(exported)
            self.assertEqual(full["processed_decoder_views"], 32)
            self.assertEqual(full["processed_labeled_views"], 64)
            self.assertEqual(full["test_accuracy"], resumed["test_accuracy"])
            self.assertGreater(full["trainable_parameter_count"], full["inference_parameter_count"])


if __name__ == "__main__":
    unittest.main()
