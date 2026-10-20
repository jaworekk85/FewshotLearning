import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from fewshot_metric.models import build_model, shared_parameter_fingerprint
from fewshot_metric.pilot import model_fingerprint, select_method
from fewshot_metric.supervised import run_supervised


class NormalizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.base=json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def test_initial_shared_weights_and_rng_match_all_three_normalizations(self):
        hashes, rngs, counts=[],[],[]
        for normalization in ("batchnorm","none","groupnorm"):
            torch.manual_seed(0)
            model=build_model("cnn",embedding_dim=128,normalization=normalization)
            hashes.append(shared_parameter_fingerprint(model))
            rngs.append(torch.get_rng_state())
            counts.append(sum(p.numel() for p in model.parameters()))
            if normalization=="batchnorm":
                self.assertEqual(model_fingerprint(model),"ff9df5909da63e26cdef8c2f124b9abc2eb6d224ab299f86e61c8d1289bbac6b")
            else:
                self.assertFalse(any(isinstance(m,nn.BatchNorm2d) for m in model.modules()))
        self.assertEqual(len(set(hashes)),1)
        self.assertEqual(counts,[111498,111050,111498])
        for state in rngs[1:]:
            torch.testing.assert_close(state,rngs[0],rtol=0,atol=0)

    def test_groupnorm_and_identity_do_not_depend_on_other_batch_images(self):
        image=torch.randn(1,32,8,8)
        others=torch.randn(3,32,8,8)+20
        for layer in (nn.GroupNorm(8,32),nn.Identity()):
            layer.train()
            torch.testing.assert_close(layer(image),layer(torch.cat([image,others]))[:1],rtol=1e-6,atol=1e-6)
        batchnorm=nn.BatchNorm2d(32)
        difference=(batchnorm(image)-batchnorm(torch.cat([image,others]))[:1]).abs().mean()
        self.assertGreater(float(difference.detach()),0.5)

    def test_all_new_variants_train_and_record_real_exposures(self):
        for normalization in ("none","groupnorm"):
            for method in ("ce_2v","ce_2v_pseudo"):
                config=select_method({**self.base,"normalization":normalization,"epochs":1,
                                      "steps_per_epoch":2,"batch_size":8,"unlabeled_batch_size":8,
                                      "embedding_dim":16},method)
                with self.subTest(normalization=normalization,method=method), tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                    root=Path(tmp)
                    result=run_supervised(config,root/"run",root,"cpu",smoke=True)
                    self.assertEqual(result["processed_labeled_views"],32)
                    self.assertEqual(result["processed_unlabeled_views"],32 if method=="ce_2v_pseudo" else 0)
                    self.assertEqual(result["batchnorm_policy"],"not_applicable")
                    self.assertEqual(result["normalization"],normalization)
                    model=build_model("cnn",embedding_dim=16,normalization=normalization)
                    model.load_state_dict(torch.load(root/"run/model.pt",weights_only=True))
                    self.assertEqual(result["inference_parameter_count"],sum(p.numel() for p in model.parameters()))

    def test_groupnorm_resume_preserves_exact_weights(self):
        config=select_method({**self.base,"normalization":"groupnorm","epochs":2,
                              "steps_per_epoch":2,"batch_size":8,"unlabeled_batch_size":8,
                              "embedding_dim":16},"ce_2v_pseudo")
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            run_supervised(config,root/"full",root,"cpu",smoke=True)
            run_supervised(config,root/"resumed",root,"cpu",smoke=True,stop_after_epoch=1)
            run_supervised(config,root/"resumed",root,"cpu",smoke=True,resume=True)
            a=torch.load(root/"full/model.pt",weights_only=True)
            b=torch.load(root/"resumed/model.pt",weights_only=True)
            for key,value in a.items():
                torch.testing.assert_close(value,b[key],rtol=0,atol=0)

    def test_queue_contains_only_two_no_normalization_runs_and_waits_for_reconstruction(self):
        from scripts.run_next_pilot import build_jobs
        plan=json.loads(Path("configs/cifar10_pilot_normalization.json").read_text())
        jobs=build_jobs(plan)
        self.assertEqual(len(jobs),2)
        self.assertEqual(len({job["name"] for job in jobs}),2)
        self.assertEqual({(job["config"]["method"],job["config"]["normalization"]) for job in jobs},
                         {(method,"none") for method in ("ce_2v","ce_2v_pseudo")})
        for job in jobs:
            self.assertEqual(job["config"]["epochs"],20)
            self.assertEqual(job["config"]["budget_per_class"],50)
        self.assertEqual(plan["expected_preceding_runs"],5)


if __name__=="__main__":
    unittest.main()
