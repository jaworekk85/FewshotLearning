import contextlib
import inspect
import io
import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from fewshot_metric.mined_triplets import conflicting_instance_pairs, mine_indices, mined_triplet_loss
from fewshot_metric.pilot import PilotTraining, select_method, synthetic_datasets, unlabeled_triplet_loss
from fewshot_metric.supervised import run_supervised, validate_config
from scripts.run_next_pilot import build_jobs, run_queue


class MinedTripletTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base = json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def fixture(self):
        z = torch.tensor([[1.,0.], [0.,1.], [.6,.8], [.8,.6], [-1.,0.], [.1,.9]])
        labels, predictions = torch.tensor([0,1,0]), torch.tensor([0,0,1])
        confidence = torch.tensor([.99,.97,.99])
        cfg = dict(metric_triplet_design="lul", metric_triplet_positive_fraction=.25,
                   metric_triplet_anchor_limit=4, mean=[0,0,0], std=[1,1,1])
        return z, labels, predictions, confidence, cfg

    def test_lul_selects_distant_pseudo_positive_nearest_true_negative_and_confidence_weight(self):
        z, labels, predictions, confidence, cfg = self.fixture()
        triples, weights = mine_indices(z, labels, predictions, confidence, torch.Generator().manual_seed(0), cfg)
        selected = (triples[:,0] == 0).nonzero().item()
        self.assertEqual(triples[selected].tolist(), [0,4,1])
        self.assertAlmostEqual(float(weights[selected]), .97, places=6)
        self.assertTrue((triples[:,0] < 3).all())
        self.assertTrue((triples[:,1] >= 3).all())
        self.assertTrue((triples[:,2] < 3).all())
        self.assertNotIn("unlabeled_labels", inspect.signature(mine_indices).parameters)

    def test_pixel_distance_changes_selection_while_loss_has_embedding_gradients(self):
        z, labels, predictions, confidence, cfg = self.fixture()
        pixels = torch.tensor([0.,.8,.3,1.,.1,.2])[:,None,None,None].expand(6,3,2,2)
        cfg.update(metric_triplet_distance="pixels", metric_triplet_warmup_epochs=0, triplet_margin=2.)
        triples, weights = mine_indices(z, labels, predictions, confidence, torch.Generator().manual_seed(0), cfg, pixels)
        self.assertEqual(triples[triples[:,0] == 0][0].tolist(), [0,3,1])
        first = z.clone().requires_grad_()
        second = (z + .03).clone().requires_grad_()
        loss, stats = mined_triplet_loss(first[:3], second[:3], first[3:], second[3:], labels,
                predictions, confidence, torch.Generator().manual_seed(0), cfg, 0, pixels)
        norm_first, norm_second = F.normalize(first,dim=1), F.normalize(second,dim=1)
        a,p,n = triples.unbind(1)
        expected = (weights * ((norm_first[a]-norm_second[p]).norm(dim=1) - (norm_first[a]-norm_first[n]).norm(dim=1) + 2.).relu()).sum()/4
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertGreater(float(first.grad.abs().sum()+second.grad.abs().sum()), 0)
        self.assertEqual(stats["pseudo_triplet_attempted"], 4)

    def test_mixed_anchors_include_L_and_U_with_detached_product_weights(self):
        z, labels, predictions, confidence, cfg = self.fixture()
        cfg.update(metric_triplet_design="mixed", metric_triplet_anchor_limit=128)
        confidence.requires_grad_()
        triples, weights = mine_indices(z, labels, predictions, confidence, torch.Generator().manual_seed(0), cfg)
        self.assertTrue((triples[:,0] < 3).any() and (triples[:,0] >= 3).any())
        self.assertFalse(weights.requires_grad)
        reliability=torch.cat([torch.ones(3),confidence.detach()])
        torch.testing.assert_close(weights, reliability[triples].prod(1))
        all_labels=torch.cat([labels,predictions])
        self.assertTrue((all_labels[triples[:,0]] == all_labels[triples[:,1]]).all())
        self.assertTrue((all_labels[triples[:,0]] != all_labels[triples[:,2]]).all())
        self.assertTrue((triples[:,0] != triples[:,1]).all())

    def test_warmup_rejection_and_no_valid_triplets_produce_differentiable_zero(self):
        z, labels, predictions, confidence, cfg = self.fixture()
        z.requires_grad_()
        loss, stats = mined_triplet_loss(z[:3],z[:3],z[3:],z[3:],labels,predictions,confidence,
                                       torch.Generator().manual_seed(0),cfg,2)
        self.assertEqual(float(loss.detach()),0)
        self.assertEqual(stats["pseudo_triplet_valid"],0)
        loss.backward()
        self.assertEqual(float(z.grad.abs().sum()),0)
        triples,_=mine_indices(z,labels,predictions,torch.zeros(3),torch.Generator().manual_seed(0),cfg)
        self.assertEqual(len(triples),0)

    def test_instance_mask_excludes_only_confident_same_class_pairs_and_handles_empty_pool(self):
        mask=conflicting_instance_pairs(torch.tensor([0,0,1,0]),torch.tensor([.99,.99,.99,.5]),.95)
        expected=torch.tensor([[True,True,False,False],[True,True,False,False],[False,False,True,False],[False,False,False,False]])
        self.assertTrue(torch.equal(mask,expected))
        first=torch.randn(4,3,requires_grad=True); second=torch.randn(4,3,requires_grad=True)
        stats={}
        loss=unlabeled_triplet_loss(first,second,None,.2,"batch_all",stats,torch.ones(4,4,dtype=torch.bool))
        self.assertEqual(stats["metric_instance_excluded_pairs"],12)
        self.assertEqual(stats["triplet_u_pairs"],0)
        self.assertEqual(float(loss.detach()),0)
        loss.backward()

    def test_all_variants_reuse_one_forward_and_all_three_components(self):
        for distance,mask in (("embedding",False),("pixels",False),("embedding",True)):
            cfg=select_method({**self.base,"unlabeled_batch_size":8},"ce_2v_pseudo_soft_mined_triplet")
            cfg.update(metric_triplet_distance=distance,metric_triplet_mask_same_pseudo=mask,
                       metric_triplet_warmup_epochs=0,metric_triplet_confidence_threshold=0.)
            train,_,_=synthetic_datasets(cfg)
            pilot=PilotTraining(cfg,train,dict(train=list(range(40)),validation=list(range(40,60))))
            self.assertIsNone(pilot.unlabeled.labels)
            pilot.start_epoch(0)
            views,labels=next(iter(DataLoader(train,batch_size=10)))
            class CountedModel(torch.nn.Module):
                def __init__(self):
                    super().__init__(); self.encoder=torch.nn.Linear(3072,8); self.head=torch.nn.Linear(8,10); self.calls=0
                def forward(self,x,normalize=False):
                    self.calls+=1; z=self.encoder(x.flatten(1)); return z,self.head(z)
            model=CountedModel()
            loss,_,_,metrics=pilot.step(model,views,labels,torch.device("cpu"))
            self.assertEqual(model.calls,1)
            self.assertGreater(metrics["triplet_l_valid"],0)
            self.assertGreater(metrics["pseudo_triplet_valid"],0)
            expected=metrics["ce_loss"]+metrics["pseudo_label_loss"]+(metrics["triplet_l_loss"]+metrics["triplet_u_loss"]+metrics["pseudo_triplet_loss"])/30
            self.assertAlmostEqual(float(loss.detach()),expected,places=5)
            loss.backward()

    def test_plan_has_four_distinct_new_jobs_two_reused_baselines_and_start_guard(self):
        plan=json.loads(Path("configs/cifar10_pilot_mined_triplets.json").read_text())
        jobs=build_jobs(plan)
        self.assertEqual(len(jobs),6)
        new=[j for j in jobs if "reuse_run" not in j]
        self.assertEqual(len(new),4)
        self.assertEqual(len({(j["config"]["metric_triplet_design"],j["config"]["metric_triplet_distance"],j["config"]["metric_triplet_mask_same_pseudo"]) for j in new}),4)
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"unstarted"
            with self.assertRaisesRegex(ValueError,"prepared only"):
                run_queue({**plan,"output_dir":str(path)})
            self.assertFalse(path.exists())
        for key,value in (("metric_triplet_distance","bad"),("metric_triplet_anchor_limit",0),("metric_triplet_confidence_threshold",2)):
            with self.assertRaises(ValueError):validate_config({**new[0]["config"],key:value})

    def test_each_new_miner_resumes_exactly_with_same_exposure(self):
        plan=json.loads(Path("configs/cifar10_pilot_mined_triplets.json").read_text())
        for job in [j for j in build_jobs(plan) if "reuse_run" not in j]:
            with self.subTest(job=job["name"]),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp)
                cfg={**job["config"],"epochs":3,"steps_per_epoch":2,"batch_size":8,"unlabeled_batch_size":8,"embedding_dim":16,
                     "metric_triplet_warmup_epochs":1,"metric_triplet_confidence_threshold":0.}
                full=run_supervised(cfg,root/"full",root,"cpu",smoke=True)
                run_supervised(cfg,root/"resume",root,"cpu",smoke=True,stop_after_epoch=1)
                self.assertFalse((root/"resume/metrics.json").exists())
                resumed=run_supervised(cfg,root/"resume",root,"cpu",smoke=True,resume=True)
                for k,v in torch.load(root/"full/model.pt",weights_only=True).items():
                    torch.testing.assert_close(v,torch.load(root/"resume/model.pt",weights_only=True)[k],rtol=0,atol=0)
                self.assertEqual(full["processed_labeled_views"],resumed["processed_labeled_views"])
                self.assertEqual(full["pseudo_triplet_valid"],resumed["pseudo_triplet_valid"])
                self.assertGreater(full["metric_anchor_l"],0)


if __name__ == "__main__":unittest.main()
