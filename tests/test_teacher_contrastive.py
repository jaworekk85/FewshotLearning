import contextlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from fewshot_metric.ema_teacher import EMATeacher
from fewshot_metric.mixed_contrastive import pair_graph,mixed_contrastive_loss
from fewshot_metric.models import build_model
from fewshot_metric.pilot import select_method
from fewshot_metric.pilot import PilotTraining,synthetic_datasets
from fewshot_metric.supervised import run_supervised,validate_config
from scripts.run_next_pilot import build_jobs,run_queue


class TeacherContrastiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.base=json.loads(Path("configs/cifar10_pilot_ssl.json").read_text())

    def tiny(self,**overrides):
        cfg=select_method({**self.base,"epochs":3,"steps_per_epoch":2,"batch_size":8,
            "unlabeled_batch_size":8,"embedding_dim":16},"ce_2v_pseudo_soft")
        cfg.update(overrides)
        return cfg

    def assert_models_equal(self,a,b):
        for k,v in torch.load(a,weights_only=True).items():
            torch.testing.assert_close(v,torch.load(b,weights_only=True)[k],rtol=0,atol=0)

    def test_ema_average_corrected_decay_buffers_and_no_rng_or_parameter_alias(self):
        student=torch.nn.Sequential(torch.nn.Linear(2,2),torch.nn.BatchNorm1d(2))
        with torch.no_grad():
            for p in student.parameters():p.fill_(1.)
        ema=EMATeacher({"ema_teacher_decay":.99})
        rng=torch.get_rng_state().clone()
        teacher=ema.initialize(student)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(teacher.training)
        self.assertTrue(all(not p.requires_grad for p in teacher.parameters()))
        with torch.no_grad():
            for p in student.parameters():p.fill_(3.)
            student[1].running_mean.fill_(7.)
        ema.update(student)
        for p in teacher.parameters():torch.testing.assert_close(p,torch.full_like(p,2.))
        torch.testing.assert_close(teacher[1].running_mean,student[1].running_mean)
        with torch.no_grad():
            for p in student.parameters():p.fill_(5.)
        ema.update(student)
        for p in teacher.parameters():torch.testing.assert_close(p,torch.full_like(p,3.))
        pending=EMATeacher({})
        pending.load_state_dict(ema.state_dict())
        pending.initialize(student)
        for k,v in teacher.state_dict().items():torch.testing.assert_close(v,pending.model.state_dict()[k])

    def test_pair_graph_mixes_known_instance_and_pseudo_pairs_without_conflicting_negatives(self):
        labels=torch.tensor([0,0,1]); pred=torch.tensor([0,0,0]); conf=torch.tensor([.99,.98,.4],requires_grad=True)
        off,old=pair_graph(labels,pred,conf,{"contrastive_pseudo_pairs":False})
        on,new=pair_graph(labels,pred,conf,{"contrastive_pseudo_pairs":True})
        self.assertEqual(old["contrastive_instance_pairs"],12)
        self.assertEqual(old["contrastive_ll_pairs"],8)
        self.assertEqual(old["contrastive_lu_pairs"],0)
        self.assertEqual(new["contrastive_lu_pairs"],32)
        self.assertEqual(new["contrastive_uu_pairs"],8)
        self.assertEqual(new["contrastive_negative_pairs"],72)
        self.assertFalse(on.requires_grad)
        self.assertEqual(float(on.diag().sum()),0)
        self.assertEqual(float(on[6,9]),1.) # same U image, weak/strong
        self.assertAlmostEqual(float(on[6,7]),.99*.98,places=6)
        self.assertEqual(float(on[6,8]),0.) # rejected pseudo label
        self.assertEqual(float(on[0,2]),0.) # different true class

    def test_contrastive_loss_analytical_value_gradients_and_warmup(self):
        z=torch.tensor([[1.,0.],[1.,0.],[0.,1.],[0.,1.]],requires_grad=True)
        args=(z,torch.tensor([0]),torch.tensor([1]),torch.tensor([.99]),{"contrastive_warmup_epochs":0})
        loss,stats=mixed_contrastive_loss(*args,0)
        expected=torch.log1p(2*torch.exp(torch.tensor(-5.)))
        torch.testing.assert_close(loss,expected,atol=1e-6,rtol=1e-5)
        self.assertEqual(stats["contrastive_active_anchors"],4)
        loss.backward();self.assertTrue(torch.isfinite(z.grad).all())
        zero,_=mixed_contrastive_loss(z,*args[1:4],{"contrastive_warmup_epochs":3},2)
        self.assertEqual(float(zero.detach()),0.)

    def test_teacher_warmup_tracking_and_extra_evaluation_do_not_change_student(self):
        with tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(tmp)
            run_supervised(self.tiny(),root/"base",root,"cpu",smoke=True)
            r=run_supervised(self.tiny(ema_teacher=True,ema_teacher_warmup_epochs=3),root/"ema",root,"cpu",smoke=True)
            self.assert_models_equal(root/"base/model.pt",root/"ema/model.pt")
            self.assertEqual(r["ema_training_views"],0)
            self.assertEqual(r["ema_updates"],6)
            self.assertEqual(r["ema_validation_views"],60)
            self.assertTrue((root/"ema/teacher_model.pt").exists())

    def test_existing_batch_views_are_reused_and_teacher_is_separate_without_gradients(self):
        from torch.utils.data import DataLoader
        for teacher in [False,True]:
            cfg=self.tiny()
            if teacher:cfg.update(ema_teacher=True,ema_teacher_warmup_epochs=0)
            else:cfg.update(contrastive_weight=.05,contrastive_warmup_epochs=0,contrastive_pseudo_pairs=True)
            train,_,_=synthetic_datasets(cfg)
            pilot=PilotTraining(cfg,train,dict(train=list(range(40)),validation=list(range(40,60))))
            pilot.start_epoch(0)
            views,labels=next(iter(DataLoader(train,batch_size=8)))
            class Model(torch.nn.Module):
                def __init__(self):
                    super().__init__();self.encoder=torch.nn.Linear(3072,8);self.head=torch.nn.Linear(8,10);self.calls=[]
                def forward(self,x,normalize=False):
                    self.calls.append(len(x));z=self.encoder(x.flatten(1));return z,self.head(z)
            model=Model()
            loss,_,_,stats=pilot.step(model,views,labels,torch.device("cpu"))
            self.assertEqual(model.calls,[32])
            self.assertEqual(stats["processed_labeled_views"],16)
            self.assertEqual(stats["processed_unlabeled_views"],16)
            loss.backward()
            self.assertGreater(float(model.encoder.weight.grad.abs().sum()),0)
            if teacher:
                self.assertEqual(pilot.ema.model.calls,[8])
                self.assertTrue(all(p.grad is None for p in pilot.ema.model.parameters()))
                self.assertEqual(stats["ema_training_views"],8)

    def test_teacher_and_both_contrastive_variants_resume_exactly(self):
        cases=[dict(ema_teacher=True,ema_teacher_decay=.99,ema_teacher_warmup_epochs=1,pseudo_temperature=t) for t in [.5,1.]]
        cases += [dict(contrastive_weight=.05,contrastive_warmup_epochs=1,contrastive_pseudo_pairs=p,contrastive_pseudo_threshold=0.) for p in [False,True]]
        for changes in cases:
            with self.subTest(changes=changes),tempfile.TemporaryDirectory() as tmp,contextlib.redirect_stdout(io.StringIO()):
                root=Path(tmp);cfg=self.tiny(**changes)
                full=run_supervised(cfg,root/"full",root,"cpu",smoke=True)
                run_supervised(cfg,root/"resume",root,"cpu",smoke=True,stop_after_epoch=1)
                self.assertFalse((root/"resume/metrics.json").exists())
                resumed=run_supervised(cfg,root/"resume",root,"cpu",smoke=True,resume=True)
                self.assert_models_equal(root/"full/model.pt",root/"resume/model.pt")
                self.assertEqual(full["processed_labeled_views"],resumed["processed_labeled_views"])
                if changes.get("ema_teacher"):
                    self.assert_models_equal(root/"full/teacher_model.pt",root/"resume/teacher_model.pt")
                    self.assertEqual(full["ema_training_views"],32)
                    self.assertEqual(resumed["ema_training_views"],32)
                    self.assertEqual(full["primary_evaluation"],"student_last")
                    final_resume=root/"after_final_epoch"
                    final_resume.mkdir()
                    shutil.copy2(root/"full/latest.pt",final_resume/"latest.pt")
                    exported=run_supervised(cfg,final_resume,root,"cpu",smoke=True,resume=True)
                    self.assert_models_equal(root/"full/teacher_model.pt",final_resume/"teacher_model.pt")
                    self.assertEqual(exported["ema_updates"],full["ema_updates"])
                else:self.assertGreater(full["contrastive_active_anchors"],0)

    def test_plan_is_five_new_jobs_one_existing_baseline_and_validates_settings(self):
        plan=json.loads(Path("configs/cifar10_pilot_teacher_contrastive.json").read_text())
        jobs=build_jobs(plan)
        self.assertEqual(len(jobs),6)
        self.assertEqual(sum("reuse_run" not in j for j in jobs),5)
        self.assertEqual(sum(j["config"].get("ema_teacher",False) for j in jobs),2)
        self.assertEqual(sum(bool(j["config"].get("contrastive_weight")) for j in jobs),2)
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/"unstarted"
            with self.assertRaisesRegex(ValueError,"prepared only"):run_queue({**plan,"output_dir":str(target)})
            self.assertFalse(target.exists())
        for changed in [dict(ema_teacher=True,ema_teacher_decay=1.),dict(contrastive_weight=.05,contrastive_temperature=0.)]:
            with self.assertRaises(ValueError):validate_config(self.tiny(**changed))

    def test_common_reference_reorders_validation_and_never_fits_full_label_probe(self):
        from fewshot_metric.representations import analyze_run
        rng=np.random.default_rng(4)
        base=SimpleNamespace(data=rng.integers(0,256,(60,32,32,3),dtype=np.uint8),
                             targets=(np.arange(60)%10).tolist(),classes=[str(i) for i in range(10)])
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/"full";run.mkdir()
            cfg=self.tiny()
            (run/"config.json").write_text(json.dumps(cfg))
            (run/"splits.json").write_text(json.dumps(dict(train=list(range(20))+list(range(40,60)),validation=list(reversed(range(20,40))))))
            model=build_model("cnn",embedding_dim=16)
            torch.save(model.state_dict(),run/"model.pt")
            torch.save(model.state_dict(),run/"teacher_model.pt")
            with patch.dict(os.environ,{"MPLCONFIGDIR":str(root/"mpl")}),patch("fewshot_metric.representations.datasets.CIFAR10",return_value=base):
                r=analyze_run(run,root/"audit",tsne=False,reference_indices=list(range(20)),validation_indices=list(range(20,40)))
                teacher=analyze_run(run,root/"audit",tsne=False,reference_indices=list(range(20)),validation_indices=list(range(20,40)),model_file="teacher_model.pt")
                self.assertEqual(r["source_training_label_budget"],40)
                self.assertEqual(r["probe_training_labels"],20)
                self.assertEqual(teacher["run"],"full__teacher")
                with np.load(root/"audit/full/embeddings.npz") as data:
                    self.assertEqual(data["validation_indices"].tolist(),list(range(20,40)))
                with self.assertRaises(ValueError):analyze_run(run,root/"bad",reference_indices=[20])


if __name__=="__main__":unittest.main()
